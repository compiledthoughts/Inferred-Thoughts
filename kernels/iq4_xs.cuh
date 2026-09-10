// iq4_xs.cuh -- IQ4_XS on the scalar cores: the shared dot product and its matmuls.
//
// Included by kernels.cu, which pulls every file into one compile unit and one
// PTX module in dependency order. Kernel names are the PTX symbols the backend
// looks up, so which file a kernel lives in changes nothing at run time.
// Kernels are in `extern "C"` blocks; templates sit outside them, since a
// template cannot have C linkage.
#pragma once

extern "C" {

// IQ4_XS x Q8_K, one warp per output row.
//
// The odd one of the three. Its accumulation is a **single** `sumf`, not eight
// lanes -- each 32-element sub-block contributes `d * (sumi1 + sumi2)` straight
// into it -- so its rounding differs from Q5_K's and Q6_K's by construction.
// That single chain is `nb * 8` long, which at n_in 2048 is 64: exactly the
// length `matmul_q8_0_warp` keeps serial for 2-7%.
//
// Lane layout is `lane = t*4 + p`. `t` picks one of the eight sub-blocks the
// oracle folds in order, and `p` splits that sub-block's integer sum four ways.
// Each sub-block's 6-bit scale is split across `scales_l` and two bits marching
// through `scales_h`, and biased by -32.
// The IQ4_XS dot product, one warp cooperating on one output row.
//
// Factored out so the plain and the grouped matmul below share **one**
// implementation. Two copies of an accumulation this order-sensitive would
// drift, and the drift would be a handful of ulps in a kernel whose whole claim
// is bit-equality with the oracle.
//
// Returns the result on lane 0; other lanes' values are partial.
__device__ __forceinline__ float dot_iq4_xs_warp(
        int nb,
        const unsigned char *__restrict__ row,
        const float *__restrict__ xs,
        const signed char *__restrict__ xq,
        int lane) {
    const int t = lane >> 2;   // which of the eight sub-blocks
    const int p = lane & 3;    // which quarter of its integer sum

    // `t` is (ib, half) flattened, and the reference walks ib = 0,2,4,6 with
    // half = 0 then 1 -- so ascending `t` is the oracle's own order.
    const int ib   = (t >> 1) * 2;
    const int half = t & 1;

    float sumf = 0.0f;

    for (int ibl = 0; ibl < nb; ++ibl) {
        const unsigned char *blk = row + (size_t)ibl * IQ4XS_BYTES;
        const float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        const unsigned int sh = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
        const unsigned char *scales_l = blk + 4;
        const unsigned char *qs = blk + 4 + QK_K / 64;
        const float d4d8 = d * xs[ibl];
        const signed char *q8 = xq + (size_t)ibl * QK_K;

        // The reference shifts `h` right by 4 once per ib-pair, so at pair
        // ib/2 it has been shifted by 2*ib.
        const unsigned int h  = sh >> (ib * 2);
        const unsigned int lo = scales_l[ib >> 1];
        const int ls = (half == 0) ? (int)((lo & 0xf) | ((h << 4) & 0x30))
                                   : (int)((lo >> 4)  | ((h << 2) & 0x30));
        const float dh = d4d8 * (float)(ls - 32);

        const int qo = ib * 16 + half * 16;
        const int ao = ib * 32 + half * 32;

        // s1 and s2 in the reference, summed together here: both are integer,
        // so joining them cannot round.
        int s = 0;
        for (int k = p * 4; k < p * 4 + 4; ++k) {
            const unsigned char b = qs[qo + k];
            s += (int)q8[ao + k]      * kvalue_iq4nl(b & 0xf);
            s += (int)q8[ao + 16 + k] * kvalue_iq4nl(b >> 4);
        }
        s += __shfl_down_sync(0xffffffff, s, 2);
        s += __shfl_down_sync(0xffffffff, s, 1);
        // Lane t*4 now holds sumi1 + sumi2 for sub-block t.

        // The one serial chain. NOT fused -- unlike Q5_K and Q6_K, the
        // reference's compiler leaves this one as a multiply and an add.
        const float term = dh * (float)s;
        for (int k = 0; k < 8; ++k) {
            const float v = __shfl_sync(0xffffffff, term, k * 4);
            if (lane == 0) sumf += v;
        }
    }
    return sumf;
}

__global__ void matmul_iq4_xs_q8_k(int n_in, int n_out,
                                   const unsigned char *__restrict__ w,
                                   const float *__restrict__ x_scales,
                                   const signed char *__restrict__ x_quants,
                                   float *__restrict__ out) {
    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;
    const int tok = blockIdx.y;

    const float sumf = dot_iq4_xs_warp(
        nb,
        w + (size_t)j * nb * IQ4XS_BYTES,
        x_scales + (size_t)tok * nb,
        x_quants + (size_t)tok * n_in,
        lane);
    if (lane == 0) out[(size_t)tok * n_out + j] = sumf;
}

// IQ4_XS x Q8_K with the batch in registers — the token-reuse variant.
//
// **The single-token kernel above re-reads its weight row once per token**, and
// its own doc says so: "prefill weight traffic scales with the batch. Correct
// first, and the reuse variant arrives as a measured change against this
// baseline." Measured, on a 4,000-token 35B prefill: `matmul_iq4_xs_q8_k` is
// **23.9% of prefill device time**, the largest single kernel, moving ~1.03 GB
// of dense weight per token — 4.1 TB across that prompt.
//
// A warp loads its slice of a superblock once and dots it against `IQ4_TOK`
// tokens, so weight traffic falls by that factor. This is what
// `matmul_q8_0_batch` already does for Q8_0, which was 1.0% of the same run
// precisely because it does it.
//
// **No shared memory, unlike the Q8_0 pair.** That kernel segments its
// cross-block sum through shared memory because it defers partials; here lane 0
// already folds each superblock into a running total, so `IQ4_TOK` running
// totals live in registers and shared memory stays out of the occupancy
// question entirely.
//
// **Bit-identical to the single-token kernel**, and for the reason recorded on
// the Q8_0 one: adding a token axis changes which outputs share a weight load,
// never how one output accumulates. Each token still walks `ibl` ascending and
// folds the same eight lanes in the same order, so the sequence of f32
// additions reaching `sumf[u]` is exactly the sequence the unbatched kernel
// produces for that token. `the_batched_iq4_matmul_is_bit_identical` demands
// equal bits rather than a tolerance.
//
// The nibble table is two `u64` immediates in registers (see `kvalue_iq4nl`),
// so hoisting the unpack out of the token loop costs nothing — which matters,
// because a `__constant__` lookup here was once 72% of this kernel.
#define IQ4_TOK 8

__global__ void matmul_iq4_xs_q8_k_batch(int n_in, int n_out, int n_tok,
                                         const unsigned char *__restrict__ w,
                                         const float *__restrict__ x_scales,
                                         const signed char *__restrict__ x_quants,
                                         float *__restrict__ out) {
    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;

    const int t0 = blockIdx.y * IQ4_TOK;
    int nt = n_tok - t0;
    if (nt > IQ4_TOK) nt = IQ4_TOK;

    // Lane layout is the single-token kernel's: `t` picks one of the eight
    // sub-blocks the oracle folds in order, `p` splits its integer sum four
    // ways. Ascending `t` is the reference's own order.
    const int t    = lane >> 2;
    const int p    = lane & 3;
    const int ib   = (t >> 1) * 2;
    const int half = t & 1;

    // A running total per token, folded in ascending `ibl` exactly as the
    // unbatched kernel folds its single one.
    float sumf[IQ4_TOK];
#pragma unroll
    for (int u = 0; u < IQ4_TOK; ++u) sumf[u] = 0.0f;

    const unsigned char *row = w + (size_t)j * nb * IQ4XS_BYTES;

    for (int ibl = 0; ibl < nb; ++ibl) {
        const unsigned char *blk = row + (size_t)ibl * IQ4XS_BYTES;
        const float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        const unsigned int sh = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
        const unsigned char *scales_l = blk + 4;
        const unsigned char *qs = blk + 4 + QK_K / 64;

        const unsigned int h  = sh >> (ib * 2);
        const unsigned int lo = scales_l[ib >> 1];
        const int ls = (half == 0) ? (int)((lo & 0xf) | ((h << 4) & 0x30))
                                   : (int)((lo >> 4)  | ((h << 2) & 0x30));

        const int qo = ib * 16 + half * 16;
        const int ao = ib * 32 + half * 32;

        // **The whole point: this lane's four weight bytes, read once.** The
        // unbatched kernel re-reads them for every token.
        int vlo[4], vhi[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const unsigned char b = qs[qo + p * 4 + k];
            vlo[k] = kvalue_iq4nl(b & 0xf);
            vhi[k] = kvalue_iq4nl(b >> 4);
        }

        for (int u = 0; u < nt; ++u) {
            const signed char *q8 =
                x_quants + (size_t)(t0 + u) * n_in + (size_t)ibl * QK_K;
            // s1 and s2 in the reference, summed together: both integer, so
            // joining them cannot round.
            int s = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                s += (int)q8[ao + p * 4 + k] * vlo[k];
                s += (int)q8[ao + 16 + p * 4 + k] * vhi[k];
            }
            s += __shfl_down_sync(0xffffffff, s, 2);
            s += __shfl_down_sync(0xffffffff, s, 1);

            // `(d * xs) * (ls - 32)`, grouped as the reference groups it:
            // `d4d8 = d * xs[ibl]` then `dh = d4d8 * (ls - 32)`.
            const float d4d8 = d * x_scales[(size_t)(t0 + u) * nb + ibl];
            const float dh = d4d8 * (float)(ls - 32);
            // NOT fused, as the reference's compiler leaves it.
            const float term = dh * (float)s;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float v = __shfl_sync(0xffffffff, term, k * 4);
                if (lane == 0) sumf[u] += v;
            }
        }
    }

    if (lane == 0) {
        for (int u = 0; u < nt; ++u) {
            out[(size_t)(t0 + u) * n_out + j] = sumf[u];
        }
    }
}

// The routed FFN's matmul: **every expert a token visits, in one launch.**
//
// Measured, and the reason this exists: issuing the eight experts one at a time
// costs 18.03 ms/token across the whole expert stage, against 4.46 for the
// grouped form. The saving is **not** launch overhead -- host issue is only
// ~2 us against ~9 us of device time per launch. It is occupancy. A
// `{2048, 512}` matmul is 128 blocks of 128 threads on a 36-SM card and
// finishes before the machine is full; eight of them stacked on `blockIdx.y`
// have eight times the parallelism for the same bytes.
//
// `x_stride_super` is what lets one kernel serve both halves of the FFN:
//
//   gate and up   every expert reads the same activation, so 0
//   down          every expert reads its own 512-element intermediate, so nb
//
// which is the seam's "derive the count from the buffer" convention applied to
// the expert axis rather than the token axis.
//
// Bit-exact by construction, for the same reason batching was: this changes
// which outputs share a launch, never how one accumulates. Each output row is
// `dot_iq4_xs_warp` over the same bytes in the same order.
__global__ void matmul_iq4_xs_q8_k_moe(int n_in, int n_out, int x_stride_super,
                                       int n_pair,
                                       const unsigned long long *__restrict__ wptrs,
                                       const float *__restrict__ x_scales,
                                       const signed char *__restrict__ x_quants,
                                       float *__restrict__ out) {
    // **`blockIdx.y` is a (token, pick) pair, not a pick.** In decode there is
    // one token and this is the eight picks exactly as before; in a batch it is
    // `n_tok * n_used`, laid out token-major so pair `t * n_used + i` is
    // token `t`'s `i`-th expert -- which is the order `moe_gather_ptrs` already
    // writes `wptrs` in.
    //
    // Nothing else moves: `wptrs`, `x` and `out` are all indexed by the same
    // pair, so this is the batch convention applied to the expert axis. Bit
    // exact by construction, for the third time: it changes which outputs share
    // a launch, never how one accumulates.
    const int e = blockIdx.y;
    if (e >= n_pair) return;
    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;

    // **The expert's address comes from device memory, not from a kernel
    // argument**, which is the whole point. `moe_gather_ptrs` filled `wptrs`
    // from the slot table using ids `moe_topk` chose, so nothing in this launch
    // depends on the host having seen the router's output -- and a CUDA graph,
    // which bakes its arguments in at record time, still routes to the experts
    // this token actually picked.
    const unsigned char *w = (const unsigned char *)wptrs[e];

    const float sumf = dot_iq4_xs_warp(
        nb,
        w + (size_t)j * nb * IQ4XS_BYTES,
        x_scales + (size_t)e * x_stride_super,
        x_quants + (size_t)e * x_stride_super * QK_K,
        lane);
    if (lane == 0) out[(size_t)e * n_out + j] = sumf;
}

// The routed FFN's gate, up and SiLU-gating in **one launch**.
//
// Three kernels became one: `matmul_experts(gate)`, `matmul_experts(up)` and
// `silu_mul` over the pair. Measured at 3.0 + 2.7 launches per layer out of 38,
// so this is 80 of a token's 1522 -- and on this platform the unit of cost is
// the launch, because WDDM serializes submissions and the CUDA driver batches
// them by heuristic. llama.cpp reaches 40 tok/s here with graphs *disabled*
// (split buffers, see ggml_cuda_graph_check_compability), so fewer and larger
// launches is a proven route rather than a guess.
//
// It also removes a round trip through VRAM: the old form wrote `n_used * n_ff`
// floats twice and read them back to combine. Here `gate` and `up` for the same
// output element are both in registers, so only the result is stored.
//
// Bit-exactness is unchanged. Each dot is `dot_iq4_xs_warp` over the same bytes
// in the same order, and `silu` was already outside the exact set for the usual
// reason -- `expf` -- with its order untouched.
__global__ void matmul_iq4_xs_q8_k_moe_glu(int n_in, int n_ff, int n_pair,
                                           int n_used,
                                           const unsigned long long *__restrict__ gptrs,
                                           const unsigned long long *__restrict__ uptrs,
                                           const float *__restrict__ x_scales,
                                           const signed char *__restrict__ x_quants,
                                           float *__restrict__ out) {
    const int e = blockIdx.y;
    if (e >= n_pair) return;
    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_ff) return;

    // **The one asymmetry with the `down` matmul above.** There, every pair has
    // its own `n_ff`-wide intermediate, so `x` is indexed by the pair. Here the
    // gate and up projections read the token's activation, which all of that
    // token's `n_used` experts share -- so `x` is indexed by the *token*, and
    // the pair layout being token-major is what makes that a division.
    //
    // Getting this wrong is silent: at `n_tok == 1` the quotient is always 0
    // and every batch shape would read token 0's activation for the whole
    // prompt. `batched_moe_prefill_equals_token_by_token` is what catches it.
    const int tok = e / n_used;

    // As `matmul_iq4_xs_q8_k_moe`: both addresses come from device memory, so
    // this launch carries no trace of which experts were chosen.
    const unsigned char *gw = (const unsigned char *)gptrs[e];
    const unsigned char *uw = (const unsigned char *)uptrs[e];
    const size_t off = (size_t)j * nb * IQ4XS_BYTES;
    const float *xs = x_scales + (size_t)tok * nb;
    const signed char *xq = x_quants + (size_t)tok * nb * QK_K;

    // Every expert of *this token* reads the same activation here -- this is
    // the gate/up half of the FFN, before any per-expert intermediate exists.
    const float g = dot_iq4_xs_warp(nb, gw + off, xs, xq, lane);
    const float u = dot_iq4_xs_warp(nb, uw + off, xs, xq, lane);

    if (lane == 0) {
        // silu(g) * u, exactly as `silu_mul` computes it.
        out[(size_t)e * n_ff + j] = g / (1.0f + expf(-g)) * u;
    }
}

}  // extern "C"
