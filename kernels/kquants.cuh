// kquants.cuh -- the k-quants: block layouts, the Q8_K quantizer, Q6_K and Q5_K.
//
// Included by kernels.cu, which pulls every file into one compile unit and one
// PTX module in dependency order. Kernel names are the PTX symbols the backend
// looks up, so which file a kernel lives in changes nothing at run time.
// Kernels are in `extern "C"` blocks; templates sit outside them, since a
// template cannot have C linkage.
#pragma once

extern "C" {

// ================================================================ the k-quants
//
// The 35B needs three matmuls the earlier two models did not: IQ4_XS on every
// routed expert, `token_embd` and the GDN projections; Q5_K on `attn_output`;
// Q6_K on `attn_q` and the LM head. ggml pairs all three with a **Q8_K**
// activation rather than Q8_0, which is why `quantize_q8_k` lives beside
// `quantize_q8_0` instead of replacing it -- one layer of the 35B feeds the
// same normed activation to a Q6_K matmul (`attn_q`) and two Q8_0 ones
// (`attn_k`, `attn_v`), so both quantizations are live at once.
//
// **All three are bit-identical to `ops::naive`.** The handoff for this work
// expected a derived tolerance; it is not needed, and the escape is the one
// `matmul_q8_0_warp` found, which generalizes further than it looked:
//
//     every k-quant dot is an INTEGER inner sum inside an f32 outer chain.
//
// The integer part cannot round, so its order is free and it can have the whole
// warp. The f32 chain is walked serially and ascending, exactly as the oracle
// walks it. Only the *shape* of that split differs per format:
//
//   Q6_K     8 f32 lanes x nb super-blocks, then an 8-way fold
//   Q5_K     the same, plus a separate `dmin` chain joined before the lanes
//   IQ4_XS   one chain of nb*8 terms -- the same length as Q8_0's at n_in 2048
//
// The integer sums provably cannot overflow i32, which is what makes them
// order-free rather than merely usually-order-free:
//
//   Q6_K    |scale| <= 127, |aux16| <= 32767, 32 terms  ->  1.3e8
//   Q5_K     scale <=  63, |aux16| <= 32767, 32 terms  ->  6.6e7
//   IQ4_XS  |q8| <= 127, |kvalue| <= 127, 32 terms     ->  5.2e5
//
// Two things had to be reproduced deliberately rather than falling out:
//
//  - **FMA is chosen per format, not per file.** `--fmad=false` turns nvcc's
//    contraction off globally, so fusion is opted back in by hand with
//    `__fmaf_rn` in exactly the two places the reference build fused: Q5_K's
//    `sums[l]` and `sumf` updates. Q6_K's `sums[l] +=` and IQ4_XS's `sumf +=`
//    look identical on the page and must NOT be fused. `src/quant/kquant.rs`
//    records the same asymmetry on the CPU side; it is a property of the
//    reference's compiler, not of the format.
//  - **The Q8_K scale's argmax breaks ties toward the lower index**, because
//    the reference scans linearly with a strict `>` and keeps the first winner.
//    A plain tree max is free to return either, and would move a whole block's
//    quants when it disagreed.

#define QK_K 256
// block_q6_K:   { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; f16 d; }
#define Q6K_BYTES 210
// block_q5_K:   { f16 d; f16 dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128] }
#define Q5K_BYTES 176
// block_iq4_xs: { f16 d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128] }
#define IQ4XS_BYTES 136

// `kvalues_iq4nl` from ggml/src/ggml-common.h, held in **registers** rather
// than `__constant__` memory. A non-uniform grid -- the "IQ" in IQ4_XS -- so a
// linear dequantization would be a different format.
//
// **This lookup was 72% of the IQ4_XS matmul.** Constant memory is
// broadcast-optimised: a read where lanes index different addresses serialises
// into up to 32 transactions, and every lane here looks up a different nibble,
// twice per byte, 64 times per output row. Decomposed against the real kernel
// at its real geometry: 143.9 us baseline, 40.6 us with only this lookup
// replaced by arithmetic, 36.9 us with all unpacking removed. The nibble split
// and the 6-bit scale assembly cost ~4 us between them; the table cost ~103.
//
// Sixteen signed bytes are 128 bits, so the whole grid fits in two registers
// and the lookup becomes a select and a shift. The values are identical, so
// this is bit-exact by inspection as well as by test.
//
// Two earlier attempts at this kernel -- an alignment repack and a restructured
// fold -- were aimed at the two things the decomposition later showed were free
// (loads cost 2 us of 144, the ordered fold 2 us).
__device__ __forceinline__ int kvalue_iq4nl(int i) {
    const unsigned long long w =
        (i < 8) ? 0xF6EADDCFBFAD9881ULL : 0x7159453526190D01ULL;
    return (int)(signed char)((w >> ((i & 7) * 8)) & 0xffULL);
}

// Round to nearest, ties to even -- `nearest_int` in ggml/src/ggml-quants.c.
//
// Adding 1.5 * 2^23 forces the fractional bits out of an f32 mantissa and IEEE
// addition resolves the tie to even while doing it. **This add must not be
// contracted with the multiply that feeds it**; `--fmad=false` guarantees that
// globally, and the caller uses `__fmul_rn` so the guarantee is local too.
__device__ __forceinline__ int nearest_int_dev(float fval) {
    float val = fval + 12582912.0f;
    return (__float_as_int(val) & 0x007fffff) - 0x00400000;
}

// Quantize an activation to Q8_K: one 256-thread block per super-block.
//
// Mirrors `quant::kquant::Q8KRow::from_f32`, which mirrors
// `quantize_row_q8_K_ref`. Four details are silent if wrong, and the module
// docs in kquant.rs carry them; the two that constrain *this* kernel are the
// argmax tie-break above and `bsums`, which only Q5_K reads -- so omitting them
// leaves two formats right and one subtly wrong.
__global__ void quantize_q8_k(int n_super, const float *__restrict__ x,
                              float *__restrict__ scales,
                              signed char *__restrict__ quants,
                              short *__restrict__ bsums) {
    __shared__ float s_av[QK_K];
    __shared__ int   s_ai[QK_K];
    __shared__ int   s_q[QK_K];

    const int sb = blockIdx.x;
    if (sb >= n_super) return;
    const int t = threadIdx.x;

    const float v  = x[(size_t)sb * QK_K + t];
    s_av[t] = fabsf(v);
    s_ai[t] = t;
    __syncthreads();

    // Tree argmax. Taking the right half only on a *strict* win keeps the
    // leftmost index at every level, so the winner is the lowest index holding
    // the maximum -- which is what the reference's linear `if (ax > amax)`
    // scan returns.
    for (int stride = QK_K / 2; stride > 0; stride >>= 1) {
        if (t < stride && s_av[t + stride] > s_av[t]) {
            s_av[t] = s_av[t + stride];
            s_ai[t] = s_ai[t + stride];
        }
        __syncthreads();
    }

    const float amax = s_av[0];
    if (amax == 0.0f) {
        quants[(size_t)sb * QK_K + t] = 0;
        if (t < QK_K / 16) bsums[(size_t)sb * (QK_K / 16) + t] = 0;
        if (t == 0) scales[sb] = 0.0f;
        return;
    }

    // The **signed** value at the largest magnitude, not the magnitude: the
    // reference divides -127 by it, so `iscale` is negative for a positive
    // extreme and the extreme element quantizes to exactly -127.
    const float mx = x[(size_t)sb * QK_K + s_ai[0]];
    const float iscale = -127.0f / mx;

    const int q = min(127, nearest_int_dev(__fmul_rn(iscale, v)));
    quants[(size_t)sb * QK_K + t] = (signed char)q;
    s_q[t] = q;
    __syncthreads();

    // Sums of 16 consecutive quants, accumulated in int and narrowed after --
    // the reference's `sum` is an int and only the store is int16.
    if (t < QK_K / 16) {
        int sum = 0;
        for (int k = 0; k < 16; ++k) sum += s_q[t * 16 + k];
        bsums[(size_t)sb * (QK_K / 16) + t] = (short)sum;
    }
    if (t == 0) scales[sb] = 1.0f / iscale;
}

// Q6_K x Q8_K, one warp per output row.
//
// Six bits per weight, split across a low nibble in `ql` and a high pair in
// `qh`, biased by -32. The 16 sub-block scales are plain int8.
//
// Lane layout is `lane = g*8 + l`. `l` is the oracle's f32 lane -- the fixed
// position within a 16-element sub-block that owns one of the eight `sums[]`
// accumulators -- and `g` splits the 16 sub-blocks four ways. The g-split is
// integer-only and therefore free; the l-split is the oracle's own.
//
// `blockIdx.y` is the batch token. Decode is `gridDim.y == 1` of the same
// kernel rather than a separate one, so prefill and decode cannot disagree.
__global__ void matmul_q6_k_q8_k(int n_in, int n_out,
                                 const unsigned char *__restrict__ w,
                                 const float *__restrict__ x_scales,
                                 const signed char *__restrict__ x_quants,
                                 float *__restrict__ out) {
    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    // Warp-uniform: every lane of a warp shares `j`, so the shuffles below
    // keep a full mask.
    if (j >= n_out) return;
    const int tok = blockIdx.y;

    const int l = lane & 7;
    const int g = lane >> 3;

    const float *xs = x_scales + (size_t)tok * nb;
    const signed char *xq = x_quants + (size_t)tok * n_in;
    const unsigned char *row = w + (size_t)j * nb * Q6K_BYTES;

    float sum_l = 0.0f;   // this lane's sums[l], serial and ascending in i

    for (int i = 0; i < nb; ++i) {
        const unsigned char *blk = row + (size_t)i * Q6K_BYTES;
        const unsigned char *ql = blk;
        const unsigned char *qh = blk + QK_K / 2;
        const signed char   *sc = (const signed char *)(blk + QK_K / 2 + QK_K / 4);
        const unsigned short d16 =
            (unsigned short)blk[Q6K_BYTES - 2] | ((unsigned short)blk[Q6K_BYTES - 1] << 8);
        const signed char *q8 = xq + (size_t)i * QK_K;

        int aux = 0;
        for (int jj = g * 4; jj < g * 4 + 4; ++jj) {
            const int scale = (int)sc[jj];
            for (int half = 0; half < 2; ++half) {
                const int idx = jj * 16 + half * 8 + l;

                // The reference unpacks all 256 weights up front in two
                // 128-element halves; this is that mapping inverted, so a lane
                // materializes only the eight it needs.
                const int j2  = idx >> 7;
                const int r   = idx & 127;
                const int sub = r >> 5;
                const int l2  = r & 31;
                const unsigned char h = qh[j2 * 32 + l2];
                int base, hb;
                if (sub == 0)      { base = ql[j2 * 64 + l2]      & 0xF; hb = (h >> 0) & 3; }
                else if (sub == 1) { base = ql[j2 * 64 + l2 + 32] & 0xF; hb = (h >> 2) & 3; }
                else if (sub == 2) { base = ql[j2 * 64 + l2]      >> 4;  hb = (h >> 4) & 3; }
                else               { base = ql[j2 * 64 + l2 + 32] >> 4;  hb = (h >> 6) & 3; }
                const int av = (base | (hb << 4)) - 32;

                // int16 in the reference. The product of an int8 quant and a
                // -32..31 weight cannot leave that range, so the narrowing is
                // faithful rather than lossy.
                const short aux16 = (short)((int)q8[idx] * av);
                aux += scale * (int)aux16;
            }
        }
        // Fold the four g-groups. Integer, so exact and order-free.
        aux += __shfl_down_sync(0xffffffff, aux, 16);
        aux += __shfl_down_sync(0xffffffff, aux, 8);

        if (lane < 8) {
            const float d = h2f(d16) * xs[i];
            // NOT fused. The reference's compiler contracts Q5_K's identical
            // line and not this one.
            sum_l += d * (float)aux;
        }
    }

    // The 8-way fold, serial and ascending -- the oracle's last loop. The
    // shuffle is outside the branch because every lane must reach it.
    float sumf = 0.0f;
    for (int k = 0; k < 8; ++k) {
        const float v = __shfl_sync(0xffffffff, sum_l, k);
        if (lane == 0) sumf += v;
    }
    if (lane == 0) out[(size_t)tok * n_out + j] = sumf;
}

// Tokens per weight load in the batched Q6_K matmul.
//
// **The unpack is the cost, and it does not depend on the token.** Six-bit
// weights arrive as a low nibble in `ql` and a high pair in `qh`, selected by a
// four-way branch on the sub-block -- a dozen instructions per weight -- and
// `matmul_q6_k_q8_k` ran the whole sequence again for every token of the batch,
// because its `blockIdx.y` is the token. At 512 tokens that is 512 unpacks of
// each weight to serve 512 outputs.
//
// Holding the unpacked value across `Q6K_TOK` tokens divides both the unpack
// count and the weight traffic by the same factor. 8 keeps `aux` and `sum_l` at
// eight registers each; the kernel is otherwise identical to the one-token one.
//
// Worth **+13.2% of whole-model prefill** on the 35B, 360.3 -> 407.7 tok/s at
// 2,240 tokens, from one bit-exact kernel.
//
// **The same change on Q5_K is written and unresolved, not rejected.** At 2,240
// tokens it measured 399 against this kernel's 402 -- inside a within-batch
// spread of 2% -- but at 11,237 tokens it measured **380.9 against 366.2**, and
// that is the depth a real session runs at. One run each, so neither settles
// it; it was reverted on the shallow number before the deep one existed, which
// is precisely the error the rest of this session was spent learning.
//
// There is a reason to expect it to gain less either way: Q5_K's unpack is a
// nibble select plus one `qh` bit, about six instructions where Q6_K's four-way
// branch over a split nibble and bit-pair is about twelve, and its twelve-byte
// scale shuffle was already amortised per super-block. Half as much to remove,
// against the same eight extra accumulator sets. That is the shape of the rule
// this file keeps confirming: the gain is proportional to the *instructions*
// deleted, not the traffic -- 09-09 cut A traffic 4x and B traffic 2x for
// nothing.
#define Q6K_TOK 8

// Q6_K x Q8_K over a batch, one warp per output row and `Q6K_TOK` tokens.
//
// **Bit-identical to `matmul_q6_k_q8_k` by construction, not by measurement.**
// Each output still keeps the oracle's eight interleaved f32 accumulators, in
// the same lane, folded in the same ascending order; the only change is how
// many outputs one weight load serves. Nothing is reassociated, so this needs
// no tolerance and stays inside the exact set -- unlike an MMA form of the same
// kernel, which cannot keep those eight accumulators at all.
//
// Padding tokens are pointed at token 0 rather than branched around, so the
// inner loop is unconditional. They accumulate into `aux`/`sum_l` slots that the
// write-back never stores.
__global__ void matmul_q6_k_q8_k_tok(int n_in, int n_out, int n_tok,
                                     const unsigned char *__restrict__ w,
                                     const float *__restrict__ x_scales,
                                     const signed char *__restrict__ x_quants,
                                     float *__restrict__ out) {
    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;
    const int t0 = blockIdx.y * Q6K_TOK;

    const int l = lane & 7;
    const int g = lane >> 3;

    const unsigned char *row = w + (size_t)j * nb * Q6K_BYTES;

    // Safe token index per slot: out-of-range slots read token 0 and are
    // discarded at the write-back, which keeps the inner loop branch-free.
    int ts[Q6K_TOK];
#pragma unroll
    for (int t = 0; t < Q6K_TOK; ++t) ts[t] = (t0 + t < n_tok) ? (t0 + t) : 0;

    float sum_l[Q6K_TOK];
#pragma unroll
    for (int t = 0; t < Q6K_TOK; ++t) sum_l[t] = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const unsigned char *blk = row + (size_t)i * Q6K_BYTES;
        const unsigned char *ql = blk;
        const unsigned char *qh = blk + QK_K / 2;
        const signed char   *sc = (const signed char *)(blk + QK_K / 2 + QK_K / 4);
        const unsigned short d16 =
            (unsigned short)blk[Q6K_BYTES - 2] | ((unsigned short)blk[Q6K_BYTES - 1] << 8);

        int aux[Q6K_TOK];
#pragma unroll
        for (int t = 0; t < Q6K_TOK; ++t) aux[t] = 0;

        for (int jj = g * 4; jj < g * 4 + 4; ++jj) {
            const int scale = (int)sc[jj];
            for (int half = 0; half < 2; ++half) {
                const int idx = jj * 16 + half * 8 + l;

                // Unpacked once, then dotted against every token below. This is
                // the same mapping `matmul_q6_k_q8_k` uses, lifted out of the
                // token loop rather than changed.
                const int j2  = idx >> 7;
                const int r   = idx & 127;
                const int sub = r >> 5;
                const int l2  = r & 31;
                const unsigned char h = qh[j2 * 32 + l2];
                int base, hb;
                if (sub == 0)      { base = ql[j2 * 64 + l2]      & 0xF; hb = (h >> 0) & 3; }
                else if (sub == 1) { base = ql[j2 * 64 + l2 + 32] & 0xF; hb = (h >> 2) & 3; }
                else if (sub == 2) { base = ql[j2 * 64 + l2]      >> 4;  hb = (h >> 4) & 3; }
                else               { base = ql[j2 * 64 + l2 + 32] >> 4;  hb = (h >> 6) & 3; }
                const int av = (base | (hb << 4)) - 32;

#pragma unroll
                for (int t = 0; t < Q6K_TOK; ++t) {
                    const signed char *q8 =
                        x_quants + (size_t)ts[t] * n_in + (size_t)i * QK_K;
                    // int16 in the reference, and faithful: an int8 quant times
                    // a -32..31 weight cannot leave that range.
                    const short aux16 = (short)((int)q8[idx] * av);
                    aux[t] += scale * (int)aux16;
                }
            }
        }

        // Fold the four g-groups. Integer, so exact and order-free.
#pragma unroll
        for (int t = 0; t < Q6K_TOK; ++t) {
            aux[t] += __shfl_down_sync(0xffffffff, aux[t], 16);
            aux[t] += __shfl_down_sync(0xffffffff, aux[t], 8);
        }

        if (lane < 8) {
            const float dv = h2f(d16);
#pragma unroll
            for (int t = 0; t < Q6K_TOK; ++t) {
                // NOT fused, as in the one-token kernel: the reference's
                // compiler contracts Q5_K's identical line and not this one.
                const float d = dv * x_scales[(size_t)ts[t] * nb + i];
                sum_l[t] += d * (float)aux[t];
            }
        }
    }

    // The 8-way fold, serial and ascending -- the oracle's last loop. The
    // shuffle is outside the branch because every lane must reach it.
#pragma unroll
    for (int t = 0; t < Q6K_TOK; ++t) {
        float sumf = 0.0f;
        for (int k = 0; k < 8; ++k) {
            const float v = __shfl_sync(0xffffffff, sum_l[t], k);
            if (lane == 0) sumf += v;
        }
        if (lane == 0 && t0 + t < n_tok) {
            out[(size_t)(t0 + t) * n_out + j] = sumf;
        }
    }
}

// Q5_K x Q8_K, one warp per output row.
//
// Five bits per weight: four in a nibble of `qs`, the fifth as a bit-plane in
// `qh`, giving an **unsigned** 0..31 with no bias -- unlike Q6_K's -32. The
// offset lives instead in a per-sub-block `min`, subtracted through the
// activation's `bsums`, which is the only reason Q8_K carries them.
//
// Same `lane = g*8 + l` layout as Q6_K, with `g` splitting eight sub-blocks
// two apiece. The extra structure is the `dmin` chain: it accumulates into
// `sumf` inside the super-block loop, ahead of the lane fold, and both of its
// updates are fused where Q6_K's are not.
__global__ void matmul_q5_k_q8_k(int n_in, int n_out,
                                 const unsigned char *__restrict__ w,
                                 const float *__restrict__ x_scales,
                                 const signed char *__restrict__ x_quants,
                                 const short *__restrict__ x_bsums,
                                 float *__restrict__ out) {
    const unsigned int KMASK1 = 0x3f3f3f3fu;
    const unsigned int KMASK2 = 0x0f0f0f0fu;
    const unsigned int KMASK3 = 0x03030303u;

    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;
    const int tok = blockIdx.y;

    const int l = lane & 7;
    const int g = lane >> 3;

    const float *xs = x_scales + (size_t)tok * nb;
    const signed char *xq = x_quants + (size_t)tok * n_in;
    const short *xb = x_bsums + (size_t)tok * nb * (QK_K / 16);
    const unsigned char *row = w + (size_t)j * nb * Q5K_BYTES;

    float sum_l = 0.0f;   // sums[l]
    float sumf  = 0.0f;   // the dmin chain, lane 0's

    for (int i = 0; i < nb; ++i) {
        const unsigned char *blk = row + (size_t)i * Q5K_BYTES;
        const unsigned short d16    = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        const unsigned short dmin16 = (unsigned short)blk[2] | ((unsigned short)blk[3] << 8);
        const unsigned char *qh = blk + 16;
        const unsigned char *qs = blk + 16 + QK_K / 8;

        // 12 bytes -> 8 six-bit scales and 8 six-bit mins, through a fixed
        // shuffle that is easier to transcribe than to re-derive. Recomputed
        // per lane rather than staged in shared memory: a dozen integer ops
        // against a barrier. The block base is 16-byte aligned and these sit
        // at +4, +8, +12, so the 32-bit loads are legal.
        unsigned int u0 = *(const unsigned int *)(blk + 4);
        unsigned int u1 = *(const unsigned int *)(blk + 8);
        unsigned int u2 = *(const unsigned int *)(blk + 12);
        const unsigned int u3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
        const unsigned int uaux = u1 & KMASK1;
        u1 = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
        u2 = uaux;
        u0 &= KMASK1;
        // scales = bytes of (u0, u1); mins = bytes of (u2, u3).

        const signed char *q8 = xq + (size_t)i * QK_K;

        int aux = 0;
        for (int jj = g * 2; jj < g * 2 + 2; ++jj) {
            const unsigned int su = (jj < 4) ? u0 : u1;
            const int scale = (int)((su >> ((jj & 3) * 8)) & 0xff);
            for (int q = 0; q < 4; ++q) {
                const int idx = jj * 32 + q * 8 + l;

                const int jj4  = idx >> 6;
                const int r    = idx & 63;
                const int half = r >> 5;
                const int l2   = r & 31;
                const unsigned char q4 = qs[jj4 * 32 + l2];
                const int base = (half == 0) ? (q4 & 0xF) : (q4 >> 4);
                // `m` marches one bit per 32-element run: 1 << (j*2 + half).
                const int av = base + (((qh[l2] >> (jj4 * 2 + half)) & 1) ? 16 : 0);

                const short aux16 = (short)((int)q8[idx] * av);
                aux += scale * (int)aux16;
            }
        }
        aux += __shfl_down_sync(0xffffffff, aux, 16);
        aux += __shfl_down_sync(0xffffffff, aux, 8);

        if (lane < 8) {
            const float d = h2f(d16) * xs[i];
            // FUSED. `fma(d, aux32[l], sums[l])` in the compiled reference.
            sum_l = __fmaf_rn(d, (float)aux, sum_l);
        }
        if (lane == 0) {
            const short *bs = xb + (size_t)i * (QK_K / 16);
            int sumi = 0;
            for (int k = 0; k < QK_K / 16; ++k) {
                const int mk = k >> 1;
                const unsigned int mu = (mk < 4) ? u2 : u3;
                sumi += (int)bs[k] * (int)((mu >> ((mk & 3) * 8)) & 0xff);
            }
            const float dmin = h2f(dmin16) * xs[i];
            // FUSED, and inside the loop, before the lanes are folded in:
            // `fma(-dmin, sumi, sumf)`, not a multiply followed by a subtract.
            sumf = __fmaf_rn(-dmin, (float)sumi, sumf);
        }
    }

    for (int k = 0; k < 8; ++k) {
        const float v = __shfl_sync(0xffffffff, sum_l, k);
        if (lane == 0) sumf += v;
    }
    if (lane == 0) out[(size_t)tok * n_out + j] = sumf;
}

// Tokens per weight load in the batched Q5_K matmul. Same reasoning and the
// same figure as `Q6K_TOK`: the five-bit unpack and the twelve-byte scale/min
// shuffle both depend only on the weight, and `matmul_q5_k_q8_k` ran them again
// for every token because its `blockIdx.y` is the token.
#define Q5K_TOK 8

// Q5_K x Q8_K over a batch, one warp per output row and `Q5K_TOK` tokens.
//
// **Bit-identical to `matmul_q5_k_q8_k` by construction.** Both of its f32
// chains are preserved per token and in order: the `dmin` chain accumulates
// into `sumf` inside the super-block loop, ahead of the eight-lane fold, and
// both of its updates stay fused. The eight interleaved `sums[l]` accumulators
// keep their lane and their ascending order. Only how many outputs one weight
// load serves changes.
//
// The token-independent work hoisted out of the token loop is the whole point:
// the `u0..u3` shuffle -- a dozen integer ops that turn 12 bytes into 8 scales
// and 8 mins -- and the per-index `av`, which reads a nibble of `qs` and a bit
// of `qh`. The mins are token-independent too, but `sumi` folds them against
// the token's own `bsums`, so that stays per token.
//
// Padding tokens read token 0 and are discarded at the write-back, as in
// `matmul_q6_k_q8_k_tok`, so the inner loop carries no branch.
__global__ void matmul_q5_k_q8_k_tok(int n_in, int n_out, int n_tok,
                                     const unsigned char *__restrict__ w,
                                     const float *__restrict__ x_scales,
                                     const signed char *__restrict__ x_quants,
                                     const short *__restrict__ x_bsums,
                                     float *__restrict__ out) {
    const unsigned int KMASK1 = 0x3f3f3f3fu;
    const unsigned int KMASK2 = 0x0f0f0f0fu;
    const unsigned int KMASK3 = 0x03030303u;

    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;
    const int t0 = blockIdx.y * Q5K_TOK;

    const int l = lane & 7;
    const int g = lane >> 3;

    int ts[Q5K_TOK];
#pragma unroll
    for (int t = 0; t < Q5K_TOK; ++t) ts[t] = (t0 + t < n_tok) ? (t0 + t) : 0;

    const unsigned char *row = w + (size_t)j * nb * Q5K_BYTES;

    float sum_l[Q5K_TOK];   // sums[l], per token
    float sumf[Q5K_TOK];    // the dmin chain, lane 0's, per token
#pragma unroll
    for (int t = 0; t < Q5K_TOK; ++t) {
        sum_l[t] = 0.0f;
        sumf[t]  = 0.0f;
    }

    for (int i = 0; i < nb; ++i) {
        const unsigned char *blk = row + (size_t)i * Q5K_BYTES;
        const unsigned short d16    = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        const unsigned short dmin16 = (unsigned short)blk[2] | ((unsigned short)blk[3] << 8);
        const unsigned char *qh = blk + 16;
        const unsigned char *qs = blk + 16 + QK_K / 8;

        // Token-independent, and this is what used to run once per token.
        unsigned int u0 = *(const unsigned int *)(blk + 4);
        unsigned int u1 = *(const unsigned int *)(blk + 8);
        unsigned int u2 = *(const unsigned int *)(blk + 12);
        const unsigned int u3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
        const unsigned int uaux = u1 & KMASK1;
        u1 = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
        u2 = uaux;
        u0 &= KMASK1;

        int aux[Q5K_TOK];
#pragma unroll
        for (int t = 0; t < Q5K_TOK; ++t) aux[t] = 0;

        for (int jj = g * 2; jj < g * 2 + 2; ++jj) {
            const unsigned int su = (jj < 4) ? u0 : u1;
            const int scale = (int)((su >> ((jj & 3) * 8)) & 0xff);
            for (int q = 0; q < 4; ++q) {
                const int idx = jj * 32 + q * 8 + l;

                const int jj4  = idx >> 6;
                const int r    = idx & 63;
                const int half = r >> 5;
                const int l2   = r & 31;
                const unsigned char q4 = qs[jj4 * 32 + l2];
                const int base = (half == 0) ? (q4 & 0xF) : (q4 >> 4);
                const int av = base + (((qh[l2] >> (jj4 * 2 + half)) & 1) ? 16 : 0);

#pragma unroll
                for (int t = 0; t < Q5K_TOK; ++t) {
                    const signed char *q8 =
                        x_quants + (size_t)ts[t] * n_in + (size_t)i * QK_K;
                    const short aux16 = (short)((int)q8[idx] * av);
                    aux[t] += scale * (int)aux16;
                }
            }
        }

#pragma unroll
        for (int t = 0; t < Q5K_TOK; ++t) {
            aux[t] += __shfl_down_sync(0xffffffff, aux[t], 16);
            aux[t] += __shfl_down_sync(0xffffffff, aux[t], 8);
        }

        if (lane < 8) {
            const float dv = h2f(d16);
#pragma unroll
            for (int t = 0; t < Q5K_TOK; ++t) {
                const float d = dv * x_scales[(size_t)ts[t] * nb + i];
                // FUSED, as in the one-token kernel.
                sum_l[t] = __fmaf_rn(d, (float)aux[t], sum_l[t]);
            }
        }
        // **The min chain, spread over sixteen lanes instead of one.**
        //
        // This was `if (lane == 0)` wrapping a 16-iteration loop per token:
        // 128 serial steps a superblock with 31 lanes masked off, against 64
        // MACs a lane in the main loop above. A warp runs at its slowest lane,
        // so the whole warp was paying for it.
        //
        // **`sumi` is an integer sum of sixteen products, so any order gives
        // the same integer** — a tree across lanes is exact rather than merely
        // close, which is the same argument that lets the MMA path split a
        // sub-block. The f32 chain it feeds is untouched: still
        // `fma(-dmin, sumi, sumf)`, still on lane 0, still once per superblock
        // in ascending order.
        {
            const float dmv = h2f(dmin16);
            const int k = lane & 15;
            const int mk = k >> 1;
            const unsigned int mu = (mk < 4) ? u2 : u3;
            const int mn = (int)((mu >> ((mk & 3) * 8)) & 0xff);
#pragma unroll
            for (int t = 0; t < Q5K_TOK; ++t) {
                const short *bs =
                    x_bsums + (size_t)ts[t] * nb * (QK_K / 16) + (size_t)i * (QK_K / 16);
                // Lanes 16-31 contribute zero and exist only to keep the
                // shuffles warp-synchronous.
                int sumi = (lane < 16) ? (int)bs[k] * mn : 0;
                sumi += __shfl_down_sync(0xffffffff, sumi, 8);
                sumi += __shfl_down_sync(0xffffffff, sumi, 4);
                sumi += __shfl_down_sync(0xffffffff, sumi, 2);
                sumi += __shfl_down_sync(0xffffffff, sumi, 1);
                if (lane == 0) {
                    const float dmin = dmv * x_scales[(size_t)ts[t] * nb + i];
                    // FUSED, and inside the loop, before the lanes are folded in.
                    sumf[t] = __fmaf_rn(-dmin, (float)sumi, sumf[t]);
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < Q5K_TOK; ++t) {
        for (int k = 0; k < 8; ++k) {
            const float v = __shfl_sync(0xffffffff, sum_l[t], k);
            if (lane == 0) sumf[t] += v;
        }
        if (lane == 0 && t0 + t < n_tok) {
            out[(size_t)(t0 + t) * n_out + j] = sumf[t];
        }
    }
}

}  // extern "C"
