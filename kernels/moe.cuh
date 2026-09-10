// moe.cuh -- the routed FFN: grouping by expert, the grouped matmuls, top-k, expert address gather and the weighted sum.
//
// Included by kernels.cu, which pulls every file into one compile unit and one
// PTX module in dependency order. Kernel names are the PTX symbols the backend
// looks up, so which file a kernel lives in changes nothing at run time.
// Kernels are in `extern "C"` blocks; templates sit outside them, since a
// template cannot have C linkage.
#pragma once

extern "C" {

// ---------------------------------------------------------------------------
// Grouping the routed FFN by expert
// ---------------------------------------------------------------------------
//
// **The measured reason.** `matmul_iq4_xs_q8_k_moe_glu` gives every (token,
// pick) pair its own `blockIdx.y` and re-reads that expert's weight row for it.
// A `MOE_CHUNK` of 128 tokens is 1,024 pairs landing on at most 256 experts, so
// each row crosses the bus about four times more often than it needs to.
//
// That this is worth fixing is a measurement, not an intuition: on a
// 2,936-token 35B prefill the kernel moves 1.14 GB in 7.88 ms -- **144 GB/s**,
// against 206.8 for the same shape with everything resident, and a 130.6 GB/s
// prediction from blending the two residency tiers at that run's 95.1/4.9
// split. It is bandwidth-bound, so removing re-reads is the whole win.
//
// The *dense* IQ4_XS matmul is not: token-tiling cut its traffic 8x for 1.54x,
// and its decomposition puts only 13% in weight loads against 24% in the
// ordered fold. Same dot product, opposite bound -- which is why grouping was
// wrongly discounted once already, by carrying the dense finding across.
//
// **Bit-exact by construction, for the fourth time.** This changes which block
// computes an output and which weight loads are shared, never how one output
// accumulates: every dot is still the arithmetic of `dot_iq4_xs_warp` over the
// same bytes in the same order. It is the argument that made batching free.

// Pairs per tile. Mean occupancy is `n_pair / n_expert` = 4 at `MOE_CHUNK` 128,
// so 8 puts all but the hot experts in one tile and captures essentially the
// whole 4x. Larger costs registers: `moe_glu_grouped` already carries two
// accumulator arrays and two unpacked weight sets.
#define MOE_TOK 8

// Tile width for the tensor-core routed FFN, and how many 8-token MMA tiles it
// is cut into.
//
// **Separate from `MOE_TOK` because the two kernels have opposite constraints.**
// The scalar grouped kernel holds a per-token register array, so 8 is already
// its ceiling; the MMA kernel holds only accumulators per sub-tile and wants the
// widest tile the routing can fill. At `MOE_CHUNK` 512 a chunk is 4,096 picks
// over 256 experts, so the mean expert collects 16 -- measured 5.17 at chunk
// 128, which is what capped the first version.
#define MOE_MMA_TOK 16
#define MOE_MMA_NTILE 2

// Sort a chunk's (token, pick) pairs by expert id, and cut the result into
// tiles a block can own.
//
// One block: the work is 256 counters over at most 1,024 pairs. `perm` holds
// the pair indices in ascending expert order; tile `t` covers `tile_n[t]` of
// them starting at `tile_first[t]`, all sharing one expert.
//
// # Deterministic on purpose
//
// A counting sort with an atomic cursor orders pairs within an expert by
// whichever thread arrives first. Nothing numeric depends on it -- each pair is
// an independent output row -- but a permutation that varies between runs makes
// a differential failure unreproducible, which is the property this repo spends
// the most to keep. So one thread owns one expert and scans pairs ascending.
//
// Shared memory is `n_pair + n_expert + 2*(n_expert+1)` ints, supplied by the
// caller: 7,176 bytes at the shapes this model uses.
extern "C" __global__ void moe_group(int n_pair, int n_expert, int e_tok,
                                     const int *__restrict__ ids,
                                     int *__restrict__ perm,
                                     int *__restrict__ tile_first,
                                     int *__restrict__ tile_n,
                                     int *__restrict__ n_tile) {
    extern __shared__ int shm[];
    int *sids  = shm;                    // n_pair
    int *count = sids + n_pair;          // n_expert
    int *scan  = count + n_expert;       // n_expert + 1, pair offsets
    int *tscan = scan + n_expert + 1;    // n_expert + 1, tile offsets

    for (int p = threadIdx.x; p < n_pair; p += blockDim.x) sids[p] = ids[p];
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) count[e] = 0;
    __syncthreads();

    for (int p = threadIdx.x; p < n_pair; p += blockDim.x) atomicAdd(&count[sids[p]], 1);
    __syncthreads();

    // Both prefix sums in one thread. `n_expert` is 256 and this is a single
    // block, so a scan network would cost more in barriers than the 512 adds it
    // removes.
    if (threadIdx.x == 0) {
        int off = 0, toff = 0;
        for (int e = 0; e < n_expert; ++e) {
            scan[e]  = off;   off  += count[e];
            tscan[e] = toff;  toff += (count[e] + e_tok - 1) / e_tok;
        }
        scan[n_expert]  = off;
        tscan[n_expert] = toff;
        *n_tile = toff;
    }
    __syncthreads();

    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        int c = scan[e];
        // A shared-memory broadcast: every lane of a warp reads the same `p`.
        for (int p = 0; p < n_pair; ++p) {
            if (sids[p] == e) perm[c++] = p;
        }
        int t = tscan[e], at = scan[e], left = count[e];
        while (left > 0) {
            const int take = left < e_tok ? left : e_tok;
            tile_first[t] = at;
            tile_n[t]     = take;
            ++t; at += take; left -= take;
        }
    }
}

// The routed FFN gate, up and SiLU gating, grouped by expert.
//
// `matmul_iq4_xs_q8_k_moe_glu` with the token tiling of
// `matmul_iq4_xs_q8_k_batch`, where the tile is "the tokens this expert was
// routed to" rather than "the next eight tokens of the batch". The weight bytes
// a lane needs are unpacked once per super-block and dotted against every token
// in the tile -- the reuse the per-pair form cannot have, because adjacent
// pairs are different experts.
//
// `n_tile` is read from device memory rather than taken as an argument, for the
// same reason the expert pointers are: it is a function of this chunk's
// routing, and an argument is baked in when a graph records it.
__global__ void matmul_iq4_xs_q8_k_moe_glu_grouped(
        int n_in, int n_ff, int n_used,
        const int *__restrict__ n_tile,
        const int *__restrict__ perm,
        const int *__restrict__ tile_first,
        const int *__restrict__ tile_n,
        const unsigned long long *__restrict__ gptrs,
        const unsigned long long *__restrict__ uptrs,
        const float *__restrict__ x_scales,
        const signed char *__restrict__ x_quants,
        float *__restrict__ out) {
    const int tl = blockIdx.y;
    if (tl >= *n_tile) return;

    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_ff) return;

    const int first = tile_first[tl];
    const int nt    = tile_n[tl];

    // Every pair here shares an expert, so one pointer serves the tile. Lanes
    // past `nt` re-read entry 0 rather than branching: the results are dropped.
    int pr[MOE_TOK], tk[MOE_TOK];
#pragma unroll
    for (int u = 0; u < MOE_TOK; ++u) {
        const int q = perm[first + (u < nt ? u : 0)];
        pr[u] = q;
        tk[u] = q / n_used;
    }

    const size_t off = (size_t)j * nb * IQ4XS_BYTES;
    const unsigned char *grow = (const unsigned char *)gptrs[pr[0]] + off;
    const unsigned char *urow = (const unsigned char *)uptrs[pr[0]] + off;

    // Lane layout is the single-token kernel, so ascending `t` is the order the
    // reference folds in.
    const int t    = lane >> 2;
    const int p    = lane & 3;
    const int ib   = (t >> 1) * 2;
    const int half = t & 1;

    float sg[MOE_TOK], su[MOE_TOK];
#pragma unroll
    for (int u = 0; u < MOE_TOK; ++u) { sg[u] = 0.0f; su[u] = 0.0f; }

    for (int ibl = 0; ibl < nb; ++ibl) {
        const unsigned char *gblk = grow + (size_t)ibl * IQ4XS_BYTES;
        const unsigned char *ublk = urow + (size_t)ibl * IQ4XS_BYTES;

        const float dg = h2f((unsigned short)gblk[0] | ((unsigned short)gblk[1] << 8));
        const float du = h2f((unsigned short)ublk[0] | ((unsigned short)ublk[1] << 8));
        const unsigned int shg = (unsigned int)gblk[2] | ((unsigned int)gblk[3] << 8);
        const unsigned int shu = (unsigned int)ublk[2] | ((unsigned int)ublk[3] << 8);

        const unsigned int hg  = shg >> (ib * 2);
        const unsigned int hu  = shu >> (ib * 2);
        const unsigned int lg  = gblk[4 + (ib >> 1)];
        const unsigned int lou = ublk[4 + (ib >> 1)];
        const int lsg = (half == 0) ? (int)((lg & 0xf) | ((hg << 4) & 0x30))
                                    : (int)((lg >> 4)  | ((hg << 2) & 0x30));
        const int lsu = (half == 0) ? (int)((lou & 0xf) | ((hu << 4) & 0x30))
                                    : (int)((lou >> 4)  | ((hu << 2) & 0x30));

        const unsigned char *gqs = gblk + 4 + QK_K / 64;
        const unsigned char *uqs = ublk + 4 + QK_K / 64;
        const int qo = ib * 16 + half * 16;
        const int ao = ib * 32 + half * 32;

        // **The reuse.** These four weight bytes of each of the two matrices
        // are read and unpacked once for the whole tile.
        int gvlo[4], gvhi[4], uvlo[4], uvhi[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const unsigned char bg = gqs[qo + p * 4 + k];
            const unsigned char bu = uqs[qo + p * 4 + k];
            gvlo[k] = kvalue_iq4nl(bg & 0xf);
            gvhi[k] = kvalue_iq4nl(bg >> 4);
            uvlo[k] = kvalue_iq4nl(bu & 0xf);
            uvhi[k] = kvalue_iq4nl(bu >> 4);
        }

        // **Fully unrolled with a predicate, not bounded by `nt`.** A runtime
        // bound makes `u` a non-constant index, so ptxas puts `pr`, `tk` and
        // the accumulators in local memory -- 128 bytes of stack frame, which
        // it does not report as a spill. Measured: that cost 9% of prefill and
        // made the first version of this change a net loss.
#pragma unroll
        for (int u = 0; u < MOE_TOK; ++u) {
            if (u >= nt) continue;
            const signed char *q8 =
                x_quants + (size_t)tk[u] * n_in + (size_t)ibl * QK_K;
            // s1 and s2 in the reference, summed together: both integer, so
            // joining them cannot round.
            int gs = 0, us = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int a0 = (int)q8[ao + p * 4 + k];
                const int a1 = (int)q8[ao + 16 + p * 4 + k];
                gs += a0 * gvlo[k]; gs += a1 * gvhi[k];
                us += a0 * uvlo[k]; us += a1 * uvhi[k];
            }
            gs += __shfl_down_sync(0xffffffff, gs, 2);
            gs += __shfl_down_sync(0xffffffff, gs, 1);
            us += __shfl_down_sync(0xffffffff, us, 2);
            us += __shfl_down_sync(0xffffffff, us, 1);

            // `d4d8 = d * xs[ibl]` then `dh = d4d8 * (ls - 32)`, grouped as the
            // reference groups it, and not fused -- `--fmad=false` is global.
            const float xs = x_scales[(size_t)tk[u] * nb + ibl];
            const float gd4d8 = dg * xs;
            const float ud4d8 = du * xs;
            const float gdh = gd4d8 * (float)(lsg - 32);
            const float udh = ud4d8 * (float)(lsu - 32);
            const float gterm = gdh * (float)gs;
            const float uterm = udh * (float)us;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float vg = __shfl_sync(0xffffffff, gterm, k * 4);
                const float vu = __shfl_sync(0xffffffff, uterm, k * 4);
                if (lane == 0) { sg[u] += vg; su[u] += vu; }
            }
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int u = 0; u < MOE_TOK; ++u) {
            if (u >= nt) continue;
            const float g = sg[u];
            out[(size_t)pr[u] * n_ff + j] = g / (1.0f + expf(-g)) * su[u];
        }
    }
}

// The routed FFN `down` matmul, grouped by expert.
//
// The one asymmetry with the gate/up half above: there every expert of a token
// reads that token's activation, so `x` is indexed by token; here each pair has
// its own `n_ff`-wide intermediate, so `x` is indexed by the pair. Getting it
// backwards is silent at `n_tok == 1`, where the two coincide -- which is what
// `batched_moe_prefill_equals_token_by_token` exists to catch.
__global__ void matmul_iq4_xs_q8_k_moe_grouped(
        int n_in, int n_out,
        const int *__restrict__ n_tile,
        const int *__restrict__ perm,
        const int *__restrict__ tile_first,
        const int *__restrict__ tile_n,
        const unsigned long long *__restrict__ wptrs,
        const float *__restrict__ x_scales,
        const signed char *__restrict__ x_quants,
        float *__restrict__ out) {
    const int tl = blockIdx.y;
    if (tl >= *n_tile) return;

    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;

    const int first = tile_first[tl];
    const int nt    = tile_n[tl];

    int pr[MOE_TOK];
#pragma unroll
    for (int u = 0; u < MOE_TOK; ++u) pr[u] = perm[first + (u < nt ? u : 0)];

    const unsigned char *row =
        (const unsigned char *)wptrs[pr[0]] + (size_t)j * nb * IQ4XS_BYTES;

    const int t    = lane >> 2;
    const int p    = lane & 3;
    const int ib   = (t >> 1) * 2;
    const int half = t & 1;

    float sumf[MOE_TOK];
#pragma unroll
    for (int u = 0; u < MOE_TOK; ++u) sumf[u] = 0.0f;

    for (int ibl = 0; ibl < nb; ++ibl) {
        const unsigned char *blk = row + (size_t)ibl * IQ4XS_BYTES;
        const float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        const unsigned int sh = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
        const unsigned char *qs = blk + 4 + QK_K / 64;

        const unsigned int h  = sh >> (ib * 2);
        const unsigned int lo = blk[4 + (ib >> 1)];
        const int ls = (half == 0) ? (int)((lo & 0xf) | ((h << 4) & 0x30))
                                   : (int)((lo >> 4)  | ((h << 2) & 0x30));

        const int qo = ib * 16 + half * 16;
        const int ao = ib * 32 + half * 32;

        int vlo[4], vhi[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const unsigned char b = qs[qo + p * 4 + k];
            vlo[k] = kvalue_iq4nl(b & 0xf);
            vhi[k] = kvalue_iq4nl(b >> 4);
        }

        // **Fully unrolled with a predicate, not bounded by `nt`.** A runtime
        // bound makes `u` a non-constant index, so ptxas puts `pr`, `tk` and
        // the accumulators in local memory -- 128 bytes of stack frame, which
        // it does not report as a spill. Measured: that cost 9% of prefill and
        // made the first version of this change a net loss.
#pragma unroll
        for (int u = 0; u < MOE_TOK; ++u) {
            if (u >= nt) continue;
            const signed char *q8 =
                x_quants + (size_t)pr[u] * n_in + (size_t)ibl * QK_K;
            int s = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                s += (int)q8[ao + p * 4 + k] * vlo[k];
                s += (int)q8[ao + 16 + p * 4 + k] * vhi[k];
            }
            s += __shfl_down_sync(0xffffffff, s, 2);
            s += __shfl_down_sync(0xffffffff, s, 1);

            const float d4d8 = d * x_scales[(size_t)pr[u] * nb + ibl];
            const float dh   = d4d8 * (float)(ls - 32);
            const float term = dh * (float)s;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float v = __shfl_sync(0xffffffff, term, k * 4);
                if (lane == 0) sumf[u] += v;
            }
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int u = 0; u < MOE_TOK; ++u) {
            if (u < nt) out[(size_t)pr[u] * n_out + j] = sumf[u];
        }
    }
}

// The whole tail of a routed FFN in one launch: weighted sum of the experts,
// Resolve this token's chosen experts to device addresses.
//
// `table` is one slot-table entry per expert of one `Experts` tensor -- 256 of
// them, fixed at first sight of that tensor and never moved -- and `ids` is
// what `moe_topk` chose. The result is the `n_used` addresses the expert
// matmuls dereference.
//
// **This is the indirection that makes a CUDA graph possible.** A kernel
// argument is baked into a graph node when the graph is recorded; a device
// buffer is read when it is replayed. Moving the expert pointers from the
// former to the latter is the whole difference between a graph that routes to
// last token's experts and one that routes to this token's.
extern "C" __global__ void moe_gather_ptrs(int n_used, int base, int n_tok,
                                           const unsigned long long *__restrict__ table,
                                           const int *__restrict__ ids,
                                           const int *__restrict__ vram,
                                           unsigned int *__restrict__ counts,
                                           unsigned long long *__restrict__ tally,
                                           unsigned long long *__restrict__ out) {
    // `n_tok * n_used` addresses, one per (token, pick), laid out token-major
    // so an expert matmul can index `wptrs[tok * n_used + e]`.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tok * n_used) return;
    const int e = i;
    const int id = ids[e];
    out[e] = table[id];

    // **Observation, not policy, and it is not optional.** The moment routing
    // moved onto the device the host stopped learning which experts were read,
    // so hit rate, host-read rate and the coverage distribution all silently
    // went to zero -- a working cache reporting nothing, which is the exact
    // failure `CLAUDE.md` catalogues four prior instances of. These two atomics
    // are the fifth one caught rather than shipped.
    //
    // Cost is 8 atomics per expert tensor per token, ~960 a token, against a
    // launch that already exists.
    atomicAdd(&counts[base + id], 1u);
    atomicAdd(&tally[vram[base + id] ? 0 : 1], 1ull);
}

// Top-k expert selection, on the device.
//
// **The one host decision left in a decode pass, and therefore the reason CUDA
// graphs are off for this model.** `moe_token` downloads the router's 256
// probabilities once per layer to choose eight experts on the CPU -- 40 syncs a
// token, in a backend that took bus crossings from 1329 to 5. A graph defers
// every kernel to `end_pass`, so that mid-pass read would return the *previous*
// token's probabilities and the model would route to the wrong experts:
// fluent-looking nonsense. Moving the decision here is what removes it.
//
// # It must reproduce the host selection exactly, and does
//
// `moe_token` runs `n_used` rounds, each scanning `e` ascending and taking a
// new best only on a *strict* `>`, so the lowest index survives a tie. Then it
// sums the chosen probabilities **in selection order** and divides each by
// `max(sum, 6.103515625e-5)`.
//
// Selection is exact under any decomposition: (value, index) with "higher value
// wins, tie to lower index" is a total order, and a max never rounds. So the
// tree reduction below is free of the usual reordering worry -- the same
// argument that made the Q8_0 warp matmul bit-identical, applied to a
// comparison rather than to an integer sum.
//
// The sum is *not* order-free, so thread 0 walks the eight picks serially in
// pick order, exactly as the host does. Eight additions on one thread is not
// worth splitting.
extern "C" __global__ void moe_topk(int n_expert, int n_used,
                                    const float *__restrict__ all_probs,
                                    int *__restrict__ all_ids,
                                    float *__restrict__ all_weights) {
    extern __shared__ unsigned char moe_topk_smem[];
    float *sv = (float *)moe_topk_smem;
    int *si = (int *)(sv + blockDim.x);

    // One block per token. Selection is independent per row -- a token's
    // experts depend only on its own probabilities -- so a prefill batch is
    // just a wider grid, and decode is `gridDim.x == 1` of the same kernel.
    const int tok = blockIdx.x;
    const float *probs = all_probs + (size_t)tok * n_expert;
    int *ids = all_ids + (size_t)tok * n_used;
    float *weights = all_weights + (size_t)tok * n_used;
    // `MAX` in `Cuda::moe_glu` and friends: the routed count this model uses is
    // 8, and every kernel downstream carries no more.
    __shared__ int picked[8];

    const int t = threadIdx.x;

    for (int r = 0; r < n_used; ++r) {
        // This thread's best over the experts it owns, skipping ones already
        // taken. Strided, so `n_expert` may exceed the block.
        float bv = 0.0f;
        int bi = -1;
        for (int e = t; e < n_expert; e += blockDim.x) {
            bool taken = false;
            for (int k = 0; k < r; ++k) {
                if (picked[k] == e) taken = true;
            }
            if (taken) continue;
            const float v = probs[e];
            // Strictly greater, so the lowest index survives a tie -- the rule
            // `moe_token` gets from scanning ascending.
            if (bi < 0 || v > bv) {
                bv = v;
                bi = e;
            }
        }
        sv[t] = bv;
        si[t] = bi;
        __syncthreads();

        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (t < s) {
                // Take the other half only if it is a real candidate and either
                // strictly larger, or an equal value at a lower index.
                const bool other_ok = si[t + s] >= 0;
                const bool mine_bad = si[t] < 0;
                const bool better =
                    other_ok && (mine_bad || sv[t + s] > sv[t] ||
                                 (sv[t + s] == sv[t] && si[t + s] < si[t]));
                if (better) {
                    sv[t] = sv[t + s];
                    si[t] = si[t + s];
                }
            }
            __syncthreads();
        }
        if (t == 0) picked[r] = si[0];
        __syncthreads();
    }

    if (t == 0) {
        // Serial, in pick order, from zero -- `Iterator::sum` on the host folds
        // left the same way, and this is the one part that would round
        // differently if it were split.
        float sum = 0.0f;
        for (int r = 0; r < n_used; ++r) sum += probs[picked[r]];
        // f16's smallest normal, guarding the division rather than the weights.
        // A ternary rather than `fmaxf` so a NaN sum yields the clamp, which is
        // what Rust's `f32::max` does.
        const float denom = sum > 6.103515625e-5f ? sum : 6.103515625e-5f;
        for (int r = 0; r < n_used; ++r) {
            ids[r] = picked[r];
            weights[r] = probs[picked[r]] / denom;
        }
    }
}

// the shared expert's sigmoid gate, and the write back into the layer's output
// row.
//
// Replaces `add_scaled_rows` + `add_scaled_sigmoid` + `scatter_chunks`, which
// were three launches per layer -- 120 of a token's 1455 -- to produce one
// vector. Per-launch cost on this platform is **20.7 us, measured**, so three
// kernels that each read and write the same 8 KB are worth removing on launch
// count alone, before the memory traffic they also save.
//
// Order is the oracle's: the experts summed in ascending pick order from zero,
// then the shared expert added last. `expf` is the only inexactness and it was
// already there.
extern "C" __global__ void moe_finish(int n, int n_used, int at,
                                      int n_tok,
                                      const float *__restrict__ scales,
                                      const float *__restrict__ rows,
                                      const float *__restrict__ shared,
                                      const float *__restrict__ logit,
                                      int logit_at,
                                      float *__restrict__ out) {
    // One thread per output element of the whole batch. Decode is
    // `n_tok == 1`, which is exactly the grid this had before.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n_tok) return;
    const int t = idx / n;
    const int j = idx - t * n;

    // From device memory rather than eight kernel arguments, for the same
    // reason the expert pointers are: `moe_topk` wrote them and the host never
    // saw them. The sum below is still serial and ascending, so the order the
    // oracle uses is untouched.
    //
    // Three indexings that were all zero at `n_tok == 1` and are all distinct
    // in a batch: this token's slice of the weights, its own block of expert
    // rows, and its own shared-expert row and gate logit.
    float v = 0.0f;
    for (int e = 0; e < n_used; ++e) {
        v += scales[t * n_used + e] * rows[((size_t)t * n_used + e) * n + j];
    }
    const float g = 1.0f / (1.0f + expf(-logit[logit_at + t]));
    v += shared[(size_t)t * n + j] * g;
    out[(size_t)at + (size_t)t * n + j] = v;
}

// Weighted sum of `n_rows` rows into one, in ascending row order.
//
// The MoE accumulation: `acc[j] = sum_e scale[e] * rows[e * n + j]`, replacing
// `n_rows` separate `add_scaled` launches over an accumulator that starts at
// zero. **Bit-identical to that loop**: the sum is walked serially and
// ascending exactly as the oracle walks its picks -- parallel over `j`, which
// the oracle already treats as independent, and serial over `e`, which it does
// not.
extern "C" __global__ void add_scaled_rows(int n, int n_rows, float s0, float s1,
                                           float s2, float s3, float s4, float s5,
                                           float s6, float s7,
                                           const float *__restrict__ rows,
                                           float *__restrict__ acc) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const float sc[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
    float a = 0.0f;
    for (int e = 0; e < n_rows; ++e) a += sc[e] * rows[(size_t)e * n + j];
    acc[j] = a;
}

}  // extern "C"
