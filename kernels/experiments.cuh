// experiments.cuh -- opt-in measured negatives kept as controls: the IQ4_XS fold-once and staged-tile kernels.
//
// Included by kernels.cu, which pulls every file into one compile unit and one
// PTX module in dependency order. Kernel names are the PTX symbols the backend
// looks up, so which file a kernel lives in changes nothing at run time.
// Kernels are in `extern "C"` blocks; templates sit outside them, since a
// template cannot have C linkage.
#pragma once

extern "C" {

// The same kernel with the f32 fold deferred to once per superblock.
//
// **A probe first, a candidate second.** Six data-movement experiments on this
// kernel have measured nothing (`c8131e5`'s two volume cuts and its fold hoist,
// the staged tile at two shapes, and Q5_K's activation staging), while the one
// dependency change that was tried paid 1.47x. This asks whether what sits
// *between* the MMAs is the cost.
//
// The arithmetic that motivates it: at 2048x8192, n_tok 512, the real kernel
// issues 2,097,152 `mma.m16n8k32.s8` in ~2.1 ms across 36 SMs, which is 0.0115
// MMA per cycle per SM against a measured ceiling of 0.278 - **24x off, the 4.3%
// of peak the bench reports**. `ptxas` says 72 registers and no spills, so 28
// warps an SM fit and occupancy is not it either. What is left per MMA is the
// unpack and the fold.
//
// # What changes, and why it is exact right up to the last step
//
// The reference folds once per 32-weight sub-block:
//
//     acc += (d * xs) * (ls - 32) * s          for t = 0..7, ascending
//
// `(ls - 32) * s` is an **integer** product — `s` is at most 32*127*127 and
// `ls - 32` is within +-31, so eight of them sum to under 1.3e8 and cannot
// overflow int32, let alone round. So `sum_t (ls_t - 32) * s_t` is exact, and
// this kernel accumulates it in int32 and folds once:
//
//     acc += (d * xs) * (float)(sum_t (ls_t - 32) * s_t)
//
// That replaces eight (convert, multiply, multiply, add) chains per superblock
// with eight integer MACs and one f32 fold.
//
// # It is NOT bit-exact, and the reason is worth stating precisely
//
// `sum_t c * a_t` and `c * sum_t a_t` are the same real number and different
// f32 numbers: the reference rounds after every sub-block, this rounds once per
// superblock. The `a_t` are exact integers either way, so the error is a *fold
// order* change of the same family as the RMSNorm tree — and, as there, this
// one is the *more* accurate of the two, since it rounds 8x less often.
//
// It is therefore off by default and outside the exact set. Whether it ships is
// not a decision this kernel gets to make; `--iq4-fold-once` exists so the cost
// can be known before anyone is asked.
__global__ void dbg_iq4_mma_foldonce(int n_in, int n_out, int n_tok,
                                     const unsigned char *__restrict__ w,
                                     const float *__restrict__ x_scales,
                                     const signed char *__restrict__ x_quants,
                                     float *__restrict__ out) {
    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const int j0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 16;
    const int t0 = blockIdx.y * (8 * MMA_NTILE);
    if (j0 >= n_out) return;

    const int g = lane >> 2;
    const int q = lane & 3;

    const int row_a = j0 + g;
    const int row_b = j0 + g + 8;

    const unsigned char *ra = w + (size_t)row_a * nb * IQ4XS_BYTES;
    const unsigned char *rb = w + (size_t)row_b * nb * IQ4XS_BYTES;

    float acc[MMA_NTILE][4];
#pragma unroll
    for (int n = 0; n < MMA_NTILE; ++n) {
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[n][k] = 0.0f;
    }

    for (int ibl = 0; ibl < nb; ++ibl) {
        const unsigned char *ba = ra + (size_t)ibl * IQ4XS_BYTES;
        const unsigned char *bb = rb + (size_t)ibl * IQ4XS_BYTES;

        const float da = h2f((unsigned short)ba[0] | ((unsigned short)ba[1] << 8));
        const float db = h2f((unsigned short)bb[0] | ((unsigned short)bb[1] << 8));
        const unsigned int sha = (unsigned int)ba[2] | ((unsigned int)ba[3] << 8);
        const unsigned int shb = (unsigned int)bb[2] | ((unsigned int)bb[3] << 8);

        float xs[MMA_NTILE][2];
#pragma unroll
        for (int n = 0; n < MMA_NTILE; ++n) {
            const int c0 = t0 + n * 8 + 2 * q;
            const int c1 = c0 + 1;
            xs[n][0] = (c0 < n_tok) ? x_scales[(size_t)c0 * nb + ibl] : 0.0f;
            xs[n][1] = (c1 < n_tok) ? x_scales[(size_t)c1 * nb + ibl] : 0.0f;
        }

        // The whole superblock's scaled products, as exact integers.
        int acci[MMA_NTILE][4];
#pragma unroll
        for (int n = 0; n < MMA_NTILE; ++n) {
#pragma unroll
            for (int k = 0; k < 4; ++k) acci[n][k] = 0;
        }

        for (int t = 0; t < 8; ++t) {
            const int ib   = (t >> 1) * 2;
            const int half = t & 1;

            const int lsa = iq4_ls(sha, ba[4 + (ib >> 1)], ib, half) - 32;
            const int lsb = iq4_ls(shb, bb[4 + (ib >> 1)], ib, half) - 32;

            const int off = 4 + QK_K / 64 + t * 16 + q * 4;
            int a[4];
            unpack_iq4_pair(*(const int *)(ba + off), a[0], a[2]);
            unpack_iq4_pair(*(const int *)(bb + off), a[1], a[3]);

#pragma unroll
            for (int n = 0; n < MMA_NTILE; ++n) {
                const int tb = t0 + n * 8 + g;
                int b[2] = {0, 0};
                if (tb < n_tok) {
                    // `+ t * 32` is load-bearing: sub-block `t` covers
                    // activations [32t, 32t+32). Dropping it makes every
                    // sub-block re-read the same 32 values, which is both wrong
                    // and *faster* — a much smaller working set — so it reads
                    // as a speedup rather than as a failure.
                    const signed char *q8 =
                        x_quants + (size_t)tb * n_in + (size_t)ibl * QK_K + t * 32;
                    b[0] = *(const int *)(q8 + q * 4);
                    b[1] = *(const int *)(q8 + 16 + q * 4);
                }

                const int zero[4] = {0, 0, 0, 0};
                int s[4];
                mma_m16n8k32_s8(s, a, b, zero);

                // Integer, so exact and order-free. This is the whole change.
                acci[n][0] += lsa * s[0];
                acci[n][1] += lsa * s[1];
                acci[n][2] += lsb * s[2];
                acci[n][3] += lsb * s[3];
            }
        }

        // One f32 fold per superblock instead of eight.
#pragma unroll
        for (int n = 0; n < MMA_NTILE; ++n) {
            acc[n][0] += (da * xs[n][0]) * (float)acci[n][0];
            acc[n][1] += (da * xs[n][1]) * (float)acci[n][1];
            acc[n][2] += (db * xs[n][0]) * (float)acci[n][2];
            acc[n][3] += (db * xs[n][1]) * (float)acci[n][3];
        }
    }

#pragma unroll
    for (int n = 0; n < MMA_NTILE; ++n) {
        const int c0 = t0 + n * 8 + 2 * q;
        const int c1 = c0 + 1;
        if (c0 < n_tok) {
            if (row_a < n_out) out[(size_t)c0 * n_out + row_a] = acc[n][0];
            if (row_b < n_out) out[(size_t)c0 * n_out + row_b] = acc[n][2];
        }
        if (c1 < n_tok) {
            if (row_a < n_out) out[(size_t)c1 * n_out + row_a] = acc[n][1];
            if (row_b < n_out) out[(size_t)c1 * n_out + row_b] = acc[n][3];
        }
    }
}

// ---------------------------------------------------------------------------
// The staged tile: both operands through shared memory, in a canonical form
// ---------------------------------------------------------------------------
//
// `matmul_iq4_xs_q8_k_mma` reads both MMA operands straight from global, one
// fragment per lane. A warp's A load is therefore **eight scattered 16-byte
// segments** -- lanes 0-3 take 16 bytes of one row, lanes 4-7 of the next, and
// so on -- and B has the same shape over eight tokens. `c8131e5` measured what
// that costs by elimination: cutting A traffic 4x (`MMA_NTILE` 16) and B
// traffic 2x (`MMA_NROW` 2) were both a wash, so **volume is not the bound and
// access shape is what is left.**
//
// So this kernel changes shape rather than volume. Each warp stages one whole
// superblock of one row -- lane `i` reads `qs + 4i`, so the 32 lanes cover 128
// contiguous bytes in **one transaction** -- unpacks it to the canonical form
// once, and the MMA loop then reads only shared. Global latency leaves the
// inner loop entirely.
//
// # The canonical form, and where it deliberately differs from llama.cpp
//
// `load_tiles_iq4_xs` in `mmq.cuh` writes two things per row: the nibbles
// already mapped through `kvalues_iq4nl` and packed four to an int, and a
// single f32 scale `d * (ls - 32)`. After that its MMA loop is quant-agnostic,
// which is how twenty formats share one GEMM.
//
// **We keep the quants and split the scale.** Pre-folding `d * (ls - 32)`
// reassociates `(d * xs) * ls` into `(d * ls) * xs`, which `c8131e5` measured
// as outside the bit-exact set -- it was tried, it was worth -1 to -2.5%, and
// it changes the rounding. Storing `d` and `ls` separately costs 1 KiB of
// shared and keeps every bit. The MMA loop is still quant-agnostic; a Q5_K or
// Q6_K loader would fill the same three arrays.
//
// # Why the row stride is 68 ints and not 64
//
// A superblock is 64 ints, and a warp's A fragment for sub-block `t` reads
// `[row g][8t + q]` over `g = 0..7, q = 0..3`. At stride 64 every row starts on
// bank 0 and that is a 32-way conflict, which gives back exactly what the
// staging saves -- the same trap as `delta_rule_batch_shared`'s
// `head_k_dim + 1`, at a different modulus because this access is four wide.
//
// 68 mod 32 is 4, so row `g` starts at bank `4g` and the lanes read banks
// `(4g + q) mod 32`, covering all 32 exactly once. Conflict-free, for four
// ints of padding per row.
//
// # The trade this makes, stated because it is the risk
//
// `c8131e5` left two candidates, access shape **and occupancy**, and this
// improves the first by worsening the second: ~36 KiB of shared is one or two
// blocks an SM where the register-tiled kernel used none. That is the ordinary
// GEMM trade -- latency hidden by ILP across `ST_TILE_N / 8` accumulator sets
// rather than by more resident warps -- but it is a trade, so the tile
// dimensions are `#define`s and the bench sweeps them.

// Tile: weight rows x tokens per block. One warp per 16 rows, so
// `ST_TILE_M / 16` warps; `ST_TILE_N / 8` MMA tiles per warp.
#define ST_TILE_M 64
#define ST_TILE_N 64
#define ST_WARPS (ST_TILE_M / 16)
#define ST_THREADS (ST_WARPS * 32)
#define ST_NTILE (ST_TILE_N / 8)

// Ints per staged row: 64 payload, 4 pad. See the bank argument above.
#define ST_STRIDE 68

__global__ void matmul_iq4_xs_q8_k_staged(int n_in, int n_out, int n_tok,
                                          const unsigned char *__restrict__ w,
                                          const float *__restrict__ x_scales,
                                          const signed char *__restrict__ x_quants,
                                          float *__restrict__ out) {
    __shared__ int sa_q[ST_TILE_M][ST_STRIDE];
    __shared__ int sb_q[ST_TILE_N][ST_STRIDE];
    __shared__ float sa_d[ST_TILE_M];
    // Stride 9, not 8: at 8 rows `g` and `g + 4` land on one bank.
    __shared__ int sa_ls[ST_TILE_M][9];
    __shared__ float sb_xs[ST_TILE_N];

    const int nb = n_in / QK_K;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int j0 = blockIdx.x * ST_TILE_M;   // first weight row
    const int t0 = blockIdx.y * ST_TILE_N;   // first token
    if (j0 >= n_out) return;

    const int g = lane >> 2;
    const int q = lane & 3;

    // Shared rows this lane accumulates, and the global rows they stand for.
    const int sra = warp * 16 + g;
    const int srb = sra + 8;
    const int row_a = j0 + sra;
    const int row_b = j0 + srb;

    float acc[ST_NTILE][4];
#pragma unroll
    for (int n = 0; n < ST_NTILE; ++n) {
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[n][k] = 0.0f;
    }

    for (int ibl = 0; ibl < nb; ++ibl) {
        // --- stage A: the weights, unpacked into the canonical form ---
        //
        // One warp per row, lane `i` taking `qs + 4i`. Sub-block `t` and quad
        // `q` come from the lane index because the `qs` offset `8 + 16t + 4q`
        // is exactly `8 + 4 * (4t + q)`, so a linear walk is also the right
        // walk.
#pragma unroll
        for (int r0 = 0; r0 < ST_TILE_M; r0 += ST_WARPS) {
            const int r = r0 + warp;
            // Clamped rather than skipped: the read stays in bounds and the
            // write-back is what drops the row. A skipped row would leave the
            // previous superblock's quants staged.
            const int jr = min(j0 + r, n_out - 1);
            const unsigned char *ba = w + ((size_t)jr * nb + ibl) * IQ4XS_BYTES;

            const int t = lane >> 2;
            const int lq = lane & 3;
            int lo, hi;
            unpack_iq4_pair(*(const int *)(ba + 4 + QK_K / 64 + lane * 4), lo, hi);
            sa_q[r][t * 8 + lq] = lo;
            sa_q[r][t * 8 + 4 + lq] = hi;

            if (lane < 8) {
                const unsigned int sh = (unsigned int)ba[2] | ((unsigned int)ba[3] << 8);
                const int ib = (lane >> 1) * 2;
                sa_ls[r][lane] = iq4_ls(sh, ba[4 + (ib >> 1)], ib, lane & 1);
                if (lane == 0) {
                    sa_d[r] = h2f((unsigned short)ba[0] | ((unsigned short)ba[1] << 8));
                }
            }
        }

        // --- stage B: the activations, already s8 and already in order ---
        //
        // Q8_K quants are 4-per-int in ascending `k`, which is the fragment
        // layout, so this is a copy. Two warps per token, 128 contiguous bytes
        // each.
#pragma unroll
        for (int c0 = 0; c0 < ST_TILE_N; c0 += ST_THREADS / 64) {
            const int c = c0 + (tid >> 6);
            const int j = tid & 63;
            const int tc = min(t0 + c, n_tok - 1);
            const signed char *q8 = x_quants + (size_t)tc * n_in + (size_t)ibl * QK_K;
            sb_q[c][j] = *(const int *)(q8 + j * 4);
            if (j == 0) {
                sb_xs[c] = x_scales[(size_t)tc * nb + ibl];
            }
        }

        __syncthreads();

        // --- the MMA loop, reading only shared ---
        //
        // Superblock-invariant, so hoisted out of the sub-block loop exactly as
        // the register-tiled kernel hoists them.
        const float da = sa_d[sra];
        const float db = sa_d[srb];
        float xs[ST_NTILE][2];
#pragma unroll
        for (int n = 0; n < ST_NTILE; ++n) {
            xs[n][0] = sb_xs[n * 8 + 2 * q];
            xs[n][1] = sb_xs[n * 8 + 2 * q + 1];
        }

#pragma unroll
        for (int t = 0; t < 8; ++t) {
            int a[4];
            a[0] = sa_q[sra][t * 8 + q];
            a[2] = sa_q[sra][t * 8 + 4 + q];
            a[1] = sa_q[srb][t * 8 + q];
            a[3] = sa_q[srb][t * 8 + 4 + q];

            const int lsa = sa_ls[sra][t];
            const int lsb = sa_ls[srb][t];

#pragma unroll
            for (int n = 0; n < ST_NTILE; ++n) {
                int b[2];
                b[0] = sb_q[n * 8 + g][t * 8 + q];
                b[1] = sb_q[n * 8 + g][t * 8 + 4 + q];

                // Zeroed per k-tile, which the format requires: IQ4_XS carries
                // a scale per 32 weights, so an int32 accumulator cannot span
                // two of them.
                const int zero[4] = {0, 0, 0, 0};
                int s[4];
                mma_m16n8k32_s8(s, a, b, zero);

                // `(d * xs) * (ls - 32)` then `* s`, the reference's grouping
                // and the register kernel's, folded ascending in `t`.
                acc[n][0] += (da * xs[n][0]) * (float)(lsa - 32) * (float)s[0];
                acc[n][1] += (da * xs[n][1]) * (float)(lsa - 32) * (float)s[1];
                acc[n][2] += (db * xs[n][0]) * (float)(lsb - 32) * (float)s[2];
                acc[n][3] += (db * xs[n][1]) * (float)(lsb - 32) * (float)s[3];
            }
        }

        // The next superblock overwrites what these reads just consumed.
        __syncthreads();
    }

#pragma unroll
    for (int n = 0; n < ST_NTILE; ++n) {
        const int c0 = t0 + n * 8 + 2 * q;
        const int c1 = c0 + 1;
        if (c0 < n_tok) {
            if (row_a < n_out) out[(size_t)c0 * n_out + row_a] = acc[n][0];
            if (row_b < n_out) out[(size_t)c0 * n_out + row_b] = acc[n][2];
        }
        if (c1 < n_tok) {
            if (row_a < n_out) out[(size_t)c1 * n_out + row_a] = acc[n][1];
            if (row_b < n_out) out[(size_t)c1 * n_out + row_b] = acc[n][3];
        }
    }
}

}  // extern "C"
