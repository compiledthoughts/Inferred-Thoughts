// mma_int8.cuh -- matmuls on the int8 tensor cores: IQ4_XS, Q5_K and the grouped MoE.
//
// Included by kernels.cu, which pulls every file into one compile unit and one
// PTX module in dependency order. Kernel names are the PTX symbols the backend
// looks up, so which file a kernel lives in changes nothing at run time.
// Kernels are in `extern "C"` blocks; templates sit outside them, since a
// template cannot have C linkage.
#pragma once

extern "C" {

// ---------------------------------------------------------------------------
// IQ4_XS through the int8 tensor cores — the exactness probe
// ---------------------------------------------------------------------------
//
// **The question this exists to answer.** Two independent experiments have now
// cut weight traffic and got a fraction of it back: token-tiling the dense
// matmul, 8x for 1.54x, and grouping the routed FFN by expert, 5.17x for 1.12x.
// Both share `dot_iq4_xs_warp`, so both say the same thing -- the kernel is
// bound by instruction issue, not by bytes. What it issues per 32-element
// sub-block per token is a scalar nibble unpack, eight `imad`, two reduction
// shuffles, and then eight `__shfl_sync` with eight *dependent* f32 adds on
// lane 0 to preserve the reference's summation order.
//
// One `mma.sync.aligned.m16n8k32.s8` replaces all of that for a 16x8 tile.
//
// **And the standing worry is that it costs bit-exactness. It should not.**
// The reference's arithmetic for one output is
//
//     for ibl in 0..nb:                       # superblocks of 256
//       for t in 0..8:                        # sub-blocks of 32, ascending
//         s   = sum of 32 int8 products       # INTEGER, so it cannot round
//         dh  = (d * xs[ibl]) * (ls_t - 32)
//         sumf += dh * s
//
// The inner sum is integer, so any decomposition of it is exact -- the same
// property that let `matmul_q8_0` split blocks across lanes and stay
// bit-identical. And `s` for one sub-block is precisely one `k = 32` MMA tile.
// The f32 chain outside it stays serial and ascending, in a register, which is
// what the reference does.
//
// The format forces that shape on everyone: IQ4_XS carries a scale per 32
// weights, so nobody can accumulate int32 across k-tiles. llama.cpp's
// `vec_dot_q8_0_q8_1_mma` declares its accumulator *inside* the k loop for the
// same reason.
//
// So this is a probe, not a product: one warp per 16x8 output tile, scalar
// scale loads, no shared-memory staging and no double buffering. If it is
// bit-identical, the direction is real and worth building properly. If it is
// not, an hour was spent rather than a session.

// Pack four already-mapped IQ4_NL values as four s8 in one register, which is
// the operand form the MMA wants.
__device__ __forceinline__ int pack_s8x4(int n0, int n1, int n2, int n3) {
    return (n0 & 0xff) | ((n1 & 0xff) << 8) | ((n2 & 0xff) << 16) | ((n3 & 0xff) << 24);
}

// `aux` is four packed IQ4_XS bytes, i.e. eight nibbles. The low nibbles are
// sub-block elements k..k+3 and the high nibbles elements k+16..k+19 -- so one
// aligned 4-byte load serves two of the four A registers.
__device__ __forceinline__ void unpack_iq4_pair(int aux, int &lo, int &hi) {
    const unsigned int u = (unsigned int)aux;
    lo = pack_s8x4(kvalue_iq4nl((int)((u >>  0) & 0xf)),
                   kvalue_iq4nl((int)((u >>  8) & 0xf)),
                   kvalue_iq4nl((int)((u >> 16) & 0xf)),
                   kvalue_iq4nl((int)((u >> 24) & 0xf)));
    hi = pack_s8x4(kvalue_iq4nl((int)((u >>  4) & 0xf)),
                   kvalue_iq4nl((int)((u >> 12) & 0xf)),
                   kvalue_iq4nl((int)((u >> 20) & 0xf)),
                   kvalue_iq4nl((int)((u >> 28) & 0xf)));
}

// `D = A * B + C` over s8 with an s32 accumulator, 16x8x32.
//
// Available from sm_80; this project targets sm_120 only, so there is no
// fallback path and no dispatch.
__device__ __forceinline__ void mma_m16n8k32_s8(
        int (&d)[4], const int (&a)[4], const int (&b)[2], const int (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

// One warp computes a 16 (weight row) x 8 (token) tile.
//
// Fragment layout for m16n8k32, from the PTX ISA. With `g = lane >> 2` and
// `q = lane & 3`:
//
//   A   a0: row g,     k = 4q + 0..3        a1: row g + 8, k = 4q + 0..3
//       a2: row g,     k = 4q + 16..19      a3: row g + 8, k = 4q + 16..19
//   B   b0: col g,     k = 4q + 0..3        b1: col g,     k = 4q + 16..19
//   C   c0: row g,     col 2q               c1: row g,     col 2q + 1
//       c2: row g + 8, col 2q               c3: row g + 8, col 2q + 1
//
// The sub-block layout falls out of it: element `k` of sub-block `t` is the low
// nibble of `qs[16t + k]` for k < 16 and the high nibble of `qs[16t + k - 16]`
// above, so a0/a2 come from a single aligned 4-byte load and so do a1/a3.
// The 6-bit sub-block scale, assembled exactly as `dot_iq4_xs_warp` does it:
// four low bits from `scales_l` and two high bits from `scales_h`, with the
// reference shifting `h` right by `2 * ib` before it reads them.
__device__ __forceinline__ int iq4_ls(unsigned int sh, unsigned int lo, int ib, int half) {
    const unsigned int h = sh >> (ib * 2);
    return (half == 0) ? (int)((lo & 0xf) | ((h << 4) & 0x30))
                       : (int)((lo >> 4)  | ((h << 2) & 0x30));
}

// Token tiles per weight load.
//
// **The A fragment is the expensive load; B is cheap.** One `mma.m16n8k32`
// covers 8 tokens, so the first version re-read and re-unpacked a weight row
// once per 8 tokens of the batch. Holding A in registers across `MMA_NTILE`
// tiles divides that traffic by the same factor, for four more accumulators and
// one more B load per tile.
//
// 4 is 32 tokens per weight load, against `MM_TOK` 8 for the warp-dot kernel it
// replaced. Higher costs accumulators: `acc` is `MMA_NTILE * 4` floats and the
// kernel already carries A, B and the scale pairs.
#define MMA_NTILE 4

__global__ void matmul_iq4_xs_q8_k_mma(int n_in, int n_out, int n_tok,
                                       const unsigned char *__restrict__ w,
                                       const float *__restrict__ x_scales,
                                       const signed char *__restrict__ x_quants,
                                       float *__restrict__ out) {
    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const int j0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 16;   // first row
    const int t0 = blockIdx.y * (8 * MMA_NTILE);                   // first token
    if (j0 >= n_out) return;

    const int g = lane >> 2;
    const int q = lane & 3;

    // This lane owns rows j0+g and j0+g+8. Within each token tile it loads B
    // for column `g` and accumulates columns `2q` and `2q+1`.
    const int row_a = j0 + g;
    const int row_b = j0 + g + 8;

    const unsigned char *ra = w + (size_t)row_a * nb * IQ4XS_BYTES;
    const unsigned char *rb = w + (size_t)row_b * nb * IQ4XS_BYTES;

    // One f32 accumulator per output, folded ascending exactly as the reference
    // folds its single one.
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

        // Activation scales are per (token, superblock), so they depend on the
        // tile but not on the sub-block: hoisted out of the `t` loop.
        float xs[MMA_NTILE][2];
#pragma unroll
        for (int n = 0; n < MMA_NTILE; ++n) {
            const int c0 = t0 + n * 8 + 2 * q;
            const int c1 = c0 + 1;
            xs[n][0] = (c0 < n_tok) ? x_scales[(size_t)c0 * nb + ibl] : 0.0f;
            xs[n][1] = (c1 < n_tok) ? x_scales[(size_t)c1 * nb + ibl] : 0.0f;
        }

        for (int t = 0; t < 8; ++t) {
            const int ib   = (t >> 1) * 2;
            const int half = t & 1;

            const int lsa = iq4_ls(sha, ba[4 + (ib >> 1)], ib, half);
            const int lsb = iq4_ls(shb, bb[4 + (ib >> 1)], ib, half);

            // **Loaded and unpacked once for every token tile below.** Sub-block
            // `t` is qs bytes [16t, 16t+16) and activations [32t, 32t+32); the
            // low nibbles of these four bytes are k < 16 and the high nibbles
            // k >= 16, so one aligned 4-byte load fills two A registers.
            const int off = 4 + QK_K / 64 + t * 16 + q * 4;
            int a[4];
            unpack_iq4_pair(*(const int *)(ba + off), a[0], a[2]);
            unpack_iq4_pair(*(const int *)(bb + off), a[1], a[3]);


#pragma unroll
            for (int n = 0; n < MMA_NTILE; ++n) {
                const int tb = t0 + n * 8 + g;
                int b[2] = {0, 0};
                if (tb < n_tok) {
                    const signed char *q8 =
                        x_quants + (size_t)tb * n_in + (size_t)ibl * QK_K + t * 32;
                    b[0] = *(const int *)(q8 + q * 4);
                    b[1] = *(const int *)(q8 + 16 + q * 4);
                }

                // Zeroed per k-tile, which the format requires anyway: IQ4_XS
                // carries a scale per 32 weights, so an int32 accumulator
                // cannot span two of them.
                const int zero[4] = {0, 0, 0, 0};
                int s[4];
                mma_m16n8k32_s8(s, a, b, zero);

                // `(d * xs) * (ls - 32)` then `* s`, the reference's grouping,
                // and not fused -- `--fmad=false` is global.
                acc[n][0] += (da * xs[n][0]) * (float)(lsa - 32) * (float)s[0];
                acc[n][1] += (da * xs[n][1]) * (float)(lsa - 32) * (float)s[1];
                acc[n][2] += (db * xs[n][0]) * (float)(lsb - 32) * (float)s[2];
                acc[n][3] += (db * xs[n][1]) * (float)(lsb - 32) * (float)s[3];
            }
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
// Q8_0 x Q8_0 on the int8 tensor cores — the dense matmuls in prefill
// ---------------------------------------------------------------------------
//
// **Why** (06-10). Layer-major prefill made the 125B's prefill compute-bound,
// and `--profile-kernels` on a 5,688-token prompt put `matmul_q8_0_batch` first
// at 31.9% of kernel time: every attention, GDN and shared-expert projection.
// That kernel is `__dp4a` on the CUDA cores and holds one weight row for
// `MM_TOK` 8 tokens, so a row is read once per 8 tokens. Here a warp holds 16
// rows for `8 * MMA_NTILE` = 32 tokens, and the products run on the tensor
// cores.
//
// **Bit-identical to `matmul_q8_0_batch`, by the IQ4_XS argument.** Per output
// the reference folds, over blocks of 32 in ascending order from zero,
//
//     acc += (float)sumi * (dw * dx)
//
// with `sumi` the integer dot of one block. A Q8_0 block is 32 int8 weights
// against 32 int8 activations, which is exactly one `m16n8k32` tile, so the MMA
// returns that same integer, and the fold below is the same expression in the
// same order in a register. `--fmad=false` is global, so nothing contracts.
// `the_q8_0_mma_matmul_is_bit_identical` holds it to the bit.
//
// Same geometry as `matmul_iq4_xs_q8_k_mma`: one warp per 16 rows by 32
// tokens, A in registers across the token tiles, no shared memory. The weight
// layout is the repacked one `matmul_q8_0_batch` reads: f16 scales `[row][block]`
// and int8 quants `[row][n_in]`.
__global__ void matmul_q8_0_mma(int n_in, int n_out, int n_tok,
                                const unsigned short *__restrict__ w_scales,
                                const signed char *__restrict__ w_quants,
                                const float *__restrict__ x_scales,
                                const signed char *__restrict__ x_quants,
                                float *__restrict__ out) {
    const int n_blocks = n_in / 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const int j0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 16;   // first row
    const int t0 = blockIdx.y * (8 * MMA_NTILE);                   // first token
    if (j0 >= n_out) return;

    const int g = lane >> 2;
    const int q = lane & 3;

    // This lane owns rows j0+g and j0+g+8 (`n_out % 16 == 0`, so both exist).
    // Within each token tile it loads B for column `g` and accumulates columns
    // `2q` and `2q+1`.
    const int row_a = j0 + g;
    const int row_b = j0 + g + 8;
    const signed char *qa = w_quants + (size_t)row_a * n_in;
    const signed char *qb = w_quants + (size_t)row_b * n_in;
    const unsigned short *sa = w_scales + (size_t)row_a * n_blocks;
    const unsigned short *sb = w_scales + (size_t)row_b * n_blocks;

    // One f32 accumulator per output, folded ascending as the reference folds.
    float acc[MMA_NTILE][4];
#pragma unroll
    for (int n = 0; n < MMA_NTILE; ++n) {
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[n][k] = 0.0f;
    }

    for (int bb = 0; bb < n_blocks; ++bb) {
        const float dwa = __half2float(__ushort_as_half(sa[bb]));
        const float dwb = __half2float(__ushort_as_half(sb[bb]));

        // **Loaded once for every token tile below.** Block `bb` is quants
        // [32 bb, 32 bb + 32) of the row; lane q takes k = 4q..4q+3 and
        // 4q+16..4q+19, as the m16n8k32 A fragment lays them out.
        const int off = bb * 32 + q * 4;
        int a[4];
        a[0] = *(const int *)(qa + off);
        a[2] = *(const int *)(qa + off + 16);
        a[1] = *(const int *)(qb + off);
        a[3] = *(const int *)(qb + off + 16);

#pragma unroll
        for (int n = 0; n < MMA_NTILE; ++n) {
            const int tb = t0 + n * 8 + g;
            int b[2] = {0, 0};
            if (tb < n_tok) {
                const signed char *xq = x_quants + (size_t)tb * n_in + (size_t)bb * 32;
                b[0] = *(const int *)(xq + q * 4);
                b[1] = *(const int *)(xq + 16 + q * 4);
            }

            // Zeroed per block: a scale per 32 weights means an int32
            // accumulator cannot span two of them.
            const int zero[4] = {0, 0, 0, 0};
            int s[4];
            mma_m16n8k32_s8(s, a, b, zero);

            const int c0 = t0 + n * 8 + 2 * q;
            const int c1 = c0 + 1;
            const float dx0 = (c0 < n_tok) ? x_scales[(size_t)c0 * n_blocks + bb] : 0.0f;
            const float dx1 = (c1 < n_tok) ? x_scales[(size_t)c1 * n_blocks + bb] : 0.0f;

            // `(float)sumi * (dw * dx)`, the reference's grouping, then added.
            acc[n][0] += (float)s[0] * (dwa * dx0);
            acc[n][1] += (float)s[1] * (dwa * dx1);
            acc[n][2] += (float)s[2] * (dwb * dx0);
            acc[n][3] += (float)s[3] * (dwb * dx1);
        }
    }

#pragma unroll
    for (int n = 0; n < MMA_NTILE; ++n) {
        const int c0 = t0 + n * 8 + 2 * q;
        const int c1 = c0 + 1;
        if (c0 < n_tok) {
            out[(size_t)c0 * n_out + row_a] = acc[n][0];
            out[(size_t)c0 * n_out + row_b] = acc[n][2];
        }
        if (c1 < n_tok) {
            out[(size_t)c1 * n_out + row_a] = acc[n][1];
            out[(size_t)c1 * n_out + row_b] = acc[n][3];
        }
    }
}

// ---------------------------------------------------------------------------
// Q5_K x Q8_K on the int8 tensor cores
// ---------------------------------------------------------------------------
//
// **The one matmul here deliberately outside the bit-exact set**, and the reason
// is worth stating before the code.
//
// Q5_K and Q6_K were the last matmuls running one warp per output row with no
// tensor cores at all: 16.0% of a prefill at **0.5% of the int8 ceiling**
// against 6-10% for the kernels on `mma`. A 12-20x gap on the same card, and the
// largest unexploited headroom in the engine.
//
// # What maps cleanly, and it is more than expected
//
// Q5_K carries **one scale per 32 weights**, which is exactly the MMA's k width,
// so one `mma.m16n8k32` covers one sub-block and one scale. And the reference
// already folds to f32 only **once per superblock** --
// `sums[l] = fma(d, aux32[l], sums[l])` after the eight sub-blocks have
// accumulated in int32. So the fold structure is reproduced fma for fma.
//
// Unpacked values are 0..31, so they fit a *signed* int8 operand and
// `mma_m16n8k32_s8` is used unchanged. A 0..31 weight against an int8 activation
// is at most 3937, inside the `i16` the reference casts to, and 32 of those are
// nowhere near int32.
//
// # What does not map, which is the whole cost of the decision
//
// The reference keeps **eight int32 lanes** -- `aux32[l]`, lane `l` owning the
// positions congruent to `l` mod 8 -- each feeding its own f32 chain across
// superblocks, the eight summed only at the very end. An `mma.m16n8k32`
// contracts all 32 k-elements into one int32 and cannot hand back per-lane
// partials, so this computes
//
//     sumf = sum_i d_i * (sum_l aux32[l,i])      one rounding per superblock
//
// where the reference computes
//
//     sumf = sum_l (sum_i d_i * aux32[l,i])      eight
//
// Same real number, different f32. **The integers are exact on both sides**, so
// the gap is a fold-order change of the same family as the RMSNorm tree -- and
// as there, this side rounds eight times less often and is the *more* accurate
// of the two. What is given up is determinism, not accuracy.
//
// `INFERRED_Q5K_SCALAR` keeps the scalar kernel and with it bit equality, and
// `q5k_scalar_restores_bit_equality` tests that the flag really buys it back
// rather than merely claiming to.
//
// # Two places Q5_K differs from IQ4_XS and the layout has to follow
//
// **The two k-halves come from different bytes.** IQ4_XS puts element `k` in the
// low nibble and `k + 16` in the high nibble of one byte, so a single 4-byte
// load fills two A registers. Q5_K puts element `t` of sub-block `sb` in nibble
// `sb & 1` of `qs[(sb >> 1) * 32 + t]`, so `t` and `t + 16` are sixteen bytes
// apart and each A register needs its own load.
//
// **The fifth bit lives in `qh`,** one bit per element position and one bit
// plane per sub-block: element `t` of sub-block `sb` takes bit `sb` of `qh[t]`.
//
// # The mins term stays scalar, and that is a measurement not an oversight
//
// The offset is `fma(-dmin, sumi, sumf)` with `sumi = sum_k bsum[k] * mins[k/2]`,
// which pairs to `sum_j mins[j] * (bsum[2j] + bsum[2j+1])` -- eight products per
// output per superblock, so 32 integer MACs a superblock against eight MMAs.
// That sounds ruinous and is not: the unpack and the f32 fold around those eight
// MMAs are a few hundred operations, so the mins term is roughly a tenth of the
// loop. It is also **exact**, which keeps the error bound to the one reordering
// named above. Moving it onto the tensor cores would mean splitting `bsum` into
// two int8 planes; do that only if it measures as the bound.

// Token tiles per weight load, as `MMA_NTILE` is for IQ4_XS. Four is 32 tokens
// per load of a weight row that costs four unpacks instead of two.
#define Q5K_MMA_NTILE 4

// Four elements of one sub-block, from one `qs` quad and one `qh` quad.
//
// `sh` is `4 * (sb & 1)`, selecting the nibble; `sb` selects the `qh` bit plane.
// Byte `b` of each quad is element `4q + b` of the sub-block, so the `qh` bit is
// at `8 * b + sb`.
__device__ __forceinline__ int unpack_q5_quad(int qs4, int qh4, int sh, int sb) {
    const unsigned int u = (unsigned int)qs4;
    const unsigned int h = (unsigned int)qh4;
    const int v0 = (int)((u >> sh) & 0xfu) | (int)(((h >> sb) & 1u) << 4);
    const int v1 = (int)((u >> (8 + sh)) & 0xfu) | (int)(((h >> (8 + sb)) & 1u) << 4);
    const int v2 = (int)((u >> (16 + sh)) & 0xfu) | (int)(((h >> (16 + sb)) & 1u) << 4);
    const int v3 = (int)((u >> (24 + sh)) & 0xfu) | (int)(((h >> (24 + sb)) & 1u) << 4);
    return pack_s8x4(v0, v1, v2, v3);
}

__global__ void matmul_q5_k_q8_k_mma(int n_in, int n_out, int n_tok,
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

    const int j0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 16;
    const int t0 = blockIdx.y * (8 * Q5K_MMA_NTILE);
    if (j0 >= n_out) return;

    const int g = lane >> 2;
    const int q = lane & 3;

    // Rows g and g+8 of the tile, clamped so a partial tile reads in bounds and
    // is dropped at write-back instead.
    const int row_a = j0 + g;
    const int row_b = j0 + g + 8;
    const unsigned char *pa = w + (size_t)min(row_a, n_out - 1) * nb * Q5K_BYTES;
    const unsigned char *pb = w + (size_t)min(row_b, n_out - 1) * nb * Q5K_BYTES;

    // `acc` is the reference's `sums` chain collapsed to one lane; `off` is the
    // dmin chain, which stays exactly as the reference has it.
    float acc[Q5K_MMA_NTILE][4], off[Q5K_MMA_NTILE][4];
#pragma unroll
    for (int n = 0; n < Q5K_MMA_NTILE; ++n) {
#pragma unroll
        for (int k = 0; k < 4; ++k) { acc[n][k] = 0.0f; off[n][k] = 0.0f; }
    }

    for (int ibl = 0; ibl < nb; ++ibl) {
        const unsigned char *ba = pa + (size_t)ibl * Q5K_BYTES;
        const unsigned char *bb = pb + (size_t)ibl * Q5K_BYTES;

        const float da = h2f((unsigned short)ba[0] | ((unsigned short)ba[1] << 8));
        const float db = h2f((unsigned short)bb[0] | ((unsigned short)bb[1] << 8));
        const float ma = h2f((unsigned short)ba[2] | ((unsigned short)ba[3] << 8));
        const float mb = h2f((unsigned short)bb[2] | ((unsigned short)bb[3] << 8));

        // The twelve-byte scale/min shuffle, transcribed from the scalar kernel
        // and through it from `ggml_vec_dot_q5_K_q8_K_generic`.
        unsigned int a0 = *(const unsigned int *)(ba + 4);
        unsigned int a1 = *(const unsigned int *)(ba + 8);
        unsigned int a2 = *(const unsigned int *)(ba + 12);
        const unsigned int a3 = ((a2 >> 4) & KMASK2) | (((a1 >> 6) & KMASK3) << 4);
        const unsigned int aaux = a1 & KMASK1;
        a1 = (a2 & KMASK2) | (((a0 >> 6) & KMASK3) << 4);
        a2 = aaux;
        a0 &= KMASK1;

        unsigned int c0v = *(const unsigned int *)(bb + 4);
        unsigned int c1v = *(const unsigned int *)(bb + 8);
        unsigned int c2v = *(const unsigned int *)(bb + 12);
        const unsigned int c3v = ((c2v >> 4) & KMASK2) | (((c1v >> 6) & KMASK3) << 4);
        const unsigned int caux = c1v & KMASK1;
        c1v = (c2v & KMASK2) | (((c0v >> 6) & KMASK3) << 4);
        c2v = caux;
        c0v &= KMASK1;

        const unsigned char *qha = ba + 16;
        const unsigned char *qhb = bb + 16;
        const unsigned char *qsa = ba + 16 + QK_K / 8;
        const unsigned char *qsb = bb + 16 + QK_K / 8;

        // Per (token, superblock), so hoisted out of the sub-block loop exactly
        // as the IQ4_XS kernel hoists its own.
        float xs[Q5K_MMA_NTILE][2];
#pragma unroll
        for (int n = 0; n < Q5K_MMA_NTILE; ++n) {
            const int t_0 = t0 + n * 8 + 2 * q, t_1 = t_0 + 1;
            xs[n][0] = (t_0 < n_tok) ? x_scales[(size_t)t_0 * nb + ibl] : 0.0f;
            xs[n][1] = (t_1 < n_tok) ? x_scales[(size_t)t_1 * nb + ibl] : 0.0f;
        }

        // The superblock's scaled products, in int32 and therefore exact.
        int acci[Q5K_MMA_NTILE][4];
#pragma unroll
        for (int n = 0; n < Q5K_MMA_NTILE; ++n) {
#pragma unroll
            for (int k = 0; k < 4; ++k) acci[n][k] = 0;
        }

        for (int sb = 0; sb < QK_K / 32; ++sb) {
            const unsigned int sua = (sb < 4) ? a0 : a1;
            const unsigned int sub = (sb < 4) ? c0v : c1v;
            const int sca = (int)((sua >> ((sb & 3) * 8)) & 0xff);
            const int scb = (int)((sub >> ((sb & 3) * 8)) & 0xff);

            const int qoff = (sb >> 1) * 32 + q * 4;
            const int hoff = q * 4;
            const int sh = 4 * (sb & 1);

            // a[0]/a[2] are row_a's low and high k-halves, a[1]/a[3] row_b's --
            // the m16n8k32 fragment order. Four loads, not two, because Q5_K's
            // halves are sixteen bytes apart.
            int a[4];
            a[0] = unpack_q5_quad(*(const int *)(qsa + qoff),
                                  *(const int *)(qha + hoff), sh, sb);
            a[2] = unpack_q5_quad(*(const int *)(qsa + qoff + 16),
                                  *(const int *)(qha + hoff + 16), sh, sb);
            a[1] = unpack_q5_quad(*(const int *)(qsb + qoff),
                                  *(const int *)(qhb + hoff), sh, sb);
            a[3] = unpack_q5_quad(*(const int *)(qsb + qoff + 16),
                                  *(const int *)(qhb + hoff + 16), sh, sb);

#pragma unroll
            for (int n = 0; n < Q5K_MMA_NTILE; ++n) {
                const int tb = t0 + n * 8 + g;
                int bfrag[2] = {0, 0};
                if (tb < n_tok) {
                    const signed char *q8 =
                        x_quants + (size_t)tb * n_in + (size_t)ibl * QK_K + sb * 32;
                    bfrag[0] = *(const int *)(q8 + q * 4);
                    bfrag[1] = *(const int *)(q8 + 16 + q * 4);
                }
                // Zeroed per sub-block: one scale per 32 weights means an int32
                // accumulator cannot span two of them without the scale.
                const int zero[4] = {0, 0, 0, 0};
                int s[4];
                mma_m16n8k32_s8(s, a, bfrag, zero);
                acci[n][0] += sca * s[0];
                acci[n][1] += sca * s[1];
                acci[n][2] += scb * s[2];
                acci[n][3] += scb * s[3];
            }
        }

        // One fma per superblock, as the reference does -- only the eight lanes
        // it would have kept are collapsed into one. This is the departure.
#pragma unroll
        for (int n = 0; n < Q5K_MMA_NTILE; ++n) {
            acc[n][0] = __fmaf_rn(da * xs[n][0], (float)acci[n][0], acc[n][0]);
            acc[n][1] = __fmaf_rn(da * xs[n][1], (float)acci[n][1], acc[n][1]);
            acc[n][2] = __fmaf_rn(db * xs[n][0], (float)acci[n][2], acc[n][2]);
            acc[n][3] = __fmaf_rn(db * xs[n][1], (float)acci[n][3], acc[n][3]);
        }

        // The mins chain, exact and in the reference's order: `sumi` is an
        // integer sum, and `fma(-dmin, sumi, off)` lands inside the superblock
        // loop before anything is folded, as `dot_q5_k` has it.
#pragma unroll
        for (int n = 0; n < Q5K_MMA_NTILE; ++n) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int c = t0 + n * 8 + 2 * q + e;
                if (c >= n_tok) continue;
                const short *bs =
                    x_bsums + (size_t)c * nb * (QK_K / 16) + (size_t)ibl * (QK_K / 16);
                int sia = 0, sib = 0;
#pragma unroll
                for (int mk = 0; mk < 8; ++mk) {
                    const unsigned int mua = (mk < 4) ? a2 : a3;
                    const unsigned int mub = (mk < 4) ? c2v : c3v;
                    // `mins[k >> 1]` is constant over each pair of bsums, so the
                    // reference's sixteen products pair into eight. Integer, so
                    // the pairing is exact.
                    const int pair = (int)bs[2 * mk] + (int)bs[2 * mk + 1];
                    sia += (int)((mua >> ((mk & 3) * 8)) & 0xff) * pair;
                    sib += (int)((mub >> ((mk & 3) * 8)) & 0xff) * pair;
                }
                off[n][e] = __fmaf_rn(-(ma * xs[n][e]), (float)sia, off[n][e]);
                off[n][2 + e] = __fmaf_rn(-(mb * xs[n][e]), (float)sib, off[n][2 + e]);
            }
        }
    }

#pragma unroll
    for (int n = 0; n < Q5K_MMA_NTILE; ++n) {
        const int t_0 = t0 + n * 8 + 2 * q, t_1 = t_0 + 1;
        if (t_0 < n_tok) {
            if (row_a < n_out) out[(size_t)t_0 * n_out + row_a] = off[n][0] + acc[n][0];
            if (row_b < n_out) out[(size_t)t_0 * n_out + row_b] = off[n][2] + acc[n][2];
        }
        if (t_1 < n_tok) {
            if (row_a < n_out) out[(size_t)t_1 * n_out + row_a] = off[n][1] + acc[n][1];
            if (row_b < n_out) out[(size_t)t_1 * n_out + row_b] = off[n][3] + acc[n][3];
        }
    }
}

// Gate, up and the SiLU gating for one expert tile, on the tensor cores.
//
// The tile is `MOE_MMA_TOK` pairs sharing one expert, cut into `MOE_MMA_NTILE`
// MMA tiles of 8 that share one A fragment -- so a weight row is loaded and
// unpacked once per `MOE_MMA_TOK` tokens rather than once per 8.
//
// `x` is indexed by *token* here (every expert of a token reads the same
// activation) where the `down` kernel indexes it by pair. That asymmetry is
// invisible at `n_tok == 1`.
__global__ void matmul_iq4_xs_q8_k_moe_glu_grouped_mma(
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
    const int j0   = (blockIdx.x * (blockDim.x >> 5) + warp) * 16;
    if (j0 >= n_ff) return;

    const int first = tile_first[tl];
    const int nt    = tile_n[tl];
    const int g = lane >> 2;
    const int q = lane & 3;

    // Slots this lane touches in each sub-tile: `g` for the B fragment it
    // loads, `2q` and `2q+1` for the columns it accumulates.
    int pb[MOE_MMA_NTILE], p0[MOE_MMA_NTILE], p1[MOE_MMA_NTILE];
    int tb[MOE_MMA_NTILE], t0i[MOE_MMA_NTILE], t1i[MOE_MMA_NTILE];
#pragma unroll
    for (int n = 0; n < MOE_MMA_NTILE; ++n) {
        const int sb = n * 8 + g, s0 = n * 8 + 2 * q, s1 = s0 + 1;
        pb[n] = (sb < nt) ? perm[first + sb] : -1;
        p0[n] = (s0 < nt) ? perm[first + s0] : -1;
        p1[n] = (s1 < nt) ? perm[first + s1] : -1;
        tb[n] = (pb[n] >= 0) ? pb[n] / n_used : -1;
        t0i[n] = (p0[n] >= 0) ? p0[n] / n_used : -1;
        t1i[n] = (p1[n] >= 0) ? p1[n] / n_used : -1;
    }

    // Every pair in the tile shares an expert, so one pointer pair serves it.
    const unsigned char *gw = (const unsigned char *)gptrs[perm[first]];
    const unsigned char *uw = (const unsigned char *)uptrs[perm[first]];
    const size_t ra = (size_t)(j0 + g) * nb * IQ4XS_BYTES;
    const size_t rb = (size_t)(j0 + g + 8) * nb * IQ4XS_BYTES;

    float ag[MOE_MMA_NTILE][4], au[MOE_MMA_NTILE][4];
#pragma unroll
    for (int n = 0; n < MOE_MMA_NTILE; ++n) {
#pragma unroll
        for (int k = 0; k < 4; ++k) { ag[n][k] = 0.0f; au[n][k] = 0.0f; }
    }

    for (int ibl = 0; ibl < nb; ++ibl) {
        const unsigned char *ga = gw + ra + (size_t)ibl * IQ4XS_BYTES;
        const unsigned char *gb = gw + rb + (size_t)ibl * IQ4XS_BYTES;
        const unsigned char *ua = uw + ra + (size_t)ibl * IQ4XS_BYTES;
        const unsigned char *ub = uw + rb + (size_t)ibl * IQ4XS_BYTES;

        const float dga = h2f((unsigned short)ga[0] | ((unsigned short)ga[1] << 8));
        const float dgb = h2f((unsigned short)gb[0] | ((unsigned short)gb[1] << 8));
        const float dua = h2f((unsigned short)ua[0] | ((unsigned short)ua[1] << 8));
        const float dub = h2f((unsigned short)ub[0] | ((unsigned short)ub[1] << 8));
        const unsigned int hga = (unsigned int)ga[2] | ((unsigned int)ga[3] << 8);
        const unsigned int hgb = (unsigned int)gb[2] | ((unsigned int)gb[3] << 8);
        const unsigned int hua = (unsigned int)ua[2] | ((unsigned int)ua[3] << 8);
        const unsigned int hub = (unsigned int)ub[2] | ((unsigned int)ub[3] << 8);

        // Gate and up read the same activation, so one scale pair per sub-tile.
        float xs[MOE_MMA_NTILE][2];
#pragma unroll
        for (int n = 0; n < MOE_MMA_NTILE; ++n) {
            xs[n][0] = (t0i[n] >= 0) ? x_scales[(size_t)t0i[n] * nb + ibl] : 0.0f;
            xs[n][1] = (t1i[n] >= 0) ? x_scales[(size_t)t1i[n] * nb + ibl] : 0.0f;
        }

        for (int t = 0; t < 8; ++t) {
            const int ib = (t >> 1) * 2, half = t & 1;
            const int lsga = iq4_ls(hga, ga[4 + (ib >> 1)], ib, half);
            const int lsgb = iq4_ls(hgb, gb[4 + (ib >> 1)], ib, half);
            const int lsua = iq4_ls(hua, ua[4 + (ib >> 1)], ib, half);
            const int lsub = iq4_ls(hub, ub[4 + (ib >> 1)], ib, half);

            // **Unpacked once for every sub-tile below.**
            const int off = 4 + QK_K / 64 + t * 16 + q * 4;
            int fg[4], fu[4];
            unpack_iq4_pair(*(const int *)(ga + off), fg[0], fg[2]);
            unpack_iq4_pair(*(const int *)(gb + off), fg[1], fg[3]);
            unpack_iq4_pair(*(const int *)(ua + off), fu[0], fu[2]);
            unpack_iq4_pair(*(const int *)(ub + off), fu[1], fu[3]);

#pragma unroll
            for (int n = 0; n < MOE_MMA_NTILE; ++n) {
                int b[2] = {0, 0};
                if (tb[n] >= 0) {
                    const signed char *q8 =
                        x_quants + (size_t)tb[n] * n_in + (size_t)ibl * QK_K + t * 32;
                    b[0] = *(const int *)(q8 + q * 4);
                    b[1] = *(const int *)(q8 + 16 + q * 4);
                }
                const int zero[4] = {0, 0, 0, 0};
                int sg[4], su[4];
                mma_m16n8k32_s8(sg, fg, b, zero);
                mma_m16n8k32_s8(su, fu, b, zero);

                ag[n][0] += (dga * xs[n][0]) * (float)(lsga - 32) * (float)sg[0];
                ag[n][1] += (dga * xs[n][1]) * (float)(lsga - 32) * (float)sg[1];
                ag[n][2] += (dgb * xs[n][0]) * (float)(lsgb - 32) * (float)sg[2];
                ag[n][3] += (dgb * xs[n][1]) * (float)(lsgb - 32) * (float)sg[3];

                au[n][0] += (dua * xs[n][0]) * (float)(lsua - 32) * (float)su[0];
                au[n][1] += (dua * xs[n][1]) * (float)(lsua - 32) * (float)su[1];
                au[n][2] += (dub * xs[n][0]) * (float)(lsub - 32) * (float)su[2];
                au[n][3] += (dub * xs[n][1]) * (float)(lsub - 32) * (float)su[3];
            }
        }
    }

    const int row_a = j0 + g, row_b = j0 + g + 8;
#pragma unroll
    for (int n = 0; n < MOE_MMA_NTILE; ++n) {
        if (p0[n] >= 0) {
            if (row_a < n_ff)
                out[(size_t)p0[n] * n_ff + row_a] = ag[n][0] / (1.0f + expf(-ag[n][0])) * au[n][0];
            if (row_b < n_ff)
                out[(size_t)p0[n] * n_ff + row_b] = ag[n][2] / (1.0f + expf(-ag[n][2])) * au[n][2];
        }
        if (p1[n] >= 0) {
            if (row_a < n_ff)
                out[(size_t)p1[n] * n_ff + row_a] = ag[n][1] / (1.0f + expf(-ag[n][1])) * au[n][1];
            if (row_b < n_ff)
                out[(size_t)p1[n] * n_ff + row_b] = ag[n][3] / (1.0f + expf(-ag[n][3])) * au[n][3];
        }
    }
}

// The routed FFN `down` matmul for one expert tile, on the tensor cores.
//
// `x` is indexed by pair: each pair has its own `n_ff`-wide intermediate.
__global__ void matmul_iq4_xs_q8_k_moe_grouped_mma(
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
    const int j0   = (blockIdx.x * (blockDim.x >> 5) + warp) * 16;
    if (j0 >= n_out) return;

    const int first = tile_first[tl];
    const int nt    = tile_n[tl];
    const int g = lane >> 2;
    const int q = lane & 3;

    int pb[MOE_MMA_NTILE], p0[MOE_MMA_NTILE], p1[MOE_MMA_NTILE];
#pragma unroll
    for (int n = 0; n < MOE_MMA_NTILE; ++n) {
        const int sb = n * 8 + g, s0 = n * 8 + 2 * q, s1 = s0 + 1;
        pb[n] = (sb < nt) ? perm[first + sb] : -1;
        p0[n] = (s0 < nt) ? perm[first + s0] : -1;
        p1[n] = (s1 < nt) ? perm[first + s1] : -1;
    }

    const unsigned char *w = (const unsigned char *)wptrs[perm[first]];
    const size_t ra = (size_t)(j0 + g) * nb * IQ4XS_BYTES;
    const size_t rb = (size_t)(j0 + g + 8) * nb * IQ4XS_BYTES;

    float acc[MOE_MMA_NTILE][4];
#pragma unroll
    for (int n = 0; n < MOE_MMA_NTILE; ++n) {
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[n][k] = 0.0f;
    }

    for (int ibl = 0; ibl < nb; ++ibl) {
        const unsigned char *ba = w + ra + (size_t)ibl * IQ4XS_BYTES;
        const unsigned char *bb = w + rb + (size_t)ibl * IQ4XS_BYTES;

        const float da = h2f((unsigned short)ba[0] | ((unsigned short)ba[1] << 8));
        const float db = h2f((unsigned short)bb[0] | ((unsigned short)bb[1] << 8));
        const unsigned int sha = (unsigned int)ba[2] | ((unsigned int)ba[3] << 8);
        const unsigned int shb = (unsigned int)bb[2] | ((unsigned int)bb[3] << 8);

        float xs[MOE_MMA_NTILE][2];
#pragma unroll
        for (int n = 0; n < MOE_MMA_NTILE; ++n) {
            xs[n][0] = (p0[n] >= 0) ? x_scales[(size_t)p0[n] * nb + ibl] : 0.0f;
            xs[n][1] = (p1[n] >= 0) ? x_scales[(size_t)p1[n] * nb + ibl] : 0.0f;
        }

        for (int t = 0; t < 8; ++t) {
            const int ib = (t >> 1) * 2, half = t & 1;
            const int lsa = iq4_ls(sha, ba[4 + (ib >> 1)], ib, half);
            const int lsb = iq4_ls(shb, bb[4 + (ib >> 1)], ib, half);

            const int off = 4 + QK_K / 64 + t * 16 + q * 4;
            int a[4];
            unpack_iq4_pair(*(const int *)(ba + off), a[0], a[2]);
            unpack_iq4_pair(*(const int *)(bb + off), a[1], a[3]);

#pragma unroll
            for (int n = 0; n < MOE_MMA_NTILE; ++n) {
                int b[2] = {0, 0};
                if (pb[n] >= 0) {
                    const signed char *q8 =
                        x_quants + (size_t)pb[n] * n_in + (size_t)ibl * QK_K + t * 32;
                    b[0] = *(const int *)(q8 + q * 4);
                    b[1] = *(const int *)(q8 + 16 + q * 4);
                }
                const int zero[4] = {0, 0, 0, 0};
                int s[4];
                mma_m16n8k32_s8(s, a, b, zero);

                acc[n][0] += (da * xs[n][0]) * (float)(lsa - 32) * (float)s[0];
                acc[n][1] += (da * xs[n][1]) * (float)(lsa - 32) * (float)s[1];
                acc[n][2] += (db * xs[n][0]) * (float)(lsb - 32) * (float)s[2];
                acc[n][3] += (db * xs[n][1]) * (float)(lsb - 32) * (float)s[3];
            }
        }
    }

    const int row_a = j0 + g, row_b = j0 + g + 8;
#pragma unroll
    for (int n = 0; n < MOE_MMA_NTILE; ++n) {
        if (p0[n] >= 0) {
            if (row_a < n_out) out[(size_t)p0[n] * n_out + row_a] = acc[n][0];
            if (row_b < n_out) out[(size_t)p0[n] * n_out + row_b] = acc[n][2];
        }
        if (p1[n] >= 0) {
            if (row_a < n_out) out[(size_t)p1[n] * n_out + row_a] = acc[n][1];
            if (row_b < n_out) out[(size_t)p1[n] * n_out + row_b] = acc[n][3];
        }
    }
}

}  // extern "C"
