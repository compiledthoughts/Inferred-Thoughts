// nvfp4.cuh -- NVFP4 weights against a Q8_0 activation: the exact path.
//
// Included by kernels.cu. ggml-cpu pairs NVFP4 with Q8_0 (its `vec_dot_type`),
// so these reproduce `ggml_vec_dot_nvfp4_q8_0_generic` (ggml-cpu/quants.c) and
// `ops::naive::dot_nvfp4_q8_0`, which transcribes it: per 16-element sub-block
// two integer sums, times `dy * d`, accumulated serially in f32 in sub-block
// order. The FP4 x FP4 tensor-core kernels are the precision departure measured
// against these.
//
// **One thread per output row, serial within it** -- correct first, as
// `matmul_q8_0` began. A block of 64 is 36 bytes: four UE4M3 scales, then 32
// bytes where byte `k` of sub-block `s` holds element `k` (low nibble) and
// element `k + 8` (high nibble). The per-tensor second scale is not part of the
// dot, as in ggml; the routed kernels apply the per-expert one to each pick's
// output, where `build_moe_ffn` applies it.
#pragma once

#ifdef INFERRED_NVFP4_BLOCK_SCALE
// `__nv_fp8_e4m3`, whose float conversion seeds the FP4 activation scale.
// `cuda_fp8.hpp` refuses to be included directly.
#include <cuda_fp8.h>
#endif

extern "C" {

// `kvalues_mxfp4` (ggml-common.h): E2M1 doubled. The magnitude by the low three
// bits, {0, 1, 2, 3, 4, 6, 8, 12}, is packed one per nibble; bit 3 is the sign.
__device__ __forceinline__ int nvfp4_kv(unsigned int q) {
    const int mag = (int)((0xC8643210u >> ((q & 7) * 4)) & 0xf);
    return (q & 8) ? -mag : mag;
}

// `ggml_ue4m3_to_fp32` (ggml-impl.h): unsigned E4M3, bias 7, halved to pair
// with the doubled table; 0 and 0x7F read as 0. Every factor is a short
// mantissa times a power of two, exact in f32, so this equals
// `quant::ue4m3_to_f32` to the bit.
__device__ __forceinline__ float nvfp4_ue4m3(unsigned int x) {
    if (x == 0 || x == 0x7f) return 0.0f;
    const int e = (int)((x >> 3) & 0xf);
    const float m = (float)(x & 7);
    const float raw = (e == 0) ? m / 512.0f
                               : (1.0f + m / 8.0f) * ((float)(1 << e) / 128.0f);
    return raw * 0.5f;
}

// One NVFP4 row against one token's Q8_0 activation. Sub-block `s` of block
// `ib` reads Q8_0 block `2 * ib + s / 2` at offset `(s % 2) * 16`.
__device__ __forceinline__ float nvfp4_q8_0_row(const unsigned char *row, int nb,
                                                 const float *xs,
                                                 const signed char *xq) {
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        const unsigned char *blk = row + (size_t)ib * 36;
        for (int s = 0; s < 4; ++s) {
            const float d  = nvfp4_ue4m3(blk[s]);
            const int   q8 = 2 * ib + (s >> 1);
            const float dy = xs[q8];
            const signed char *qa = xq + (size_t)q8 * 32 + (s & 1) * 16;
            int lo = 0, hi = 0;
            for (int k = 0; k < 8; ++k) {
                const unsigned int qv = blk[4 + s * 8 + k];
                lo += (int)qa[k] * nvfp4_kv(qv & 0xf);
                hi += (int)qa[k + 8] * nvfp4_kv(qv >> 4);
            }
            sumf += dy * d * (float)(lo + hi);
        }
    }
    return sumf;
}

// Dense NVFP4 matmul: the shared expert and the LM head. Batch on `blockIdx.y`,
// token-major, as `matmul_f32_t`.
__global__ void matmul_nvfp4_q8_0(int n_in, int n_out,
                                  const unsigned char *__restrict__ w,
                                  const float *__restrict__ x_scales,
                                  const signed char *__restrict__ x_quants,
                                  float *__restrict__ out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;
    const int tok = blockIdx.y;
    const int nb  = n_in / 64;
    out[(size_t)tok * n_out + j] =
        nvfp4_q8_0_row(w + (size_t)j * nb * 36, nb,
                       x_scales + (size_t)tok * (n_in / 32),
                       x_quants + (size_t)tok * n_in);
}

// Gate, up and the SiLU gating for one expert tile: `Ops::moe_glu`'s default --
// `matmul_experts` for gate and for up, each pick's row scaled by its expert's
// second scale, then `silu_mul` -- grouped by expert as
// `matmul_iq4_xs_q8_k_moe_glu_grouped` groups it. `x` is one row per token.
//
// `ids` is the route's expert ids, pair-major; every pair in a tile shares one
// expert, so pair `perm[first]` names it. A scale pointer is only read when its
// `has_*` flag is set: `convert_hf_to_gguf.py` omits the tensor when every
// expert's scale is 1.0.
__global__ void matmul_nvfp4_q8_0_moe_glu_grouped(
        int n_in, int n_ff, int n_used, int has_gs, int has_us,
        const int *__restrict__ n_tile,
        const int *__restrict__ perm,
        const int *__restrict__ tile_first,
        const int *__restrict__ tile_n,
        const int *__restrict__ ids,
        const unsigned long long *__restrict__ gptrs,
        const unsigned long long *__restrict__ uptrs,
        const float *__restrict__ gscale,
        const float *__restrict__ uscale,
        const float *__restrict__ x_scales,
        const signed char *__restrict__ x_quants,
        float *__restrict__ out) {
    const int tl = blockIdx.y;
    if (tl >= *n_tile) return;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_ff) return;

    const int first = tile_first[tl];
    const int nt    = tile_n[tl];
    const int nb    = n_in / 64;
    const int p0    = perm[first];
    const int e     = ids[p0];
    const size_t off = (size_t)j * nb * 36;
    const unsigned char *grow = (const unsigned char *)gptrs[p0] + off;
    const unsigned char *urow = (const unsigned char *)uptrs[p0] + off;

    for (int u = 0; u < nt; ++u) {
        const int q   = perm[first + u];
        const int tok = q / n_used;
        const float *xs = x_scales + (size_t)tok * (n_in / 32);
        const signed char *xq = x_quants + (size_t)tok * n_in;
        float g  = nvfp4_q8_0_row(grow, nb, xs, xq);
        float up = nvfp4_q8_0_row(urow, nb, xs, xq);
        if (has_gs) g *= gscale[e];
        if (has_us) up *= uscale[e];
        out[(size_t)q * n_ff + j] = g / (1.0f + expf(-g)) * up;
    }
}

// The routed `down` matmul for one expert tile. `x` is one row per *pair*, the
// asymmetry `matmul_iq4_xs_q8_k_moe_grouped` documents.
__global__ void matmul_nvfp4_q8_0_moe_grouped(
        int n_in, int n_out, int has_s,
        const int *__restrict__ n_tile,
        const int *__restrict__ perm,
        const int *__restrict__ tile_first,
        const int *__restrict__ tile_n,
        const int *__restrict__ ids,
        const unsigned long long *__restrict__ wptrs,
        const float *__restrict__ scale,
        const float *__restrict__ x_scales,
        const signed char *__restrict__ x_quants,
        float *__restrict__ out) {
    const int tl = blockIdx.y;
    if (tl >= *n_tile) return;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;

    const int first = tile_first[tl];
    const int nt    = tile_n[tl];
    const int nb    = n_in / 64;
    const int p0    = perm[first];
    const int e     = ids[p0];
    const unsigned char *row =
        (const unsigned char *)wptrs[p0] + (size_t)j * nb * 36;

    for (int u = 0; u < nt; ++u) {
        const int q = perm[first + u];
        float v = nvfp4_q8_0_row(row, nb,
                                 x_scales + (size_t)q * (n_in / 32),
                                 x_quants + (size_t)q * n_in);
        if (has_s) v *= scale[e];
        out[(size_t)q * n_out + j] = v;
    }
}

// ------------------------------------------------------------ FP4 x FP4
//
// The tensor-core path, and the NVFP4 default once it is proven against
// `ops::naive::{Fp4Row, dot_nvfp4_fp4}`. Only in an `sm_120a` build: the
// block-scaled FP4 `mma` is architecture-specific (build.rs).
#ifdef INFERRED_NVFP4_BLOCK_SCALE

// `D = (A * scale_A) * (B * scale_B) + D` over a 16x8x64 tile, in place.
//
// Operand layout, PTX ISA 9.7.16.5.11 with the scale-A pair order from
// llama.cpp's `vec_dot_fp4_fp4_mma`, verified value by value by
// `the_nvfp4_block_scaled_mma_follows_the_isa`. With `g = lane >> 2` and
// `t = lane & 3`:
//
//   A   a0: row g,     k = 8t + 0..7        a1: row g + 8, k = 8t + 0..7
//       a2: row g,     k = 32 + 8t + 0..7   a3: row g + 8, k = 32 + 8t + 0..7
//   B   b0: col g,     k = 8t + 0..7        b1: col g,     k = 32 + 8t + 0..7
//   D   d0, d1: row g, cols 2t, 2t + 1      d2, d3: row g + 8, cols 2t, 2t + 1
//   scale_A for row g from lane 4g, for row g + 8 from lane 4g + 1;
//   scale_B for col c from lane 4c; byte c of a scale register is chunk c.
__device__ __forceinline__ void mma_nvfp4_inplace(
        float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2],
        unsigned sa, unsigned sb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sa), "r"(sb));
}

// `ggml_cuda_float_to_fp4_e2m1` (ggml-cuda/common.cuh): nearest E2M1 magnitude
// to `|x| * e`, the first index winning a tie, sign in bit 3 when `x < 0`.
__device__ __forceinline__ unsigned nvfp4_e2m1_code(float x, float e) {
    const float pos[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float ax = fabsf(x) * e;
    int best_i = 0;
    float best_err = fabsf(ax - pos[0]);
    for (int i = 1; i < 8; ++i) {
        const float err = fabsf(ax - pos[i]);
        if (err < best_err) {
            best_err = err;
            best_i = i;
        }
    }
    return (unsigned)best_i | (x < 0.0f ? 8u : 0u);
}

// An activation to FP4, one thread per 16-element sub-block over the whole
// batch: `quantize_mmq_nvfp4` (ggml-cuda/quantize.cu), and `ops::naive::Fp4Row`
// on the host, bit for bit.
//
// Output layout is the NVFP4 weight's, so the tensor-core kernel loads both
// operands as raw words: `x_d` one UE4M3 code per sub-block, `x_qs` 8 bytes per
// sub-block whose byte `k` is element `k` in the low nibble and `k + 8` in the
// high. `n_in` is a multiple of 64, so a token's sub-blocks tile its blocks.
__global__ void quantize_nvfp4_act(int n_sub,
                                   const float *__restrict__ x,
                                   unsigned char *__restrict__ x_d,
                                   unsigned char *__restrict__ x_qs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sub) return;
    const float *v = x + (size_t)i * 16;

    float amax = 0.0f;
    for (int k = 0; k < 16; ++k) amax = fmaxf(amax, fabsf(v[k]));
    // `ggml_cuda_fp32_to_ue4m3`: CUDA's FP8 conversion, round to nearest even,
    // saturating at 448.
    const float s0 = amax / 6.0f;
    int seed = 0;
    if (s0 > 0.0f) {
        const __nv_fp8_e4m3 f8(s0);
        seed = (int)f8.__x;
    }

    const int offs[5] = {0, -1, 1, -2, 2};
    float best_err = 3.40282347e+38f;
    float best_scale = 0.0f;
    unsigned best_code = 0;
    for (int o = 0; o < 5; ++o) {
        const int code = seed + offs[o];
        if (code < 0 || code > 0x7e) continue;
        const float scale = nvfp4_ue4m3((unsigned)code);
        const float inv = scale > 0.0f ? 0.5f / scale : 0.0f;
        float err = 0.0f;
        for (int k = 0; k < 16; ++k) {
            const unsigned q = nvfp4_e2m1_code(v[k], inv);
            const float d = fabsf(v[k]) - fabsf((float)nvfp4_kv(q & 7)) * scale;
            err = __fmaf_rn(d, d, err);
        }
        if (err < best_err) {
            best_err = err;
            best_code = (unsigned)code;
            best_scale = scale;
        }
    }

    const float inv = best_scale > 0.0f ? 0.5f / best_scale : 0.0f;
    x_d[i] = (unsigned char)best_code;
    for (int k = 0; k < 8; ++k) {
        x_qs[(size_t)i * 8 + k] = (unsigned char)(nvfp4_e2m1_code(v[k], inv) |
                                                  (nvfp4_e2m1_code(v[k + 8], inv) << 4));
    }
}

// Dense FP4 x FP4 on the tensor cores: one warp per 16 weight rows x 8 tokens,
// one `mma` per 64-element block. The shared expert and the LM head.
//
// **Every operand is a raw 4-byte load.** Word `4t` of a block's nibbles is the
// A register for `k = 8t..8t+7` and word `16 + 4t` for `k = 32 + 8t..`, and the
// activation packs its codes the same way, so within every 16-element chunk
// weight and activation share one permutation of `k` and the dot product is
// the one `ops::naive::dot_nvfp4_fp4` defines. The f32 accumulation order is the
// core's, so this answers to the reference within the chain bound, not to the
// bit. Rows and tokens past the end read the last valid one and are dropped.
__global__ void matmul_nvfp4_fp4_mma(int n_in, int n_out, int n_tok,
                                     const unsigned char *__restrict__ w,
                                     const unsigned char *__restrict__ x_d,
                                     const unsigned char *__restrict__ x_qs,
                                     float *__restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 16;
    const int t0 = blockIdx.y * 8;
    if (j0 >= n_out || t0 >= n_tok) return;

    const int nb = n_in / 64;
    const int g = lane >> 2, t = lane & 3;
    const int ra = (j0 + g < n_out) ? j0 + g : n_out - 1;
    const int rb = (j0 + g + 8 < n_out) ? j0 + g + 8 : n_out - 1;
    const int tk = (t0 + g < n_tok) ? t0 + g : n_tok - 1;
    const unsigned char *wa = w + (size_t)ra * nb * 36;
    const unsigned char *wb = w + (size_t)rb * nb * 36;
    const unsigned char *xq = x_qs + (size_t)tk * nb * 32;
    const unsigned char *xd = x_d + (size_t)tk * nb * 4;

    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int ib = 0; ib < nb; ++ib) {
        const unsigned char *ba = wa + (size_t)ib * 36;
        const unsigned char *bb = wb + (size_t)ib * 36;
        const unsigned char *bq = xq + (size_t)ib * 32;
        const unsigned a[4] = {
            *(const unsigned *)(ba + 4 + 4 * t),  *(const unsigned *)(bb + 4 + 4 * t),
            *(const unsigned *)(ba + 20 + 4 * t), *(const unsigned *)(bb + 20 + 4 * t),
        };
        const unsigned b[2] = {*(const unsigned *)(bq + 4 * t), *(const unsigned *)(bq + 16 + 4 * t)};
        // Lane 4g supplies row g's scales and lane 4g + 1 row g + 8's; the
        // other two lanes' are not read. Lane 4c supplies column c's.
        const unsigned sa = *(const unsigned *)((t == 1) ? bb : ba);
        const unsigned sb = *(const unsigned *)(xd + (size_t)ib * 4);
        mma_nvfp4_inplace(d, a, b, sa, sb);
    }

    for (int i = 0; i < 4; ++i) {
        const int row = (i < 2) ? j0 + g : j0 + g + 8;
        const int tok = t0 + 2 * t + (i & 1);
        if (row < n_out && tok < n_tok) out[(size_t)tok * n_out + row] = d[i];
    }
}

// The routed `down` matmul for one expert tile as FP4 x FP4: the dense kernel's
// warp over `MOE_MMA_TOK` pairs cut into `MOE_MMA_NTILE` sub-tiles of 8, which
// share one weight load. `x` is one row per pair.
//
// Sub-tile `n`'s B column `g` is the activation of pair `perm[first + 8n + g]`
// and its D column `2t + (i & 1)` the output of pair `perm[first + 8n + 2t +
// (i & 1)]`. A slot past the tile loads zeros, which only reach D columns whose
// own slots are past the tile, and those are dropped.
__global__ void matmul_nvfp4_fp4_moe_grouped_mma(
        int n_in, int n_out, int has_s,
        const int *__restrict__ n_tile,
        const int *__restrict__ perm,
        const int *__restrict__ tile_first,
        const int *__restrict__ tile_n,
        const int *__restrict__ ids,
        const unsigned long long *__restrict__ wptrs,
        const float *__restrict__ scale,
        const unsigned char *__restrict__ x_d,
        const unsigned char *__restrict__ x_qs,
        float *__restrict__ out) {
    const int tl = blockIdx.y;
    if (tl >= *n_tile) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 16;
    if (j0 >= n_out) return;

    const int nb = n_in / 64;
    const int first = tile_first[tl];
    const int nt = tile_n[tl];
    const int g = lane >> 2, t = lane & 3;
    const int p0 = perm[first];
    const int e = ids[p0];

    int pb[MOE_MMA_NTILE];
#pragma unroll
    for (int n = 0; n < MOE_MMA_NTILE; ++n) {
        const int sb = n * 8 + g;
        pb[n] = (sb < nt) ? perm[first + sb] : -1;
    }

    // Every pair in the tile shares an expert, so one pointer serves it.
    const unsigned char *w = (const unsigned char *)wptrs[p0];
    const int ra = (j0 + g < n_out) ? j0 + g : n_out - 1;
    const int rb = (j0 + g + 8 < n_out) ? j0 + g + 8 : n_out - 1;
    const unsigned char *wa = w + (size_t)ra * nb * 36;
    const unsigned char *wb = w + (size_t)rb * nb * 36;

    float d[MOE_MMA_NTILE][4];
#pragma unroll
    for (int n = 0; n < MOE_MMA_NTILE; ++n) {
#pragma unroll
        for (int k = 0; k < 4; ++k) d[n][k] = 0.0f;
    }

    for (int ib = 0; ib < nb; ++ib) {
        const unsigned char *ba = wa + (size_t)ib * 36;
        const unsigned char *bb = wb + (size_t)ib * 36;
        const unsigned a[4] = {
            *(const unsigned *)(ba + 4 + 4 * t),  *(const unsigned *)(bb + 4 + 4 * t),
            *(const unsigned *)(ba + 20 + 4 * t), *(const unsigned *)(bb + 20 + 4 * t),
        };
        const unsigned sa = *(const unsigned *)((t == 1) ? bb : ba);
#pragma unroll
        for (int n = 0; n < MOE_MMA_NTILE; ++n) {
            unsigned b[2] = {0u, 0u};
            unsigned sb = 0u;
            if (pb[n] >= 0) {
                const unsigned char *bq = x_qs + (size_t)pb[n] * nb * 32 + (size_t)ib * 32;
                b[0] = *(const unsigned *)(bq + 4 * t);
                b[1] = *(const unsigned *)(bq + 16 + 4 * t);
                sb = *(const unsigned *)(x_d + (size_t)pb[n] * nb * 4 + (size_t)ib * 4);
            }
            mma_nvfp4_inplace(d[n], a, b, sa, sb);
        }
    }

    const float s = has_s ? scale[e] : 1.0f;
#pragma unroll
    for (int n = 0; n < MOE_MMA_NTILE; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int slot = n * 8 + 2 * t + (i & 1);
            const int row = (i < 2) ? j0 + g : j0 + g + 8;
            if (slot < nt && row < n_out) {
                float v = d[n][i];
                if (has_s) v *= s;
                out[(size_t)perm[first + slot] * n_out + row] = v;
            }
        }
    }
}

// Gate, up and the SiLU gating for one expert tile as FP4 x FP4. `x` is one
// row per *token*: sub-tile `n`'s B column `g` reads the token of pair
// `perm[first + 8n + g]`, and that one load serves both matrices. Each pick's
// gate and up are scaled by its expert's second scale before the gating reads
// them, as `Ops::moe_glu`'s default does through `matmul_experts`.
__global__ void matmul_nvfp4_fp4_moe_glu_grouped_mma(
        int n_in, int n_ff, int n_used, int has_gs, int has_us,
        const int *__restrict__ n_tile,
        const int *__restrict__ perm,
        const int *__restrict__ tile_first,
        const int *__restrict__ tile_n,
        const int *__restrict__ ids,
        const unsigned long long *__restrict__ gptrs,
        const unsigned long long *__restrict__ uptrs,
        const float *__restrict__ gscale,
        const float *__restrict__ uscale,
        const unsigned char *__restrict__ x_d,
        const unsigned char *__restrict__ x_qs,
        float *__restrict__ out) {
    const int tl = blockIdx.y;
    if (tl >= *n_tile) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 16;
    if (j0 >= n_ff) return;

    const int nb = n_in / 64;
    const int first = tile_first[tl];
    const int nt = tile_n[tl];
    const int g = lane >> 2, t = lane & 3;
    const int p0 = perm[first];
    const int e = ids[p0];

    int tb[MOE_MMA_NTILE];
#pragma unroll
    for (int n = 0; n < MOE_MMA_NTILE; ++n) {
        const int sb = n * 8 + g;
        tb[n] = (sb < nt) ? perm[first + sb] / n_used : -1;
    }

    const unsigned char *gw = (const unsigned char *)gptrs[p0];
    const unsigned char *uw = (const unsigned char *)uptrs[p0];
    const int ra = (j0 + g < n_ff) ? j0 + g : n_ff - 1;
    const int rb = (j0 + g + 8 < n_ff) ? j0 + g + 8 : n_ff - 1;
    const size_t oa = (size_t)ra * nb * 36, ob = (size_t)rb * nb * 36;

    float dg[MOE_MMA_NTILE][4], du[MOE_MMA_NTILE][4];
#pragma unroll
    for (int n = 0; n < MOE_MMA_NTILE; ++n) {
#pragma unroll
        for (int k = 0; k < 4; ++k) { dg[n][k] = 0.0f; du[n][k] = 0.0f; }
    }

    for (int ib = 0; ib < nb; ++ib) {
        const size_t o = (size_t)ib * 36;
        const unsigned char *gba = gw + oa + o, *gbb = gw + ob + o;
        const unsigned char *uba = uw + oa + o, *ubb = uw + ob + o;
        const unsigned ga[4] = {
            *(const unsigned *)(gba + 4 + 4 * t),  *(const unsigned *)(gbb + 4 + 4 * t),
            *(const unsigned *)(gba + 20 + 4 * t), *(const unsigned *)(gbb + 20 + 4 * t),
        };
        const unsigned ua[4] = {
            *(const unsigned *)(uba + 4 + 4 * t),  *(const unsigned *)(ubb + 4 + 4 * t),
            *(const unsigned *)(uba + 20 + 4 * t), *(const unsigned *)(ubb + 20 + 4 * t),
        };
        const unsigned sga = *(const unsigned *)((t == 1) ? gbb : gba);
        const unsigned sua = *(const unsigned *)((t == 1) ? ubb : uba);
#pragma unroll
        for (int n = 0; n < MOE_MMA_NTILE; ++n) {
            unsigned b[2] = {0u, 0u};
            unsigned sb = 0u;
            if (tb[n] >= 0) {
                const unsigned char *bq = x_qs + (size_t)tb[n] * nb * 32 + (size_t)ib * 32;
                b[0] = *(const unsigned *)(bq + 4 * t);
                b[1] = *(const unsigned *)(bq + 16 + 4 * t);
                sb = *(const unsigned *)(x_d + (size_t)tb[n] * nb * 4 + (size_t)ib * 4);
            }
            mma_nvfp4_inplace(dg[n], ga, b, sga, sb);
            mma_nvfp4_inplace(du[n], ua, b, sua, sb);
        }
    }

    const float gs = has_gs ? gscale[e] : 1.0f;
    const float us = has_us ? uscale[e] : 1.0f;
#pragma unroll
    for (int n = 0; n < MOE_MMA_NTILE; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int slot = n * 8 + 2 * t + (i & 1);
            const int row = (i < 2) ? j0 + g : j0 + g + 8;
            if (slot < nt && row < n_ff) {
                float gv = dg[n][i];
                float uv = du[n][i];
                if (has_gs) gv *= gs;
                if (has_us) uv *= us;
                out[(size_t)perm[first + slot] * n_ff + row] = gv / (1.0f + expf(-gv)) * uv;
            }
        }
    }
}

#endif  // INFERRED_NVFP4_BLOCK_SCALE

}  // extern "C"
