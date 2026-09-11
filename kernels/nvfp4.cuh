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

}  // extern "C"
