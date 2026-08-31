// CUDA kernels, compiled to PTX by build.rs and loaded through the driver API.
//
// Compiled with --fmad=false. That is not a tuning choice: nvcc contracts
// `a * b + c` into an FMA by default, rounding once where the CPU rounds twice,
// and the first kernel here is meant to be compared against the CPU oracle for
// *exact* equality. Every kernel in this file must keep that property or say
// loudly that it does not.

#include <cuda_fp16.h>

extern "C" {

// Toolchain proof. Nothing depends on this; it exists so a failure to build,
// load, launch, or copy back is diagnosed on its own rather than inside a
// matmul.
__global__ void saxpy(int n, float a, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

// Q8_0 matrix-vector, one thread per output row.
//
// **Deliberately serial within a row.** Each thread walks its whole row in
// index order and accumulates into one f32, which is exactly what
// `ops::naive::dot_q8_0_q8_0` does. A block-wide reduction would be far faster
// and would also change the f32 accumulation order, so the results could no
// longer be compared bit for bit against the oracle. Getting a correct kernel
// first, and paying for speed knowingly, is the order this project works in.
//
// The cost of that choice is real: adjacent threads read rows that are
// `n_in/32 * 34` bytes apart, so the loads do not coalesce. Fixing that is the
// next kernel, and it is where the bit-exactness conversation has to happen.
//
// Layout matches the file: each row is `n_in/32` blocks of [f16 scale][32 i8].
// The activation arrives already quantized to Q8_0 by the caller, because that
// is what ggml does and what makes this comparable to the CPU path.
__global__ void matmul_q8_0(int n_in, int n_out,
                            const unsigned char *__restrict__ w,
                            const float *__restrict__ x_scales,
                            const signed char *__restrict__ x_quants,
                            float *__restrict__ out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;

    const int n_blocks = n_in / 32;
    const unsigned char *row = w + (size_t)j * (size_t)n_blocks * 34;

    float sumf = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char *blk = row + (size_t)b * 34;
        // Little-endian f16 scale, as stored.
        unsigned short dbits =
            (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        float dw = __half2float(__ushort_as_half(dbits));

        int sumi = 0;
        const signed char *xq = x_quants + b * 32;
        for (int k = 0; k < 32; ++k) {
            sumi += (int)((signed char)blk[2 + k]) * (int)xq[k];
        }
        sumf += (float)sumi * (dw * x_scales[b]);
    }
    out[j] = sumf;
}

// ---------------------------------------------------------------- the rest of
// the forward pass. Every kernel below mirrors `ops::naive` statement for
// statement, including accumulation order, so results can be compared against
// the CPU oracle rather than merely sanity-checked.
//
// Where exactness is *not* attainable it is called out on the kernel. The one
// systematic obstacle is `expf`: CUDA's is not obliged to agree with glibc's to
// the last bit, so anything exponential (softmax, silu, attention) can differ
// by an ulp for reasons that have nothing to do with this code.

__device__ inline float h2f(unsigned short bits) {
    return __half2float(__ushort_as_half(bits));
}

// RMSNorm, one block for the whole vector.
//
// Thread 0 accumulates the sum of squares serially in *double*, which is what
// ggml_compute_forward_rms_norm_f32 does and what ops::naive::rms_scale
// reproduces. That accumulator is load-bearing: summing 1024 squares in f32
// shifts the scale enough to move activations across Q8_0 boundaries in every
// matmul downstream. A parallel tree reduction would be far faster and would
// not be the same number, so the block waits.
__global__ void rms_norm(int n, const float *__restrict__ x,
                         const float *__restrict__ w, float eps,
                         float *__restrict__ out) {
    __shared__ float scale;
    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            float v = x[i];
            sum += (double)(v * v);
        }
        float mean = (float)(sum / (double)n);
        scale = 1.0f / sqrtf(mean + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = x[i] * scale * w[i];
    }
}

// Per-head RMSNorm, in place. One block per head, same serial-double rule.
__global__ void rms_norm_heads(int head_dim, const float *__restrict__ w,
                               float eps, float *__restrict__ x) {
    __shared__ float scale;
    float *head = x + (size_t)blockIdx.x * head_dim;
    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int i = 0; i < head_dim; ++i) {
            float v = head[i];
            sum += (double)(v * v);
        }
        float mean = (float)(sum / (double)head_dim);
        scale = 1.0f / sqrtf(mean + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        head[i] = head[i] * scale * w[i];
    }
}

// NEOX RoPE, in place.
//
// The cosines and sines arrive precomputed from the host. That is not an
// optimization: ops::naive derives theta with f64 powf and sin_cos from glibc,
// and CUDA's double-precision pow and sincos are not obliged to return the same
// bits. Computing the table once on the CPU costs head_dim/2 transcendentals
// per call and makes this kernel exactly the oracle's arithmetic.
__global__ void rope_neox(int head_dim, int n_heads,
                          const float *__restrict__ cosv,
                          const float *__restrict__ sinv, float *__restrict__ x) {
    const int half = head_dim / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_heads * half) return;

    float *head = x + (size_t)(idx / half) * head_dim;
    const int i = idx % half;
    const float c = cosv[i], s = sinv[i];

    // NEOX pairs i with i + head_dim/2, not with i + 1.
    const float x0 = head[i];
    const float x1 = head[i + half];
    head[i] = x0 * c - x1 * s;
    head[i + half] = x0 * s + x1 * c;
}

// Softmax over each of `n_rows` contiguous rows, one thread per row.
//
// Serial within a row, max subtracted first, exactly as
// ops::naive::softmax_in_place. Not bit-exact against the CPU: expf.
__global__ void softmax_rows(int n, int n_rows, float *__restrict__ x) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    float *row = x + (size_t)r * n;

    float mx = -INFINITY;
    for (int i = 0; i < n; ++i) mx = fmaxf(mx, row[i]);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        row[i] = expf(row[i] - mx);
        sum += row[i];
    }
    for (int i = 0; i < n; ++i) row[i] /= sum;
}

// Whole-token attention, **one thread per query head**.
//
// That is a deliberately poor decomposition for a GPU — 16 threads for the
// 0.6B, so 16 of 36 SMs hold one warp lane each — but it is the decomposition
// that reproduces the oracle: every dot product and every weighted sum keeps
// its serial index order, and each query head owns disjoint output. Splitting a
// head across threads is the next kernel, and it is where the accumulation
// order has to change.
//
// `scores` is caller-provided scratch, n_head * n_pos floats.
__global__ void attend(int n_pos, int kv_dim, int head_dim, int n_head,
                       int n_head_kv, float scale, const float *__restrict__ q,
                       const unsigned short *__restrict__ k,
                       const unsigned short *__restrict__ v,
                       float *__restrict__ scores, float *__restrict__ out) {
    int hq = blockIdx.x * blockDim.x + threadIdx.x;
    if (hq >= n_head) return;

    const int group = n_head / n_head_kv;
    const int off = (hq / group) * head_dim;   // this query head's kv head
    const float *qh = q + (size_t)hq * head_dim;
    float *sc = scores + (size_t)hq * n_pos;

    for (int s = 0; s < n_pos; ++s) {
        const unsigned short *key = k + (size_t)s * kv_dim + off;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i) dot += qh[i] * h2f(key[i]);
        sc[s] = dot * scale;
    }

    float mx = -INFINITY;
    for (int s = 0; s < n_pos; ++s) mx = fmaxf(mx, sc[s]);
    float sum = 0.0f;
    for (int s = 0; s < n_pos; ++s) {
        sc[s] = expf(sc[s] - mx);
        sum += sc[s];
    }
    for (int s = 0; s < n_pos; ++s) sc[s] /= sum;

    float *o = out + (size_t)hq * head_dim;
    for (int i = 0; i < head_dim; ++i) o[i] = 0.0f;
    for (int s = 0; s < n_pos; ++s) {
        const unsigned short *val = v + (size_t)s * kv_dim + off;
        const float w = sc[s];
        for (int i = 0; i < head_dim; ++i) o[i] += w * h2f(val[i]);
    }
}

// SwiGLU: gate = silu(gate) * up, in place. Not bit-exact: expf.
__global__ void silu_mul(int n, float *__restrict__ gate,
                         const float *__restrict__ up) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    gate[i] = g / (1.0f + expf(-g)) * up[i];
}

// Residual add, in place.
__global__ void add_assign(int n, float *__restrict__ a,
                           const float *__restrict__ b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// Q8_0 matrix-vector, **one warp per output row, one Q8_0 block per lane**.
//
// This is the fast kernel, and it is still bit-exact. The earlier attempt to
// get coalescing by repacking the weight was measured and reverted for being
// slower; the mistake there was treating the layout as the problem. The layout
// is fine — what was wrong was giving a whole row to a single thread.
//
// The trick is that Q8_0 has a natural unit of work that is *already* exact.
// Within one 32-element block the sum of products is an **integer** sum, so it
// cannot round at all, and the order it happens in does not matter. The only
// f32 accumulation in the whole dot product is the one across blocks:
//
//     sumf += (float)sumi * (dw * x_scales[b])      for b = 0, 1, 2, ...
//
// So: lane `b % 32` computes block `b` and leaves its f32 contribution in
// shared memory, then lane 0 adds those up **in ascending b** — the oracle's
// order, exactly. Parallel where the arithmetic is order-free, serial where it
// is not.
//
// The reads come out right as a side effect. At any instant the warp's 32 lanes
// are inside 32 consecutive 34-byte blocks, so they cover a 1088-byte window
// that is fully used, instead of 32 separate rows a kilobyte apart.
//
// Shared memory is `warps_per_block * n_blocks` floats, sized at launch.
__global__ void matmul_q8_0_warp(int n_in, int n_out,
                                 const unsigned char *__restrict__ w,
                                 const float *__restrict__ x_scales,
                                 const signed char *__restrict__ x_quants,
                                 float *__restrict__ out) {
    extern __shared__ float partial[];

    const int n_blocks = n_in / 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;

    float *mine = partial + (size_t)warp * n_blocks;
    const unsigned char *row = w + (size_t)j * (size_t)n_blocks * 34;

    for (int b = lane; b < n_blocks; b += 32) {
        const unsigned char *blk = row + (size_t)b * 34;
        unsigned short dbits =
            (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        float dw = __half2float(__ushort_as_half(dbits));

        // Integer, so exact and order-free.
        int sumi = 0;
        const signed char *xq = x_quants + b * 32;
        for (int k = 0; k < 32; ++k) {
            sumi += (int)((signed char)blk[2 + k]) * (int)xq[k];
        }
        mine[b] = (float)sumi * (dw * x_scales[b]);
    }
    __syncwarp();

    // The one ordered accumulation, kept serial and ascending.
    if (lane == 0) {
        float sumf = 0.0f;
        for (int b = 0; b < n_blocks; ++b) sumf += mine[b];
        out[j] = sumf;
    }
}

// Quantize an activation to Q8_0, one thread per 32-element block.
//
// This exists so a matmul's input never has to come back to the host. It was
// done on the CPU at first precisely because it is the one place a rounding
// mode could differ: the scale is stored as f16 and read back, so `d` must
// round the way `quant::half::f32_to_f16` rounds it. Both are round-to-nearest-
// even, and `tests/cuda_ops.rs` checks that on real data rather than trusting
// the docs.
//
// Mirrors `quantize_row_q8_0_ref` in ggml-quants.c, as ops::naive does.
__global__ void quantize_q8_0(int n_blocks, const float *__restrict__ x,
                              float *__restrict__ scales,
                              signed char *__restrict__ quants) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    const float *blk = x + (size_t)b * 32;
    float amax = 0.0f;
    for (int k = 0; k < 32; ++k) amax = fmaxf(amax, fabsf(blk[k]));

    const float d = amax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    // Stored as f16 and read back, exactly as the reference does.
    scales[b] = __half2float(__float2half(d));

    signed char *q = quants + (size_t)b * 32;
    for (int k = 0; k < 32; ++k) {
        // roundf is half-away-from-zero, which is what Rust's f32::round does.
        q[k] = (signed char)roundf(blk[k] * id);
    }
}

} // extern "C"
