// core.cuh -- the small kernels everything else leans on: h2f, RMSNorm, RoPE, softmax, SwiGLU and residual adds, gather/scatter, noop.
//
// Included by kernels.cu, which pulls every file into one compile unit and one
// PTX module in dependency order. Kernel names are the PTX symbols the backend
// looks up, so which file a kernel lives in changes nothing at run time.
// Kernels are in `extern "C"` blocks; templates sit outside them, since a
// template cannot have C linkage.
#pragma once

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

// RMSNorm, one block for the whole vector — the **serial** pair, reached with
// `--rms-serial`. `rms_norm_tree` below is the default; this is kept because it
// is the only version bit-identical to `ops::naive`, and determinism is hard to
// get back once it is given up.
//
// The sum of squares is accumulated serially in *double*, which is what
// ggml_compute_forward_rms_norm_f32 does and what ops::naive::rms_scale
// reproduces. The double accumulator is load-bearing and the serial order is
// load-bearing, for two different reasons.
//
// **Double, because f32 is a known bug.** Summing 1024 squares in f32 shifts
// the scale by ~1e-5 relative, which is invisible in a printed tensor and is
// enough to move activations across Q8_0 boundaries in every matmul
// downstream. That was found the hard way and is recorded in `CLAUDE.md`.
//
// **Serial, because the rule is that a redistributing backend reproduces the
// oracle exactly**, which is what lets every differential test here demand
// equal bits rather than a tolerance.
//
// # What that costs, measured
//
// `tests/cuda_ops.rs::why_is_the_rms_reduction_slow` varies one thing at a
// time, in us for 1024 elements:
//
//     serial f64, global   62.5      serial f32, global   13.0
//     serial f64, shared   51.6      serial f32, shared    8.4
//     tree f64             11.7
//
// So the cost is **FP64 latency on a dependent chain** — 4.8x on the same
// memory path — and not the compiler, the loads, or occupancy, which is what
// was assumed the first time. FP64 *throughput* here is 1/64 of FP32, but a
// dependent chain is a latency problem and the two are not the same number.
//
// Staging through shared memory is worth a real 18% and changes no bit, since
// the squares are per-element and independent; only the sum is ordered. That
// is taken below. The remaining 4x needs the chain broken, which is a tree —
// a different answer rather than a faster one, and the reason this kernel still
// exists. See `rms_norm_tree` for what that trade costs and why it was taken.
__global__ void rms_norm(int n, const float *__restrict__ x,
                         const float *__restrict__ w, float eps,
                         float *__restrict__ out) {
    extern __shared__ float sq[];
    __shared__ float scale;

    // One block per row of the batch. Each row normalizes against its own mean,
    // so nothing accumulates across the block boundary and a batch is
    // bit-identical to the same rows done one at a time.
    x += (size_t)blockIdx.x * n;
    out += (size_t)blockIdx.x * n;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        sq[i] = v * v;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) sum += (double)sq[i];
        float mean = (float)(sum / (double)n);
        scale = 1.0f / sqrtf(mean + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = x[i] * scale * w[i];
    }
}

// Per-head RMSNorm, in place. One block per head, same rule and same shape.
// Serial, so also behind `--rms-serial`.
__global__ void rms_norm_heads(int head_dim, const float *__restrict__ w,
                               float eps, float *__restrict__ x) {
    extern __shared__ float sq[];
    __shared__ float scale;

    float *head = x + (size_t)blockIdx.x * head_dim;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = head[i];
        sq[i] = v * v;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int i = 0; i < head_dim; ++i) sum += (double)sq[i];
        float mean = (float)(sum / (double)head_dim);
        scale = 1.0f / sqrtf(mean + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        head[i] = head[i] * scale * w[i];
    }
}

// The same two, with the sum of squares reduced as a tree. **These are the
// default**; the serial pair above is kept behind `--rms-serial`.
//
// # Why the default changed
//
// The serial f64 chain was ~40% of device time, and the cost is dependent-chain
// FP64 latency: `tests/cuda_ops.rs::why_is_the_rms_reduction_slow` measures it
// linear in n at 50.4 / 97.9 / 374.1 us for n = 1024 / 2048 / 8192, against
// ~12 us for the tree at every size. Only breaking the chain removes it.
//
// # What that costs, stated exactly
//
// A tree is a *different answer*, not a faster one -- f64 addition rounds, so
// it is not associative, and the same test shows this kernel's shape returning
// **four different results at four block sizes** on an adversarial input. It
// therefore cannot be bit-identical to `ops::naive`, and `rms_norm` leaves the
// project's bit-exact set.
//
// The tolerance is n * 2^-53, the worst-case relative error of f64 summation
// over n terms -- ~1.1e-13 at n = 1024. It is *derived*, not fitted to what
// passes. For scale: `attend` already carries a derived tolerance three to four
// orders of magnitude looser, and any real defect here (a wrong index, a missed
// element, the wrong eps) misses by 1e-3 or more.
//
// Note what the f32 cast of `mean` does and does not do. It snaps a 53-bit
// value onto a grid spaced 2^-23 = 1.19e-7 apart, roughly a million times
// coarser than the reorder, so two orders usually land on the same f32 -- but
// "usually" is the honest word. They differ whenever a rounding boundary falls
// between them, about once in 1e9 calls at the measured 1.5e-16. Hidden at the
// rate we sample, not absent.
//
// This is **not** the f64-to-f32 question. That is a precision change of ~1e-5
// which diverges from llama.cpp itself, and is a bug this project already found
// and fixed.
//
// The tree needs no dynamic shared memory: staging the squares existed to feed
// the serial walk, and there is no serial walk here.
__global__ void rms_norm_tree(int n, const float *__restrict__ x,
                              const float *__restrict__ w, float eps,
                              float *__restrict__ out) {
    // One block per row of the batch, as in `rms_norm` above.
    x += (size_t)blockIdx.x * n;
    out += (size_t)blockIdx.x * n;
    __shared__ double p[256];
    __shared__ float scale;

    double acc = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        acc += (double)(v * v);
    }
    p[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) p[threadIdx.x] += p[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float mean = (float)(p[0] / (double)n);
        scale = 1.0f / sqrtf(mean + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = x[i] * scale * w[i];
    }
}

// Per-head, in place. One block per head, same trade and same tolerance.
__global__ void rms_norm_heads_tree(int head_dim, const float *__restrict__ w,
                                    float eps, float *__restrict__ x) {
    __shared__ double p[256];
    __shared__ float scale;

    float *head = x + (size_t)blockIdx.x * head_dim;

    double acc = 0.0;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = head[i];
        acc += (double)(v * v);
    }
    p[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) p[threadIdx.x] += p[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float mean = (float)(p[0] / (double)head_dim);
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
__global__ void rope_neox(int head_dim, int n_rot, int n_heads, int n_tok,
                          const float *__restrict__ cosv,
                          const float *__restrict__ sinv, float *__restrict__ x) {
    // Partial RoPE: only the first `n_rot` of each head rotate and the rest
    // pass through, so the pair stride is n_rot/2 and the head stride stays
    // head_dim. qwen35 rotates 64 of 256; qwen3 passes n_rot == head_dim.
    const int half = n_rot / 2;
    const int per_row = n_heads * half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tok * per_row) return;

    // Consecutive rows are consecutive absolute positions, so the caller sends
    // `n_tok` stacked cos/sin tables and each row reads its own. Rotating every
    // row at row 0's position is the classic KV cache bug -- invisible in a
    // prefill from zero, wrong for everything decoded after.
    const int t = idx / per_row;
    const int within = idx % per_row;

    float *head = x + (size_t)t * n_heads * head_dim + (size_t)(within / half) * head_dim;
    const int i = within % half;
    const float c = cosv[(size_t)t * half + i], s = sinv[(size_t)t * half + i];

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

// Does nothing, launched with the real grid and block. See `--null-kernels`.
//
// **The instrument that separates the two halves of a token.** A decode step
// measures 53.6 ms of which the shape bench accounts for 18.05 ms of kernel
// time, and five hypotheses about the remaining 35 ms have been falsified by
// subtracting one number from another. Replacing every kernel with this one,
// while keeping the launch count, order, grid and block identical, makes the
// residual something the clock reports directly: whatever the token still costs
// is what the work was never responsible for.
//
// The output is garbage, deliberately. This is a stopwatch, not a mode.
extern "C" __global__ void noop() {}

// Pull one `chunk`-sized run out of every `stride` of `src`, starting at
// `offset`. Generic, but it exists for one thing: qwen35's `attn_q` emits query
// and gate interleaved per head, so the two are strided views of one matmul
// result and the model would otherwise de-interleave them on the host --
// dragging the activation home mid-layer and breaking graph capture.
extern "C" __global__ void gather_chunks(int n_out, int chunk, int stride,
                                         int offset,
                                         const float *__restrict__ src,
                                         float *__restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_out) return;
    int c = i / chunk;
    int j = i - c * chunk;
    out[i] = src[c * stride + offset + j];
}

// a += b * scale, elementwise.
//
// The MoE expert accumulation: a routed expert's output is weighted by its
// router probability before it joins the sum. Separate from `add_assign`
// because doing the scale as its own pass would read and write `b` an extra
// time for each of the eight experts a token visits.
//
// A multiply and an add, not an FMA -- `--fmad=false` keeps it that way, which
// is what `ops::naive`'s `a[i] += b[i] * scale` compiles to.
extern "C" __global__ void add_scaled(int n, float scale,
                                      float *__restrict__ a,
                                      const float *__restrict__ b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i] * scale;
}

// acc += b * sigmoid(logit[0]) -- the shared expert's gate.
//
// The scale is read from device memory rather than passed as an argument, and
// that is the whole point: the logit is a matmul result, so taking it as a
// kernel argument would mean copying it to the host first, which drains the
// pipeline. `qwen35moe` did that once per layer per token.
//
// Inexact for the usual reason: expf. Same class as softmax, silu_mul and
// sigmoid_mul, which the exactness boundary already covers.
extern "C" __global__ void add_scaled_sigmoid(int n, const float *__restrict__ logit,
                                              float *__restrict__ acc,
                                              const float *__restrict__ b) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = 1.0f / (1.0f + expf(-logit[0]));
    acc[i] += b[i] * s;
}

// The dual of gather_chunks: contiguous `src` written back into `dst` every
// `stride`, starting at `offset`.
//
// One thread per *source* element, because src is the shorter buffer and every
// one of its elements lands exactly once -- so this cannot race, and nothing
// outside the written windows is touched.
extern "C" __global__ void scatter_chunks(int n_src, int chunk, int stride,
                                          int offset,
                                          const float *__restrict__ src,
                                          float *__restrict__ dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_src) return;
    int c = i / chunk;
    int j = i - c * chunk;
    dst[c * stride + offset + j] = src[i];
}

// x *= sigmoid(g), elementwise. The sibling of silu_mul, and inexact for the
// same reason: expf.
extern "C" __global__ void sigmoid_mul(int n, float *__restrict__ x,
                                       const float *__restrict__ g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= 1.0f / (1.0f + expf(-g[i]));
}

}  // extern "C"
