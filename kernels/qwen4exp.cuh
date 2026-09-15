// qwen4exp.cuh -- Qwen3.8-Flash-Next's hyper-connections and PLE block.
//
// Included by kernels.cu, which pulls every file into one compile unit and one
// PTX module in dependency order. Kernel names are the PTX symbols the backend
// looks up, so which file a kernel lives in changes nothing at run time.
//
// The device forms of the seam's qwen4exp ops, whose scalar trait defaults in
// src/ops/mod.rs are the oracle (src/model/qwen4exp.md, "Our implementation").
// Nothing the 35B runs launches any of these.
//
// Exactness, per kernel, against those defaults:
//   bit-identical   mul_rows, mul_streams, row_dot, dilated_conv(_state)
//   expf class      silu_f32, sigmoid_f32, signed_sqrt_sigmoid
// The first four are f32 products, an f64 serial sum and an f32 tap chain in the
// oracle's order, which --fmad=false keeps from contracting. The last three call
// expf, which CUDA is not obliged to round as glibc does -- the same class as
// silu_mul and sigmoid_mul.
#pragma once

extern "C" {

// x[i] *= w[i % n_w], in place: ggml_mul by a broadcast weight. A
// hyper-connection norm's weight spans all four streams (hc_dim), so it is
// applied after a per-stream RMSNorm with a unit weight.
extern "C" __global__ void mul_rows(int n, int n_w, float *__restrict__ x,
                                    const float *__restrict__ w) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= w[i % n_w];
}

// x = silu(x), in place: ggml_silu_f32, x / (1 + exp(-x)). expf class.
extern "C" __global__ void silu_f32(int n, float *__restrict__ x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = v / (1.0f + expf(-v));
}

// x = sigmoid(x), in place: ggml_vec_sigmoid_f32, 1 / (1 + exp(-x)). expf class.
extern "C" __global__ void sigmoid_f32(int n, float *__restrict__ x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = 1.0f / (1.0f + expf(-x[i]));
}

// out[t][s][j] = h[t][j] * w[t][s]: one row per token broadcast over the streams
// and scaled per stream -- ggml_mul(ggml_repeat_4d(h), w). One thread per output
// element, so each is one product and nothing accumulates.
extern "C" __global__ void mul_streams(int n_out, int n_embd, int n_stream,
                                       const float *__restrict__ h,
                                       const float *__restrict__ w,
                                       float *__restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_out) return;
    const int row = i / n_embd;          // t * n_stream + s
    const int j = i - row * n_embd;
    const int t = row / n_stream;
    out[i] = h[(size_t)t * n_embd + j] * w[row];
}

// out[r] = sum_j a[r][j] * b[r][j]: ggml_mul then ggml_sum_rows, whose sum is
// ggml_float. The products are f32 and the accumulator f64, walked serially and
// ascending, then narrowed -- the oracle's order exactly. One thread per row.
extern "C" __global__ void row_dot(int n_rows, int width,
                                   const float *__restrict__ a,
                                   const float *__restrict__ b,
                                   float *__restrict__ out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const float *ar = a + (size_t)r * width;
    const float *br = b + (size_t)r * width;
    double sum = 0.0;
    for (int j = 0; j < width; ++j) sum += (double)(ar[j] * br[j]);
    out[r] = (float)sum;
}

// PLE's gate, in place: sigmoid(sgn(s) * sqrt(max(|s|, 1e-6))), the chain
// ggml_abs, ggml_clamp, ggml_sqrt, ggml_sgn, ggml_mul, ggml_sigmoid in build_ple
// (qwen4exp.cpp:1219-1220). fabsf, fmaxf and sqrtf are correctly rounded, so
// only the expf differs from the oracle. sgn(0) = 0.
extern "C" __global__ void signed_sqrt_sigmoid(int n, float *__restrict__ s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = s[i];
    const float mag = sqrtf(fmaxf(fabsf(v), 1e-6f));
    const float sgn = v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f);
    const float g = sgn * mag;
    s[i] = 1.0f / (1.0f + expf(-g));
}

// PLE's conv over a whole batch: depthwise, causal, dilated, no activation
// (build_ple, qwen4exp.cpp:1235-1276). One thread per (channel, token).
//
// The oracle pads the batch with the state in front, `pad = state ++ x`, and
// reads tap k of token t at `pad[hist + t - (kernel-1-k) * dilation]`. Relative to
// the batch that is `p = t - (kernel-1-k) * dilation`: `p < 0` is a sample the
// last pass left in the state, at `state[c][hist + p]`. Taps are summed in k order
// starting from tap 0's product, exactly as the oracle's chain.
//
// Outputs read only the incoming state, so they are independent; the state is
// advanced by `dilated_conv_state`, launched after this on the same stream.
extern "C" __global__ void dilated_conv(int n_channels, int kernel, int dilation, int n_tok,
                                        const float *__restrict__ state,
                                        const float *__restrict__ x,
                                        const float *__restrict__ w,
                                        float *__restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;
    const int t = blockIdx.y;
    const int hist = (kernel - 1) * dilation;
    const float *past = state + (size_t)c * hist;
    const float *wc = w + (size_t)c * kernel;

    float acc = 0.0f;
    for (int k = 0; k < kernel; ++k) {
        const int p = t - (kernel - 1 - k) * dilation;
        const float v = (p < 0) ? past[hist + p] : x[(size_t)p * n_channels + c];
        if (k == 0) {
            acc = v * wc[0];
        } else {
            acc += v * wc[k];
        }
    }
    out[(size_t)t * n_channels + c] = acc;
}

// The history the next pass inherits: the last `hist` samples of `state ++ x`.
//
// New `state[i]` is `pad[n_tok + i]`, which is `state[n_tok + i]` while that
// index is still inside the state and `x[n_tok + i - hist]` past it. Because
// `n_tok >= 1`, the state index read is always ahead of the one written, so an
// ascending in-place walk never reads a sample it has already overwritten -- no
// staging buffer, and no cap on `hist`.
extern "C" __global__ void dilated_conv_state(int n_channels, int kernel, int dilation,
                                              int n_tok,
                                              float *__restrict__ state,
                                              const float *__restrict__ x) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;
    const int hist = (kernel - 1) * dilation;
    float *past = state + (size_t)c * hist;
    for (int i = 0; i < hist; ++i) {
        const int j = n_tok + i;
        past[i] = (j < hist) ? past[j] : x[(size_t)(j - hist) * n_channels + c];
    }
}

}  // extern "C"
