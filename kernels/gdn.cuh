// gdn.cuh -- GatedDeltaNet: l2 norm, the short convolution and the delta rule.
//
// Included by kernels.cu, which pulls every file into one compile unit and one
// PTX module in dependency order. Kernel names are the PTX symbols the backend
// looks up, so which file a kernel lives in changes nothing at run time.
// Kernels are in `extern "C"` blocks; templates sit outside them, since a
// template cannot have C linkage.
#pragma once

extern "C" {

// ---------------------------------------------------------------- GatedDeltaNet
//
// The three primitives the qwen35 recurrent layer needs. Two are trivial; the
// third is where the architecture's cost lives.
//
// All three take this layer's state slab and leave it on the device. That is
// the point of the seam taking a slab rather than an assembled window: the
// state is ~2 MB per layer and is read and written every token, so shuttling it
// home would cost ~96 MB a token across 24 recurrent layers -- about 3.4 ms at
// the measured 28.6 GB/s, against a whole-token budget of roughly 20 ms on the
// 9B.

// L2 normalization per head, in place. No weight, no division by n, and eps
// clamps the *norm* rather than sitting under the root -- see
// ggml_compute_forward_l2_norm_f32. Confusing it with RMSNorm is wrong by
// exactly sqrt(n).
//
// **Serial f64, so this stays bit-identical to the oracle.** Unlike rms_norm,
// which walks 1024 or 2048 elements and was worth breaking the chain for, a
// head here is 128 elements and 16 heads run as 16 concurrent blocks. The whole
// op is ~6 us a call, so exactness is nearly free and is kept.
extern "C" __global__ void l2_norm_heads(int head_dim, float eps,
                                         float *__restrict__ x) {
    __shared__ float scale;
    float *head = x + (size_t)blockIdx.x * head_dim;

    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int i = 0; i < head_dim; ++i) {
            float v = head[i];
            sum += (double)(v * v);
        }
        // sqrtf takes a float, so `sum` narrows before the root here exactly as
        // it does in the reference. That narrowing is the reference's, not an
        // accident of this transcription.
        scale = 1.0f / fmaxf(sqrtf((float)sum), eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        head[i] *= scale;
    }
}

// Depthwise causal conv1d over the stored window and this token, then silu,
// advancing the state.
//
// One thread per channel. Depthwise means no mixing, so there is nothing to
// reduce and nothing to share -- each thread reads its own `kernel - 1` stored
// samples, its own new sample and its own `kernel` weights.
//
// The accumulator is f32 because ggml_compute_forward_ssm_conv_f32 says
// outright that it avoids ggml_vec_dot_f32 "because its sum is in double
// precision". Bit-identical to the oracle.
extern "C" __global__ void ssm_conv(int n_channels, int kernel,
                                    float *__restrict__ state,
                                    const float *__restrict__ x,
                                    const float *__restrict__ w,
                                    float *__restrict__ out) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;

    const int keep = kernel - 1;
    float *past = state + (size_t)c * keep;
    const float *wc = w + (size_t)c * kernel;

    // Oldest sample first, so tap `keep` is always this token and never comes
    // from the state.
    float sum = 0.0f;
    for (int t = 0; t < keep; ++t) sum += past[t] * wc[t];
    sum += x[c] * wc[keep];
    out[c] = sum / (1.0f + expf(-sum));

    for (int t = 0; t < keep - 1; ++t) past[t] = past[t + 1];
    past[keep - 1] = x[c];
}

// `ssm_conv` over a whole batch, in one launch instead of `n_tok`.
//
// **The per-token loop was never a data dependency.** This is a causal
// depthwise convolution: token `t`'s output reads a fixed window of the `keep`
// samples before it and its own, and nothing token `t` computes feeds token
// `t+1`. What was sequential is only the *state shift* the single-token kernel
// performs after each output, which is why it had to run in order.
//
// Split those apart and the outputs are independent: one thread per (token,
// channel), reading the window from the incoming state where the index falls
// before the batch and from `x` where it does not. `ssm_conv_state` then writes
// the final window once, after every output has read the old one.
//
// Measured on a 4,000-token 35B prefill: `ssm_conv` was 7.8% of device time
// across **120,090 launches** — one per token per GDN layer. This makes it two
// per layer per pass.
//
// Bit-identical: the taps are summed oldest-first and the current sample added
// last, which is the single-token kernel's order exactly. Only which outputs
// share a launch changes.
extern "C" __global__ void ssm_conv_batch(int n_channels, int kernel, int n_tok,
                                          const float *__restrict__ state,
                                          const float *__restrict__ x,
                                          const float *__restrict__ w,
                                          float *__restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;
    const int t = blockIdx.y;

    const int keep = kernel - 1;
    const float *past = state + (size_t)c * keep;
    const float *wc = w + (size_t)c * kernel;

    // Oldest tap first. Index `p` is relative to the batch, so `p < 0` names a
    // sample the previous pass left in the state: `past[0]` is position -keep.
    float sum = 0.0f;
    for (int i = 0; i < keep; ++i) {
        const int p = t - keep + i;
        const float v = (p < 0) ? past[keep + p] : x[(size_t)p * n_channels + c];
        sum += v * wc[i];
    }
    sum += x[(size_t)t * n_channels + c] * wc[keep];
    out[(size_t)t * n_channels + c] = sum / (1.0f + expf(-sum));
}

// The window the next pass inherits: the last `keep` samples of this batch.
//
// A separate launch because every output above must read the *old* state
// first, and kernels on one stream are ordered while threads within one are
// not.
extern "C" __global__ void ssm_conv_state(int n_channels, int kernel, int n_tok,
                                          float *__restrict__ state,
                                          const float *__restrict__ x) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_channels) return;

    const int keep = kernel - 1;
    float *past = state + (size_t)c * keep;

    // Read the whole new window before writing any of it: a short batch takes
    // some of it from the window being overwritten.
    float next[8];
    for (int i = 0; i < keep; ++i) {
        const int p = n_tok - keep + i;
        next[i] = (p < 0) ? past[keep + p] : x[(size_t)p * n_channels + c];
    }
    for (int i = 0; i < keep; ++i) past[i] = next[i];
}

// The gated delta rule: one token, every value head, state updated in place.
//
// # The decomposition
//
// One block per value head, one thread per *value row* of that head's state.
// Row j touches only k, v[j], q and its own 128 floats, so the rows are
// independent and each thread keeps the oracle's serial ascending sum over the
// key axis. That is the same escape the Q8_0 matmul found: parallelize over
// independent outputs and the arithmetic never has to be reordered.
//
// # Two passes, not four
//
// The oracle decays the whole state, reads, corrects and reads again -- four
// sweeps of 2 MB per layer. Folding the decay into the two reads gives the same
// arithmetic in two sweeps: `(s*g)*k` is what the oracle computes, and
// `(s*g) + k*d` is what it stores. With --fmad=false nothing contracts, so this
// is bit-for-bit the oracle's order, at half the traffic.
//
// # What is *not* exact, and why it is accepted here
//
// `g` and `beta` need expf and logf, and CUDA is not obliged to round them as
// glibc does. That puts this kernel in the same class as `softmax` and
// `silu_mul` -- about one ulp -- rather than in the exact set.
//
// The alternative was to precompute both on the host, as `rope_neox` does with
// its sin/cos table. It was rejected: alpha and beta are matmul outputs, so
// they are already on the device, and bringing 64 floats home per layer would
// be 48 round trips a token and would break graph capture, which needs an
// identical launch sequence. Trading one ulp for that is a bad trade.
extern "C" __global__ void delta_rule(int head_k_dim, int head_v_dim,
                                      int n_k_heads, float q_scale,
                                      const float *__restrict__ q,
                                      const float *__restrict__ k,
                                      const float *__restrict__ v,
                                      const float *__restrict__ alpha,
                                      const float *__restrict__ beta_raw,
                                      const float *__restrict__ ssm_a,
                                      const float *__restrict__ dt_bias,
                                      float *__restrict__ state,
                                      float *__restrict__ out) {
    extern __shared__ float sh[];
    float *qs = sh;                 // head_k_dim
    float *ks = sh + head_k_dim;    // head_k_dim
    __shared__ float g, beta;

    const int h = blockIdx.x;
    // Modulo, not division. Value head h reads key head h % n_k_heads: the
    // unfused reference path tiles via ggml_repeat, and the fused kernel writes
    // `iq1 = iv1 % neq1`. Blocked grouping agrees only for h = 0 and h = 1.
    const int kh = h % n_k_heads;

    for (int i = threadIdx.x; i < head_k_dim; i += blockDim.x) {
        qs[i] = q[kh * head_k_dim + i];
        ks[i] = k[kh * head_k_dim + i];
    }
    if (threadIdx.x == 0) {
        float a = alpha[h] + dt_bias[h];
        // The 20.0 cutoff is the reference's (ggml_compute_softplus_f32), not a
        // guard invented here: above it, log(1 + exp(x)) is x to f32 precision.
        float sp = (a > 20.0f) ? a : logf(1.0f + expf(a));
        g = expf(sp * ssm_a[h]);
        beta = 1.0f / (1.0f + expf(-beta_raw[h]));
    }
    __syncthreads();

    const size_t per_head = (size_t)head_k_dim * head_v_dim;
    for (int j = threadIdx.x; j < head_v_dim; j += blockDim.x) {
        float *row = state + (size_t)h * per_head + (size_t)j * head_k_dim;

        float pred = 0.0f;
        for (int i = 0; i < head_k_dim; ++i) pred += (row[i] * g) * ks[i];

        const float d = beta * (v[h * head_v_dim + j] - pred);

        float o = 0.0f;
        for (int i = 0; i < head_k_dim; ++i) {
            float s = row[i] * g + ks[i] * d;
            row[i] = s;
            o += s * (qs[i] * q_scale);
        }
        out[h * head_v_dim + j] = o;
    }
}

// The delta rule over a whole batch, one launch instead of one per token.
//
// **The sequential dependency is along tokens, and only along tokens.** Token
// `t`'s rank-1 correction is token `t+1`'s stored state, so the token loop must
// stay ordered -- but `state` is per value head, `out` is per (token, head,
// dim), and no block ever reads another block's state. So the ordering lives
// *inside* a block and the heads stay a grid dimension, exactly as before.
//
// What that removes is a launch per token per layer. On the 35B at 11,237
// tokens `--profile-kernels` counted **337,140 delta_rule launches**, 30 GDN
// layers times every token, which is ~14% of prefill in kernel time and a
// further ~8% in pure launch overhead. This makes it 30.
//
// It also stops the state round-tripping. Each launch previously re-read
// `head_k_dim * head_v_dim` floats per head from global and wrote them back;
// now the block stays resident across the batch and those rows stay hot in L1.
//
// **Bit-identical to `delta_rule` by construction.** Every token performs the
// same reads, the same products and the same accumulations in the same order,
// on the same thread. Only the loop that drives them moved from the host into
// the kernel.
//
// The trailing `__syncthreads()` is load-bearing: `qs`, `ks`, `g` and `beta` are
// reused every iteration, so a thread racing ahead to the next token would
// overwrite the staging buffer while a slower one still reads it. The launch
// boundary used to provide that barrier for free.
extern "C" __global__ void delta_rule_batch(int head_k_dim, int head_v_dim,
                                            int n_k_heads, int n_tokens,
                                            float q_scale,
                                            const float *__restrict__ q,
                                            const float *__restrict__ k,
                                            const float *__restrict__ v,
                                            const float *__restrict__ alpha,
                                            const float *__restrict__ beta_raw,
                                            const float *__restrict__ ssm_a,
                                            const float *__restrict__ dt_bias,
                                            float *__restrict__ state,
                                            float *__restrict__ out) {
    extern __shared__ float sh[];
    float *qs = sh;                 // head_k_dim
    float *ks = sh + head_k_dim;    // head_k_dim
    __shared__ float g, beta;

    const int h = blockIdx.x;
    // Modulo, not division, as in `delta_rule`: the fused reference writes
    // `iq1 = iv1 % neq1`, and blocked grouping agrees only for h = 0 and h = 1.
    const int kh = h % n_k_heads;
    const int n_v_heads = gridDim.x;

    const size_t kper = (size_t)n_k_heads * head_k_dim;
    const size_t vper = (size_t)n_v_heads * head_v_dim;
    const size_t per_head = (size_t)head_k_dim * head_v_dim;
    float *const head_state = state + (size_t)h * per_head;

    for (int t = 0; t < n_tokens; ++t) {
        const float *qt = q + (size_t)t * kper;
        const float *kt = k + (size_t)t * kper;
        const float *vt = v + (size_t)t * vper;
        float *ot = out + (size_t)t * vper;

        for (int i = threadIdx.x; i < head_k_dim; i += blockDim.x) {
            qs[i] = qt[kh * head_k_dim + i];
            ks[i] = kt[kh * head_k_dim + i];
        }
        if (threadIdx.x == 0) {
            float a = alpha[(size_t)t * n_v_heads + h] + dt_bias[h];
            // The 20.0 cutoff is the reference's (ggml_compute_softplus_f32).
            float sp = (a > 20.0f) ? a : logf(1.0f + expf(a));
            g = expf(sp * ssm_a[h]);
            beta = 1.0f / (1.0f + expf(-beta_raw[(size_t)t * n_v_heads + h]));
        }
        __syncthreads();

        for (int j = threadIdx.x; j < head_v_dim; j += blockDim.x) {
            float *row = head_state + (size_t)j * head_k_dim;

            float pred = 0.0f;
            for (int i = 0; i < head_k_dim; ++i) pred += (row[i] * g) * ks[i];

            const float d = beta * (vt[h * head_v_dim + j] - pred);

            float o = 0.0f;
            for (int i = 0; i < head_k_dim; ++i) {
                float s = row[i] * g + ks[i] * d;
                row[i] = s;
                o += s * (qs[i] * q_scale);
            }
            ot[h * head_v_dim + j] = o;
        }
        // `qs`, `ks`, `g` and `beta` are about to be rewritten for token t+1.
        __syncthreads();
    }
}

// The delta rule with the recurrent state held in shared memory.
//
// **This is the kernel's actual cost, and the 09-09 batching missed it.** The
// state is `head_k_dim * head_v_dim` floats per head — 64 KiB at the 35B's
// 128x128 — and the global-memory form reads *and writes* all of it once per
// token. At 512 tokens a call that is 2.1 GB moved in 13.1 ms: **164 GB/s, 37%
// of this card's 448.** Every other kernel here runs at 2-10% of bandwidth and
// 1-5% of compute; this one was bandwidth-bound the whole time, and moving the
// token loop into the kernel (which removed 337,140 launches) left it untouched.
//
// Staged once at entry and written back once at exit, the traffic becomes
// `2 * per_head` floats for the whole batch instead of per token — 2.1 GB
// becomes ~8 MiB.
//
// # Why this fits, and why we thought it did not
//
// 64 KiB exceeds the 48 KiB a block gets by default, and 09-09 recorded that as
// a hard limit from memory. It is not: `cudaDeviceProp` reports **99 KiB
// opt-in** out of 100 KiB per SM, and `cached_function` now asks for it.
//
// Occupancy is not the trade it appears to be. The grid is one block per value
// head — **32 blocks on 36 SMs** — so an SM already holds at most one block and
// 4 warps of its 48. Taking 65 KiB of its 100 KiB costs nothing that was being
// used.
//
// # The +1 on the row stride
//
// Thread `j` walks row `j`. At a stride of `head_k_dim` = 128 floats, every
// thread of a warp lands on the same bank — a 32-way conflict that would give
// back what the staging saves. At `head_k_dim + 1` the bank is `(j + i) % 32`,
// so a warp touches 32 distinct banks.
//
// **Bit-identical to `delta_rule_batch`.** Same arithmetic, same order, same
// thread; only where the state lives changes.
extern "C" __global__ void delta_rule_batch_shared(int head_k_dim, int head_v_dim,
                                                   int n_k_heads, int n_tokens,
                                                   float q_scale,
                                                   const float *__restrict__ q,
                                                   const float *__restrict__ k,
                                                   const float *__restrict__ v,
                                                   const float *__restrict__ alpha,
                                                   const float *__restrict__ beta_raw,
                                                   const float *__restrict__ ssm_a,
                                                   const float *__restrict__ dt_bias,
                                                   float *__restrict__ state,
                                                   float *__restrict__ out) {
    extern __shared__ float sh[];
    float *qs = sh;                        // head_k_dim
    float *ks = sh + head_k_dim;           // head_k_dim
    float *st = sh + 2 * head_k_dim;       // head_v_dim x (head_k_dim + 1)
    __shared__ float g, beta;

    const int h = blockIdx.x;
    const int kh = h % n_k_heads;
    const int n_v_heads = gridDim.x;
    const int stride = head_k_dim + 1;     // padded, see above

    const size_t kper = (size_t)n_k_heads * head_k_dim;
    const size_t vper = (size_t)n_v_heads * head_v_dim;
    const size_t per_head = (size_t)head_k_dim * head_v_dim;
    float *const head_state = state + (size_t)h * per_head;

    // Linear in the global index so the reads coalesce; the scatter lands in
    // shared, which tolerates it. Once per batch, not once per token.
    for (size_t idx = threadIdx.x; idx < per_head; idx += blockDim.x) {
        const size_t j = idx / (size_t)head_k_dim;
        const size_t i = idx - j * (size_t)head_k_dim;
        st[j * (size_t)stride + i] = head_state[idx];
    }
    __syncthreads();

    for (int t = 0; t < n_tokens; ++t) {
        const float *qt = q + (size_t)t * kper;
        const float *kt = k + (size_t)t * kper;
        const float *vt = v + (size_t)t * vper;
        float *ot = out + (size_t)t * vper;

        for (int i = threadIdx.x; i < head_k_dim; i += blockDim.x) {
            qs[i] = qt[kh * head_k_dim + i];
            ks[i] = kt[kh * head_k_dim + i];
        }
        if (threadIdx.x == 0) {
            float a = alpha[(size_t)t * n_v_heads + h] + dt_bias[h];
            // The 20.0 cutoff is the reference's (ggml_compute_softplus_f32).
            float sp = (a > 20.0f) ? a : logf(1.0f + expf(a));
            g = expf(sp * ssm_a[h]);
            beta = 1.0f / (1.0f + expf(-beta_raw[(size_t)t * n_v_heads + h]));
        }
        __syncthreads();

        for (int j = threadIdx.x; j < head_v_dim; j += blockDim.x) {
            float *row = st + (size_t)j * stride;

            float pred = 0.0f;
            for (int i = 0; i < head_k_dim; ++i) pred += (row[i] * g) * ks[i];

            const float d = beta * (vt[h * head_v_dim + j] - pred);

            float o = 0.0f;
            for (int i = 0; i < head_k_dim; ++i) {
                float s = row[i] * g + ks[i] * d;
                row[i] = s;
                o += s * (qs[i] * q_scale);
            }
            ot[h * head_v_dim + j] = o;
        }
        // `qs`, `ks`, `g` and `beta` are about to be rewritten for token t+1.
        __syncthreads();
    }

    __syncthreads();
    for (size_t idx = threadIdx.x; idx < per_head; idx += blockDim.x) {
        const size_t j = idx / (size_t)head_k_dim;
        const size_t i = idx - j * (size_t)head_k_dim;
        head_state[idx] = st[j * (size_t)stride + i];
    }
}

}  // extern "C"
