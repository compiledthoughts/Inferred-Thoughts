// f32.cuh -- F32 matmuls: the router and the paired and staged variants.
//
// Included by kernels.cu, which pulls every file into one compile unit and one
// PTX module in dependency order. Kernel names are the PTX symbols the backend
// looks up, so which file a kernel lives in changes nothing at run time.
// Kernels are in `extern "C"` blocks; templates sit outside them, since a
// template cannot have C linkage.
#pragma once

extern "C" {

// F32 matrix-vector, one thread per output row.
//
// **The MoE router, and nothing else.** `ffn_gate_inp` is F32 in the 35B --
// llama.cpp keeps the routing logits unquantized because they decide *which*
// experts run, and a quant flip there changes the answer categorically rather
// than by an ulp. `CLAUDE.md` leans on that: the router being exact is what
// lets an expert choice be compared against llama.cpp directly.
//
// So this kernel is deliberately the slow shape -- one thread walking a whole
// row in ascending order, which is `ops::naive::dot_row` statement for
// statement. It is bit-identical for the same reason the very first Q8_0
// kernel was. That costs nothing worth measuring here: the router is
// 2048 x 256, about 2 MB against the ~630 MB of experts the same token reads.
// If an F32 matmul ever lands somewhere that matters, it needs the warp
// treatment and a decision about its serial chain -- this one does not.
extern "C" __global__ void matmul_f32(int n_in, int n_out,
                                      const float *__restrict__ w,
                                      const float *__restrict__ x,
                                      float *__restrict__ out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;
    const int tok = blockIdx.y;

    const float *row = w + (size_t)j * n_in;
    const float *xt = x + (size_t)tok * n_in;

    float sum = 0.0f;
    for (int k = 0; k < n_in; ++k) sum += row[k] * xt[k];
    out[(size_t)tok * n_out + j] = sum;
}

// F32 matrix-vector against a **column-major** weight, one thread per row.
//
// Replaces `matmul_f32`, which was the largest single kernel in the 35B: 22% of
// device time for 2.3 MB of weights, at ~20 GB/s on a 448 GB/s card.
//
// **The serial chain was never the problem.** An exact f32 dot must accumulate
// in order, so parallelism is capped at `n_out` and a 2048-long chain of
// 4-cycle adds is ~3.6 us -- but the kernel measured 64-99 us, 18-28x that. The
// cost was the access pattern: thread `j` walked its own 8 KB row, so 32
// threads were 32 uncoalesced streams a kilobyte apart.
//
// Transposed, step `k` has threads 0..31 reading `wt[k*n_out + j]` -- 128
// contiguous bytes, one transaction. **The accumulation order is untouched**,
// `k` ascending exactly as `ops::naive::dot_row` walks it, so this is
// bit-identical and the exactness that made the router worth keeping costs
// nothing.
//
// `x[k]` is the same address for every thread, which is a broadcast rather than
// 32 loads.
extern "C" __global__ void matmul_f32_t(int n_in, int n_out,
                                        const float *__restrict__ wt,
                                        const float *__restrict__ x,
                                        float *__restrict__ out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;
    const int tok = blockIdx.y;
    const float *xt = x + (size_t)tok * n_in;

    // The remaining floor is memory latency on a single thread, and it is
    // where exactness stops paying. Every shape measures ~40 us whatever
    // `n_out` is -- 1, 32 or 256 -- because 2048 loads with ~8 in flight at a
    // few hundred cycles each is ~44 us however few threads there are.
    // **Hoisting the loads by hand into an unrolled batch of eight changed
    // nothing** (43.4 vs 41.6 us), so nvcc already does it. Going below this
    // needs more loads in flight per output, which means splitting the
    // reduction, which is the exactness the router is kept for.
    float sum = 0.0f;
    for (int k = 0; k < n_in; ++k) sum += wt[(size_t)k * n_out + j] * xt[k];
    out[(size_t)tok * n_out + j] = sum;
}

// Two F32 matmuls over the **same** activation, in one launch.
//
// **The cost of this kernel is reading `x`, not producing outputs.** Its own
// decomposition says loads are 81% of the call, and the measured times say the
// rest: 39.3 us at `n_out` 1, 41.8 at 32, 43.3 at 256. Twelve times the output
// for 10% more time, because every thread walks the same 2048-element
// activation and that walk is the kernel.
//
// So two matmuls that read one activation pay for it twice. On this model there
// are two such pairs per layer and they are 5.51 ms of a 34 ms token:
//
//   ssm_alpha + ssm_beta        {2048,32} each, both read `normed`
//   ffn_gate_inp + _shexp       {2048,256} + {2048,1}, both read the MoE input
//
// **Two destinations, not one concatenated buffer**, which is what keeps this
// change local. `out_a` and `out_b` stay the exact buffers the model already
// owns, so nothing downstream sees a stride: `Delta` keeps separate `alpha` and
// `beta` slices, and `softmax` keeps a 256-wide row rather than needing to skip
// a 257th element. The alternative -- one buffer read at two offsets -- would
// have reached `Delta`, `softmax` and every backend's `delta_rule`.
//
// `wt` is the two weights interleaved into one column-major stack of `n_a +
// n_b` rows, built once at upload. Interleaved rather than appended because
// column-major strides by the output width: row `j` of super-block `k` lives at
// `k * (n_a + n_b) + j`.
//
// Bit-identical by construction, and for the same reason batching and the warp
// matmul were: each output is still one thread walking `k` ascending over the
// same values in the same order. Only which outputs share a launch changes.
extern "C" __global__ void matmul_f32_t_pair(int n_in, int n_a, int n_b,
                                             const float *__restrict__ wt,
                                             const float *__restrict__ x,
                                             float *__restrict__ out_a,
                                             float *__restrict__ out_b) {
    const int n_out = n_a + n_b;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;
    const int tok = blockIdx.y;
    const float *xt = x + (size_t)tok * n_in;

    float sum = 0.0f;
    for (int k = 0; k < n_in; ++k) sum += wt[(size_t)k * n_out + j] * xt[k];

    // The split. Thread `j` belongs to whichever weight its row came from, and
    // writes into that weight's own output at that output's own stride.
    if (j < n_a) {
        out_a[(size_t)tok * n_a + j] = sum;
    } else {
        out_b[(size_t)tok * n_b + (j - n_a)] = sum;
    }
}

// The candidate: cooperative staging, serial accumulation.
//
// **Loads in flight and accumulation order are independent quantities**, and
// `matmul_f32_t`'s comment conflates them. The floor there is one thread per
// output row issuing 2048 dependent-ish loads with ~8 outstanding, on a block
// of `n_out` threads -- 32 of them for `ssm_alpha`, one warp on one SM of
// thirty-six. Nothing about that is arithmetic.
//
// So: every thread in the block helps stage a tile of `wt` and `x` into shared
// memory, which is a fully parallel, perfectly coalesced read (`wt` is
// column-major, so a k-range is contiguous). Then thread `j` walks *its own*
// row of the tile serially in ascending `k`, exactly as `ops::naive::dot_row`
// does. **The accumulation order is untouched, so this is bit-identical**; only
// who fetched the bytes changed.
//
// `kt` is the tile height, chosen on the host so `kt * (n_out + 1)` floats fit
// the dynamic shared allocation.
extern "C" __global__ void matmul_f32_t_staged(int n_in, int n_out, int kt,
                                               const float *__restrict__ wt,
                                               const float *__restrict__ x,
                                               float *__restrict__ out) {
    extern __shared__ float f32_stage[];
    float *sw = f32_stage;                     // kt * n_out
    float *sx = f32_stage + (size_t)kt * n_out;  // kt

    const int t = threadIdx.x;
    const int nthreads = blockDim.x;
    const int tok = blockIdx.y;
    const float *xt = x + (size_t)tok * n_in;

    float sum = 0.0f;
    for (int k0 = 0; k0 < n_in; k0 += kt) {
        const int kn = min(kt, n_in - k0);
        // Column-major, so `wt[k0 * n_out .. (k0 + kn) * n_out]` is one
        // contiguous run and the whole block streams it.
        for (int i = t; i < kn * n_out; i += nthreads)
            sw[i] = wt[(size_t)k0 * n_out + i];
        for (int i = t; i < kn; i += nthreads) sx[i] = xt[k0 + i];
        __syncthreads();

        if (t < n_out) {
            // Serial, ascending, one row -- the oracle's order exactly.
            for (int k = 0; k < kn; ++k) sum += sw[(size_t)k * n_out + t] * sx[k];
        }
        __syncthreads();
    }
    if (t < n_out) out[(size_t)tok * n_out + t] = sum;
}

}  // extern "C"
