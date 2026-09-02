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
__global__ void rope_neox(int head_dim, int n_rot, int n_heads,
                          const float *__restrict__ cosv,
                          const float *__restrict__ sinv, float *__restrict__ x) {
    // Partial RoPE: only the first `n_rot` of each head rotate and the rest
    // pass through, so the pair stride is n_rot/2 and the head stride stays
    // head_dim. qwen35 rotates 64 of 256; qwen3 passes n_rot == head_dim.
    const int half = n_rot / 2;
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

// Flash-decoding attention: split over the KV sequence, online softmax.
//
// # Why the previous shape had to go
//
// Attention here ran as three kernels — scores, softmax, weighted sum — each
// parallel over something the oracle treats as independent, which kept the
// accumulation order and therefore the bit-exactness. The trouble is what it
// was parallel *over*: `n_head`, which is 16. `softmax_rows` ran sixteen
// threads on a thirty-six SM card, each making three serial passes over every
// position, so its cost grew with context while its parallelism did not.
// Measured at ~29 us a call near depth 500 and ~120 us near depth 1000: a
// cliff, not a slope.
//
// The published answer is to split along the **KV sequence** instead
// (Flash-Decoding). Each block takes the same query and a different chunk of
// positions, so parallelism scales with context rather than being capped by
// head count — which is precisely the regime batch-1 decode lives in.
//
// # Online softmax
//
// Milakov & Gimelshein's trick makes one pass do what three did. A chunk
// computes its own max `m` and its own sum of exponentials `l` relative to that
// max, plus a partial output. Combining two chunks is then exact algebra:
// rescale each by `exp(m_chunk - m_global)` and add. So the scores array never
// has to exist — memory goes from O(n_pos) to O(head_dim) per chunk — and the
// three launches per layer become two.
//
// # What this costs in numerics, stated plainly
//
// This is the first kernel here that does **not** reproduce the oracle's
// summation order, and it cannot: chunked accumulation is the whole point.
// `attend` was already outside the bit-exact set because it calls `expf`, so
// the change is inexact -> differently-inexact rather than exact -> inexact.
//
// The error is bounded and derived, not measured-then-blessed. Within a chunk
// the max is a tree, which is exact because max never rounds; the sum is a tree
// over 128, so ~7 eps; the combine is serial over `n_split`, so ~n_split eps.
// Total relative error is about `(7 + n_split + a few) * eps_f32`. The test
// derives its tolerance from that expression rather than from what passes.
//
// Worth noting the tree sums are *more* accurate than the oracle's serial one —
// error grows as O(log n) rather than O(n). They are simply different.

#define FD_CHUNK 128

// One block per (query head, chunk of positions).
//
// Shared memory: head_dim floats for the query, FD_CHUNK for the exponentials,
// FD_CHUNK for the reduction scratch.
__global__ void attn_flash(int n_pos, int kv_dim, int head_dim, int n_head,
                           int n_head_kv, float scale,
                           const float *__restrict__ q,
                           const unsigned short *__restrict__ k,
                           const unsigned short *__restrict__ v,
                           float *__restrict__ part_acc,
                           float *__restrict__ part_m,
                           float *__restrict__ part_l) {
    extern __shared__ float smem[];
    float *sq = smem;                 // [head_dim]  the query, read n_pos times
    float *se = sq + head_dim;        // [FD_CHUNK]  exp(score - m) per position
    float *red = se + FD_CHUNK;       // [FD_CHUNK]  reduction scratch

    const int hq = blockIdx.x;
    const int split = blockIdx.y;
    const int lo = split * FD_CHUNK;
    const int len = min(FD_CHUNK, n_pos - lo);
    const int off = (hq / (n_head / n_head_kv)) * head_dim;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        sq[i] = q[(size_t)hq * head_dim + i];
    }
    __syncthreads();

    // One position per thread; the dot over head_dim stays serial in-thread.
    float score = -INFINITY;
    if (threadIdx.x < len) {
        const unsigned short *key =
            k + (size_t)(lo + threadIdx.x) * kv_dim + off;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i) dot += sq[i] * h2f(key[i]);
        score = dot * scale;
    }

    // Chunk max. A tree here is *exact* — max never rounds — so this costs
    // nothing in accuracy.
    red[threadIdx.x] = score;
    __syncthreads();
    for (int s = FD_CHUNK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        }
        __syncthreads();
    }
    const float m = red[0];
    __syncthreads();

    const float e = (threadIdx.x < len) ? expf(score - m) : 0.0f;
    se[threadIdx.x] = e;
    red[threadIdx.x] = e;
    __syncthreads();
    for (int s = FD_CHUNK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float l = red[0];

    // Weighted sum of this chunk's values. Adjacent threads hold adjacent
    // elements and read adjacent halves of V, so the reads coalesce.
    const size_t base = (size_t)hq * gridDim.y + split;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < len; ++t) {
            acc += se[t] * h2f(v[(size_t)(lo + t) * kv_dim + off + i]);
        }
        part_acc[base * head_dim + i] = acc;
    }
    if (threadIdx.x == 0) {
        part_m[base] = m;
        part_l[base] = l;
    }
}

// Combine the per-chunk partials. One block per query head.
//
// The algebra is exact: a chunk's numbers are relative to its own max, so
// rescaling by exp(m_chunk - m_global) puts them all on one reference before
// they are added.
__global__ void attn_flash_combine(int n_split, int head_dim,
                                   const float *__restrict__ part_acc,
                                   const float *__restrict__ part_m,
                                   const float *__restrict__ part_l,
                                   float *__restrict__ out) {
    extern __shared__ float w[];   // [n_split] rescaling weights
    __shared__ float total;

    const int hq = blockIdx.x;
    const float *pm = part_m + (size_t)hq * n_split;
    const float *pl = part_l + (size_t)hq * n_split;

    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int s = 0; s < n_split; ++s) m = fmaxf(m, pm[s]);
        float l = 0.0f;
        for (int s = 0; s < n_split; ++s) {
            w[s] = expf(pm[s] - m);
            l += pl[s] * w[s];
        }
        total = l;
    }
    __syncthreads();

    const float *pa = part_acc + (size_t)hq * n_split * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_split; ++s) acc += pa[(size_t)s * head_dim + i] * w[s];
        out[(size_t)hq * head_dim + i] = acc / total;
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
                                 const unsigned short *__restrict__ w_scales,
                                 const signed char *__restrict__ w_quants,
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

    for (int b = lane; b < n_blocks; b += 32) {
        float dw = __half2float(__ushort_as_half(w_scales[(size_t)j * n_blocks + b]));

        // Both operands are 16-byte aligned in the repacked layout, so a lane
        // reads its whole 32-element block as two `int4` and reduces it with
        // eight `__dp4a`. On disk the block is {f16 scale; int8 q[32]} = 34
        // bytes, which puts the quants at 34b+2 -- even, but never a multiple
        // of four, so neither wide loads nor __dp4a were reachable.
        const int4 *wq = (const int4 *)(w_quants + (size_t)j * n_in + (size_t)b * 32);
        const int4 *xq = (const int4 *)(x_quants + (size_t)b * 32);
        int4 w0 = wq[0], w1 = wq[1];
        int4 a0 = xq[0], a1 = xq[1];

        // Integer, so exact and order-free -- __dp4a accumulates four int8
        // products into an int and cannot round.
        int sumi = 0;
        sumi = __dp4a(w0.x, a0.x, sumi);
        sumi = __dp4a(w0.y, a0.y, sumi);
        sumi = __dp4a(w0.z, a0.z, sumi);
        sumi = __dp4a(w0.w, a0.w, sumi);
        sumi = __dp4a(w1.x, a1.x, sumi);
        sumi = __dp4a(w1.y, a1.y, sumi);
        sumi = __dp4a(w1.z, a1.z, sumi);
        sumi = __dp4a(w1.w, a1.w, sumi);

        mine[b] = (float)sumi * (dw * x_scales[b]);
    }
    __syncwarp();

    // The one ordered accumulation, kept serial and ascending.
    //
    // Measured cost of keeping it: with the old byte loads a warp tree was
    // worth 11-21%, which was tempting. With aligned loads it is worth 2-7%,
    // because the tail was only ever visible behind slow loads. Bit-exactness
    // here is close to free, so it is kept.
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

// Round f32 to f16 and store, which is how K and V enter the cache.
//
// This exists so a GPU layer's keys and values never leave the card. They used
// to: the model computed them on the device, shipped them home so host code
// could convert and write the cache, and then shipped them straight back up for
// attention — 56 needless bus crossings a token, and a host barrier in the
// middle of every layer that no CUDA graph could span.
//
// `__float2half` rounds to nearest even, which is what `quant::half::f32_to_f16`
// does. That equivalence is not assumed: the same pairing is already relied on
// for the Q8_0 scale in `quantize_q8_0` and checked against the oracle there.
__global__ void kv_write_f16(int n, const float *__restrict__ src,
                             unsigned short *__restrict__ dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half_as_ushort(__float2half(src[i]));
}

// ------------------------------------------------------------- diagnostics
//
// Not used by the forward pass. These exist to answer one question: why does a
// serial f64 sum of 1024 values cost ~59 us, when that is ~164 cycles per add
// at 2.84 GHz and an FP64 add should be nowhere near that?
//
// FP64 *throughput* on a consumer part is 1/64 of FP32, but a dependent chain
// is a latency problem, not a throughput one. If the answer is memory or
// occupancy rather than FP64 latency, the cost is recoverable without touching
// the arithmetic — which is the outcome worth checking for before trading any
// exactness away.
//
// The five below vary one thing at a time:
//   serial_f64_global  what the real kernel does
//   serial_f32_global  the same with an f32 accumulator, isolating FP64
//   shared_f64         the same values, staged through shared memory first
//   shared_f32         both changes, for the corner of the square
//   tree_f64           the parallel alternative, as a floor

__global__ void bench_serial_f64_global(int n, const float *__restrict__ x,
                                        float *__restrict__ out) {
    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            float v = x[i];
            sum += (double)(v * v);
        }
        ((double *)out)[0] = (double)sum;
    }
}

__global__ void bench_serial_f32_global(int n, const float *__restrict__ x,
                                        float *__restrict__ out) {
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            float v = x[i];
            sum += v * v;
        }
        ((double *)out)[0] = (double)sum;
    }
}

__global__ void bench_shared_f64(int n, const float *__restrict__ x,
                                 float *__restrict__ out) {
    extern __shared__ float sq[];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        sq[i] = v * v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) sum += (double)sq[i];
        ((double *)out)[0] = (double)sum;
    }
}

__global__ void bench_shared_f32(int n, const float *__restrict__ x,
                                 float *__restrict__ out) {
    extern __shared__ float sq[];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        sq[i] = v * v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) sum += sq[i];
        ((double *)out)[0] = (double)sum;
    }
}

__global__ void bench_tree_f64(int n, const float *__restrict__ x,
                               float *__restrict__ out) {
    __shared__ double p[256];
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
    if (threadIdx.x == 0) ((double *)out)[0] = p[0];
}

// The floor. Every variant above is timed as a batch of launches against one
// synchronize, so nothing can measure faster than the driver can issue work.
// A variant reporting near this number is launch-bound and its true cost is
// unresolved -- which is why the table also runs at n=8192.
__global__ void bench_empty(int n, const float *__restrict__ x,
                            float *__restrict__ out) {
    if (threadIdx.x == 0 && n < 0) ((double *)out)[0] = (double)x[0];
}

} // extern "C"

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

// x *= sigmoid(g), elementwise. The sibling of silu_mul, and inexact for the
// same reason: expf.
extern "C" __global__ void sigmoid_mul(int n, float *__restrict__ x,
                                       const float *__restrict__ g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= 1.0f / (1.0f + expf(-g[i]));
}

// ------------------------------------------------------- matmul diagnostics
//
// `matmul_q8_0_warp` is 66.5% of device time on the 9B and moves ~235 GB/s
// against the card's 448. Two things in it could be responsible and they call
// for opposite fixes, so they are separated here rather than guessed at.
//
//   base   what the real kernel does
//   tree   same loads, but the cross-block sum is a warp tree instead of one
//          lane walking it serially -- inexact, and a diagnostic only
//   u16    same serial tail, but the quants are read two bytes at a time
//
// If `tree` is much faster, the cost is the dependent f32 chain at the end,
// which is `rms_norm`'s problem again. If `u16` is, the cost is load
// instruction count. If neither moves, it is neither.

extern "C" __global__ void bench_mm_base(int n_in, int n_out,
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
        unsigned short dbits = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        float dw = __half2float(__ushort_as_half(dbits));
        int sumi = 0;
        const signed char *xq = x_quants + b * 32;
        for (int k = 0; k < 32; ++k) sumi += (int)((signed char)blk[2 + k]) * (int)xq[k];
        mine[b] = (float)sumi * (dw * x_scales[b]);
    }
    __syncwarp();

    if (lane == 0) {
        float sumf = 0.0f;
        for (int b = 0; b < n_blocks; ++b) sumf += mine[b];
        out[j] = sumf;
    }
}

extern "C" __global__ void bench_mm_tree(int n_in, int n_out,
                                         const unsigned char *__restrict__ w,
                                         const float *__restrict__ x_scales,
                                         const signed char *__restrict__ x_quants,
                                         float *__restrict__ out) {
    const int n_blocks = n_in / 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;

    const unsigned char *row = w + (size_t)j * (size_t)n_blocks * 34;

    // No shared staging: each lane keeps a running partial and the warp folds
    // them at the end. Different summation order, hence diagnostic only.
    float acc = 0.0f;
    for (int b = lane; b < n_blocks; b += 32) {
        const unsigned char *blk = row + (size_t)b * 34;
        unsigned short dbits = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        float dw = __half2float(__ushort_as_half(dbits));
        int sumi = 0;
        const signed char *xq = x_quants + b * 32;
        for (int k = 0; k < 32; ++k) sumi += (int)((signed char)blk[2 + k]) * (int)xq[k];
        acc += (float)sumi * (dw * x_scales[b]);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) out[j] = acc;
}

extern "C" __global__ void bench_mm_u16(int n_in, int n_out,
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
        unsigned short dbits = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        float dw = __half2float(__ushort_as_half(dbits));

        // A block starts at 34*b from a 256-byte aligned base, and 34 is even,
        // so the quants at +2 are 2-byte aligned but never 4-byte aligned.
        // ushort is the widest load the layout allows without repacking.
        const unsigned short *q16 = (const unsigned short *)(blk + 2);
        const signed char *xq = x_quants + b * 32;
        int sumi = 0;
        for (int k = 0; k < 16; ++k) {
            unsigned short pair = q16[k];
            sumi += (int)(signed char)(pair & 0xff) * (int)xq[2 * k];
            sumi += (int)(signed char)(pair >> 8) * (int)xq[2 * k + 1];
        }
        mine[b] = (float)sumi * (dw * x_scales[b]);
    }
    __syncwarp();

    if (lane == 0) {
        float sumf = 0.0f;
        for (int b = 0; b < n_blocks; ++b) sumf += mine[b];
        out[j] = sumf;
    }
}

// The fourth variant, and the only one that changes the weight *layout*.
//
// A Q8_0 block on disk is {f16 scale; int8 q[32]} = 34 bytes, so block b sits
// at 34b and its quants at 34b+2. 34 is even but never a multiple of four, so
// the widest legal load is two bytes and `__dp4a` -- which wants 4-byte
// operands -- is unreachable.
//
// Splitting the tensor at upload into an aligned scale array and an aligned
// quant array fixes both: quants for block b start at 32b, so a lane reads its
// whole block as two `int4` and does eight `__dp4a` instead of 32 byte loads
// and 32 multiplies.
//
// Still bit-exact. `__dp4a` accumulates four int8 products into an int, which
// is integer arithmetic and therefore order-free, and the cross-block sum is
// left serial and ascending exactly as before.
//
// Note this is *not* the repack that failed earlier. That one went element-major
// to chase warp coalescing and gave up per-thread locality. This keeps the
// block-major order and only separates two fields so they can be addressed.
extern "C" __global__ void bench_mm_packed(int n_in, int n_out,
                                           const unsigned short *__restrict__ w_scales,
                                           const signed char *__restrict__ w_quants,
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

    for (int b = lane; b < n_blocks; b += 32) {
        float dw = __half2float(__ushort_as_half(w_scales[(size_t)j * n_blocks + b]));

        const int4 *wq = (const int4 *)(w_quants + (size_t)j * n_in + (size_t)b * 32);
        const int4 *xq = (const int4 *)(x_quants + (size_t)b * 32);
        int4 w0 = wq[0], w1 = wq[1];
        int4 a0 = xq[0], a1 = xq[1];

        int sumi = 0;
        sumi = __dp4a(w0.x, a0.x, sumi);
        sumi = __dp4a(w0.y, a0.y, sumi);
        sumi = __dp4a(w0.z, a0.z, sumi);
        sumi = __dp4a(w0.w, a0.w, sumi);
        sumi = __dp4a(w1.x, a1.x, sumi);
        sumi = __dp4a(w1.y, a1.y, sumi);
        sumi = __dp4a(w1.z, a1.z, sumi);
        sumi = __dp4a(w1.w, a1.w, sumi);

        mine[b] = (float)sumi * (dw * x_scales[b]);
    }
    __syncwarp();

    if (lane == 0) {
        float sumf = 0.0f;
        for (int b = 0; b < n_blocks; ++b) sumf += mine[b];
        out[j] = sumf;
    }
}

// Both fixes at once: aligned wide loads *and* a warp-tree tail. Inexact,
// so this is the ceiling rather than a candidate.
extern "C" __global__ void bench_mm_packed_tree(int n_in, int n_out,
                                                const unsigned short *__restrict__ w_scales,
                                                const signed char *__restrict__ w_quants,
                                                const float *__restrict__ x_scales,
                                                const signed char *__restrict__ x_quants,
                                                float *__restrict__ out) {
    const int n_blocks = n_in / 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;

    float acc = 0.0f;
    for (int b = lane; b < n_blocks; b += 32) {
        float dw = __half2float(__ushort_as_half(w_scales[(size_t)j * n_blocks + b]));
        const int4 *wq = (const int4 *)(w_quants + (size_t)j * n_in + (size_t)b * 32);
        const int4 *xq = (const int4 *)(x_quants + (size_t)b * 32);
        int4 w0 = wq[0], w1 = wq[1];
        int4 a0 = xq[0], a1 = xq[1];
        int sumi = 0;
        sumi = __dp4a(w0.x, a0.x, sumi);
        sumi = __dp4a(w0.y, a0.y, sumi);
        sumi = __dp4a(w0.z, a0.z, sumi);
        sumi = __dp4a(w0.w, a0.w, sumi);
        sumi = __dp4a(w1.x, a1.x, sumi);
        sumi = __dp4a(w1.y, a1.y, sumi);
        sumi = __dp4a(w1.z, a1.z, sumi);
        sumi = __dp4a(w1.w, a1.w, sumi);
        acc += (float)sumi * (dw * x_scales[b]);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) out[j] = acc;
}
