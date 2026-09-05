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

// The same dot product as `matmul_q8_0_warp`, with the batch in registers.
//
// **This is the kernel batched prefill exists for.** The single-token version
// reads the whole weight from VRAM for every token, which is why prefill cost
// what generating the prompt would: at 20k prompt tokens the 9B's weights would
// cross the bus 20,000 times. Here a warp loads its slice of a weight row once
// and dots it against `MM_TOK` tokens held in registers, so weight traffic --
// the dominant term, and the one the card is actually limited by after the
// repack took it to 85-89% of peak -- falls by that factor.
//
// Bit-exactness is untouched, and for the reason recorded above: the block sum
// is integer and order-free, and each token's cross-block sum is still walked
// serially in ascending `b`. Adding a token axis changes which outputs share a
// weight load, never how one output accumulates. So the batched and unbatched
// kernels agree exactly, which `tests/cuda_ops.rs` asserts rather than assumes.
//
// Shared memory is `warps_per_block * MM_TOK * n_blocks` floats. That is what
// caps MM_TOK at 4: the 9B's `ffn_down` has n_in 12288, so n_blocks is 384 and
// four warps need 24.5 KB, already half the 48 KB a block may ask for.
// Tokens one warp holds while it loads a weight row once, and how many blocks
// of that row are in flight at a time.
//
// Shared memory is `warps * MM_TOK * MM_SEG` floats and **does not depend on
// n_in**, which is the point of segmenting: an earlier version kept all
// `n_blocks` partials live, so the 9B's `ffn_down` (n_in 12288, 384 blocks)
// forced the block down to one or two warps at MM_TOK 8 or 16 and lost more to
// occupancy than reuse gained.
#define MM_TOK 8
#define MM_SEG 64

// The same dot product as `matmul_q8_0_warp`, with the batch in registers.
//
// **This is the kernel batched prefill exists for.** The single-token version
// reads the whole weight from VRAM for every token, which is why prefill cost
// what generating the prompt would. Here a warp loads its slice of a weight row
// once and dots it against `MM_TOK` tokens, so weight traffic -- the dominant
// term, and the one the card is actually limited by -- falls by that factor.
//
// It matters because the matmul is *bandwidth-saturated*: at MM_TOK 4 the 9B
// moves 2.11 GB of weights per token in 4.53 ms, which is ~465 GB/s against a
// 448 GB/s card (the excess is L2 catching reuse across the token tiles). There
// is no arithmetic left to win. The only lever is moving fewer bytes, which is
// exactly what raising MM_TOK does.
//
// **The reduction is segmented so that lever is not capped by shared memory.**
// The cross-block sum is f32 and must stay serial and ascending to reproduce
// the oracle. It does not have to be *deferred*: lane 0 keeps a running total
// and folds each segment into it in order, so only MM_SEG partials per token
// are ever live. Same additions, same order, same bits -- and shared memory
// becomes a constant instead of growing with the tensor.
//
// Within a block the sum of products is integer (`__dp4a` accumulates four int8
// products into an int and cannot round), so it is order-free and the lanes may
// split it however they like.
__global__ void matmul_q8_0_batch(int n_in, int n_out, int n_tok,
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
    const int t0 = blockIdx.y * MM_TOK;
    if (j >= n_out) return;

    // The tail tile of the batch carries fewer than MM_TOK tokens.
    int nt = n_tok - t0;
    if (nt > MM_TOK) nt = MM_TOK;

    float *mine = partial + (size_t)warp * MM_TOK * MM_SEG;

    // Running totals, one per token, ascending in `b`. Unrolled over the
    // constant MM_TOK so they stay in registers rather than spilling to local.
    float acc[MM_TOK];
#pragma unroll
    for (int u = 0; u < MM_TOK; ++u) acc[u] = 0.0f;

    for (int seg = 0; seg < n_blocks; seg += MM_SEG) {
        const int len = min(MM_SEG, n_blocks - seg);

        for (int b = lane; b < len; b += 32) {
            const int bb = seg + b;
            // Loaded once, used `nt` times. This is the whole point.
            const float dw =
                __half2float(__ushort_as_half(w_scales[(size_t)j * n_blocks + bb]));
            const int4 *wq = (const int4 *)(w_quants + (size_t)j * n_in + (size_t)bb * 32);
            const int4 w0 = wq[0], w1 = wq[1];

#pragma unroll
            for (int u = 0; u < MM_TOK; ++u) {
                if (u >= nt) break;
                const int t = t0 + u;
                const int4 *xq =
                    (const int4 *)(x_quants + (size_t)t * n_in + (size_t)bb * 32);
                const int4 a0 = xq[0], a1 = xq[1];

                int sumi = 0;
                sumi = __dp4a(w0.x, a0.x, sumi);
                sumi = __dp4a(w0.y, a0.y, sumi);
                sumi = __dp4a(w0.z, a0.z, sumi);
                sumi = __dp4a(w0.w, a0.w, sumi);
                sumi = __dp4a(w1.x, a1.x, sumi);
                sumi = __dp4a(w1.y, a1.y, sumi);
                sumi = __dp4a(w1.z, a1.z, sumi);
                sumi = __dp4a(w1.w, a1.w, sumi);

                mine[u * MM_SEG + b] =
                    (float)sumi * (dw * x_scales[(size_t)t * n_blocks + bb]);
            }
        }
        __syncwarp();

        // Fold this segment into the running totals, in ascending `b`. Across
        // segments this visits every block exactly once, in order, so the sum
        // is the oracle's.
        if (lane == 0) {
#pragma unroll
            for (int u = 0; u < MM_TOK; ++u) {
                if (u >= nt) break;
                const float *p = mine + u * MM_SEG;
                float sumf = acc[u];
                for (int b = 0; b < len; ++b) sumf += p[b];
                acc[u] = sumf;
            }
        }
        __syncwarp();
    }

    if (lane == 0) {
#pragma unroll
        for (int u = 0; u < MM_TOK; ++u) {
            if (u >= nt) break;
            out[(size_t)(t0 + u) * n_out + j] = acc[u];
        }
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


// ================================================================ the k-quants
//
// The 35B needs three matmuls the earlier two models did not: IQ4_XS on every
// routed expert, `token_embd` and the GDN projections; Q5_K on `attn_output`;
// Q6_K on `attn_q` and the LM head. ggml pairs all three with a **Q8_K**
// activation rather than Q8_0, which is why `quantize_q8_k` lives beside
// `quantize_q8_0` instead of replacing it -- one layer of the 35B feeds the
// same normed activation to a Q6_K matmul (`attn_q`) and two Q8_0 ones
// (`attn_k`, `attn_v`), so both quantizations are live at once.
//
// **All three are bit-identical to `ops::naive`.** The handoff for this work
// expected a derived tolerance; it is not needed, and the escape is the one
// `matmul_q8_0_warp` found, which generalizes further than it looked:
//
//     every k-quant dot is an INTEGER inner sum inside an f32 outer chain.
//
// The integer part cannot round, so its order is free and it can have the whole
// warp. The f32 chain is walked serially and ascending, exactly as the oracle
// walks it. Only the *shape* of that split differs per format:
//
//   Q6_K     8 f32 lanes x nb super-blocks, then an 8-way fold
//   Q5_K     the same, plus a separate `dmin` chain joined before the lanes
//   IQ4_XS   one chain of nb*8 terms -- the same length as Q8_0's at n_in 2048
//
// The integer sums provably cannot overflow i32, which is what makes them
// order-free rather than merely usually-order-free:
//
//   Q6_K    |scale| <= 127, |aux16| <= 32767, 32 terms  ->  1.3e8
//   Q5_K     scale <=  63, |aux16| <= 32767, 32 terms  ->  6.6e7
//   IQ4_XS  |q8| <= 127, |kvalue| <= 127, 32 terms     ->  5.2e5
//
// Two things had to be reproduced deliberately rather than falling out:
//
//  - **FMA is chosen per format, not per file.** `--fmad=false` turns nvcc's
//    contraction off globally, so fusion is opted back in by hand with
//    `__fmaf_rn` in exactly the two places the reference build fused: Q5_K's
//    `sums[l]` and `sumf` updates. Q6_K's `sums[l] +=` and IQ4_XS's `sumf +=`
//    look identical on the page and must NOT be fused. `src/quant/kquant.rs`
//    records the same asymmetry on the CPU side; it is a property of the
//    reference's compiler, not of the format.
//  - **The Q8_K scale's argmax breaks ties toward the lower index**, because
//    the reference scans linearly with a strict `>` and keeps the first winner.
//    A plain tree max is free to return either, and would move a whole block's
//    quants when it disagreed.

#define QK_K 256
// block_q6_K:   { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; f16 d; }
#define Q6K_BYTES 210
// block_q5_K:   { f16 d; f16 dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128] }
#define Q5K_BYTES 176
// block_iq4_xs: { f16 d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128] }
#define IQ4XS_BYTES 136

// `kvalues_iq4nl` from ggml/src/ggml-common.h. A non-uniform grid -- the "IQ"
// in IQ4_XS -- so a linear dequantization would be a different format.
__constant__ signed char kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

// Round to nearest, ties to even -- `nearest_int` in ggml/src/ggml-quants.c.
//
// Adding 1.5 * 2^23 forces the fractional bits out of an f32 mantissa and IEEE
// addition resolves the tie to even while doing it. **This add must not be
// contracted with the multiply that feeds it**; `--fmad=false` guarantees that
// globally, and the caller uses `__fmul_rn` so the guarantee is local too.
__device__ __forceinline__ int nearest_int_dev(float fval) {
    float val = fval + 12582912.0f;
    return (__float_as_int(val) & 0x007fffff) - 0x00400000;
}

// Quantize an activation to Q8_K: one 256-thread block per super-block.
//
// Mirrors `quant::kquant::Q8KRow::from_f32`, which mirrors
// `quantize_row_q8_K_ref`. Four details are silent if wrong, and the module
// docs in kquant.rs carry them; the two that constrain *this* kernel are the
// argmax tie-break above and `bsums`, which only Q5_K reads -- so omitting them
// leaves two formats right and one subtly wrong.
__global__ void quantize_q8_k(int n_super, const float *__restrict__ x,
                              float *__restrict__ scales,
                              signed char *__restrict__ quants,
                              short *__restrict__ bsums) {
    __shared__ float s_av[QK_K];
    __shared__ int   s_ai[QK_K];
    __shared__ int   s_q[QK_K];

    const int sb = blockIdx.x;
    if (sb >= n_super) return;
    const int t = threadIdx.x;

    const float v  = x[(size_t)sb * QK_K + t];
    s_av[t] = fabsf(v);
    s_ai[t] = t;
    __syncthreads();

    // Tree argmax. Taking the right half only on a *strict* win keeps the
    // leftmost index at every level, so the winner is the lowest index holding
    // the maximum -- which is what the reference's linear `if (ax > amax)`
    // scan returns.
    for (int stride = QK_K / 2; stride > 0; stride >>= 1) {
        if (t < stride && s_av[t + stride] > s_av[t]) {
            s_av[t] = s_av[t + stride];
            s_ai[t] = s_ai[t + stride];
        }
        __syncthreads();
    }

    const float amax = s_av[0];
    if (amax == 0.0f) {
        quants[(size_t)sb * QK_K + t] = 0;
        if (t < QK_K / 16) bsums[(size_t)sb * (QK_K / 16) + t] = 0;
        if (t == 0) scales[sb] = 0.0f;
        return;
    }

    // The **signed** value at the largest magnitude, not the magnitude: the
    // reference divides -127 by it, so `iscale` is negative for a positive
    // extreme and the extreme element quantizes to exactly -127.
    const float mx = x[(size_t)sb * QK_K + s_ai[0]];
    const float iscale = -127.0f / mx;

    const int q = min(127, nearest_int_dev(__fmul_rn(iscale, v)));
    quants[(size_t)sb * QK_K + t] = (signed char)q;
    s_q[t] = q;
    __syncthreads();

    // Sums of 16 consecutive quants, accumulated in int and narrowed after --
    // the reference's `sum` is an int and only the store is int16.
    if (t < QK_K / 16) {
        int sum = 0;
        for (int k = 0; k < 16; ++k) sum += s_q[t * 16 + k];
        bsums[(size_t)sb * (QK_K / 16) + t] = (short)sum;
    }
    if (t == 0) scales[sb] = 1.0f / iscale;
}

// Q6_K x Q8_K, one warp per output row.
//
// Six bits per weight, split across a low nibble in `ql` and a high pair in
// `qh`, biased by -32. The 16 sub-block scales are plain int8.
//
// Lane layout is `lane = g*8 + l`. `l` is the oracle's f32 lane -- the fixed
// position within a 16-element sub-block that owns one of the eight `sums[]`
// accumulators -- and `g` splits the 16 sub-blocks four ways. The g-split is
// integer-only and therefore free; the l-split is the oracle's own.
//
// `blockIdx.y` is the batch token. Decode is `gridDim.y == 1` of the same
// kernel rather than a separate one, so prefill and decode cannot disagree.
__global__ void matmul_q6_k_q8_k(int n_in, int n_out,
                                 const unsigned char *__restrict__ w,
                                 const float *__restrict__ x_scales,
                                 const signed char *__restrict__ x_quants,
                                 float *__restrict__ out) {
    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    // Warp-uniform: every lane of a warp shares `j`, so the shuffles below
    // keep a full mask.
    if (j >= n_out) return;
    const int tok = blockIdx.y;

    const int l = lane & 7;
    const int g = lane >> 3;

    const float *xs = x_scales + (size_t)tok * nb;
    const signed char *xq = x_quants + (size_t)tok * n_in;
    const unsigned char *row = w + (size_t)j * nb * Q6K_BYTES;

    float sum_l = 0.0f;   // this lane's sums[l], serial and ascending in i

    for (int i = 0; i < nb; ++i) {
        const unsigned char *blk = row + (size_t)i * Q6K_BYTES;
        const unsigned char *ql = blk;
        const unsigned char *qh = blk + QK_K / 2;
        const signed char   *sc = (const signed char *)(blk + QK_K / 2 + QK_K / 4);
        const unsigned short d16 =
            (unsigned short)blk[Q6K_BYTES - 2] | ((unsigned short)blk[Q6K_BYTES - 1] << 8);
        const signed char *q8 = xq + (size_t)i * QK_K;

        int aux = 0;
        for (int jj = g * 4; jj < g * 4 + 4; ++jj) {
            const int scale = (int)sc[jj];
            for (int half = 0; half < 2; ++half) {
                const int idx = jj * 16 + half * 8 + l;

                // The reference unpacks all 256 weights up front in two
                // 128-element halves; this is that mapping inverted, so a lane
                // materializes only the eight it needs.
                const int j2  = idx >> 7;
                const int r   = idx & 127;
                const int sub = r >> 5;
                const int l2  = r & 31;
                const unsigned char h = qh[j2 * 32 + l2];
                int base, hb;
                if (sub == 0)      { base = ql[j2 * 64 + l2]      & 0xF; hb = (h >> 0) & 3; }
                else if (sub == 1) { base = ql[j2 * 64 + l2 + 32] & 0xF; hb = (h >> 2) & 3; }
                else if (sub == 2) { base = ql[j2 * 64 + l2]      >> 4;  hb = (h >> 4) & 3; }
                else               { base = ql[j2 * 64 + l2 + 32] >> 4;  hb = (h >> 6) & 3; }
                const int av = (base | (hb << 4)) - 32;

                // int16 in the reference. The product of an int8 quant and a
                // -32..31 weight cannot leave that range, so the narrowing is
                // faithful rather than lossy.
                const short aux16 = (short)((int)q8[idx] * av);
                aux += scale * (int)aux16;
            }
        }
        // Fold the four g-groups. Integer, so exact and order-free.
        aux += __shfl_down_sync(0xffffffff, aux, 16);
        aux += __shfl_down_sync(0xffffffff, aux, 8);

        if (lane < 8) {
            const float d = h2f(d16) * xs[i];
            // NOT fused. The reference's compiler contracts Q5_K's identical
            // line and not this one.
            sum_l += d * (float)aux;
        }
    }

    // The 8-way fold, serial and ascending -- the oracle's last loop. The
    // shuffle is outside the branch because every lane must reach it.
    float sumf = 0.0f;
    for (int k = 0; k < 8; ++k) {
        const float v = __shfl_sync(0xffffffff, sum_l, k);
        if (lane == 0) sumf += v;
    }
    if (lane == 0) out[(size_t)tok * n_out + j] = sumf;
}

// Q5_K x Q8_K, one warp per output row.
//
// Five bits per weight: four in a nibble of `qs`, the fifth as a bit-plane in
// `qh`, giving an **unsigned** 0..31 with no bias -- unlike Q6_K's -32. The
// offset lives instead in a per-sub-block `min`, subtracted through the
// activation's `bsums`, which is the only reason Q8_K carries them.
//
// Same `lane = g*8 + l` layout as Q6_K, with `g` splitting eight sub-blocks
// two apiece. The extra structure is the `dmin` chain: it accumulates into
// `sumf` inside the super-block loop, ahead of the lane fold, and both of its
// updates are fused where Q6_K's are not.
__global__ void matmul_q5_k_q8_k(int n_in, int n_out,
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
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;
    const int tok = blockIdx.y;

    const int l = lane & 7;
    const int g = lane >> 3;

    const float *xs = x_scales + (size_t)tok * nb;
    const signed char *xq = x_quants + (size_t)tok * n_in;
    const short *xb = x_bsums + (size_t)tok * nb * (QK_K / 16);
    const unsigned char *row = w + (size_t)j * nb * Q5K_BYTES;

    float sum_l = 0.0f;   // sums[l]
    float sumf  = 0.0f;   // the dmin chain, lane 0's

    for (int i = 0; i < nb; ++i) {
        const unsigned char *blk = row + (size_t)i * Q5K_BYTES;
        const unsigned short d16    = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        const unsigned short dmin16 = (unsigned short)blk[2] | ((unsigned short)blk[3] << 8);
        const unsigned char *qh = blk + 16;
        const unsigned char *qs = blk + 16 + QK_K / 8;

        // 12 bytes -> 8 six-bit scales and 8 six-bit mins, through a fixed
        // shuffle that is easier to transcribe than to re-derive. Recomputed
        // per lane rather than staged in shared memory: a dozen integer ops
        // against a barrier. The block base is 16-byte aligned and these sit
        // at +4, +8, +12, so the 32-bit loads are legal.
        unsigned int u0 = *(const unsigned int *)(blk + 4);
        unsigned int u1 = *(const unsigned int *)(blk + 8);
        unsigned int u2 = *(const unsigned int *)(blk + 12);
        const unsigned int u3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
        const unsigned int uaux = u1 & KMASK1;
        u1 = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
        u2 = uaux;
        u0 &= KMASK1;
        // scales = bytes of (u0, u1); mins = bytes of (u2, u3).

        const signed char *q8 = xq + (size_t)i * QK_K;

        int aux = 0;
        for (int jj = g * 2; jj < g * 2 + 2; ++jj) {
            const unsigned int su = (jj < 4) ? u0 : u1;
            const int scale = (int)((su >> ((jj & 3) * 8)) & 0xff);
            for (int q = 0; q < 4; ++q) {
                const int idx = jj * 32 + q * 8 + l;

                const int jj4  = idx >> 6;
                const int r    = idx & 63;
                const int half = r >> 5;
                const int l2   = r & 31;
                const unsigned char q4 = qs[jj4 * 32 + l2];
                const int base = (half == 0) ? (q4 & 0xF) : (q4 >> 4);
                // `m` marches one bit per 32-element run: 1 << (j*2 + half).
                const int av = base + (((qh[l2] >> (jj4 * 2 + half)) & 1) ? 16 : 0);

                const short aux16 = (short)((int)q8[idx] * av);
                aux += scale * (int)aux16;
            }
        }
        aux += __shfl_down_sync(0xffffffff, aux, 16);
        aux += __shfl_down_sync(0xffffffff, aux, 8);

        if (lane < 8) {
            const float d = h2f(d16) * xs[i];
            // FUSED. `fma(d, aux32[l], sums[l])` in the compiled reference.
            sum_l = __fmaf_rn(d, (float)aux, sum_l);
        }
        if (lane == 0) {
            const short *bs = xb + (size_t)i * (QK_K / 16);
            int sumi = 0;
            for (int k = 0; k < QK_K / 16; ++k) {
                const int mk = k >> 1;
                const unsigned int mu = (mk < 4) ? u2 : u3;
                sumi += (int)bs[k] * (int)((mu >> ((mk & 3) * 8)) & 0xff);
            }
            const float dmin = h2f(dmin16) * xs[i];
            // FUSED, and inside the loop, before the lanes are folded in:
            // `fma(-dmin, sumi, sumf)`, not a multiply followed by a subtract.
            sumf = __fmaf_rn(-dmin, (float)sumi, sumf);
        }
    }

    for (int k = 0; k < 8; ++k) {
        const float v = __shfl_sync(0xffffffff, sum_l, k);
        if (lane == 0) sumf += v;
    }
    if (lane == 0) out[(size_t)tok * n_out + j] = sumf;
}

// IQ4_XS x Q8_K, one warp per output row.
//
// The odd one of the three. Its accumulation is a **single** `sumf`, not eight
// lanes -- each 32-element sub-block contributes `d * (sumi1 + sumi2)` straight
// into it -- so its rounding differs from Q5_K's and Q6_K's by construction.
// That single chain is `nb * 8` long, which at n_in 2048 is 64: exactly the
// length `matmul_q8_0_warp` keeps serial for 2-7%.
//
// Lane layout is `lane = t*4 + p`. `t` picks one of the eight sub-blocks the
// oracle folds in order, and `p` splits that sub-block's integer sum four ways.
// Each sub-block's 6-bit scale is split across `scales_l` and two bits marching
// through `scales_h`, and biased by -32.
// The IQ4_XS dot product, one warp cooperating on one output row.
//
// Factored out so the plain and the grouped matmul below share **one**
// implementation. Two copies of an accumulation this order-sensitive would
// drift, and the drift would be a handful of ulps in a kernel whose whole claim
// is bit-equality with the oracle.
//
// Returns the result on lane 0; other lanes' values are partial.
__device__ __forceinline__ float dot_iq4_xs_warp(
        int nb,
        const unsigned char *__restrict__ row,
        const float *__restrict__ xs,
        const signed char *__restrict__ xq,
        int lane) {
    const int t = lane >> 2;   // which of the eight sub-blocks
    const int p = lane & 3;    // which quarter of its integer sum

    // `t` is (ib, half) flattened, and the reference walks ib = 0,2,4,6 with
    // half = 0 then 1 -- so ascending `t` is the oracle's own order.
    const int ib   = (t >> 1) * 2;
    const int half = t & 1;

    float sumf = 0.0f;

    for (int ibl = 0; ibl < nb; ++ibl) {
        const unsigned char *blk = row + (size_t)ibl * IQ4XS_BYTES;
        const float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        const unsigned int sh = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
        const unsigned char *scales_l = blk + 4;
        const unsigned char *qs = blk + 4 + QK_K / 64;
        const float d4d8 = d * xs[ibl];
        const signed char *q8 = xq + (size_t)ibl * QK_K;

        // The reference shifts `h` right by 4 once per ib-pair, so at pair
        // ib/2 it has been shifted by 2*ib.
        const unsigned int h  = sh >> (ib * 2);
        const unsigned int lo = scales_l[ib >> 1];
        const int ls = (half == 0) ? (int)((lo & 0xf) | ((h << 4) & 0x30))
                                   : (int)((lo >> 4)  | ((h << 2) & 0x30));
        const float dh = d4d8 * (float)(ls - 32);

        const int qo = ib * 16 + half * 16;
        const int ao = ib * 32 + half * 32;

        // s1 and s2 in the reference, summed together here: both are integer,
        // so joining them cannot round.
        int s = 0;
        for (int k = p * 4; k < p * 4 + 4; ++k) {
            const unsigned char b = qs[qo + k];
            s += (int)q8[ao + k]      * (int)kvalues_iq4nl[b & 0xf];
            s += (int)q8[ao + 16 + k] * (int)kvalues_iq4nl[b >> 4];
        }
        s += __shfl_down_sync(0xffffffff, s, 2);
        s += __shfl_down_sync(0xffffffff, s, 1);
        // Lane t*4 now holds sumi1 + sumi2 for sub-block t.

        // The one serial chain. NOT fused -- unlike Q5_K and Q6_K, the
        // reference's compiler leaves this one as a multiply and an add.
        const float term = dh * (float)s;
        for (int k = 0; k < 8; ++k) {
            const float v = __shfl_sync(0xffffffff, term, k * 4);
            if (lane == 0) sumf += v;
        }
    }
    return sumf;
}

__global__ void matmul_iq4_xs_q8_k(int n_in, int n_out,
                                   const unsigned char *__restrict__ w,
                                   const float *__restrict__ x_scales,
                                   const signed char *__restrict__ x_quants,
                                   float *__restrict__ out) {
    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;
    const int tok = blockIdx.y;

    const float sumf = dot_iq4_xs_warp(
        nb,
        w + (size_t)j * nb * IQ4XS_BYTES,
        x_scales + (size_t)tok * nb,
        x_quants + (size_t)tok * n_in,
        lane);
    if (lane == 0) out[(size_t)tok * n_out + j] = sumf;
}

// The routed FFN's matmul: **every expert a token visits, in one launch.**
//
// Measured, and the reason this exists: issuing the eight experts one at a time
// costs 18.03 ms/token across the whole expert stage, against 4.46 for the
// grouped form. The saving is **not** launch overhead -- host issue is only
// ~2 us against ~9 us of device time per launch. It is occupancy. A
// `{2048, 512}` matmul is 128 blocks of 128 threads on a 36-SM card and
// finishes before the machine is full; eight of them stacked on `blockIdx.y`
// have eight times the parallelism for the same bytes.
//
// `x_stride_super` is what lets one kernel serve both halves of the FFN:
//
//   gate and up   every expert reads the same activation, so 0
//   down          every expert reads its own 512-element intermediate, so nb
//
// which is the seam's "derive the count from the buffer" convention applied to
// the expert axis rather than the token axis.
//
// Bit-exact by construction, for the same reason batching was: this changes
// which outputs share a launch, never how one accumulates. Each output row is
// `dot_iq4_xs_warp` over the same bytes in the same order.
__global__ void matmul_iq4_xs_q8_k_moe(int n_in, int n_out, int x_stride_super,
                                       int n_used,
                                       const unsigned char *__restrict__ w0,
                                       const unsigned char *__restrict__ w1,
                                       const unsigned char *__restrict__ w2,
                                       const unsigned char *__restrict__ w3,
                                       const unsigned char *__restrict__ w4,
                                       const unsigned char *__restrict__ w5,
                                       const unsigned char *__restrict__ w6,
                                       const unsigned char *__restrict__ w7,
                                       const float *__restrict__ x_scales,
                                       const signed char *__restrict__ x_quants,
                                       float *__restrict__ out) {
    const int e = blockIdx.y;
    if (e >= n_used) return;
    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;

    // Eight pointers as kernel arguments rather than a device array: an array
    // would need an upload per launch, 120 a token, to save nothing. When
    // expert selection moves onto the device this becomes a slot-table lookup
    // and the whole signature changes anyway.
    const unsigned char *ws[8] = {w0, w1, w2, w3, w4, w5, w6, w7};

    const float sumf = dot_iq4_xs_warp(
        nb,
        ws[e] + (size_t)j * nb * IQ4XS_BYTES,
        x_scales + (size_t)e * x_stride_super,
        x_quants + (size_t)e * x_stride_super * QK_K,
        lane);
    if (lane == 0) out[(size_t)e * n_out + j] = sumf;
}

// Weighted sum of `n_rows` rows into one, in ascending row order.
//
// The MoE accumulation: `acc[j] = sum_e scale[e] * rows[e * n + j]`, replacing
// `n_rows` separate `add_scaled` launches over an accumulator that starts at
// zero. **Bit-identical to that loop**: the sum is walked serially and
// ascending exactly as the oracle walks its picks -- parallel over `j`, which
// the oracle already treats as independent, and serial over `e`, which it does
// not.
extern "C" __global__ void add_scaled_rows(int n, int n_rows, float s0, float s1,
                                           float s2, float s3, float s4, float s5,
                                           float s6, float s7,
                                           const float *__restrict__ rows,
                                           float *__restrict__ acc) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const float sc[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
    float a = 0.0f;
    for (int e = 0; e < n_rows; ++e) a += sc[e] * rows[(size_t)e * n + j];
    acc[j] = a;
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
