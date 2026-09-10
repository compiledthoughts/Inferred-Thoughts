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
__global__ void attn_flash(int n_pos_first, int kv_dim, int head_dim, int n_head,
                           int n_head_kv, int use_warp, int part_stride, float scale,
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

    // **The query row is a grid dimension now, folded into `blockIdx.x`.**
    //
    // One launch per row read the whole K/V window from DRAM for that row
    // alone: 29,390 launches on a 2,936-token prefill, and a measured 3.01 TB
    // of re-reads on a 5,369-token turn at depth 27,410. Rows of one group run
    // concurrently here, so the blocks sharing a chunk hit L2 rather than DRAM.
    //
    // Row `r` of the group attends over `n_pos_first + r` positions -- rows are
    // consecutive, which is `Attn::n_pos_of`. Nothing about one row's
    // arithmetic changes, so this is bit-identical rather than merely within
    // the derived tolerance.
    const int hq = blockIdx.x % n_head;
    const int r = blockIdx.x / n_head;
    const int n_pos = n_pos_first + r;
    const int split = blockIdx.y;
    const int lo = split * FD_CHUNK;
    // Chunks past this row's causal window: its combine never reads them.
    if (lo >= n_pos) return;
    const int len = min(FD_CHUNK, n_pos - lo);
    const int off = (hq / (n_head / n_head_kv)) * head_dim;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        sq[i] = q[((size_t)r * n_head + hq) * head_dim + i];
    }
    __syncthreads();

    // **Two score phases in one kernel, chosen by `use_warp`.**
    //
    // They were briefly two kernels and that was a bug: the choice depends on
    // `n_pos`, so a long run crossed the threshold mid-generation, the launch
    // name changed, and a recorded CUDA graph cannot express a changing
    // sequence. Replay caught it -- "launch 131 is attn_flash_warp, but the
    // recorded step has attn_flash there". One kernel with a grid-uniform
    // branch keeps the sequence identical every pass.
    float score = -INFINITY;
    if (use_warp) {
        // **One warp per position, because K was being read transposed.** With
        // one thread per position, adjacent threads are `kv_dim` apart -- 1024
        // bytes on this model -- so a warp's load touches 32 cache lines to use
        // two bytes from each. Here lanes stride `head_dim` instead, so lane
        // `l` reads `key[l]`, `key[l + 32]`, ...: consecutive lanes,
        // consecutive addresses. Measured 43 -> 133 GB/s, ~3x at 20k context.
        //
        // The dot becomes eight serial terms per lane plus a five-level tree,
        // which accumulates error as O(log n) where the serial walk of 256 is
        // O(n) -- more accurate, not less. `attend_tolerance` is unchanged and
        // `the_warp_attention_agrees_with_the_oracle` checks it at both models'
        // shapes, twelve depths and both batch shapes.
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int nwarps = blockDim.x >> 5;
        // `p` is warp-uniform, so the branch never diverges within a warp and
        // every lane reaches the shuffles.
        for (int p = warp; p < FD_CHUNK; p += nwarps) {
            float dot = 0.0f;
            if (p < len) {
                const unsigned short *key = k + (size_t)(lo + p) * kv_dim + off;
                for (int i = lane; i < head_dim; i += 32) dot += sq[i] * h2f(key[i]);
#pragma unroll
                for (int sh = 16; sh > 0; sh >>= 1)
                    dot += __shfl_down_sync(0xffffffff, dot, sh);
            }
            if (lane == 0) red[p] = (p < len) ? dot * scale : -INFINITY;
        }
        __syncthreads();
        score = red[threadIdx.x];
    } else {
        // One position per thread; the dot over head_dim stays serial in-thread.
        if (threadIdx.x < len) {
            const unsigned short *key =
                k + (size_t)(lo + threadIdx.x) * kv_dim + off;
            float dot = 0.0f;
            for (int i = 0; i < head_dim; ++i) dot += sq[i] * h2f(key[i]);
            score = dot * scale;
        }
        red[threadIdx.x] = score;
    }

    // Chunk max. A tree here is *exact* — max never rounds — so this costs
    // nothing in accuracy.
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
    const size_t base = ((size_t)r * n_head + hq) * part_stride + split;
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
__global__ void attn_flash_combine(int n_pos_first, int head_dim, int n_head,
                                   int part_stride, int chunk,
                                   const float *__restrict__ part_acc,
                                   const float *__restrict__ part_m,
                                   const float *__restrict__ part_l,
                                   float *__restrict__ out) {
    extern __shared__ float w[];   // [n_split] rescaling weights
    __shared__ float total;

    // As `attn_flash`: the query row is folded into `blockIdx.x`, and each row
    // has its own split count because it has its own causal window.
    const int hq = blockIdx.x % n_head;
    const int r = blockIdx.x / n_head;
    const int n_split = (n_pos_first + r + chunk - 1) / chunk;
    const size_t row = ((size_t)r * n_head + hq) * part_stride;
    const float *pm = part_m + row;
    const float *pl = part_l + row;

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

    const float *pa = part_acc + row * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_split; ++s) acc += pa[(size_t)s * head_dim + i] * w[s];
        out[((size_t)r * n_head + hq) * head_dim + i] = acc / total;
    }
}

// ---------------------------------------------------------------------------
// `attn_flash` and its combine, decomposed
// ---------------------------------------------------------------------------
//
// **Copies of the two production attention kernels, each able to leave one
// piece out**, so a bench can price the pieces rather than a session guessing
// at them. Attention is linear in depth from d8192 to d65536 at 4.6% of the
// fp32 cores, and a one-kv-head arm reading half the bytes costs the same, so
// it is bound by work. Reading the kernel ranks its pieces by instruction count
// -- the same kind of reading that sent two IQ4_XS attempts after parts worth
// 2 us of 144, where a decomposition found 72% in one pass.
//
// **The mask is a template parameter, not a kernel argument.** A runtime branch
// around something as small as an `expf` or a load can be if-converted, and an
// arm that still paid for the work it claims to remove would report that work
// as free -- a defect presenting as a finding. Instances fold every branch at
// compile time. `S == 0` is production's arithmetic statement for statement,
// and `what_attention_is_made_of` asserts its bits equal production's before
// timing anything.
//
// A removed piece leaves values that are finite and of ordinary size, so no arm
// can be timing an exceptional path instead of an absence of work.
#define DBG_ATT_NO_QLOAD     1    // query constant: no load from global; stores and barrier kept
#define DBG_ATT_NO_SCORE     2    // no K loads, no dot, no shuffle tree, and its barrier
#define DBG_ATT_NO_MAX       4    // no max tree and none of its eight barriers
#define DBG_ATT_MAX_BARRIERS 8    // the max tree's barriers, one private store between each
#define DBG_ATT_NO_EXP       16   // no expf
#define DBG_ATT_NO_SUM       32   // no sum tree and none of its seven barriers
#define DBG_ATT_NO_V         64   // no weighted sum over V
#define DBG_ATT_NO_CLOOP     128  // combine: no lane-0 max and expf loop over the chunks
#define DBG_ATT_NO_CV        256  // combine: no weighted sum over the partials

}  // extern "C" -- a template cannot have C linkage; reopened for the instances.

template <int S>
__device__ __forceinline__ void attn_flash_body(
        float *sq, float *se, float *red,
        int n_pos_first, int kv_dim, int head_dim, int n_head,
        int n_head_kv, int use_warp, int part_stride, float scale,
        const float *__restrict__ q,
        const unsigned short *__restrict__ k,
        const unsigned short *__restrict__ v,
        float *__restrict__ part_acc,
        float *__restrict__ part_m,
        float *__restrict__ part_l) {
    const int hq = blockIdx.x % n_head;
    const int r = blockIdx.x / n_head;
    const int n_pos = n_pos_first + r;
    const int split = blockIdx.y;
    const int lo = split * FD_CHUNK;
    if (lo >= n_pos) return;
    const int len = min(FD_CHUNK, n_pos - lo);
    const int off = (hq / (n_head / n_head_kv)) * head_dim;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        if (S & DBG_ATT_NO_QLOAD) {
            sq[i] = 0.25f;
        } else {
            sq[i] = q[((size_t)r * n_head + hq) * head_dim + i];
        }
    }
    __syncthreads();

    float score = -INFINITY;
    if (S & DBG_ATT_NO_SCORE) {
        // A small ramp rather than a constant, so the max and the sum still see
        // distinct values of ordinary size.
        if (threadIdx.x < len) score = (float)(threadIdx.x & 15) * 0.125f;
        red[threadIdx.x] = score;
    } else if (use_warp) {
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int nwarps = blockDim.x >> 5;
        for (int p = warp; p < FD_CHUNK; p += nwarps) {
            float dot = 0.0f;
            if (p < len) {
                const unsigned short *key = k + (size_t)(lo + p) * kv_dim + off;
                for (int i = lane; i < head_dim; i += 32) dot += sq[i] * h2f(key[i]);
#pragma unroll
                for (int sh = 16; sh > 0; sh >>= 1)
                    dot += __shfl_down_sync(0xffffffff, dot, sh);
            }
            if (lane == 0) red[p] = (p < len) ? dot * scale : -INFINITY;
        }
        __syncthreads();
        score = red[threadIdx.x];
    } else {
        if (threadIdx.x < len) {
            const unsigned short *key =
                k + (size_t)(lo + threadIdx.x) * kv_dim + off;
            float dot = 0.0f;
            for (int i = 0; i < head_dim; ++i) dot += sq[i] * h2f(key[i]);
            score = dot * scale;
        }
        red[threadIdx.x] = score;
    }

    __syncthreads();
    float m = 0.0f;
    if (!(S & DBG_ATT_NO_MAX)) {
        for (int s = FD_CHUNK / 2; s > 0; s >>= 1) {
            if (S & DBG_ATT_MAX_BARRIERS) {
                // One real store per level, so the barriers cannot be merged
                // away as having nothing between them. `se` is rewritten below
                // before anything reads it.
                se[threadIdx.x] = (float)s;
            } else if (threadIdx.x < s) {
                red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
            }
            __syncthreads();
        }
        if (!(S & DBG_ATT_MAX_BARRIERS)) m = red[0];
        __syncthreads();
    }

    float e;
    if (S & DBG_ATT_NO_EXP) {
        e = (threadIdx.x < len) ? 0.5f : 0.0f;
    } else {
        e = (threadIdx.x < len) ? expf(score - m) : 0.0f;
    }
    se[threadIdx.x] = e;
    red[threadIdx.x] = e;
    __syncthreads();
    float l = 1.0f;
    if (!(S & DBG_ATT_NO_SUM)) {
        for (int s = FD_CHUNK / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        l = red[0];
    }

    const size_t base = ((size_t)r * n_head + hq) * part_stride + split;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float acc = 0.0f;
        if (!(S & DBG_ATT_NO_V)) {
            for (int t = 0; t < len; ++t) {
                acc += se[t] * h2f(v[(size_t)(lo + t) * kv_dim + off + i]);
            }
        }
        part_acc[base * head_dim + i] = acc;
    }
    if (threadIdx.x == 0) {
        part_m[base] = m;
        part_l[base] = l;
    }
}

template <int S>
__device__ __forceinline__ void dbg_attn_flash_combine_body(
        float *w, float *total,
        int n_pos_first, int head_dim, int n_head, int part_stride, int chunk,
        const float *__restrict__ part_acc,
        const float *__restrict__ part_m,
        const float *__restrict__ part_l,
        float *__restrict__ out) {
    const int hq = blockIdx.x % n_head;
    const int r = blockIdx.x / n_head;
    const int n_split = (n_pos_first + r + chunk - 1) / chunk;
    const size_t row = ((size_t)r * n_head + hq) * part_stride;
    const float *pm = part_m + row;
    const float *pl = part_l + row;

    if (S & DBG_ATT_NO_CLOOP) {
        // Every chunk weighted 1, stored in parallel, so the sum below still
        // reads a shared array of the same length.
        for (int s = threadIdx.x; s < n_split; s += blockDim.x) w[s] = 1.0f;
        if (threadIdx.x == 0) *total = 1.0f;
    } else if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int s = 0; s < n_split; ++s) m = fmaxf(m, pm[s]);
        float l = 0.0f;
        for (int s = 0; s < n_split; ++s) {
            w[s] = expf(pm[s] - m);
            l += pl[s] * w[s];
        }
        *total = l;
    }
    __syncthreads();

    const float *pa = part_acc + row * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float acc = 0.0f;
        if (!(S & DBG_ATT_NO_CV)) {
            for (int s = 0; s < n_split; ++s) acc += pa[(size_t)s * head_dim + i] * w[s];
        }
        out[((size_t)r * n_head + hq) * head_dim + i] = acc / *total;
    }
}

extern "C" {

// The instances `Cuda::attn_dbg` can name. Same signatures as production, so
// routing to one changes a kernel name and nothing else.
#define DBG_ATTN_FLASH_INSTANCE(S)                                                  \
extern "C" __global__ void dbg_attn_flash_##S(                                      \
        int n_pos_first, int kv_dim, int head_dim, int n_head, int n_head_kv,     \
        int use_warp, int part_stride, float scale,                               \
        const float *__restrict__ q, const unsigned short *__restrict__ k,        \
        const unsigned short *__restrict__ v, float *__restrict__ part_acc,       \
        float *__restrict__ part_m, float *__restrict__ part_l) {                 \
    extern __shared__ float smem[];                                               \
    attn_flash_body<S>(smem, smem + head_dim, smem + head_dim + FD_CHUNK,     \
        n_pos_first, kv_dim, head_dim, n_head, n_head_kv, use_warp, part_stride,  \
        scale, q, k, v, part_acc, part_m, part_l);                                \
}

#define DBG_ATTN_COMBINE_INSTANCE(S)                                                \
extern "C" __global__ void dbg_attn_flash_combine_##S(                              \
        int n_pos_first, int head_dim, int n_head, int part_stride, int chunk,    \
        const float *__restrict__ part_acc, const float *__restrict__ part_m,      \
        const float *__restrict__ part_l, float *__restrict__ out) {              \
    extern __shared__ float w[];                                                  \
    __shared__ float total;                                                       \
    dbg_attn_flash_combine_body<S>(w, &total, n_pos_first, head_dim, n_head,      \
        part_stride, chunk, part_acc, part_m, part_l, out);                       \
}

DBG_ATTN_FLASH_INSTANCE(0)
DBG_ATTN_FLASH_INSTANCE(1)
DBG_ATTN_FLASH_INSTANCE(2)
DBG_ATTN_FLASH_INSTANCE(4)
DBG_ATTN_FLASH_INSTANCE(8)
DBG_ATTN_FLASH_INSTANCE(16)
DBG_ATTN_FLASH_INSTANCE(32)
DBG_ATTN_FLASH_INSTANCE(64)
DBG_ATTN_FLASH_INSTANCE(119)
DBG_ATTN_COMBINE_INSTANCE(0)
DBG_ATTN_COMBINE_INSTANCE(128)
DBG_ATTN_COMBINE_INSTANCE(256)
DBG_ATTN_COMBINE_INSTANCE(384)


// Attention for a batch: the whole KV walk in one block, per (query row, head).
//
// **The partials were the cost, not the keys.** `attn_flash` splits the KV
// sequence across blocks and writes each chunk's `(acc, m, l)` to global for
// `attn_flash_combine` to read back. At n_q 128, n_pos 8192 that is
// `n_q * n_head * n_split * head_dim` floats -- **537 MiB written and read back
// per attending layer**, about 17% of this card's bandwidth spent on scratch.
// And it grows with `n_split`, i.e. with context, which is exactly the shape of
// the depth problem: llama.cpp holds ~1,040 tok/s flat from 2.9k to 13.5k while
// this engine falls 40%.
//
// Splitting the sequence exists because **decode has one query row** and no
// other parallelism to fill 36 SMs with. A batch has `n_q * n_head` blocks
// before splitting anything, so the split buys nothing and costs the scratch.
// Gated on `n_q > 1`; decode keeps the two-kernel path unchanged.
//
// The online-softmax merge moves from `attn_flash_combine` into this loop. Same
// algebra -- a chunk's numbers are relative to its own max, so rescaling by
// `exp(m_chunk - m_new)` puts them on one reference before they are added --
// but associated left-to-right across chunks instead of all at once against a
// global max, so it is a different rounding of the same quantity.
__global__ void attn_flash_fused(int n_pos_first, int kv_dim, int head_dim,
                                 int n_head, int n_head_kv, int use_warp,
                                 float scale,
                                 const float *__restrict__ q,
                                 const unsigned short *__restrict__ k,
                                 const unsigned short *__restrict__ v,
                                 float *__restrict__ out) {
    extern __shared__ float smem[];
    float *sq  = smem;                  // [head_dim] the query
    float *se  = sq + head_dim;         // [FD_CHUNK] exp(score - m) per position
    float *red = se + FD_CHUNK;         // [FD_CHUNK] reduction scratch
    float *acc = red + FD_CHUNK;        // [head_dim] running output, unnormalized
    __shared__ float run_m, run_l, ra, rb;

    const int hq = blockIdx.x % n_head;
    const int r  = blockIdx.x / n_head;
    // Row `r` attends over `n_pos_first + r` positions -- consecutive rows,
    // which is `Attn::n_pos_of`.
    const int n_pos = n_pos_first + r;
    const int off = (hq / (n_head / n_head_kv)) * head_dim;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        sq[i] = q[((size_t)r * n_head + hq) * head_dim + i];
        acc[i] = 0.0f;
    }
    if (threadIdx.x == 0) { run_m = -INFINITY; run_l = 0.0f; }
    __syncthreads();

    const int n_split = (n_pos + FD_CHUNK - 1) / FD_CHUNK;
    for (int split = 0; split < n_split; ++split) {
        const int lo = split * FD_CHUNK;
        const int len = min(FD_CHUNK, n_pos - lo);

        // --- scores, exactly as `attn_flash` computes them ---
        float score = -INFINITY;
        if (use_warp) {
            const int lane = threadIdx.x & 31;
            const int warp = threadIdx.x >> 5;
            const int nwarps = blockDim.x >> 5;
            for (int p = warp; p < FD_CHUNK; p += nwarps) {
                float dot = 0.0f;
                if (p < len) {
                    const unsigned short *key = k + (size_t)(lo + p) * kv_dim + off;
                    for (int i = lane; i < head_dim; i += 32) dot += sq[i] * h2f(key[i]);
#pragma unroll
                    for (int sh = 16; sh > 0; sh >>= 1)
                        dot += __shfl_down_sync(0xffffffff, dot, sh);
                }
                if (lane == 0) red[p] = (p < len) ? dot * scale : -INFINITY;
            }
            __syncthreads();
            score = red[threadIdx.x];
        } else {
            if (threadIdx.x < len) {
                const unsigned short *key =
                    k + (size_t)(lo + threadIdx.x) * kv_dim + off;
                float dot = 0.0f;
                for (int i = 0; i < head_dim; ++i) dot += sq[i] * h2f(key[i]);
                score = dot * scale;
            }
            red[threadIdx.x] = score;
        }

        // Chunk max. A tree is exact here -- max never rounds.
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

        // --- merge this chunk into the running total ---
        //
        // One thread decides the rescaling so every thread applies the same
        // two numbers. `run_m` starts at -INFINITY, so the first chunk gets
        // `ra = exp(-inf) = 0` against `acc = 0` and contributes exactly
        // itself.
        if (threadIdx.x == 0) {
            const float nm = fmaxf(run_m, m);
            ra = expf(run_m - nm);
            rb = expf(m - nm);
            run_l = run_l * ra + l * rb;
            run_m = nm;
        }
        __syncthreads();

        // Weighted sum of this chunk's values, folded straight into `acc`.
        // Adjacent threads hold adjacent elements and read adjacent halves of
        // V, so the reads coalesce as they did before.
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            float chunk = 0.0f;
            for (int t = 0; t < len; ++t) {
                chunk += se[t] * h2f(v[(size_t)(lo + t) * kv_dim + off + i]);
            }
            acc[i] = acc[i] * ra + chunk * rb;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        out[((size_t)r * n_head + hq) * head_dim + i] = acc[i] / run_l;
    }
}



// ---------------------------------------------------------------------------
// Attention with the score matrix on the tensor cores
// ---------------------------------------------------------------------------
//
// **Attention is 33% of a 15,000-token prefill and runs at ~5% of this card's
// fp32 peak.** `attn_flash` computes `S = Q K^T` one element at a time: a warp
// does a 256-long dot product per (query row, key position) pair, eight FMAs a
// lane plus a five-step shuffle tree -- about thirteen warp-instructions for a
// single output. One `mma.m16n8k16` produces a 16x8 tile of `S`, 128 elements,
// so the same work is sixteen instructions rather than roughly seventeen
// hundred.
//
// This is FlashAttention-2's shape, not FlashAttention-3's. FA-3's gains come
// from WGMMA, TMA and warp-specialized producer/consumer pipelines, all sm_90a;
// its own ablation puts that asynchrony at 570 -> 661 TFLOPs, about 16%. The
// rest of FA-2's advantage over a scalar kernel is simply having the matmuls on
// tensor cores, and `mma.sync` is available from sm_75 -- the same gate that
// lets llama.cpp's `fattn-mma-f16` run on this card.
//
// # What it costs
//
// The GEMM operands must be f16. **K and V already are**, because the KV cache
// stores them that way to match what llama.cpp computes attention over, so only
// Q converts. Everything numerically load-bearing stays f32: the running max,
// the running sum, the rescaling, and the output accumulator. That is the split
// FlashAttention makes, and their measured RMSE against an fp64 reference is
// 1.9e-4 for FP16 FlashAttention against 3.2e-4 for a standard implementation
// -- better than the naive kernel, because the softmax stays in f32.
//
// It is still a precision change rather than a reordering, so this is off by
// default and `attend_tolerance` does not cover it.

// Query rows and key positions per block tile.
//
// 16 x 32 keeps shared memory near 26 KiB: an f16 query tile, one f16 staging
// buffer reused for K and then V, and the f32 score tile. Four warps each own
// one 16x8 quadrant of `S`, and `ATT_KC` equal to the warp size makes the
// per-row softmax a single warp reduction.
#define ATT_QT 16
#define ATT_KC 32

// Head dimensions each thread carries an accumulator for, one per query row.
#define ATT_DPER 2

__device__ __forceinline__ unsigned short f2h(float x) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return h;
}

// D = A * B + C over f16 with an f32 accumulator, 16x8x16. sm_75 and later.
__device__ __forceinline__ void mma_m16n8k16_f16(
        float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2],
        const float (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// The score matrix on the tensor cores, keeping the KV-sequence split.
//
// **Both halves of that sentence are measured decisions.** The tensor cores are
// there because attention runs at ~5% of fp32 peak with a scalar dot product
// per (query, key) pair. The split stays because removing it -- which
// `attn_flash_fused` did, taking the partial buffers with it -- was worth
// 0.96x: 537 MiB per layer of partial traffic costs nothing on a kernel using
// 4% of this card's bandwidth, while the parallelism it buys is everything. A
// first version of this kernel dropped the split and fell to 0.57x at n_pos
// 32768, running 128 blocks against the split path's tens of thousands.
//
// # Why a block owns 128 positions but tiles them 32 at a time
//
// Two limits pull opposite ways. The MMA tile wants K staged in shared memory,
// and 128 positions x 256 head dims of f16 is 64 KiB -- too much. The partial
// buffers want *few* splits, because they cost
// `rows * n_head * n_split * head_dim` floats: at 20k context a 32-position
// split is 671 MiB where a 128-position one is 168.
//
// So the block owns a `FD_CHUNK` chunk, as the split path does, and walks it as
// four `ATT_KC` sub-tiles with the online-softmax merge held in registers --
// emitting a single partial. `attn_flash_combine` is unchanged apart from
// taking the chunk width, since the two paths now use different ones.
__global__ void attn_flash_mma(int n_pos_first, int n_rows, int kv_dim,
                               int head_dim, int n_head, int n_head_kv,
                               int part_stride, float scale,
                               const float *__restrict__ q,
                               const unsigned short *__restrict__ k,
                               const unsigned short *__restrict__ v,
                               float *__restrict__ part_acc,
                               float *__restrict__ part_m,
                               float *__restrict__ part_l) {
    extern __shared__ char smem_raw[];
    unsigned short *sq  = (unsigned short *)smem_raw;      // ATT_QT x head_dim
    unsigned short *skv = sq + ATT_QT * head_dim;          // ATT_KC x head_dim
    float *ss   = (float *)(skv + ATT_KC * head_dim);      // ATT_QT x ATT_KC
    float *smax = ss + ATT_QT * ATT_KC;                    // running max
    float *ssum = smax + ATT_QT;                           // running sum
    float *sra  = ssum + ATT_QT;                           // rescale, running
    float *srb  = sra + ATT_QT;                            // rescale, sub-tile

    const int hq = blockIdx.x;
    const int split = blockIdx.y;
    const int base_lo = split * FD_CHUNK;
    const int off = (hq / (n_head / n_head_kv)) * head_dim;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;

    // Row `r` attends over `n_pos_first + r` positions -- consecutive, as
    // `Attn::n_pos_of` says -- so this is the widest window in the tile.
    const int n_pos_max = n_pos_first + n_rows - 1;
    if (base_lo >= n_pos_max) return;

    for (int i = threadIdx.x; i < ATT_QT * head_dim; i += blockDim.x) {
        const int r = i / head_dim, d = i % head_dim;
        const int rr = (r < n_rows) ? r : 0;   // padding rows, masked below
        sq[i] = f2h(q[((size_t)rr * n_head + hq) * head_dim + d]);
    }
    if (threadIdx.x < ATT_QT) {
        smax[threadIdx.x] = -INFINITY;
        ssum[threadIdx.x] = 0.0f;
    }
    // This thread's output accumulators: head dimensions
    // `threadIdx.x + i * blockDim.x`, one float per query row each.
    float acc[ATT_DPER][ATT_QT];
#pragma unroll
    for (int di = 0; di < ATT_DPER; ++di) {
#pragma unroll
        for (int r = 0; r < ATT_QT; ++r) acc[di][r] = 0.0f;
    }
    __syncthreads();

    for (int sub = 0; sub < FD_CHUNK / ATT_KC; ++sub) {
        const int lo = base_lo + sub * ATT_KC;
        // Block-uniform, so the barriers below are not divergent.
        if (lo >= n_pos_max) break;

        for (int i = threadIdx.x; i < ATT_KC * head_dim; i += blockDim.x) {
            const int p = i / head_dim, d = i % head_dim;
            skv[i] = (lo + p < n_pos_max) ? k[(size_t)(lo + p) * kv_dim + off + d] : 0;
        }
        __syncthreads();

        // --- S = Q K^T. Warp `w` owns columns [8w, 8w+8) of the 16x32 tile.
        {
            float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const int col = warp * 8 + g;
            for (int k0 = 0; k0 < head_dim; k0 += 16) {
                unsigned a[4], b[2];
                a[0] = *(const unsigned *)&sq[(size_t)g * head_dim + k0 + 2 * t];
                a[1] = *(const unsigned *)&sq[(size_t)(g + 8) * head_dim + k0 + 2 * t];
                a[2] = *(const unsigned *)&sq[(size_t)g * head_dim + k0 + 2 * t + 8];
                a[3] = *(const unsigned *)&sq[(size_t)(g + 8) * head_dim + k0 + 2 * t + 8];
                b[0] = *(const unsigned *)&skv[(size_t)col * head_dim + k0 + 2 * t];
                b[1] = *(const unsigned *)&skv[(size_t)col * head_dim + k0 + 2 * t + 8];
                float dd[4];
                mma_m16n8k16_f16(dd, a, b, c);
#pragma unroll
                for (int i = 0; i < 4; ++i) c[i] = dd[i];
            }
            const int c0 = warp * 8 + 2 * t;
            ss[(size_t)g * ATT_KC + c0]           = c[0];
            ss[(size_t)g * ATT_KC + c0 + 1]       = c[1];
            ss[(size_t)(g + 8) * ATT_KC + c0]     = c[2];
            ss[(size_t)(g + 8) * ATT_KC + c0 + 1] = c[3];
        }
        __syncthreads();

        // --- softmax over the sub-tile, and the merge into the running total.
        // Warp `w` owns rows [4w, 4w+4); `ATT_KC` is the warp size, so one warp
        // reduction covers a row.
        for (int rr = 0; rr < ATT_QT / 4; ++rr) {
            const int r = warp * (ATT_QT / 4) + rr;
            const int n_pos_r = n_pos_first + r;
            const bool live = (r < n_rows) && (lo + lane < n_pos_r);
            float s = live ? ss[(size_t)r * ATT_KC + lane] * scale : -INFINITY;
            float m = s;
#pragma unroll
            for (int sh = 16; sh > 0; sh >>= 1)
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, sh));
            const float e = live ? expf(s - m) : 0.0f;
            float l = e;
#pragma unroll
            for (int sh = 16; sh > 0; sh >>= 1)
                l += __shfl_xor_sync(0xffffffff, l, sh);
            ss[(size_t)r * ATT_KC + lane] = e;
            if (lane == 0) {
                // `m` is -INFINITY when the sub-tile is entirely past this
                // row's window; then `rb` is 0 and nothing is contributed.
                const float om = smax[r], ol = ssum[r];
                const float nm = fmaxf(om, m);
                const float a = expf(om - nm);
                const float b = (m == -INFINITY) ? 0.0f : expf(m - nm);
                smax[r] = nm;
                ssum[r] = ol * a + l * b;
                sra[r] = a;
                srb[r] = b;
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < ATT_KC * head_dim; i += blockDim.x) {
            const int p = i / head_dim, d = i % head_dim;
            skv[i] = (lo + p < n_pos_max) ? v[(size_t)(lo + p) * kv_dim + off + d] : 0;
        }
        __syncthreads();

        // --- O += P V, still scalar. The second GEMM needs V transposed in
        // shared to meet the k-major operand layout, which is the next step;
        // this isolates what the score GEMM is worth with parallelism intact.
#pragma unroll
        for (int di = 0; di < ATT_DPER; ++di) {
            const int d = threadIdx.x + di * blockDim.x;
            if (d >= head_dim) continue;
#pragma unroll
            for (int r = 0; r < ATT_QT; ++r) {
                float chunk = 0.0f;
                for (int p = 0; p < ATT_KC; ++p) {
                    chunk += ss[(size_t)r * ATT_KC + p] * h2f(skv[(size_t)p * head_dim + d]);
                }
                acc[di][r] = acc[di][r] * sra[r] + chunk * srb[r];
            }
        }
        __syncthreads();
    }

    // One partial per (row, head, chunk), in the layout `attn_flash_combine`
    // already reads.
#pragma unroll
    for (int di = 0; di < ATT_DPER; ++di) {
        const int d = threadIdx.x + di * blockDim.x;
        if (d >= head_dim) continue;
        // Fully unrolled with a predicate: a runtime bound makes `r` a
        // non-constant index and ptxas puts `acc` in local memory -- 128 bytes
        // of stack frame, which it does not report as a spill. Same trap the
        // grouped MoE kernels hit this morning.
#pragma unroll
        for (int r = 0; r < ATT_QT; ++r) {
            if (r >= n_rows) continue;
            part_acc[(((size_t)r * n_head + hq) * part_stride + split) * head_dim + d] =
                acc[di][r];
        }
    }
    if (threadIdx.x < n_rows) {
        const size_t base = ((size_t)threadIdx.x * n_head + hq) * part_stride + split;
        part_m[base] = smax[threadIdx.x];
        part_l[base] = ssum[threadIdx.x];
    }
}

// V as an MMA operand without a transposing write: `ldmatrix.x4.trans`.
//
// **This is the instruction the 08-09 attempt was missing.** `O += P V` needs
// the *position* axis to be the contraction axis, but the KV cache stores
// `[position][head_dim]`, so staging a transposed copy makes consecutive
// threads write `head_dim` apart -- a scattered shared write with heavy bank
// conflicts, measured at 0.63-0.94x and reverted.
//
// `ldmatrix.sync.aligned.m8n8.x4.trans.b16` transposes *in the load path*: four
// 8x8 tiles are read from shared in their natural layout and delivered to
// registers already transposed. Nothing is written back, so there is no scatter
// to conflict. It is sm_75 -- the same gate as `mma.sync`. FlashAttention-3
// section 3.3 reaches for `stmatrix` as well, which is sm_90a and out of reach
// here, but that is the *store* half; the load half is what removes the
// transpose, and llama.cpp's `fattn-mma-f16` gets in on exactly this one.
//
// Each lane supplies the address of one source row: lanes 0-15 address rows
// 0-15 of the left eight columns, lanes 16-31 the same rows of the right eight.
// **The destination registers come back permuted, 0 2 1 3.** The transposed
// form orders the fragment differently from the plain one, and getting it wrong
// yields a plausible wrong answer rather than an error.
__device__ __forceinline__ void ldmatrix_x4_trans(
        unsigned (&a)[4], const unsigned short *src, int stride_h) {
    const int lane = threadIdx.x & 31;
    const unsigned *p =
        (const unsigned *)src + (size_t)(lane & 15) * (stride_h >> 1) + (lane >> 4) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(a[0]), "=r"(a[2]), "=r"(a[1]), "=r"(a[3])
        : "l"(p));
}

// Dim-blocks of 16 head dimensions each warp carries an accumulator for.
// Four warps x 4 blocks x 16 dims caps `head_dim` at 256, which the launch
// checks. Raising it costs registers: `acc` is `ATT_DBLK * 2 * 4` floats.
#define ATT_DBLK 4

// Both attention GEMMs on the tensor cores.
//
// `attn_flash_mma` put `S = Q K^T` there and left `O += P V` scalar, because
// the second GEMM's operand layout looked to need a transposed V in shared.
// It does not. The contraction runs the other way round:
//
//     O^T[d][r] += sum_p V[p][d] * P[r][p]
//
// so **head dimensions are M, query rows are N, and positions are K**. V is the
// A operand, read straight out of `skv` in its stored `[position][head_dim]`
// layout by `ldmatrix.x4.trans`. P is the B operand, and `mma.m16n8k16.row.col`
// wants B's contraction axis contiguous per column -- which is exactly
// `[row][position]`, the layout the score phase already produces. So neither
// operand needs a transposing write, and unlike llama.cpp we need no
// `movmatrix` either: their scores stay in registers, ours already pass through
// shared memory, and passing through is where the layout change is free.
//
// The running-sum rescale folds into P before the f16 cast rather than
// multiplying the accumulator afterwards -- one rounding instead of two, and it
// takes a per-element multiply out of the inner loop.
// Sixteen-byte K and V staging. Defined further down beside the decomposed copy
// of this kernel, whose `S == 0` instance keeps the old staging as the control.
__device__ __forceinline__ void vmma_stage16(
        unsigned short *__restrict__ dst_tile, const unsigned short *__restrict__ src,
        int lo, int n_pos_max, int kv_dim, int off, int head_dim);

__global__ void attn_flash_mma_v(int n_pos_first, int n_rows, int kv_dim,
                                 int head_dim, int n_head, int n_head_kv,
                                 int part_stride, float scale,
                                 const float *__restrict__ q,
                                 const unsigned short *__restrict__ k,
                                 const unsigned short *__restrict__ v,
                                 float *__restrict__ part_acc,
                                 float *__restrict__ part_m,
                                 float *__restrict__ part_l) {
    extern __shared__ char smem_raw[];
    unsigned short *sq  = (unsigned short *)smem_raw;      // ATT_QT x head_dim
    unsigned short *skv = sq + ATT_QT * head_dim;          // ATT_KC x head_dim
    unsigned short *sp  = skv + ATT_KC * head_dim;         // ATT_QT x ATT_KC, f16 P
    float *ss   = (float *)(sp + ATT_QT * ATT_KC);         // ATT_QT x ATT_KC
    float *smax = ss + ATT_QT * ATT_KC;                    // running max
    float *ssum = smax + ATT_QT;                           // running sum
    float *sra  = ssum + ATT_QT;                           // rescale, running

    const int hq = blockIdx.x;
    const int split = blockIdx.y;
    const int base_lo = split * FD_CHUNK;
    const int off = (hq / (n_head / n_head_kv)) * head_dim;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int n_dblk = head_dim >> 4;

    const int n_pos_max = n_pos_first + n_rows - 1;
    if (base_lo >= n_pos_max) return;

    // **Staged by stride, not by division -- here, and for K and V below.**
    //
    // The first version computed `i / head_dim` and `i % head_dim` for every
    // element it staged, and decomposed at n_q 512 that was the kernel: K and V
    // staging ~43% each, the two GEMMs 5%. With all three moved to strided
    // copies a call went 255 -> 58 ms at d32768 and 49 -> 13 at d8192, **4.3x
    // `attn_flash`** where it had been parity. Bit-identical, because the same
    // bytes move; `dbg_attn_mma_v_0` keeps the old staging and
    // `the_restaged_tensor_core_attention_is_bit_identical` holds the two equal.
    //
    // Lane `l` casts head dimensions [8l, 8l + 8) of the rows its warp owns.
    // Each element still needs its own f32 -> f16 cast; only the division and
    // the scatter are gone.
    {
        const int nwarps = blockDim.x >> 5;
        const int d8 = lane * 8;
        if (d8 < head_dim) {
            for (int r = warp; r < ATT_QT; r += nwarps) {
                const int rr = (r < n_rows) ? r : 0;   // padding rows, masked below
                const float *src = q + ((size_t)rr * n_head + hq) * head_dim + d8;
                unsigned short *dst = sq + (size_t)r * head_dim + d8;
#pragma unroll
                for (int j = 0; j < 8; ++j) dst[j] = f2h(src[j]);
            }
        }
    }
    if (threadIdx.x < ATT_QT) {
        smax[threadIdx.x] = -INFINITY;
        ssum[threadIdx.x] = 0.0f;
    }

    // `acc[j][rg]` holds the m16n8 output tile for dim-block `warp + 4j` and
    // query rows `8rg .. 8rg+7`: elements 0,1 are dim `g` at rows `2t, 2t+1`;
    // elements 2,3 are dim `g+8` at the same rows.
    float acc[ATT_DBLK][2][4];
#pragma unroll
    for (int j = 0; j < ATT_DBLK; ++j)
#pragma unroll
        for (int rg = 0; rg < 2; ++rg)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[j][rg][i] = 0.0f;
    __syncthreads();

    for (int sub = 0; sub < FD_CHUNK / ATT_KC; ++sub) {
        const int lo = base_lo + sub * ATT_KC;
        if (lo >= n_pos_max) break;

        vmma_stage16(skv, k, lo, n_pos_max, kv_dim, off, head_dim);
        __syncthreads();

        // --- S = Q K^T, exactly as `attn_flash_mma` computes it.
        {
            float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const int col = warp * 8 + g;
            for (int k0 = 0; k0 < head_dim; k0 += 16) {
                unsigned a[4], b[2];
                a[0] = *(const unsigned *)&sq[(size_t)g * head_dim + k0 + 2 * t];
                a[1] = *(const unsigned *)&sq[(size_t)(g + 8) * head_dim + k0 + 2 * t];
                a[2] = *(const unsigned *)&sq[(size_t)g * head_dim + k0 + 2 * t + 8];
                a[3] = *(const unsigned *)&sq[(size_t)(g + 8) * head_dim + k0 + 2 * t + 8];
                b[0] = *(const unsigned *)&skv[(size_t)col * head_dim + k0 + 2 * t];
                b[1] = *(const unsigned *)&skv[(size_t)col * head_dim + k0 + 2 * t + 8];
                float dd[4];
                mma_m16n8k16_f16(dd, a, b, c);
#pragma unroll
                for (int i = 0; i < 4; ++i) c[i] = dd[i];
            }
            const int c0 = warp * 8 + 2 * t;
            ss[(size_t)g * ATT_KC + c0]           = c[0];
            ss[(size_t)g * ATT_KC + c0 + 1]       = c[1];
            ss[(size_t)(g + 8) * ATT_KC + c0]     = c[2];
            ss[(size_t)(g + 8) * ATT_KC + c0 + 1] = c[3];
        }
        __syncthreads();

        // --- softmax over the sub-tile, and the merge into the running total.
        // Warp `w` owns rows [4w, 4w+4); `ATT_KC` is the warp size, so one warp
        // reduction covers a row and the sub-tile rescale broadcasts by shuffle
        // rather than through shared memory.
        for (int rr = 0; rr < ATT_QT / 4; ++rr) {
            const int r = warp * (ATT_QT / 4) + rr;
            const int n_pos_r = n_pos_first + r;
            const bool live = (r < n_rows) && (lo + lane < n_pos_r);
            float s = live ? ss[(size_t)r * ATT_KC + lane] * scale : -INFINITY;
            float m = s;
#pragma unroll
            for (int sh = 16; sh > 0; sh >>= 1)
                m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, sh));
            const float e = live ? expf(s - m) : 0.0f;
            float l = e;
#pragma unroll
            for (int sh = 16; sh > 0; sh >>= 1)
                l += __shfl_xor_sync(0xffffffff, l, sh);
            float b = 0.0f;
            if (lane == 0) {
                // `m` is -INFINITY when the sub-tile is entirely past this
                // row's window; then `b` is 0 and nothing is contributed.
                const float om = smax[r], ol = ssum[r];
                const float nm = fmaxf(om, m);
                const float a = expf(om - nm);
                b = (m == -INFINITY) ? 0.0f : expf(m - nm);
                smax[r] = nm;
                ssum[r] = ol * a + l * b;
                sra[r] = a;
            }
            const float bb = __shfl_sync(0xffffffff, b, 0);
            sp[(size_t)r * ATT_KC + lane] = f2h(e * bb);
        }
        __syncthreads();

        vmma_stage16(skv, v, lo, n_pos_max, kv_dim, off, head_dim);
        __syncthreads();

        // --- O^T += V^T P^T on the tensor cores. The A fragment depends on the
        // dim-block and the k-step but not on the row group, so one
        // `ldmatrix.trans` feeds both halves of the query tile.
        const float ra0 = sra[2 * t],     ra1 = sra[2 * t + 1];
        const float rb0 = sra[8 + 2 * t], rb1 = sra[8 + 2 * t + 1];
#pragma unroll
        for (int j = 0; j < ATT_DBLK; ++j) {
            acc[j][0][0] *= ra0; acc[j][0][1] *= ra1;
            acc[j][0][2] *= ra0; acc[j][0][3] *= ra1;
            acc[j][1][0] *= rb0; acc[j][1][1] *= rb1;
            acc[j][1][2] *= rb0; acc[j][1][3] *= rb1;
        }
#pragma unroll
        for (int j = 0; j < ATT_DBLK; ++j) {
            const int db = warp + j * 4;
            if (db >= n_dblk) continue;
#pragma unroll
            for (int ks = 0; ks < ATT_KC / 16; ++ks) {
                unsigned a[4];
                ldmatrix_x4_trans(a, skv + (size_t)(ks * 16) * head_dim + db * 16, head_dim);
#pragma unroll
                for (int rg = 0; rg < 2; ++rg) {
                    const unsigned short *pr =
                        sp + (size_t)(rg * 8 + g) * ATT_KC + ks * 16 + 2 * t;
                    unsigned b[2];
                    b[0] = *(const unsigned *)pr;
                    b[1] = *(const unsigned *)(pr + 8);
                    float dd[4];
                    mma_m16n8k16_f16(dd, a, b, acc[j][rg]);
#pragma unroll
                    for (int i = 0; i < 4; ++i) acc[j][rg][i] = dd[i];
                }
            }
        }
        __syncthreads();
    }

    // One partial per (row, head, chunk), in the layout `attn_flash_combine`
    // already reads. Fully unrolled with predicates: a runtime index into `acc`
    // puts it in local memory, which ptxas does not report as a spill.
#pragma unroll
    for (int j = 0; j < ATT_DBLK; ++j) {
        const int db = warp + j * 4;
        if (db >= n_dblk) continue;
#pragma unroll
        for (int rg = 0; rg < 2; ++rg) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int d = db * 16 + ((i < 2) ? g : g + 8);
                const int r = rg * 8 + 2 * t + (i & 1);
                if (r >= n_rows) continue;
                part_acc[(((size_t)r * n_head + hq) * part_stride + split) * head_dim + d] =
                    acc[j][rg][i];
            }
        }
    }
    if (threadIdx.x < n_rows) {
        const size_t base = ((size_t)threadIdx.x * n_head + hq) * part_stride + split;
        part_m[base] = smax[threadIdx.x];
        part_l[base] = ssum[threadIdx.x];
    }
}




// ---------------------------------------------------------------------------
// `attn_flash_mma_v`, decomposed
// ---------------------------------------------------------------------------
//
// **Why the tensor-core kernel does not pay.** It stages K once per 16 query
// vectors where `attn_flash` walks K once per vector, and it issues a fraction
// of the instructions -- yet at d32768 both cost ~117 ns per (row, head, chunk)
// of work. Something other than load count or instruction count binds it, and
// llama.cpp's MMA kernel names four candidates it had to fix: bank conflicts on
// consecutive 4-byte shared loads, synchronous staging, scalar fragment loads in
// place of `ldmatrix`, and narrow tiles. Reading cannot say which; this prices
// the pieces. Same method, and the same reason for template instances, as the
// `attn_flash` decomposition above.
//
// Staged constants are f16 0.25 (`0x3400`) so every arm computes over ordinary
// finite values.
#define DBG_VMMA_NO_QSTAGE  1   // query staged as a constant: no global load, no f16 cast
#define DBG_VMMA_NO_KSTAGE  2   // K staged as a constant: no global loads, stores kept
#define DBG_VMMA_NO_SCORE   4   // no score GEMM: no fragment loads, no MMAs
#define DBG_VMMA_NO_SOFTMAX 8   // no max, expf, sum, merge or broadcast
#define DBG_VMMA_NO_VSTAGE  16  // V staged as a constant: no global loads, stores kept
#define DBG_VMMA_NO_VGEMM   32  // no V GEMM: no rescale, no ldmatrix, no fragment loads, no MMAs
// Not removals: the same bytes moved a faster way, held to bit-equality with `mma_v`.
#define DBG_VMMA_FAST_KSTAGE 64   // K staged sixteen bytes per instruction, no division
#define DBG_VMMA_FAST_VSTAGE 128  // V staged the same way
#define DBG_VMMA_FAST_QSTAGE 256  // query staged by stride rather than by per-element division
// Not a removal either: a tile's slots are the query heads that share one kv head, in
// one row -- decode's layout. Requires DBG_VMMA_FAST_QSTAGE.
#define DBG_VMMA_GQA_COLS    512

// Stage `ATT_KC` positions of one kv head into shared memory, sixteen bytes per
// instruction -- what llama.cpp's `flash_attn_ext_f16_load_tile` does.
//
// **The kernel this replaces spent ~43% of its time on each of K and V here**,
// because for every f16 it divided and took a remainder by a runtime `head_dim`,
// compared, selected, and moved two bytes. Lane `l` now moves head dimensions
// [8l, 8l + 8) of the positions its warp owns, so a warp's lanes read one
// contiguous row per iteration and nothing is computed per element.
//
// Positions at or past `n_pos_max` are zeroed rather than loaded, exactly as the
// original stages them: near the end of the cache they lie outside the slab.
// The branch is warp-uniform, since `p` is the warp's and `lo` the block's.
__device__ __forceinline__ void vmma_stage16(
        unsigned short *__restrict__ dst_tile, const unsigned short *__restrict__ src,
        int lo, int n_pos_max, int kv_dim, int off, int head_dim) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int d8 = lane * 8;
    if (d8 >= head_dim) return;
    for (int p = warp; p < ATT_KC; p += nwarps) {
        int *dst = (int *)(dst_tile + (size_t)p * head_dim + d8);
        if (lo + p < n_pos_max) {
            *(int4 *)dst = *(const int4 *)(src + (size_t)(lo + p) * kv_dim + off + d8);
        } else {
            dst[0] = 0;
            dst[1] = 0;
            dst[2] = 0;
            dst[3] = 0;
        }
    }
}

}  // extern "C" -- a template cannot have C linkage; reopened for the instances.

template <int S>
__device__ __forceinline__ void attn_mma_v_body(
        char *smem_raw,
        int n_pos_first, int n_rows, int kv_dim, int head_dim, int n_head, int n_head_kv,
        int part_stride, float scale,
        const float *__restrict__ q,
        const unsigned short *__restrict__ k,
        const unsigned short *__restrict__ v,
        float *__restrict__ part_acc,
        float *__restrict__ part_m,
        float *__restrict__ part_l) {
    unsigned short *sq  = (unsigned short *)smem_raw;      // ATT_QT x head_dim
    unsigned short *skv = sq + ATT_QT * head_dim;          // ATT_KC x head_dim
    unsigned short *sp  = skv + ATT_KC * head_dim;         // ATT_QT x ATT_KC, f16 P
    float *ss   = (float *)(sp + ATT_QT * ATT_KC);         // ATT_QT x ATT_KC
    float *smax = ss + ATT_QT * ATT_KC;                    // running max
    float *ssum = smax + ATT_QT;                           // running sum
    float *sra  = ssum + ATT_QT;                           // rescale, running

    const int hq = blockIdx.x;
    const int split = blockIdx.y;
    const int base_lo = split * FD_CHUNK;
    // Blocks and slots. Normally a block is a query head and its tile's slots
    // are rows of it. With `DBG_VMMA_GQA_COLS` a block is a kv head and its
    // slots are the query heads that share it, all in one row -- decode, where
    // there is only one row to fill a tile with.
    const int gqa = n_head / n_head_kv;
    const int off = (S & DBG_VMMA_GQA_COLS) ? blockIdx.x * head_dim : (hq / gqa) * head_dim;
    const int n_live = (S & DBG_VMMA_GQA_COLS) ? gqa : n_rows;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int n_dblk = head_dim >> 4;

    const int n_pos_max = n_pos_first + n_rows - 1;
    if (base_lo >= n_pos_max) return;

    if (S & DBG_VMMA_FAST_QSTAGE) {
        // The same elements as the loop below, addressed by stride: lane `l`
        // casts head dimensions [8l, 8l + 8) of the rows its warp owns. Each
        // element still needs its own f32 -> f16 cast, so only the division and
        // the scatter go.
        const int nwarps = blockDim.x >> 5;
        const int d8 = lane * 8;
        if (d8 < head_dim) {
            for (int r = warp; r < ATT_QT; r += nwarps) {
                const int rr = (r < n_live) ? r : 0;   // padding slots, masked below
                const size_t vec = (S & DBG_VMMA_GQA_COLS) ? (size_t)(blockIdx.x * gqa + rr)
                                                           : (size_t)rr * n_head + hq;
                const float *src = q + vec * head_dim + d8;
                unsigned short *dst = sq + (size_t)r * head_dim + d8;
#pragma unroll
                for (int j = 0; j < 8; ++j) dst[j] = f2h(src[j]);
            }
        }
    } else {
        for (int i = threadIdx.x; i < ATT_QT * head_dim; i += blockDim.x) {
            if (S & DBG_VMMA_NO_QSTAGE) {
                sq[i] = (unsigned short)0x3400;
            } else {
                const int r = i / head_dim, d = i % head_dim;
                const int rr = (r < n_rows) ? r : 0;   // padding rows, masked below
                sq[i] = f2h(q[((size_t)rr * n_head + hq) * head_dim + d]);
            }
        }
    }
    if (threadIdx.x < ATT_QT) {
        smax[threadIdx.x] = -INFINITY;
        ssum[threadIdx.x] = 0.0f;
    }

    float acc[ATT_DBLK][2][4];
#pragma unroll
    for (int j = 0; j < ATT_DBLK; ++j)
#pragma unroll
        for (int rg = 0; rg < 2; ++rg)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[j][rg][i] = 0.0f;
    __syncthreads();

    for (int sub = 0; sub < FD_CHUNK / ATT_KC; ++sub) {
        const int lo = base_lo + sub * ATT_KC;
        if (lo >= n_pos_max) break;

        if (S & DBG_VMMA_FAST_KSTAGE) {
            vmma_stage16(skv, k, lo, n_pos_max, kv_dim, off, head_dim);
        } else {
            for (int i = threadIdx.x; i < ATT_KC * head_dim; i += blockDim.x) {
                if (S & DBG_VMMA_NO_KSTAGE) {
                    skv[i] = (unsigned short)0x3400;
                } else {
                    const int p = i / head_dim, d = i % head_dim;
                    skv[i] = (lo + p < n_pos_max) ? k[(size_t)(lo + p) * kv_dim + off + d] : 0;
                }
            }
        }
        __syncthreads();

        {
            float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (S & DBG_VMMA_NO_SCORE) {
                c[0] = 0.5f;
                c[1] = 0.25f;
                c[2] = 0.125f;
                c[3] = 0.375f;
            } else {
                const int col = warp * 8 + g;
                for (int k0 = 0; k0 < head_dim; k0 += 16) {
                    unsigned a[4], b[2];
                    a[0] = *(const unsigned *)&sq[(size_t)g * head_dim + k0 + 2 * t];
                    a[1] = *(const unsigned *)&sq[(size_t)(g + 8) * head_dim + k0 + 2 * t];
                    a[2] = *(const unsigned *)&sq[(size_t)g * head_dim + k0 + 2 * t + 8];
                    a[3] = *(const unsigned *)&sq[(size_t)(g + 8) * head_dim + k0 + 2 * t + 8];
                    b[0] = *(const unsigned *)&skv[(size_t)col * head_dim + k0 + 2 * t];
                    b[1] = *(const unsigned *)&skv[(size_t)col * head_dim + k0 + 2 * t + 8];
                    float dd[4];
                    mma_m16n8k16_f16(dd, a, b, c);
#pragma unroll
                    for (int i = 0; i < 4; ++i) c[i] = dd[i];
                }
            }
            const int c0 = warp * 8 + 2 * t;
            ss[(size_t)g * ATT_KC + c0]           = c[0];
            ss[(size_t)g * ATT_KC + c0 + 1]       = c[1];
            ss[(size_t)(g + 8) * ATT_KC + c0]     = c[2];
            ss[(size_t)(g + 8) * ATT_KC + c0 + 1] = c[3];
        }
        __syncthreads();

        for (int rr = 0; rr < ATT_QT / 4; ++rr) {
            const int r = warp * (ATT_QT / 4) + rr;
            if (S & DBG_VMMA_NO_SOFTMAX) {
                if (lane == 0) {
                    smax[r] = 0.0f;
                    ssum[r] = 1.0f;
                    sra[r] = 1.0f;
                }
                sp[(size_t)r * ATT_KC + lane] = (unsigned short)0x3400;
            } else {
                const int n_pos_r = (S & DBG_VMMA_GQA_COLS) ? n_pos_first : n_pos_first + r;
                const bool live = (r < n_live) && (lo + lane < n_pos_r);
                float s = live ? ss[(size_t)r * ATT_KC + lane] * scale : -INFINITY;
                float m = s;
#pragma unroll
                for (int sh = 16; sh > 0; sh >>= 1)
                    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, sh));
                const float e = live ? expf(s - m) : 0.0f;
                float l = e;
#pragma unroll
                for (int sh = 16; sh > 0; sh >>= 1)
                    l += __shfl_xor_sync(0xffffffff, l, sh);
                float b = 0.0f;
                if (lane == 0) {
                    const float om = smax[r], ol = ssum[r];
                    const float nm = fmaxf(om, m);
                    const float a = expf(om - nm);
                    b = (m == -INFINITY) ? 0.0f : expf(m - nm);
                    smax[r] = nm;
                    ssum[r] = ol * a + l * b;
                    sra[r] = a;
                }
                const float bb = __shfl_sync(0xffffffff, b, 0);
                sp[(size_t)r * ATT_KC + lane] = f2h(e * bb);
            }
        }
        __syncthreads();

        if (S & DBG_VMMA_FAST_VSTAGE) {
            vmma_stage16(skv, v, lo, n_pos_max, kv_dim, off, head_dim);
        } else {
            for (int i = threadIdx.x; i < ATT_KC * head_dim; i += blockDim.x) {
                if (S & DBG_VMMA_NO_VSTAGE) {
                    skv[i] = (unsigned short)0x3400;
                } else {
                    const int p = i / head_dim, d = i % head_dim;
                    skv[i] = (lo + p < n_pos_max) ? v[(size_t)(lo + p) * kv_dim + off + d] : 0;
                }
            }
        }
        __syncthreads();

        if (!(S & DBG_VMMA_NO_VGEMM)) {
            const float ra0 = sra[2 * t],     ra1 = sra[2 * t + 1];
            const float rb0 = sra[8 + 2 * t], rb1 = sra[8 + 2 * t + 1];
#pragma unroll
            for (int j = 0; j < ATT_DBLK; ++j) {
                acc[j][0][0] *= ra0; acc[j][0][1] *= ra1;
                acc[j][0][2] *= ra0; acc[j][0][3] *= ra1;
                acc[j][1][0] *= rb0; acc[j][1][1] *= rb1;
                acc[j][1][2] *= rb0; acc[j][1][3] *= rb1;
            }
#pragma unroll
            for (int j = 0; j < ATT_DBLK; ++j) {
                const int db = warp + j * 4;
                if (db >= n_dblk) continue;
#pragma unroll
                for (int ks = 0; ks < ATT_KC / 16; ++ks) {
                    unsigned a[4];
                    ldmatrix_x4_trans(a, skv + (size_t)(ks * 16) * head_dim + db * 16, head_dim);
#pragma unroll
                    for (int rg = 0; rg < 2; ++rg) {
                        const unsigned short *pr =
                            sp + (size_t)(rg * 8 + g) * ATT_KC + ks * 16 + 2 * t;
                        unsigned b[2];
                        b[0] = *(const unsigned *)pr;
                        b[1] = *(const unsigned *)(pr + 8);
                        float dd[4];
                        mma_m16n8k16_f16(dd, a, b, acc[j][rg]);
#pragma unroll
                        for (int i = 0; i < 4; ++i) acc[j][rg][i] = dd[i];
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < ATT_DBLK; ++j) {
        const int db = warp + j * 4;
        if (db >= n_dblk) continue;
#pragma unroll
        for (int rg = 0; rg < 2; ++rg) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int d = db * 16 + ((i < 2) ? g : g + 8);
                const int r = rg * 8 + 2 * t + (i & 1);
                if (r >= n_live) continue;
                const size_t vec = (S & DBG_VMMA_GQA_COLS) ? (size_t)(blockIdx.x * gqa + r)
                                                           : (size_t)r * n_head + hq;
                part_acc[(vec * part_stride + split) * head_dim + d] =
                    acc[j][rg][i];
            }
        }
    }
    if (threadIdx.x < n_live) {
        const size_t vec = (S & DBG_VMMA_GQA_COLS) ? (size_t)(blockIdx.x * gqa + threadIdx.x)
                                                   : (size_t)threadIdx.x * n_head + hq;
        const size_t base = vec * part_stride + split;
        part_m[base] = smax[threadIdx.x];
        part_l[base] = ssum[threadIdx.x];
    }
}

extern "C" {

#define DBG_ATTN_MMA_V_INSTANCE(S)                                                   \
extern "C" __global__ void dbg_attn_mma_v_##S(                                      \
        int n_pos_first, int n_rows, int kv_dim, int head_dim, int n_head,        \
        int n_head_kv, int part_stride, float scale,                              \
        const float *__restrict__ q, const unsigned short *__restrict__ k,        \
        const unsigned short *__restrict__ v, float *__restrict__ part_acc,       \
        float *__restrict__ part_m, float *__restrict__ part_l) {                 \
    extern __shared__ char smem_raw[];                                            \
    attn_mma_v_body<S>(smem_raw, n_pos_first, n_rows, kv_dim, head_dim,       \
        n_head, n_head_kv, part_stride, scale, q, k, v, part_acc, part_m, part_l); \
}

DBG_ATTN_MMA_V_INSTANCE(0)
DBG_ATTN_MMA_V_INSTANCE(1)
DBG_ATTN_MMA_V_INSTANCE(2)
DBG_ATTN_MMA_V_INSTANCE(4)
DBG_ATTN_MMA_V_INSTANCE(8)
DBG_ATTN_MMA_V_INSTANCE(16)
DBG_ATTN_MMA_V_INSTANCE(32)
DBG_ATTN_MMA_V_INSTANCE(18)
DBG_ATTN_MMA_V_INSTANCE(36)
DBG_ATTN_MMA_V_INSTANCE(63)
DBG_ATTN_MMA_V_INSTANCE(192)
DBG_ATTN_MMA_V_INSTANCE(448)

// ---------------------------------------------------------------------------
// Decode attention: one kernel, two modes
// ---------------------------------------------------------------------------
//
// **Decode is a CUDA graph, and a graph replays a fixed kernel sequence**, so
// decode cannot switch kernels partway through a generation -- the warp score
// phase became an argument for the same reason. So the tensor-core mode is an
// argument too: `attn_flash`'s arithmetic below the depth where it pays, and
// `attn_flash_mma_v`'s above it, with the tile's slots holding the query heads
// that share a kv head. That is grouped-query attention as tile columns, which
// is what llama.cpp's MMA kernel does for decode at `gqa_ratio > 4 && KV >= 8192`.
//
// The modes want different grids -- (query head, chunk) against (kv head,
// chunk) -- and different shared memory, and graph replay updates both in
// place, so the sequence stays this launch and `attn_flash_combine` after it.
// Both modes write the same partials, so the combine does not change.
#define ATT_DECODE_MMA_S (DBG_VMMA_FAST_KSTAGE | DBG_VMMA_FAST_VSTAGE | \
                          DBG_VMMA_FAST_QSTAGE | DBG_VMMA_GQA_COLS)

// ---------------------------------------------------------------------------
// The 8-slot decode tile
// ---------------------------------------------------------------------------
//
// **The 16-slot decode tile was half empty.** `mma.m16n8k16` computes a 16x8
// score tile, and `attn_flash_mma_v` puts query vectors on its 16 side and
// positions on its 8 -- right for prefill, where a block has sixteen rows, and
// wasteful for decode, where a kv head has eight query heads and nothing else.
// Decode's tensor-core mode measured 1.6-1.9x the warp phase against ~4x for
// prefill per unit of work.
//
// So the roles swap, as llama.cpp's MMA kernel does at `ncols == 8`: positions
// on the 16 side, the kv head's query heads on the 8. A warp scores sixteen
// positions against every query head at once, a sub-tile is 64 positions, and
// the V GEMM needs one MMA per (dim block, position step) instead of two. The
// lane conventions are `attn_flash_mma_v`'s, which its tests validate, with the
// meaning of the axes exchanged:
//
//   score  A rows g, g+8 -> positions p0+g, p0+g+8   B column g -> query head g
//          D -> S[query 2t, 2t+1][position p0+g, p0+g+8]
//   value  A = V^T at positions ks*16 (unchanged)    B column g -> query head g
//          D -> O^T[dim db*16 + g, g+8][query 2t, 2t+1]
//
// The softmax runs per query head over 64 positions as two 32-lane halves, each
// reduced by shuffles, merged in registers, then merged into the running total.
// Rounding order differs from the 16-slot mode, so this is held to the derived
// bound rather than to equal bits.
#define D8_QT 8     // slots: the query heads of one kv head
#define D8_KC 64    // positions per sub-tile, sixteen per warp

// Stage `n_tile` positions of one kv head sixteen bytes per instruction; the
// same as `vmma_stage16` for a tile of any height.
__device__ __forceinline__ void stage16_rows(
        unsigned short *__restrict__ dst_tile, const unsigned short *__restrict__ src,
        int lo, int n_pos, int kv_dim, int off, int head_dim, int n_tile) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int d8 = lane * 8;
    if (d8 >= head_dim) return;
    for (int p = warp; p < n_tile; p += nwarps) {
        int *dst = (int *)(dst_tile + (size_t)p * head_dim + d8);
        if (lo + p < n_pos) {
            *(int4 *)dst = *(const int4 *)(src + (size_t)(lo + p) * kv_dim + off + d8);
        } else {
            dst[0] = 0;
            dst[1] = 0;
            dst[2] = 0;
            dst[3] = 0;
        }
    }
}

__device__ __forceinline__ void attn_decode_mma8_body(
        char *smem_raw, int n_pos, int kv_dim, int head_dim, int n_head, int n_head_kv,
        int part_stride, float scale,
        const float *__restrict__ q,
        const unsigned short *__restrict__ k,
        const unsigned short *__restrict__ v,
        float *__restrict__ part_acc,
        float *__restrict__ part_m,
        float *__restrict__ part_l) {
    unsigned short *sq  = (unsigned short *)smem_raw;      // D8_QT x head_dim
    unsigned short *skv = sq + D8_QT * head_dim;           // D8_KC x head_dim
    unsigned short *sp  = skv + D8_KC * head_dim;          // D8_QT x D8_KC, f16 P
    float *ss   = (float *)(sp + D8_QT * D8_KC);           // D8_QT x D8_KC scores
    float *smax = ss + D8_QT * D8_KC;                      // running max
    float *ssum = smax + D8_QT;                            // running sum
    float *sra  = ssum + D8_QT;                            // rescale, running

    const int hk = blockIdx.x;
    const int split = blockIdx.y;
    const int base_lo = split * FD_CHUNK;
    const int gqa = n_head / n_head_kv;
    const int off = hk * head_dim;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int n_dblk = head_dim >> 4;
    if (base_lo >= n_pos) return;

    // Query heads hk*gqa .. hk*gqa + gqa - 1 into slots; lane l casts head
    // dimensions [8l, 8l + 8) of the two slots its warp owns.
    {
        const int d8 = lane * 8;
        if (d8 < head_dim) {
            for (int r = warp; r < D8_QT; r += nwarps) {
                const int c = (r < gqa) ? r : 0;   // padding slots, masked below
                const float *src = q + (size_t)(hk * gqa + c) * head_dim + d8;
                unsigned short *dst = sq + (size_t)r * head_dim + d8;
#pragma unroll
                for (int j = 0; j < 8; ++j) dst[j] = f2h(src[j]);
            }
        }
    }
    if (threadIdx.x < D8_QT) {
        smax[threadIdx.x] = -INFINITY;
        ssum[threadIdx.x] = 0.0f;
    }

    // O^T accumulators: dim-block warp + 4j, dims g and g + 8, query heads 2t
    // and 2t + 1.
    float acc[ATT_DBLK][4];
#pragma unroll
    for (int j = 0; j < ATT_DBLK; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;
    __syncthreads();

    for (int sub = 0; sub < FD_CHUNK / D8_KC; ++sub) {
        const int lo = base_lo + sub * D8_KC;
        if (lo >= n_pos) break;

        stage16_rows(skv, k, lo, n_pos, kv_dim, off, head_dim, D8_KC);
        __syncthreads();

        // --- S = K Q: warp `w` scores positions [16w, 16w + 16) against every
        // query head.
        {
            float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const int p0 = warp * 16;
            for (int k0 = 0; k0 < head_dim; k0 += 16) {
                unsigned a[4], b[2];
                a[0] = *(const unsigned *)&skv[(size_t)(p0 + g) * head_dim + k0 + 2 * t];
                a[1] = *(const unsigned *)&skv[(size_t)(p0 + g + 8) * head_dim + k0 + 2 * t];
                a[2] = *(const unsigned *)&skv[(size_t)(p0 + g) * head_dim + k0 + 2 * t + 8];
                a[3] = *(const unsigned *)&skv[(size_t)(p0 + g + 8) * head_dim + k0 + 2 * t + 8];
                b[0] = *(const unsigned *)&sq[(size_t)g * head_dim + k0 + 2 * t];
                b[1] = *(const unsigned *)&sq[(size_t)g * head_dim + k0 + 2 * t + 8];
                float dd[4];
                mma_m16n8k16_f16(dd, a, b, c);
#pragma unroll
                for (int i = 0; i < 4; ++i) c[i] = dd[i];
            }
            ss[(size_t)(2 * t) * D8_KC + p0 + g]         = c[0];
            ss[(size_t)(2 * t + 1) * D8_KC + p0 + g]     = c[1];
            ss[(size_t)(2 * t) * D8_KC + p0 + g + 8]     = c[2];
            ss[(size_t)(2 * t + 1) * D8_KC + p0 + g + 8] = c[3];
        }
        __syncthreads();

        // --- softmax per query head over the sub-tile's 64 positions. Warp `w`
        // owns query heads w and w + 4; lanes are positions, two halves of 32.
        for (int r = warp; r < D8_QT; r += nwarps) {
            const bool slot = r < gqa;
            float s0 = (slot && lo + lane < n_pos) ? ss[(size_t)r * D8_KC + lane] * scale : -INFINITY;
            float s1 = (slot && lo + 32 + lane < n_pos) ? ss[(size_t)r * D8_KC + 32 + lane] * scale : -INFINITY;
            float m0 = s0, m1 = s1;
#pragma unroll
            for (int sh = 16; sh > 0; sh >>= 1) {
                m0 = fmaxf(m0, __shfl_xor_sync(0xffffffff, m0, sh));
                m1 = fmaxf(m1, __shfl_xor_sync(0xffffffff, m1, sh));
            }
            const float e0 = (s0 != -INFINITY) ? expf(s0 - m0) : 0.0f;
            const float e1 = (s1 != -INFINITY) ? expf(s1 - m1) : 0.0f;
            float l0 = e0, l1 = e1;
#pragma unroll
            for (int sh = 16; sh > 0; sh >>= 1) {
                l0 += __shfl_xor_sync(0xffffffff, l0, sh);
                l1 += __shfl_xor_sync(0xffffffff, l1, sh);
            }
            float b0 = 0.0f, b1 = 0.0f;
            if (lane == 0) {
                // Halves into the sub-tile, the sub-tile into the running total.
                // A half with no live position has m == -INFINITY and must
                // contribute nothing rather than exp(-inf - -inf).
                const float m = fmaxf(m0, m1);
                const float w0 = (m0 == -INFINITY) ? 0.0f : expf(m0 - m);
                const float w1 = (m1 == -INFINITY) ? 0.0f : expf(m1 - m);
                const float l = l0 * w0 + l1 * w1;
                const float om = smax[r], ol = ssum[r];
                const float nm = fmaxf(om, m);
                const float a = expf(om - nm);
                b0 = (m0 == -INFINITY) ? 0.0f : expf(m0 - nm);
                b1 = (m1 == -INFINITY) ? 0.0f : expf(m1 - nm);
                const float bm = (m == -INFINITY) ? 0.0f : expf(m - nm);
                smax[r] = nm;
                ssum[r] = ol * a + l * bm;
                sra[r] = a;
            }
            const float bb0 = __shfl_sync(0xffffffff, b0, 0);
            const float bb1 = __shfl_sync(0xffffffff, b1, 0);
            sp[(size_t)r * D8_KC + lane]      = f2h(e0 * bb0);
            sp[(size_t)r * D8_KC + 32 + lane] = f2h(e1 * bb1);
        }
        __syncthreads();

        stage16_rows(skv, v, lo, n_pos, kv_dim, off, head_dim, D8_KC);
        __syncthreads();

        // --- O^T += V^T P^T: one MMA per (dim block, 16-position step), the
        // query heads all in its eight columns.
        {
            const float ra0 = sra[2 * t], ra1 = sra[2 * t + 1];
#pragma unroll
            for (int j = 0; j < ATT_DBLK; ++j) {
                acc[j][0] *= ra0; acc[j][1] *= ra1;
                acc[j][2] *= ra0; acc[j][3] *= ra1;
            }
#pragma unroll
            for (int j = 0; j < ATT_DBLK; ++j) {
                const int db = warp + j * 4;
                if (db >= n_dblk) continue;
#pragma unroll
                for (int ks = 0; ks < D8_KC / 16; ++ks) {
                    unsigned a[4];
                    ldmatrix_x4_trans(a, skv + (size_t)(ks * 16) * head_dim + db * 16, head_dim);
                    const unsigned short *pr = sp + (size_t)g * D8_KC + ks * 16 + 2 * t;
                    unsigned b[2];
                    b[0] = *(const unsigned *)pr;
                    b[1] = *(const unsigned *)(pr + 8);
                    float dd[4];
                    mma_m16n8k16_f16(dd, a, b, acc[j]);
#pragma unroll
                    for (int i = 0; i < 4; ++i) acc[j][i] = dd[i];
                }
            }
        }
        __syncthreads();
    }

    // Partials in the layout `attn_flash_combine` reads, row 0.
#pragma unroll
    for (int j = 0; j < ATT_DBLK; ++j) {
        const int db = warp + j * 4;
        if (db >= n_dblk) continue;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = db * 16 + ((i < 2) ? g : g + 8);
            const int r = 2 * t + (i & 1);
            if (r >= gqa) continue;
            part_acc[((size_t)(hk * gqa + r) * part_stride + split) * head_dim + d] = acc[j][i];
        }
    }
    if (threadIdx.x < gqa) {
        const size_t base = (size_t)(hk * gqa + threadIdx.x) * part_stride + split;
        part_m[base] = smax[threadIdx.x];
        part_l[base] = ssum[threadIdx.x];
    }
}

// `mode`: 0 the scalar path, 1 the 16-slot tile, 2 the 8-slot tile.
__global__ void attn_decode(int mode, int n_pos_first, int kv_dim, int head_dim,
                            int n_head, int n_head_kv, int use_warp, int part_stride,
                            float scale,
                            const float *__restrict__ q,
                            const unsigned short *__restrict__ k,
                            const unsigned short *__restrict__ v,
                            float *__restrict__ part_acc,
                            float *__restrict__ part_m,
                            float *__restrict__ part_l) {
    extern __shared__ char smem_raw[];
    if (mode == 2) {
        attn_decode_mma8_body(smem_raw, n_pos_first, kv_dim, head_dim, n_head, n_head_kv,
            part_stride, scale, q, k, v, part_acc, part_m, part_l);
    } else if (mode == 1) {
        attn_mma_v_body<ATT_DECODE_MMA_S>(smem_raw, n_pos_first, 1, kv_dim, head_dim,
            n_head, n_head_kv, part_stride, scale, q, k, v, part_acc, part_m, part_l);
    } else {
        float *sf = (float *)smem_raw;
        attn_flash_body<0>(sf, sf + head_dim, sf + head_dim + FD_CHUNK,
            n_pos_first, kv_dim, head_dim, n_head, n_head_kv, use_warp, part_stride,
            scale, q, k, v, part_acc, part_m, part_l);
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

// `kvalues_iq4nl` from ggml/src/ggml-common.h, held in **registers** rather
// than `__constant__` memory. A non-uniform grid -- the "IQ" in IQ4_XS -- so a
// linear dequantization would be a different format.
//
// **This lookup was 72% of the IQ4_XS matmul.** Constant memory is
// broadcast-optimised: a read where lanes index different addresses serialises
// into up to 32 transactions, and every lane here looks up a different nibble,
// twice per byte, 64 times per output row. Decomposed against the real kernel
// at its real geometry: 143.9 us baseline, 40.6 us with only this lookup
// replaced by arithmetic, 36.9 us with all unpacking removed. The nibble split
// and the 6-bit scale assembly cost ~4 us between them; the table cost ~103.
//
// Sixteen signed bytes are 128 bits, so the whole grid fits in two registers
// and the lookup becomes a select and a shift. The values are identical, so
// this is bit-exact by inspection as well as by test.
//
// Two earlier attempts at this kernel -- an alignment repack and a restructured
// fold -- were aimed at the two things the decomposition later showed were free
// (loads cost 2 us of 144, the ordered fold 2 us).
__device__ __forceinline__ int kvalue_iq4nl(int i) {
    const unsigned long long w =
        (i < 8) ? 0xF6EADDCFBFAD9881ULL : 0x7159453526190D01ULL;
    return (int)(signed char)((w >> ((i & 7) * 8)) & 0xffULL);
}

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

// Tokens per weight load in the batched Q6_K matmul.
//
// **The unpack is the cost, and it does not depend on the token.** Six-bit
// weights arrive as a low nibble in `ql` and a high pair in `qh`, selected by a
// four-way branch on the sub-block -- a dozen instructions per weight -- and
// `matmul_q6_k_q8_k` ran the whole sequence again for every token of the batch,
// because its `blockIdx.y` is the token. At 512 tokens that is 512 unpacks of
// each weight to serve 512 outputs.
//
// Holding the unpacked value across `Q6K_TOK` tokens divides both the unpack
// count and the weight traffic by the same factor. 8 keeps `aux` and `sum_l` at
// eight registers each; the kernel is otherwise identical to the one-token one.
//
// Worth **+13.2% of whole-model prefill** on the 35B, 360.3 -> 407.7 tok/s at
// 2,240 tokens, from one bit-exact kernel.
//
// **The same change on Q5_K is written and unresolved, not rejected.** At 2,240
// tokens it measured 399 against this kernel's 402 -- inside a within-batch
// spread of 2% -- but at 11,237 tokens it measured **380.9 against 366.2**, and
// that is the depth a real session runs at. One run each, so neither settles
// it; it was reverted on the shallow number before the deep one existed, which
// is precisely the error the rest of this session was spent learning.
//
// There is a reason to expect it to gain less either way: Q5_K's unpack is a
// nibble select plus one `qh` bit, about six instructions where Q6_K's four-way
// branch over a split nibble and bit-pair is about twelve, and its twelve-byte
// scale shuffle was already amortised per super-block. Half as much to remove,
// against the same eight extra accumulator sets. That is the shape of the rule
// this file keeps confirming: the gain is proportional to the *instructions*
// deleted, not the traffic -- 09-09 cut A traffic 4x and B traffic 2x for
// nothing.
#define Q6K_TOK 8

// Q6_K x Q8_K over a batch, one warp per output row and `Q6K_TOK` tokens.
//
// **Bit-identical to `matmul_q6_k_q8_k` by construction, not by measurement.**
// Each output still keeps the oracle's eight interleaved f32 accumulators, in
// the same lane, folded in the same ascending order; the only change is how
// many outputs one weight load serves. Nothing is reassociated, so this needs
// no tolerance and stays inside the exact set -- unlike an MMA form of the same
// kernel, which cannot keep those eight accumulators at all.
//
// Padding tokens are pointed at token 0 rather than branched around, so the
// inner loop is unconditional. They accumulate into `aux`/`sum_l` slots that the
// write-back never stores.
__global__ void matmul_q6_k_q8_k_tok(int n_in, int n_out, int n_tok,
                                     const unsigned char *__restrict__ w,
                                     const float *__restrict__ x_scales,
                                     const signed char *__restrict__ x_quants,
                                     float *__restrict__ out) {
    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;
    const int t0 = blockIdx.y * Q6K_TOK;

    const int l = lane & 7;
    const int g = lane >> 3;

    const unsigned char *row = w + (size_t)j * nb * Q6K_BYTES;

    // Safe token index per slot: out-of-range slots read token 0 and are
    // discarded at the write-back, which keeps the inner loop branch-free.
    int ts[Q6K_TOK];
#pragma unroll
    for (int t = 0; t < Q6K_TOK; ++t) ts[t] = (t0 + t < n_tok) ? (t0 + t) : 0;

    float sum_l[Q6K_TOK];
#pragma unroll
    for (int t = 0; t < Q6K_TOK; ++t) sum_l[t] = 0.0f;

    for (int i = 0; i < nb; ++i) {
        const unsigned char *blk = row + (size_t)i * Q6K_BYTES;
        const unsigned char *ql = blk;
        const unsigned char *qh = blk + QK_K / 2;
        const signed char   *sc = (const signed char *)(blk + QK_K / 2 + QK_K / 4);
        const unsigned short d16 =
            (unsigned short)blk[Q6K_BYTES - 2] | ((unsigned short)blk[Q6K_BYTES - 1] << 8);

        int aux[Q6K_TOK];
#pragma unroll
        for (int t = 0; t < Q6K_TOK; ++t) aux[t] = 0;

        for (int jj = g * 4; jj < g * 4 + 4; ++jj) {
            const int scale = (int)sc[jj];
            for (int half = 0; half < 2; ++half) {
                const int idx = jj * 16 + half * 8 + l;

                // Unpacked once, then dotted against every token below. This is
                // the same mapping `matmul_q6_k_q8_k` uses, lifted out of the
                // token loop rather than changed.
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

#pragma unroll
                for (int t = 0; t < Q6K_TOK; ++t) {
                    const signed char *q8 =
                        x_quants + (size_t)ts[t] * n_in + (size_t)i * QK_K;
                    // int16 in the reference, and faithful: an int8 quant times
                    // a -32..31 weight cannot leave that range.
                    const short aux16 = (short)((int)q8[idx] * av);
                    aux[t] += scale * (int)aux16;
                }
            }
        }

        // Fold the four g-groups. Integer, so exact and order-free.
#pragma unroll
        for (int t = 0; t < Q6K_TOK; ++t) {
            aux[t] += __shfl_down_sync(0xffffffff, aux[t], 16);
            aux[t] += __shfl_down_sync(0xffffffff, aux[t], 8);
        }

        if (lane < 8) {
            const float dv = h2f(d16);
#pragma unroll
            for (int t = 0; t < Q6K_TOK; ++t) {
                // NOT fused, as in the one-token kernel: the reference's
                // compiler contracts Q5_K's identical line and not this one.
                const float d = dv * x_scales[(size_t)ts[t] * nb + i];
                sum_l[t] += d * (float)aux[t];
            }
        }
    }

    // The 8-way fold, serial and ascending -- the oracle's last loop. The
    // shuffle is outside the branch because every lane must reach it.
#pragma unroll
    for (int t = 0; t < Q6K_TOK; ++t) {
        float sumf = 0.0f;
        for (int k = 0; k < 8; ++k) {
            const float v = __shfl_sync(0xffffffff, sum_l[t], k);
            if (lane == 0) sumf += v;
        }
        if (lane == 0 && t0 + t < n_tok) {
            out[(size_t)(t0 + t) * n_out + j] = sumf;
        }
    }
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

// Tokens per weight load in the batched Q5_K matmul. Same reasoning and the
// same figure as `Q6K_TOK`: the five-bit unpack and the twelve-byte scale/min
// shuffle both depend only on the weight, and `matmul_q5_k_q8_k` ran them again
// for every token because its `blockIdx.y` is the token.
#define Q5K_TOK 8

// Q5_K x Q8_K over a batch, one warp per output row and `Q5K_TOK` tokens.
//
// **Bit-identical to `matmul_q5_k_q8_k` by construction.** Both of its f32
// chains are preserved per token and in order: the `dmin` chain accumulates
// into `sumf` inside the super-block loop, ahead of the eight-lane fold, and
// both of its updates stay fused. The eight interleaved `sums[l]` accumulators
// keep their lane and their ascending order. Only how many outputs one weight
// load serves changes.
//
// The token-independent work hoisted out of the token loop is the whole point:
// the `u0..u3` shuffle -- a dozen integer ops that turn 12 bytes into 8 scales
// and 8 mins -- and the per-index `av`, which reads a nibble of `qs` and a bit
// of `qh`. The mins are token-independent too, but `sumi` folds them against
// the token's own `bsums`, so that stays per token.
//
// Padding tokens read token 0 and are discarded at the write-back, as in
// `matmul_q6_k_q8_k_tok`, so the inner loop carries no branch.
__global__ void matmul_q5_k_q8_k_tok(int n_in, int n_out, int n_tok,
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
    const int t0 = blockIdx.y * Q5K_TOK;

    const int l = lane & 7;
    const int g = lane >> 3;

    int ts[Q5K_TOK];
#pragma unroll
    for (int t = 0; t < Q5K_TOK; ++t) ts[t] = (t0 + t < n_tok) ? (t0 + t) : 0;

    const unsigned char *row = w + (size_t)j * nb * Q5K_BYTES;

    float sum_l[Q5K_TOK];   // sums[l], per token
    float sumf[Q5K_TOK];    // the dmin chain, lane 0's, per token
#pragma unroll
    for (int t = 0; t < Q5K_TOK; ++t) {
        sum_l[t] = 0.0f;
        sumf[t]  = 0.0f;
    }

    for (int i = 0; i < nb; ++i) {
        const unsigned char *blk = row + (size_t)i * Q5K_BYTES;
        const unsigned short d16    = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
        const unsigned short dmin16 = (unsigned short)blk[2] | ((unsigned short)blk[3] << 8);
        const unsigned char *qh = blk + 16;
        const unsigned char *qs = blk + 16 + QK_K / 8;

        // Token-independent, and this is what used to run once per token.
        unsigned int u0 = *(const unsigned int *)(blk + 4);
        unsigned int u1 = *(const unsigned int *)(blk + 8);
        unsigned int u2 = *(const unsigned int *)(blk + 12);
        const unsigned int u3 = ((u2 >> 4) & KMASK2) | (((u1 >> 6) & KMASK3) << 4);
        const unsigned int uaux = u1 & KMASK1;
        u1 = (u2 & KMASK2) | (((u0 >> 6) & KMASK3) << 4);
        u2 = uaux;
        u0 &= KMASK1;

        int aux[Q5K_TOK];
#pragma unroll
        for (int t = 0; t < Q5K_TOK; ++t) aux[t] = 0;

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
                const int av = base + (((qh[l2] >> (jj4 * 2 + half)) & 1) ? 16 : 0);

#pragma unroll
                for (int t = 0; t < Q5K_TOK; ++t) {
                    const signed char *q8 =
                        x_quants + (size_t)ts[t] * n_in + (size_t)i * QK_K;
                    const short aux16 = (short)((int)q8[idx] * av);
                    aux[t] += scale * (int)aux16;
                }
            }
        }

#pragma unroll
        for (int t = 0; t < Q5K_TOK; ++t) {
            aux[t] += __shfl_down_sync(0xffffffff, aux[t], 16);
            aux[t] += __shfl_down_sync(0xffffffff, aux[t], 8);
        }

        if (lane < 8) {
            const float dv = h2f(d16);
#pragma unroll
            for (int t = 0; t < Q5K_TOK; ++t) {
                const float d = dv * x_scales[(size_t)ts[t] * nb + i];
                // FUSED, as in the one-token kernel.
                sum_l[t] = __fmaf_rn(d, (float)aux[t], sum_l[t]);
            }
        }
        // **The min chain, spread over sixteen lanes instead of one.**
        //
        // This was `if (lane == 0)` wrapping a 16-iteration loop per token:
        // 128 serial steps a superblock with 31 lanes masked off, against 64
        // MACs a lane in the main loop above. A warp runs at its slowest lane,
        // so the whole warp was paying for it.
        //
        // **`sumi` is an integer sum of sixteen products, so any order gives
        // the same integer** — a tree across lanes is exact rather than merely
        // close, which is the same argument that lets the MMA path split a
        // sub-block. The f32 chain it feeds is untouched: still
        // `fma(-dmin, sumi, sumf)`, still on lane 0, still once per superblock
        // in ascending order.
        {
            const float dmv = h2f(dmin16);
            const int k = lane & 15;
            const int mk = k >> 1;
            const unsigned int mu = (mk < 4) ? u2 : u3;
            const int mn = (int)((mu >> ((mk & 3) * 8)) & 0xff);
#pragma unroll
            for (int t = 0; t < Q5K_TOK; ++t) {
                const short *bs =
                    x_bsums + (size_t)ts[t] * nb * (QK_K / 16) + (size_t)i * (QK_K / 16);
                // Lanes 16-31 contribute zero and exist only to keep the
                // shuffles warp-synchronous.
                int sumi = (lane < 16) ? (int)bs[k] * mn : 0;
                sumi += __shfl_down_sync(0xffffffff, sumi, 8);
                sumi += __shfl_down_sync(0xffffffff, sumi, 4);
                sumi += __shfl_down_sync(0xffffffff, sumi, 2);
                sumi += __shfl_down_sync(0xffffffff, sumi, 1);
                if (lane == 0) {
                    const float dmin = dmv * x_scales[(size_t)ts[t] * nb + i];
                    // FUSED, and inside the loop, before the lanes are folded in.
                    sumf[t] = __fmaf_rn(-dmin, (float)sumi, sumf[t]);
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < Q5K_TOK; ++t) {
        for (int k = 0; k < 8; ++k) {
            const float v = __shfl_sync(0xffffffff, sum_l[t], k);
            if (lane == 0) sumf[t] += v;
        }
        if (lane == 0 && t0 + t < n_tok) {
            out[(size_t)(t0 + t) * n_out + j] = sumf[t];
        }
    }
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
            s += (int)q8[ao + k]      * kvalue_iq4nl(b & 0xf);
            s += (int)q8[ao + 16 + k] * kvalue_iq4nl(b >> 4);
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

// IQ4_XS x Q8_K with the batch in registers — the token-reuse variant.
//
// **The single-token kernel above re-reads its weight row once per token**, and
// its own doc says so: "prefill weight traffic scales with the batch. Correct
// first, and the reuse variant arrives as a measured change against this
// baseline." Measured, on a 4,000-token 35B prefill: `matmul_iq4_xs_q8_k` is
// **23.9% of prefill device time**, the largest single kernel, moving ~1.03 GB
// of dense weight per token — 4.1 TB across that prompt.
//
// A warp loads its slice of a superblock once and dots it against `IQ4_TOK`
// tokens, so weight traffic falls by that factor. This is what
// `matmul_q8_0_batch` already does for Q8_0, which was 1.0% of the same run
// precisely because it does it.
//
// **No shared memory, unlike the Q8_0 pair.** That kernel segments its
// cross-block sum through shared memory because it defers partials; here lane 0
// already folds each superblock into a running total, so `IQ4_TOK` running
// totals live in registers and shared memory stays out of the occupancy
// question entirely.
//
// **Bit-identical to the single-token kernel**, and for the reason recorded on
// the Q8_0 one: adding a token axis changes which outputs share a weight load,
// never how one output accumulates. Each token still walks `ibl` ascending and
// folds the same eight lanes in the same order, so the sequence of f32
// additions reaching `sumf[u]` is exactly the sequence the unbatched kernel
// produces for that token. `the_batched_iq4_matmul_is_bit_identical` demands
// equal bits rather than a tolerance.
//
// The nibble table is two `u64` immediates in registers (see `kvalue_iq4nl`),
// so hoisting the unpack out of the token loop costs nothing — which matters,
// because a `__constant__` lookup here was once 72% of this kernel.
#define IQ4_TOK 8

__global__ void matmul_iq4_xs_q8_k_batch(int n_in, int n_out, int n_tok,
                                         const unsigned char *__restrict__ w,
                                         const float *__restrict__ x_scales,
                                         const signed char *__restrict__ x_quants,
                                         float *__restrict__ out) {
    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;

    const int t0 = blockIdx.y * IQ4_TOK;
    int nt = n_tok - t0;
    if (nt > IQ4_TOK) nt = IQ4_TOK;

    // Lane layout is the single-token kernel's: `t` picks one of the eight
    // sub-blocks the oracle folds in order, `p` splits its integer sum four
    // ways. Ascending `t` is the reference's own order.
    const int t    = lane >> 2;
    const int p    = lane & 3;
    const int ib   = (t >> 1) * 2;
    const int half = t & 1;

    // A running total per token, folded in ascending `ibl` exactly as the
    // unbatched kernel folds its single one.
    float sumf[IQ4_TOK];
#pragma unroll
    for (int u = 0; u < IQ4_TOK; ++u) sumf[u] = 0.0f;

    const unsigned char *row = w + (size_t)j * nb * IQ4XS_BYTES;

    for (int ibl = 0; ibl < nb; ++ibl) {
        const unsigned char *blk = row + (size_t)ibl * IQ4XS_BYTES;
        const float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        const unsigned int sh = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
        const unsigned char *scales_l = blk + 4;
        const unsigned char *qs = blk + 4 + QK_K / 64;

        const unsigned int h  = sh >> (ib * 2);
        const unsigned int lo = scales_l[ib >> 1];
        const int ls = (half == 0) ? (int)((lo & 0xf) | ((h << 4) & 0x30))
                                   : (int)((lo >> 4)  | ((h << 2) & 0x30));

        const int qo = ib * 16 + half * 16;
        const int ao = ib * 32 + half * 32;

        // **The whole point: this lane's four weight bytes, read once.** The
        // unbatched kernel re-reads them for every token.
        int vlo[4], vhi[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const unsigned char b = qs[qo + p * 4 + k];
            vlo[k] = kvalue_iq4nl(b & 0xf);
            vhi[k] = kvalue_iq4nl(b >> 4);
        }

        for (int u = 0; u < nt; ++u) {
            const signed char *q8 =
                x_quants + (size_t)(t0 + u) * n_in + (size_t)ibl * QK_K;
            // s1 and s2 in the reference, summed together: both integer, so
            // joining them cannot round.
            int s = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                s += (int)q8[ao + p * 4 + k] * vlo[k];
                s += (int)q8[ao + 16 + p * 4 + k] * vhi[k];
            }
            s += __shfl_down_sync(0xffffffff, s, 2);
            s += __shfl_down_sync(0xffffffff, s, 1);

            // `(d * xs) * (ls - 32)`, grouped as the reference groups it:
            // `d4d8 = d * xs[ibl]` then `dh = d4d8 * (ls - 32)`.
            const float d4d8 = d * x_scales[(size_t)(t0 + u) * nb + ibl];
            const float dh = d4d8 * (float)(ls - 32);
            // NOT fused, as the reference's compiler leaves it.
            const float term = dh * (float)s;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float v = __shfl_sync(0xffffffff, term, k * 4);
                if (lane == 0) sumf[u] += v;
            }
        }
    }

    if (lane == 0) {
        for (int u = 0; u < nt; ++u) {
            out[(size_t)(t0 + u) * n_out + j] = sumf[u];
        }
    }
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
                                       int n_pair,
                                       const unsigned long long *__restrict__ wptrs,
                                       const float *__restrict__ x_scales,
                                       const signed char *__restrict__ x_quants,
                                       float *__restrict__ out) {
    // **`blockIdx.y` is a (token, pick) pair, not a pick.** In decode there is
    // one token and this is the eight picks exactly as before; in a batch it is
    // `n_tok * n_used`, laid out token-major so pair `t * n_used + i` is
    // token `t`'s `i`-th expert -- which is the order `moe_gather_ptrs` already
    // writes `wptrs` in.
    //
    // Nothing else moves: `wptrs`, `x` and `out` are all indexed by the same
    // pair, so this is the batch convention applied to the expert axis. Bit
    // exact by construction, for the third time: it changes which outputs share
    // a launch, never how one accumulates.
    const int e = blockIdx.y;
    if (e >= n_pair) return;
    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;

    // **The expert's address comes from device memory, not from a kernel
    // argument**, which is the whole point. `moe_gather_ptrs` filled `wptrs`
    // from the slot table using ids `moe_topk` chose, so nothing in this launch
    // depends on the host having seen the router's output -- and a CUDA graph,
    // which bakes its arguments in at record time, still routes to the experts
    // this token actually picked.
    const unsigned char *w = (const unsigned char *)wptrs[e];

    const float sumf = dot_iq4_xs_warp(
        nb,
        w + (size_t)j * nb * IQ4XS_BYTES,
        x_scales + (size_t)e * x_stride_super,
        x_quants + (size_t)e * x_stride_super * QK_K,
        lane);
    if (lane == 0) out[(size_t)e * n_out + j] = sumf;
}

// The routed FFN's gate, up and SiLU-gating in **one launch**.
//
// Three kernels became one: `matmul_experts(gate)`, `matmul_experts(up)` and
// `silu_mul` over the pair. Measured at 3.0 + 2.7 launches per layer out of 38,
// so this is 80 of a token's 1522 -- and on this platform the unit of cost is
// the launch, because WDDM serializes submissions and the CUDA driver batches
// them by heuristic. llama.cpp reaches 40 tok/s here with graphs *disabled*
// (split buffers, see ggml_cuda_graph_check_compability), so fewer and larger
// launches is a proven route rather than a guess.
//
// It also removes a round trip through VRAM: the old form wrote `n_used * n_ff`
// floats twice and read them back to combine. Here `gate` and `up` for the same
// output element are both in registers, so only the result is stored.
//
// Bit-exactness is unchanged. Each dot is `dot_iq4_xs_warp` over the same bytes
// in the same order, and `silu` was already outside the exact set for the usual
// reason -- `expf` -- with its order untouched.
__global__ void matmul_iq4_xs_q8_k_moe_glu(int n_in, int n_ff, int n_pair,
                                           int n_used,
                                           const unsigned long long *__restrict__ gptrs,
                                           const unsigned long long *__restrict__ uptrs,
                                           const float *__restrict__ x_scales,
                                           const signed char *__restrict__ x_quants,
                                           float *__restrict__ out) {
    const int e = blockIdx.y;
    if (e >= n_pair) return;
    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_ff) return;

    // **The one asymmetry with the `down` matmul above.** There, every pair has
    // its own `n_ff`-wide intermediate, so `x` is indexed by the pair. Here the
    // gate and up projections read the token's activation, which all of that
    // token's `n_used` experts share -- so `x` is indexed by the *token*, and
    // the pair layout being token-major is what makes that a division.
    //
    // Getting this wrong is silent: at `n_tok == 1` the quotient is always 0
    // and every batch shape would read token 0's activation for the whole
    // prompt. `batched_moe_prefill_equals_token_by_token` is what catches it.
    const int tok = e / n_used;

    // As `matmul_iq4_xs_q8_k_moe`: both addresses come from device memory, so
    // this launch carries no trace of which experts were chosen.
    const unsigned char *gw = (const unsigned char *)gptrs[e];
    const unsigned char *uw = (const unsigned char *)uptrs[e];
    const size_t off = (size_t)j * nb * IQ4XS_BYTES;
    const float *xs = x_scales + (size_t)tok * nb;
    const signed char *xq = x_quants + (size_t)tok * nb * QK_K;

    // Every expert of *this token* reads the same activation here -- this is
    // the gate/up half of the FFN, before any per-expert intermediate exists.
    const float g = dot_iq4_xs_warp(nb, gw + off, xs, xq, lane);
    const float u = dot_iq4_xs_warp(nb, uw + off, xs, xq, lane);

    if (lane == 0) {
        // silu(g) * u, exactly as `silu_mul` computes it.
        out[(size_t)e * n_ff + j] = g / (1.0f + expf(-g)) * u;
    }
}

// ---------------------------------------------------------------------------
// Grouping the routed FFN by expert
// ---------------------------------------------------------------------------
//
// **The measured reason.** `matmul_iq4_xs_q8_k_moe_glu` gives every (token,
// pick) pair its own `blockIdx.y` and re-reads that expert's weight row for it.
// A `MOE_CHUNK` of 128 tokens is 1,024 pairs landing on at most 256 experts, so
// each row crosses the bus about four times more often than it needs to.
//
// That this is worth fixing is a measurement, not an intuition: on a
// 2,936-token 35B prefill the kernel moves 1.14 GB in 7.88 ms -- **144 GB/s**,
// against 206.8 for the same shape with everything resident, and a 130.6 GB/s
// prediction from blending the two residency tiers at that run's 95.1/4.9
// split. It is bandwidth-bound, so removing re-reads is the whole win.
//
// The *dense* IQ4_XS matmul is not: token-tiling cut its traffic 8x for 1.54x,
// and its decomposition puts only 13% in weight loads against 24% in the
// ordered fold. Same dot product, opposite bound -- which is why grouping was
// wrongly discounted once already, by carrying the dense finding across.
//
// **Bit-exact by construction, for the fourth time.** This changes which block
// computes an output and which weight loads are shared, never how one output
// accumulates: every dot is still the arithmetic of `dot_iq4_xs_warp` over the
// same bytes in the same order. It is the argument that made batching free.

// Pairs per tile. Mean occupancy is `n_pair / n_expert` = 4 at `MOE_CHUNK` 128,
// so 8 puts all but the hot experts in one tile and captures essentially the
// whole 4x. Larger costs registers: `moe_glu_grouped` already carries two
// accumulator arrays and two unpacked weight sets.
#define MOE_TOK 8

// Tile width for the tensor-core routed FFN, and how many 8-token MMA tiles it
// is cut into.
//
// **Separate from `MOE_TOK` because the two kernels have opposite constraints.**
// The scalar grouped kernel holds a per-token register array, so 8 is already
// its ceiling; the MMA kernel holds only accumulators per sub-tile and wants the
// widest tile the routing can fill. At `MOE_CHUNK` 512 a chunk is 4,096 picks
// over 256 experts, so the mean expert collects 16 -- measured 5.17 at chunk
// 128, which is what capped the first version.
#define MOE_MMA_TOK 16
#define MOE_MMA_NTILE 2

// Sort a chunk's (token, pick) pairs by expert id, and cut the result into
// tiles a block can own.
//
// One block: the work is 256 counters over at most 1,024 pairs. `perm` holds
// the pair indices in ascending expert order; tile `t` covers `tile_n[t]` of
// them starting at `tile_first[t]`, all sharing one expert.
//
// # Deterministic on purpose
//
// A counting sort with an atomic cursor orders pairs within an expert by
// whichever thread arrives first. Nothing numeric depends on it -- each pair is
// an independent output row -- but a permutation that varies between runs makes
// a differential failure unreproducible, which is the property this repo spends
// the most to keep. So one thread owns one expert and scans pairs ascending.
//
// Shared memory is `n_pair + n_expert + 2*(n_expert+1)` ints, supplied by the
// caller: 7,176 bytes at the shapes this model uses.
extern "C" __global__ void moe_group(int n_pair, int n_expert, int e_tok,
                                     const int *__restrict__ ids,
                                     int *__restrict__ perm,
                                     int *__restrict__ tile_first,
                                     int *__restrict__ tile_n,
                                     int *__restrict__ n_tile) {
    extern __shared__ int shm[];
    int *sids  = shm;                    // n_pair
    int *count = sids + n_pair;          // n_expert
    int *scan  = count + n_expert;       // n_expert + 1, pair offsets
    int *tscan = scan + n_expert + 1;    // n_expert + 1, tile offsets

    for (int p = threadIdx.x; p < n_pair; p += blockDim.x) sids[p] = ids[p];
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) count[e] = 0;
    __syncthreads();

    for (int p = threadIdx.x; p < n_pair; p += blockDim.x) atomicAdd(&count[sids[p]], 1);
    __syncthreads();

    // Both prefix sums in one thread. `n_expert` is 256 and this is a single
    // block, so a scan network would cost more in barriers than the 512 adds it
    // removes.
    if (threadIdx.x == 0) {
        int off = 0, toff = 0;
        for (int e = 0; e < n_expert; ++e) {
            scan[e]  = off;   off  += count[e];
            tscan[e] = toff;  toff += (count[e] + e_tok - 1) / e_tok;
        }
        scan[n_expert]  = off;
        tscan[n_expert] = toff;
        *n_tile = toff;
    }
    __syncthreads();

    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        int c = scan[e];
        // A shared-memory broadcast: every lane of a warp reads the same `p`.
        for (int p = 0; p < n_pair; ++p) {
            if (sids[p] == e) perm[c++] = p;
        }
        int t = tscan[e], at = scan[e], left = count[e];
        while (left > 0) {
            const int take = left < e_tok ? left : e_tok;
            tile_first[t] = at;
            tile_n[t]     = take;
            ++t; at += take; left -= take;
        }
    }
}

// The routed FFN gate, up and SiLU gating, grouped by expert.
//
// `matmul_iq4_xs_q8_k_moe_glu` with the token tiling of
// `matmul_iq4_xs_q8_k_batch`, where the tile is "the tokens this expert was
// routed to" rather than "the next eight tokens of the batch". The weight bytes
// a lane needs are unpacked once per super-block and dotted against every token
// in the tile -- the reuse the per-pair form cannot have, because adjacent
// pairs are different experts.
//
// `n_tile` is read from device memory rather than taken as an argument, for the
// same reason the expert pointers are: it is a function of this chunk's
// routing, and an argument is baked in when a graph records it.
__global__ void matmul_iq4_xs_q8_k_moe_glu_grouped(
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
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_ff) return;

    const int first = tile_first[tl];
    const int nt    = tile_n[tl];

    // Every pair here shares an expert, so one pointer serves the tile. Lanes
    // past `nt` re-read entry 0 rather than branching: the results are dropped.
    int pr[MOE_TOK], tk[MOE_TOK];
#pragma unroll
    for (int u = 0; u < MOE_TOK; ++u) {
        const int q = perm[first + (u < nt ? u : 0)];
        pr[u] = q;
        tk[u] = q / n_used;
    }

    const size_t off = (size_t)j * nb * IQ4XS_BYTES;
    const unsigned char *grow = (const unsigned char *)gptrs[pr[0]] + off;
    const unsigned char *urow = (const unsigned char *)uptrs[pr[0]] + off;

    // Lane layout is the single-token kernel, so ascending `t` is the order the
    // reference folds in.
    const int t    = lane >> 2;
    const int p    = lane & 3;
    const int ib   = (t >> 1) * 2;
    const int half = t & 1;

    float sg[MOE_TOK], su[MOE_TOK];
#pragma unroll
    for (int u = 0; u < MOE_TOK; ++u) { sg[u] = 0.0f; su[u] = 0.0f; }

    for (int ibl = 0; ibl < nb; ++ibl) {
        const unsigned char *gblk = grow + (size_t)ibl * IQ4XS_BYTES;
        const unsigned char *ublk = urow + (size_t)ibl * IQ4XS_BYTES;

        const float dg = h2f((unsigned short)gblk[0] | ((unsigned short)gblk[1] << 8));
        const float du = h2f((unsigned short)ublk[0] | ((unsigned short)ublk[1] << 8));
        const unsigned int shg = (unsigned int)gblk[2] | ((unsigned int)gblk[3] << 8);
        const unsigned int shu = (unsigned int)ublk[2] | ((unsigned int)ublk[3] << 8);

        const unsigned int hg  = shg >> (ib * 2);
        const unsigned int hu  = shu >> (ib * 2);
        const unsigned int lg  = gblk[4 + (ib >> 1)];
        const unsigned int lou = ublk[4 + (ib >> 1)];
        const int lsg = (half == 0) ? (int)((lg & 0xf) | ((hg << 4) & 0x30))
                                    : (int)((lg >> 4)  | ((hg << 2) & 0x30));
        const int lsu = (half == 0) ? (int)((lou & 0xf) | ((hu << 4) & 0x30))
                                    : (int)((lou >> 4)  | ((hu << 2) & 0x30));

        const unsigned char *gqs = gblk + 4 + QK_K / 64;
        const unsigned char *uqs = ublk + 4 + QK_K / 64;
        const int qo = ib * 16 + half * 16;
        const int ao = ib * 32 + half * 32;

        // **The reuse.** These four weight bytes of each of the two matrices
        // are read and unpacked once for the whole tile.
        int gvlo[4], gvhi[4], uvlo[4], uvhi[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const unsigned char bg = gqs[qo + p * 4 + k];
            const unsigned char bu = uqs[qo + p * 4 + k];
            gvlo[k] = kvalue_iq4nl(bg & 0xf);
            gvhi[k] = kvalue_iq4nl(bg >> 4);
            uvlo[k] = kvalue_iq4nl(bu & 0xf);
            uvhi[k] = kvalue_iq4nl(bu >> 4);
        }

        // **Fully unrolled with a predicate, not bounded by `nt`.** A runtime
        // bound makes `u` a non-constant index, so ptxas puts `pr`, `tk` and
        // the accumulators in local memory -- 128 bytes of stack frame, which
        // it does not report as a spill. Measured: that cost 9% of prefill and
        // made the first version of this change a net loss.
#pragma unroll
        for (int u = 0; u < MOE_TOK; ++u) {
            if (u >= nt) continue;
            const signed char *q8 =
                x_quants + (size_t)tk[u] * n_in + (size_t)ibl * QK_K;
            // s1 and s2 in the reference, summed together: both integer, so
            // joining them cannot round.
            int gs = 0, us = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int a0 = (int)q8[ao + p * 4 + k];
                const int a1 = (int)q8[ao + 16 + p * 4 + k];
                gs += a0 * gvlo[k]; gs += a1 * gvhi[k];
                us += a0 * uvlo[k]; us += a1 * uvhi[k];
            }
            gs += __shfl_down_sync(0xffffffff, gs, 2);
            gs += __shfl_down_sync(0xffffffff, gs, 1);
            us += __shfl_down_sync(0xffffffff, us, 2);
            us += __shfl_down_sync(0xffffffff, us, 1);

            // `d4d8 = d * xs[ibl]` then `dh = d4d8 * (ls - 32)`, grouped as the
            // reference groups it, and not fused -- `--fmad=false` is global.
            const float xs = x_scales[(size_t)tk[u] * nb + ibl];
            const float gd4d8 = dg * xs;
            const float ud4d8 = du * xs;
            const float gdh = gd4d8 * (float)(lsg - 32);
            const float udh = ud4d8 * (float)(lsu - 32);
            const float gterm = gdh * (float)gs;
            const float uterm = udh * (float)us;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float vg = __shfl_sync(0xffffffff, gterm, k * 4);
                const float vu = __shfl_sync(0xffffffff, uterm, k * 4);
                if (lane == 0) { sg[u] += vg; su[u] += vu; }
            }
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int u = 0; u < MOE_TOK; ++u) {
            if (u >= nt) continue;
            const float g = sg[u];
            out[(size_t)pr[u] * n_ff + j] = g / (1.0f + expf(-g)) * su[u];
        }
    }
}

// The routed FFN `down` matmul, grouped by expert.
//
// The one asymmetry with the gate/up half above: there every expert of a token
// reads that token's activation, so `x` is indexed by token; here each pair has
// its own `n_ff`-wide intermediate, so `x` is indexed by the pair. Getting it
// backwards is silent at `n_tok == 1`, where the two coincide -- which is what
// `batched_moe_prefill_equals_token_by_token` exists to catch.
__global__ void matmul_iq4_xs_q8_k_moe_grouped(
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
    const int j    = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= n_out) return;

    const int first = tile_first[tl];
    const int nt    = tile_n[tl];

    int pr[MOE_TOK];
#pragma unroll
    for (int u = 0; u < MOE_TOK; ++u) pr[u] = perm[first + (u < nt ? u : 0)];

    const unsigned char *row =
        (const unsigned char *)wptrs[pr[0]] + (size_t)j * nb * IQ4XS_BYTES;

    const int t    = lane >> 2;
    const int p    = lane & 3;
    const int ib   = (t >> 1) * 2;
    const int half = t & 1;

    float sumf[MOE_TOK];
#pragma unroll
    for (int u = 0; u < MOE_TOK; ++u) sumf[u] = 0.0f;

    for (int ibl = 0; ibl < nb; ++ibl) {
        const unsigned char *blk = row + (size_t)ibl * IQ4XS_BYTES;
        const float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        const unsigned int sh = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
        const unsigned char *qs = blk + 4 + QK_K / 64;

        const unsigned int h  = sh >> (ib * 2);
        const unsigned int lo = blk[4 + (ib >> 1)];
        const int ls = (half == 0) ? (int)((lo & 0xf) | ((h << 4) & 0x30))
                                   : (int)((lo >> 4)  | ((h << 2) & 0x30));

        const int qo = ib * 16 + half * 16;
        const int ao = ib * 32 + half * 32;

        int vlo[4], vhi[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const unsigned char b = qs[qo + p * 4 + k];
            vlo[k] = kvalue_iq4nl(b & 0xf);
            vhi[k] = kvalue_iq4nl(b >> 4);
        }

        // **Fully unrolled with a predicate, not bounded by `nt`.** A runtime
        // bound makes `u` a non-constant index, so ptxas puts `pr`, `tk` and
        // the accumulators in local memory -- 128 bytes of stack frame, which
        // it does not report as a spill. Measured: that cost 9% of prefill and
        // made the first version of this change a net loss.
#pragma unroll
        for (int u = 0; u < MOE_TOK; ++u) {
            if (u >= nt) continue;
            const signed char *q8 =
                x_quants + (size_t)pr[u] * n_in + (size_t)ibl * QK_K;
            int s = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                s += (int)q8[ao + p * 4 + k] * vlo[k];
                s += (int)q8[ao + 16 + p * 4 + k] * vhi[k];
            }
            s += __shfl_down_sync(0xffffffff, s, 2);
            s += __shfl_down_sync(0xffffffff, s, 1);

            const float d4d8 = d * x_scales[(size_t)pr[u] * nb + ibl];
            const float dh   = d4d8 * (float)(ls - 32);
            const float term = dh * (float)s;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float v = __shfl_sync(0xffffffff, term, k * 4);
                if (lane == 0) sumf[u] += v;
            }
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int u = 0; u < MOE_TOK; ++u) {
            if (u < nt) out[(size_t)pr[u] * n_out + j] = sumf[u];
        }
    }
}

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


// The same kernel with the f32 fold deferred to once per superblock.
//
// **A probe first, a candidate second.** Six data-movement experiments on this
// kernel have measured nothing (`c8131e5`'s two volume cuts and its fold hoist,
// the staged tile at two shapes, and Q5_K's activation staging), while the one
// dependency change that was tried paid 1.47x. This asks whether what sits
// *between* the MMAs is the cost.
//
// The arithmetic that motivates it: at 2048x8192, n_tok 512, the real kernel
// issues 2,097,152 `mma.m16n8k32.s8` in ~2.1 ms across 36 SMs, which is 0.0115
// MMA per cycle per SM against a measured ceiling of 0.278 - **24x off, the 4.3%
// of peak the bench reports**. `ptxas` says 72 registers and no spills, so 28
// warps an SM fit and occupancy is not it either. What is left per MMA is the
// unpack and the fold.
//
// # What changes, and why it is exact right up to the last step
//
// The reference folds once per 32-weight sub-block:
//
//     acc += (d * xs) * (ls - 32) * s          for t = 0..7, ascending
//
// `(ls - 32) * s` is an **integer** product — `s` is at most 32*127*127 and
// `ls - 32` is within +-31, so eight of them sum to under 1.3e8 and cannot
// overflow int32, let alone round. So `sum_t (ls_t - 32) * s_t` is exact, and
// this kernel accumulates it in int32 and folds once:
//
//     acc += (d * xs) * (float)(sum_t (ls_t - 32) * s_t)
//
// That replaces eight (convert, multiply, multiply, add) chains per superblock
// with eight integer MACs and one f32 fold.
//
// # It is NOT bit-exact, and the reason is worth stating precisely
//
// `sum_t c * a_t` and `c * sum_t a_t` are the same real number and different
// f32 numbers: the reference rounds after every sub-block, this rounds once per
// superblock. The `a_t` are exact integers either way, so the error is a *fold
// order* change of the same family as the RMSNorm tree — and, as there, this
// one is the *more* accurate of the two, since it rounds 8x less often.
//
// It is therefore off by default and outside the exact set. Whether it ships is
// not a decision this kernel gets to make; `--iq4-fold-once` exists so the cost
// can be known before anyone is asked.
__global__ void dbg_iq4_mma_foldonce(int n_in, int n_out, int n_tok,
                                     const unsigned char *__restrict__ w,
                                     const float *__restrict__ x_scales,
                                     const signed char *__restrict__ x_quants,
                                     float *__restrict__ out) {
    const int nb   = n_in / QK_K;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    const int j0 = (blockIdx.x * (blockDim.x >> 5) + warp) * 16;
    const int t0 = blockIdx.y * (8 * MMA_NTILE);
    if (j0 >= n_out) return;

    const int g = lane >> 2;
    const int q = lane & 3;

    const int row_a = j0 + g;
    const int row_b = j0 + g + 8;

    const unsigned char *ra = w + (size_t)row_a * nb * IQ4XS_BYTES;
    const unsigned char *rb = w + (size_t)row_b * nb * IQ4XS_BYTES;

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

        float xs[MMA_NTILE][2];
#pragma unroll
        for (int n = 0; n < MMA_NTILE; ++n) {
            const int c0 = t0 + n * 8 + 2 * q;
            const int c1 = c0 + 1;
            xs[n][0] = (c0 < n_tok) ? x_scales[(size_t)c0 * nb + ibl] : 0.0f;
            xs[n][1] = (c1 < n_tok) ? x_scales[(size_t)c1 * nb + ibl] : 0.0f;
        }

        // The whole superblock's scaled products, as exact integers.
        int acci[MMA_NTILE][4];
#pragma unroll
        for (int n = 0; n < MMA_NTILE; ++n) {
#pragma unroll
            for (int k = 0; k < 4; ++k) acci[n][k] = 0;
        }

        for (int t = 0; t < 8; ++t) {
            const int ib   = (t >> 1) * 2;
            const int half = t & 1;

            const int lsa = iq4_ls(sha, ba[4 + (ib >> 1)], ib, half) - 32;
            const int lsb = iq4_ls(shb, bb[4 + (ib >> 1)], ib, half) - 32;

            const int off = 4 + QK_K / 64 + t * 16 + q * 4;
            int a[4];
            unpack_iq4_pair(*(const int *)(ba + off), a[0], a[2]);
            unpack_iq4_pair(*(const int *)(bb + off), a[1], a[3]);

#pragma unroll
            for (int n = 0; n < MMA_NTILE; ++n) {
                const int tb = t0 + n * 8 + g;
                int b[2] = {0, 0};
                if (tb < n_tok) {
                    // `+ t * 32` is load-bearing: sub-block `t` covers
                    // activations [32t, 32t+32). Dropping it makes every
                    // sub-block re-read the same 32 values, which is both wrong
                    // and *faster* — a much smaller working set — so it reads
                    // as a speedup rather than as a failure.
                    const signed char *q8 =
                        x_quants + (size_t)tb * n_in + (size_t)ibl * QK_K + t * 32;
                    b[0] = *(const int *)(q8 + q * 4);
                    b[1] = *(const int *)(q8 + 16 + q * 4);
                }

                const int zero[4] = {0, 0, 0, 0};
                int s[4];
                mma_m16n8k32_s8(s, a, b, zero);

                // Integer, so exact and order-free. This is the whole change.
                acci[n][0] += lsa * s[0];
                acci[n][1] += lsa * s[1];
                acci[n][2] += lsb * s[2];
                acci[n][3] += lsb * s[3];
            }
        }

        // One f32 fold per superblock instead of eight.
#pragma unroll
        for (int n = 0; n < MMA_NTILE; ++n) {
            acc[n][0] += (da * xs[n][0]) * (float)acci[n][0];
            acc[n][1] += (da * xs[n][1]) * (float)acci[n][1];
            acc[n][2] += (db * xs[n][0]) * (float)acci[n][2];
            acc[n][3] += (db * xs[n][1]) * (float)acci[n][3];
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
// The staged tile: both operands through shared memory, in a canonical form
// ---------------------------------------------------------------------------
//
// `matmul_iq4_xs_q8_k_mma` reads both MMA operands straight from global, one
// fragment per lane. A warp's A load is therefore **eight scattered 16-byte
// segments** -- lanes 0-3 take 16 bytes of one row, lanes 4-7 of the next, and
// so on -- and B has the same shape over eight tokens. `c8131e5` measured what
// that costs by elimination: cutting A traffic 4x (`MMA_NTILE` 16) and B
// traffic 2x (`MMA_NROW` 2) were both a wash, so **volume is not the bound and
// access shape is what is left.**
//
// So this kernel changes shape rather than volume. Each warp stages one whole
// superblock of one row -- lane `i` reads `qs + 4i`, so the 32 lanes cover 128
// contiguous bytes in **one transaction** -- unpacks it to the canonical form
// once, and the MMA loop then reads only shared. Global latency leaves the
// inner loop entirely.
//
// # The canonical form, and where it deliberately differs from llama.cpp
//
// `load_tiles_iq4_xs` in `mmq.cuh` writes two things per row: the nibbles
// already mapped through `kvalues_iq4nl` and packed four to an int, and a
// single f32 scale `d * (ls - 32)`. After that its MMA loop is quant-agnostic,
// which is how twenty formats share one GEMM.
//
// **We keep the quants and split the scale.** Pre-folding `d * (ls - 32)`
// reassociates `(d * xs) * ls` into `(d * ls) * xs`, which `c8131e5` measured
// as outside the bit-exact set -- it was tried, it was worth -1 to -2.5%, and
// it changes the rounding. Storing `d` and `ls` separately costs 1 KiB of
// shared and keeps every bit. The MMA loop is still quant-agnostic; a Q5_K or
// Q6_K loader would fill the same three arrays.
//
// # Why the row stride is 68 ints and not 64
//
// A superblock is 64 ints, and a warp's A fragment for sub-block `t` reads
// `[row g][8t + q]` over `g = 0..7, q = 0..3`. At stride 64 every row starts on
// bank 0 and that is a 32-way conflict, which gives back exactly what the
// staging saves -- the same trap as `delta_rule_batch_shared`'s
// `head_k_dim + 1`, at a different modulus because this access is four wide.
//
// 68 mod 32 is 4, so row `g` starts at bank `4g` and the lanes read banks
// `(4g + q) mod 32`, covering all 32 exactly once. Conflict-free, for four
// ints of padding per row.
//
// # The trade this makes, stated because it is the risk
//
// `c8131e5` left two candidates, access shape **and occupancy**, and this
// improves the first by worsening the second: ~36 KiB of shared is one or two
// blocks an SM where the register-tiled kernel used none. That is the ordinary
// GEMM trade -- latency hidden by ILP across `ST_TILE_N / 8` accumulator sets
// rather than by more resident warps -- but it is a trade, so the tile
// dimensions are `#define`s and the bench sweeps them.

// Tile: weight rows x tokens per block. One warp per 16 rows, so
// `ST_TILE_M / 16` warps; `ST_TILE_N / 8` MMA tiles per warp.
#define ST_TILE_M 64
#define ST_TILE_N 64
#define ST_WARPS (ST_TILE_M / 16)
#define ST_THREADS (ST_WARPS * 32)
#define ST_NTILE (ST_TILE_N / 8)

// Ints per staged row: 64 payload, 4 pad. See the bank argument above.
#define ST_STRIDE 68

__global__ void matmul_iq4_xs_q8_k_staged(int n_in, int n_out, int n_tok,
                                          const unsigned char *__restrict__ w,
                                          const float *__restrict__ x_scales,
                                          const signed char *__restrict__ x_quants,
                                          float *__restrict__ out) {
    __shared__ int sa_q[ST_TILE_M][ST_STRIDE];
    __shared__ int sb_q[ST_TILE_N][ST_STRIDE];
    __shared__ float sa_d[ST_TILE_M];
    // Stride 9, not 8: at 8 rows `g` and `g + 4` land on one bank.
    __shared__ int sa_ls[ST_TILE_M][9];
    __shared__ float sb_xs[ST_TILE_N];

    const int nb = n_in / QK_K;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int j0 = blockIdx.x * ST_TILE_M;   // first weight row
    const int t0 = blockIdx.y * ST_TILE_N;   // first token
    if (j0 >= n_out) return;

    const int g = lane >> 2;
    const int q = lane & 3;

    // Shared rows this lane accumulates, and the global rows they stand for.
    const int sra = warp * 16 + g;
    const int srb = sra + 8;
    const int row_a = j0 + sra;
    const int row_b = j0 + srb;

    float acc[ST_NTILE][4];
#pragma unroll
    for (int n = 0; n < ST_NTILE; ++n) {
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[n][k] = 0.0f;
    }

    for (int ibl = 0; ibl < nb; ++ibl) {
        // --- stage A: the weights, unpacked into the canonical form ---
        //
        // One warp per row, lane `i` taking `qs + 4i`. Sub-block `t` and quad
        // `q` come from the lane index because the `qs` offset `8 + 16t + 4q`
        // is exactly `8 + 4 * (4t + q)`, so a linear walk is also the right
        // walk.
#pragma unroll
        for (int r0 = 0; r0 < ST_TILE_M; r0 += ST_WARPS) {
            const int r = r0 + warp;
            // Clamped rather than skipped: the read stays in bounds and the
            // write-back is what drops the row. A skipped row would leave the
            // previous superblock's quants staged.
            const int jr = min(j0 + r, n_out - 1);
            const unsigned char *ba = w + ((size_t)jr * nb + ibl) * IQ4XS_BYTES;

            const int t = lane >> 2;
            const int lq = lane & 3;
            int lo, hi;
            unpack_iq4_pair(*(const int *)(ba + 4 + QK_K / 64 + lane * 4), lo, hi);
            sa_q[r][t * 8 + lq] = lo;
            sa_q[r][t * 8 + 4 + lq] = hi;

            if (lane < 8) {
                const unsigned int sh = (unsigned int)ba[2] | ((unsigned int)ba[3] << 8);
                const int ib = (lane >> 1) * 2;
                sa_ls[r][lane] = iq4_ls(sh, ba[4 + (ib >> 1)], ib, lane & 1);
                if (lane == 0) {
                    sa_d[r] = h2f((unsigned short)ba[0] | ((unsigned short)ba[1] << 8));
                }
            }
        }

        // --- stage B: the activations, already s8 and already in order ---
        //
        // Q8_K quants are 4-per-int in ascending `k`, which is the fragment
        // layout, so this is a copy. Two warps per token, 128 contiguous bytes
        // each.
#pragma unroll
        for (int c0 = 0; c0 < ST_TILE_N; c0 += ST_THREADS / 64) {
            const int c = c0 + (tid >> 6);
            const int j = tid & 63;
            const int tc = min(t0 + c, n_tok - 1);
            const signed char *q8 = x_quants + (size_t)tc * n_in + (size_t)ibl * QK_K;
            sb_q[c][j] = *(const int *)(q8 + j * 4);
            if (j == 0) {
                sb_xs[c] = x_scales[(size_t)tc * nb + ibl];
            }
        }

        __syncthreads();

        // --- the MMA loop, reading only shared ---
        //
        // Superblock-invariant, so hoisted out of the sub-block loop exactly as
        // the register-tiled kernel hoists them.
        const float da = sa_d[sra];
        const float db = sa_d[srb];
        float xs[ST_NTILE][2];
#pragma unroll
        for (int n = 0; n < ST_NTILE; ++n) {
            xs[n][0] = sb_xs[n * 8 + 2 * q];
            xs[n][1] = sb_xs[n * 8 + 2 * q + 1];
        }

#pragma unroll
        for (int t = 0; t < 8; ++t) {
            int a[4];
            a[0] = sa_q[sra][t * 8 + q];
            a[2] = sa_q[sra][t * 8 + 4 + q];
            a[1] = sa_q[srb][t * 8 + q];
            a[3] = sa_q[srb][t * 8 + 4 + q];

            const int lsa = sa_ls[sra][t];
            const int lsb = sa_ls[srb][t];

#pragma unroll
            for (int n = 0; n < ST_NTILE; ++n) {
                int b[2];
                b[0] = sb_q[n * 8 + g][t * 8 + q];
                b[1] = sb_q[n * 8 + g][t * 8 + 4 + q];

                // Zeroed per k-tile, which the format requires: IQ4_XS carries
                // a scale per 32 weights, so an int32 accumulator cannot span
                // two of them.
                const int zero[4] = {0, 0, 0, 0};
                int s[4];
                mma_m16n8k32_s8(s, a, b, zero);

                // `(d * xs) * (ls - 32)` then `* s`, the reference's grouping
                // and the register kernel's, folded ascending in `t`.
                acc[n][0] += (da * xs[n][0]) * (float)(lsa - 32) * (float)s[0];
                acc[n][1] += (da * xs[n][1]) * (float)(lsa - 32) * (float)s[1];
                acc[n][2] += (db * xs[n][0]) * (float)(lsb - 32) * (float)s[2];
                acc[n][3] += (db * xs[n][1]) * (float)(lsb - 32) * (float)s[3];
            }
        }

        // The next superblock overwrites what these reads just consumed.
        __syncthreads();
    }

#pragma unroll
    for (int n = 0; n < ST_NTILE; ++n) {
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





// The whole tail of a routed FFN in one launch: weighted sum of the experts,
// Resolve this token's chosen experts to device addresses.
//
// `table` is one slot-table entry per expert of one `Experts` tensor -- 256 of
// them, fixed at first sight of that tensor and never moved -- and `ids` is
// what `moe_topk` chose. The result is the `n_used` addresses the expert
// matmuls dereference.
//
// **This is the indirection that makes a CUDA graph possible.** A kernel
// argument is baked into a graph node when the graph is recorded; a device
// buffer is read when it is replayed. Moving the expert pointers from the
// former to the latter is the whole difference between a graph that routes to
// last token's experts and one that routes to this token's.
extern "C" __global__ void moe_gather_ptrs(int n_used, int base, int n_tok,
                                           const unsigned long long *__restrict__ table,
                                           const int *__restrict__ ids,
                                           const int *__restrict__ vram,
                                           unsigned int *__restrict__ counts,
                                           unsigned long long *__restrict__ tally,
                                           unsigned long long *__restrict__ out) {
    // `n_tok * n_used` addresses, one per (token, pick), laid out token-major
    // so an expert matmul can index `wptrs[tok * n_used + e]`.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tok * n_used) return;
    const int e = i;
    const int id = ids[e];
    out[e] = table[id];

    // **Observation, not policy, and it is not optional.** The moment routing
    // moved onto the device the host stopped learning which experts were read,
    // so hit rate, host-read rate and the coverage distribution all silently
    // went to zero -- a working cache reporting nothing, which is the exact
    // failure `CLAUDE.md` catalogues four prior instances of. These two atomics
    // are the fifth one caught rather than shipped.
    //
    // Cost is 8 atomics per expert tensor per token, ~960 a token, against a
    // launch that already exists.
    atomicAdd(&counts[base + id], 1u);
    atomicAdd(&tally[vram[base + id] ? 0 : 1], 1ull);
}

// Top-k expert selection, on the device.
//
// **The one host decision left in a decode pass, and therefore the reason CUDA
// graphs are off for this model.** `moe_token` downloads the router's 256
// probabilities once per layer to choose eight experts on the CPU -- 40 syncs a
// token, in a backend that took bus crossings from 1329 to 5. A graph defers
// every kernel to `end_pass`, so that mid-pass read would return the *previous*
// token's probabilities and the model would route to the wrong experts:
// fluent-looking nonsense. Moving the decision here is what removes it.
//
// # It must reproduce the host selection exactly, and does
//
// `moe_token` runs `n_used` rounds, each scanning `e` ascending and taking a
// new best only on a *strict* `>`, so the lowest index survives a tie. Then it
// sums the chosen probabilities **in selection order** and divides each by
// `max(sum, 6.103515625e-5)`.
//
// Selection is exact under any decomposition: (value, index) with "higher value
// wins, tie to lower index" is a total order, and a max never rounds. So the
// tree reduction below is free of the usual reordering worry -- the same
// argument that made the Q8_0 warp matmul bit-identical, applied to a
// comparison rather than to an integer sum.
//
// The sum is *not* order-free, so thread 0 walks the eight picks serially in
// pick order, exactly as the host does. Eight additions on one thread is not
// worth splitting.
extern "C" __global__ void moe_topk(int n_expert, int n_used,
                                    const float *__restrict__ all_probs,
                                    int *__restrict__ all_ids,
                                    float *__restrict__ all_weights) {
    extern __shared__ unsigned char moe_topk_smem[];
    float *sv = (float *)moe_topk_smem;
    int *si = (int *)(sv + blockDim.x);

    // One block per token. Selection is independent per row -- a token's
    // experts depend only on its own probabilities -- so a prefill batch is
    // just a wider grid, and decode is `gridDim.x == 1` of the same kernel.
    const int tok = blockIdx.x;
    const float *probs = all_probs + (size_t)tok * n_expert;
    int *ids = all_ids + (size_t)tok * n_used;
    float *weights = all_weights + (size_t)tok * n_used;
    // `MAX` in `Cuda::moe_glu` and friends: the routed count this model uses is
    // 8, and every kernel downstream carries no more.
    __shared__ int picked[8];

    const int t = threadIdx.x;

    for (int r = 0; r < n_used; ++r) {
        // This thread's best over the experts it owns, skipping ones already
        // taken. Strided, so `n_expert` may exceed the block.
        float bv = 0.0f;
        int bi = -1;
        for (int e = t; e < n_expert; e += blockDim.x) {
            bool taken = false;
            for (int k = 0; k < r; ++k) {
                if (picked[k] == e) taken = true;
            }
            if (taken) continue;
            const float v = probs[e];
            // Strictly greater, so the lowest index survives a tie -- the rule
            // `moe_token` gets from scanning ascending.
            if (bi < 0 || v > bv) {
                bv = v;
                bi = e;
            }
        }
        sv[t] = bv;
        si[t] = bi;
        __syncthreads();

        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (t < s) {
                // Take the other half only if it is a real candidate and either
                // strictly larger, or an equal value at a lower index.
                const bool other_ok = si[t + s] >= 0;
                const bool mine_bad = si[t] < 0;
                const bool better =
                    other_ok && (mine_bad || sv[t + s] > sv[t] ||
                                 (sv[t + s] == sv[t] && si[t + s] < si[t]));
                if (better) {
                    sv[t] = sv[t + s];
                    si[t] = si[t + s];
                }
            }
            __syncthreads();
        }
        if (t == 0) picked[r] = si[0];
        __syncthreads();
    }

    if (t == 0) {
        // Serial, in pick order, from zero -- `Iterator::sum` on the host folds
        // left the same way, and this is the one part that would round
        // differently if it were split.
        float sum = 0.0f;
        for (int r = 0; r < n_used; ++r) sum += probs[picked[r]];
        // f16's smallest normal, guarding the division rather than the weights.
        // A ternary rather than `fmaxf` so a NaN sum yields the clamp, which is
        // what Rust's `f32::max` does.
        const float denom = sum > 6.103515625e-5f ? sum : 6.103515625e-5f;
        for (int r = 0; r < n_used; ++r) {
            ids[r] = picked[r];
            weights[r] = probs[picked[r]] / denom;
        }
    }
}

// the shared expert's sigmoid gate, and the write back into the layer's output
// row.
//
// Replaces `add_scaled_rows` + `add_scaled_sigmoid` + `scatter_chunks`, which
// were three launches per layer -- 120 of a token's 1455 -- to produce one
// vector. Per-launch cost on this platform is **20.7 us, measured**, so three
// kernels that each read and write the same 8 KB are worth removing on launch
// count alone, before the memory traffic they also save.
//
// Order is the oracle's: the experts summed in ascending pick order from zero,
// then the shared expert added last. `expf` is the only inexactness and it was
// already there.
extern "C" __global__ void moe_finish(int n, int n_used, int at,
                                      int n_tok,
                                      const float *__restrict__ scales,
                                      const float *__restrict__ rows,
                                      const float *__restrict__ shared,
                                      const float *__restrict__ logit,
                                      int logit_at,
                                      float *__restrict__ out) {
    // One thread per output element of the whole batch. Decode is
    // `n_tok == 1`, which is exactly the grid this had before.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n_tok) return;
    const int t = idx / n;
    const int j = idx - t * n;

    // From device memory rather than eight kernel arguments, for the same
    // reason the expert pointers are: `moe_topk` wrote them and the host never
    // saw them. The sum below is still serial and ascending, so the order the
    // oracle uses is untouched.
    //
    // Three indexings that were all zero at `n_tok == 1` and are all distinct
    // in a batch: this token's slice of the weights, its own block of expert
    // rows, and its own shared-expert row and gate logit.
    float v = 0.0f;
    for (int e = 0; e < n_used; ++e) {
        v += scales[t * n_used + e] * rows[((size_t)t * n_used + e) * n + j];
    }
    const float g = 1.0f / (1.0f + expf(-logit[logit_at + t]));
    v += shared[(size_t)t * n + j] * g;
    out[(size_t)at + (size_t)t * n + j] = v;
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

// ------------------------------------------------- IQ4_XS, decomposed
//
// Three variants of `matmul_iq4_xs_q8_k`, each removing exactly one thing, so
// the question "what binds this kernel" is answered by measurement instead of
// by a third guess. They are launched with the **recorded arguments of a real
// launch**, at the real geometry, through the same replay path as the baseline.
//
// Two attempts to speed this kernel up have already failed — a repack that was
// ruled out on an alignment argument covering only one of the three things a
// repack does, and a fold restructure that came out 3% slower. Both were judged
// by whole-token time, where this kernel is 8% of a token and a 20% change is
// inside run-to-run noise.
//
// The answers are exclusive:
//
//   nounpack much faster  ->  the 4-bit split and 6-bit scale extraction cost,
//                             and a pre-unpacked repack is the fix
//   nofold much faster    ->  the ordered f32 chain costs, and exactness is
//                             what we are paying for
//   noweight much faster  ->  weight loads dominate, and coalescing is the fix
//   none of them          ->  occupancy or launch latency, and every repack
//                             would fail like the last two
//
// Output is garbage. These are stopwatches.

// No bit-extraction: the weight byte is used directly instead of being split
// into two nibbles and looked up, and the sub-block scale is taken as 1 instead
// of being assembled from `scales_l` and `scales_h`. Same loads, same number of
// products, same fold.
extern "C" __global__ void dbg_iq4_nounpack(int n_in, int n_out,
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

    const int t = lane >> 2, p = lane & 3;
    const int ib = (t >> 1) * 2, half = t & 1;
    const float *xs = x_scales + (size_t)tok * nb;
    const signed char *xq = x_quants + (size_t)tok * n_in;
    const unsigned char *row = w + (size_t)j * nb * IQ4XS_BYTES;

    float sumf = 0.0f;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const unsigned char *blk = row + (size_t)ibl * IQ4XS_BYTES;
        const float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        const unsigned char *qs = blk + 4 + QK_K / 64;
        const signed char *q8 = xq + (size_t)ibl * QK_K;
        const float dh = d * xs[ibl];              // no (ls - 32)
        const int qo = ib * 16 + half * 16;
        const int ao = ib * 32 + half * 32;
        int s = 0;
        for (int k = p * 4; k < p * 4 + 4; ++k) {
            const int b = (int)(signed char)qs[qo + k];   // no nibble split, no table
            s += (int)q8[ao + k]      * b;
            s += (int)q8[ao + 16 + k] * b;
        }
        s += __shfl_down_sync(0xffffffff, s, 2);
        s += __shfl_down_sync(0xffffffff, s, 1);
        const float term = dh * (float)s;
        for (int k = 0; k < 8; ++k) {
            const float v = __shfl_sync(0xffffffff, term, k * 4);
            if (lane == 0) sumf += v;
        }
    }
    if (lane == 0) out[(size_t)tok * n_out + j] = sumf;
}

// No ordered fold: the eight `__shfl_sync` per super-block that walk the terms
// in the oracle's order are gone. Everything else is the real kernel.
extern "C" __global__ void dbg_iq4_nofold(int n_in, int n_out,
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

    const int t = lane >> 2, p = lane & 3;
    const int ib = (t >> 1) * 2, half = t & 1;
    const float *xs = x_scales + (size_t)tok * nb;
    const signed char *xq = x_quants + (size_t)tok * n_in;
    const unsigned char *row = w + (size_t)j * nb * IQ4XS_BYTES;

    float sumf = 0.0f;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const unsigned char *blk = row + (size_t)ibl * IQ4XS_BYTES;
        const float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        const unsigned int sh = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
        const unsigned char *scales_l = blk + 4;
        const unsigned char *qs = blk + 4 + QK_K / 64;
        const signed char *q8 = xq + (size_t)ibl * QK_K;
        const unsigned int h  = sh >> (ib * 2);
        const unsigned int lo = scales_l[ib >> 1];
        const int ls = (half == 0) ? (int)((lo & 0xf) | ((h << 4) & 0x30))
                                   : (int)((lo >> 4)  | ((h << 2) & 0x30));
        const float dh = d * xs[ibl] * (float)(ls - 32);
        const int qo = ib * 16 + half * 16;
        const int ao = ib * 32 + half * 32;
        int s = 0;
        for (int k = p * 4; k < p * 4 + 4; ++k) {
            const unsigned char b = qs[qo + k];
            s += (int)q8[ao + k]      * kvalue_iq4nl(b & 0xf);
            s += (int)q8[ao + 16 + k] * kvalue_iq4nl(b >> 4);
        }
        s += __shfl_down_sync(0xffffffff, s, 2);
        s += __shfl_down_sync(0xffffffff, s, 1);
        if (lane == 0) sumf += dh * (float)s;      // no ordered 8-way walk
    }
    if (lane == 0) out[(size_t)tok * n_out + j] = sumf;
}

// No weight loads: `qs` is replaced by a constant, so the only bytes read are
// the block header and the activation. Everything else is the real kernel.
extern "C" __global__ void dbg_iq4_noweight(int n_in, int n_out,
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

    const int t = lane >> 2, p = lane & 3;
    const int ib = (t >> 1) * 2, half = t & 1;
    const float *xs = x_scales + (size_t)tok * nb;
    const signed char *xq = x_quants + (size_t)tok * n_in;
    const unsigned char *row = w + (size_t)j * nb * IQ4XS_BYTES;

    float sumf = 0.0f;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const unsigned char *blk = row + (size_t)ibl * IQ4XS_BYTES;
        const float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        const unsigned int sh = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
        const unsigned char *scales_l = blk + 4;
        const signed char *q8 = xq + (size_t)ibl * QK_K;
        const unsigned int h  = sh >> (ib * 2);
        const unsigned int lo = scales_l[ib >> 1];
        const int ls = (half == 0) ? (int)((lo & 0xf) | ((h << 4) & 0x30))
                                   : (int)((lo >> 4)  | ((h << 2) & 0x30));
        const float dh = d * xs[ibl] * (float)(ls - 32);
        const int ao = ib * 32 + half * 32;
        int s = 0;
        for (int k = p * 4; k < p * 4 + 4; ++k) {
            const unsigned char b = 0x5a;                 // no qs load
            s += (int)q8[ao + k]      * kvalue_iq4nl(b & 0xf);
            s += (int)q8[ao + 16 + k] * kvalue_iq4nl(b >> 4);
        }
        s += __shfl_down_sync(0xffffffff, s, 2);
        s += __shfl_down_sync(0xffffffff, s, 1);
        const float term = dh * (float)s;
        for (int k = 0; k < 8; ++k) {
            const float v = __shfl_sync(0xffffffff, term, k * 4);
            if (lane == 0) sumf += v;
        }
    }
    if (lane == 0) out[(size_t)tok * n_out + j] = sumf;
}

// Keeps the nibble split and the 6-bit scale assembly, replaces only the
// `kvalues_iq4nl[...]` lookup with arithmetic on the nibble.
//
// **Constant memory is broadcast-optimised.** A `__constant__` read where lanes
// index different addresses is serialised into up to 32 transactions, and every
// lane here looks up a different nibble, twice per byte. If this variant is the
// fast one, the table is the cost and the fix is to hold it in registers.
extern "C" __global__ void dbg_iq4_notable(int n_in, int n_out,
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

    const int t = lane >> 2, p = lane & 3;
    const int ib = (t >> 1) * 2, half = t & 1;
    const float *xs = x_scales + (size_t)tok * nb;
    const signed char *xq = x_quants + (size_t)tok * n_in;
    const unsigned char *row = w + (size_t)j * nb * IQ4XS_BYTES;

    float sumf = 0.0f;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const unsigned char *blk = row + (size_t)ibl * IQ4XS_BYTES;
        const float d = h2f((unsigned short)blk[0] | ((unsigned short)blk[1] << 8));
        const unsigned int sh = (unsigned int)blk[2] | ((unsigned int)blk[3] << 8);
        const unsigned char *scales_l = blk + 4;
        const unsigned char *qs = blk + 4 + QK_K / 64;
        const signed char *q8 = xq + (size_t)ibl * QK_K;
        const unsigned int h  = sh >> (ib * 2);
        const unsigned int lo = scales_l[ib >> 1];
        const int ls = (half == 0) ? (int)((lo & 0xf) | ((h << 4) & 0x30))
                                   : (int)((lo >> 4)  | ((h << 2) & 0x30));
        const float dh = d * xs[ibl] * (float)(ls - 32);
        const int qo = ib * 16 + half * 16;
        const int ao = ib * 32 + half * 32;
        int s = 0;
        for (int k = p * 4; k < p * 4 + 4; ++k) {
            const unsigned char b = qs[qo + k];
            // arithmetic stand-in for the non-uniform grid, same shape of work
            s += (int)q8[ao + k]      * (((int)(b & 0xf) << 4) - 120);
            s += (int)q8[ao + 16 + k] * (((int)(b >> 4)  << 4) - 120);
        }
        s += __shfl_down_sync(0xffffffff, s, 2);
        s += __shfl_down_sync(0xffffffff, s, 1);
        const float term = dh * (float)s;
        for (int k = 0; k < 8; ++k) {
            const float v = __shfl_sync(0xffffffff, term, k * 4);
            if (lane == 0) sumf += v;
        }
    }
    if (lane == 0) out[(size_t)tok * n_out + j] = sumf;
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

// ---------------------------------------------------------------- f32 debug
//
// Decomposing `matmul_f32_t`, which measures 6.5 GB/s on a card that reads
// VRAM at 409.6. Each variant removes exactly one thing and is replayed at the
// real recorded geometry with the real recorded arguments, because the last
// three attempts to speed up a kernel by reasoning about its parts were wrong
// and the one decomposition was right in a single pass.
//
// Its own comment claims the ~40 us floor is "memory latency on a single
// thread" and that going below it "needs more loads in flight per output, which
// means splitting the reduction". The first half is a hypothesis these variants
// test. The second half is the interesting claim, and `matmul_f32_t_staged`
// below is the argument that it is false.

// No weight load. If this is most of the cost, the floor is the `wt` stream.
extern "C" __global__ void dbg_f32_now(int n_in, int n_out,
                                       const float *__restrict__ wt,
                                       const float *__restrict__ x,
                                       float *__restrict__ out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;
    const int tok = blockIdx.y;
    const float *xt = x + (size_t)tok * n_in;
    float sum = 0.0f;
    for (int k = 0; k < n_in; ++k) sum += 1.0f * xt[k];
    out[(size_t)tok * n_out + j] = sum;
}

// No activation load. `x[k]` is the same address for every thread, so this
// should be a broadcast and nearly free; if it is not, that is the finding.
extern "C" __global__ void dbg_f32_nox(int n_in, int n_out,
                                       const float *__restrict__ wt,
                                       const float *__restrict__ x,
                                       float *__restrict__ out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;
    const int tok = blockIdx.y;
    float sum = 0.0f;
    for (int k = 0; k < n_in; ++k) sum += wt[(size_t)k * n_out + j] * 1.0f;
    out[(size_t)tok * n_out + j] = sum;
}

// Neither load: the launch and the loop alone.
extern "C" __global__ void dbg_f32_noloads(int n_in, int n_out,
                                           const float *__restrict__ wt,
                                           const float *__restrict__ x,
                                           float *__restrict__ out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;
    const int tok = blockIdx.y;
    float sum = 0.0f;
    for (int k = 0; k < n_in; ++k) sum += 1.0f;
    out[(size_t)tok * n_out + j] = sum;
}

// Both loads, but the serial dependency broken into eight partials. Wrong
// answer by construction -- it is timing the f32 accumulation chain, which the
// exactness rule forbids changing, so this variant exists only to say how much
// that rule costs.
extern "C" __global__ void dbg_f32_nochain(int n_in, int n_out,
                                           const float *__restrict__ wt,
                                           const float *__restrict__ x,
                                           float *__restrict__ out) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;
    const int tok = blockIdx.y;
    const float *xt = x + (size_t)tok * n_in;
    float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < n_in; k += 8) {
#pragma unroll
        for (int u = 0; u < 8; ++u)
            s[u] += wt[(size_t)(k + u) * n_out + j] * xt[k + u];
    }
    out[(size_t)tok * n_out + j] = ((s[0] + s[1]) + (s[2] + s[3])) +
                                   ((s[4] + s[5]) + (s[6] + s[7]));
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
