// attention.cuh -- scalar attention: attn_flash, its combine, the body template decode reuses, and the fused variant.
//
// Included by kernels.cu, which pulls every file into one compile unit and one
// PTX module in dependency order. Kernel names are the PTX symbols the backend
// looks up, so which file a kernel lives in changes nothing at run time.
// Kernels are in `extern "C"` blocks; templates sit outside them, since a
// template cannot have C linkage.
#pragma once

extern "C" {

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

}  // extern "C"

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

extern "C" {

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

}  // extern "C"
