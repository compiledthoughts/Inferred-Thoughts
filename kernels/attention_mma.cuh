// attention_mma.cuh -- attention on the fp16 tensor cores: attn_flash_mma, attn_flash_mma_v and the body template decode reuses.
//
// Included by kernels.cu, which pulls every file into one compile unit and one
// PTX module in dependency order. Kernel names are the PTX symbols the backend
// looks up, so which file a kernel lives in changes nothing at run time.
// Kernels are in `extern "C"` blocks; templates sit outside them, since a
// template cannot have C linkage.
#pragma once

extern "C" {

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

}  // extern "C"

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
