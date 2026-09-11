// diagnostics.cuh -- stopwatches, not modes: the dbg_ decompositions and bench_ kernels; their outputs are wrong by design.
//
// Included by kernels.cu, which pulls every file into one compile unit and one
// PTX module in dependency order. Kernel names are the PTX symbols the backend
// looks up, so which file a kernel lives in changes nothing at run time.
// Kernels are in `extern "C"` blocks; templates sit outside them, since a
// template cannot have C linkage.
#pragma once

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

// ---------------------------------------------------- tensor-core ceilings
//
// The tensor-core ceilings, measured rather than quoted: `mma` back to back on
// constant register operands, with no memory in the loop. `BENCHMARKS.md`'s
// 09-09 (ceilings) took int8 and fp16 this way with a harness that was never
// committed; these re-take both beside the NVFP4 form, so all three come from
// one sitting and one method. Each thread accumulates in place and never stores:
// the instruction is the only work, and its time is the point.
//
// One instruction is 16 x 8 x k element products -- 4,096 for int8 `m16n8k32`,
// 2,048 for fp16 `m16n8k16`, 8,192 for NVFP4 `m16n8k64` -- which is the unit
// `what_the_fp4_tensor_cores_can_do` divides by. Operands are small and nonzero,
// so no accumulator overflows at any iteration count the caller uses and no
// all-zero shortcut is available that a real kernel would not get.

// **Every ceiling kernel stores its accumulators once, at the end.** The first
// version stored nothing and timed 1.5M int8 iterations over 1,728 warps in
// 7.6 us -- 7,353x the recorded ceiling -- so its loop cannot have run. The store
// makes the work observable, and `what_the_fp4_tensor_cores_can_do` checks every
// accumulator is exactly `iters` times what one iteration leaves before it
// reports a rate.
__global__ void bench_mma_ceiling_s8(int iters, int *__restrict__ out) {
    // s8 1 and 2, four to a register.
    const int a[4] = {0x02010201, 0x01020102, 0x02010201, 0x01020102};
    const int b[2] = {0x01020102, 0x02010201};
    int d[4] = {0, 0, 0, 0};
    for (int n = 0; n < iters; ++n) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < 4; ++i) out[t * 4 + i] = d[i];
}

__global__ void bench_mma_ceiling_f16(int iters, float *__restrict__ out) {
    // f16 0.5 (0x3800), two to a register.
    const unsigned a[4] = {0x38003800u, 0x38003800u, 0x38003800u, 0x38003800u};
    const unsigned b[2] = {0x38003800u, 0x38003800u};
    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int n = 0; n < iters; ++n) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    }
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < 4; ++i) out[t * 4 + i] = d[i];
}

// The NVFP4 block-scaled form, as llama.cpp's `mma_block_scaled_fp4` issues it
// (`ggml-cuda/mma.cuh`), which matches the PTX ISA 9.4 grammar in 9.7.16.3:
// `d, a, b, c, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data,
// {byte-id-b, thread-id-b}`, byte-ids 0 as `scale_vec::4X` requires.
//
// **Built only for `sm_120a`.** `sm_120` is the forwards-compatible target and
// leaves this instruction out: CUDA 12.8's `ptxas -arch=sm_120` rejects it --
// "Instruction 'mma with block scale' not supported on .target 'sm_120'", and
// the same for `.kind::mxf4nvf4`, `.block_scale` and `.scale_vec::4X` -- and the
// driver then refuses the whole module, not just these kernels. PTX ISA 9.4
// lists `.kind::mxf4nvf4` for `sm_120a` and `sm_121a`; `ptxas -arch=sm_120a`
// accepts it, and this card's driver loads the result. `build.rs` defines
// `INFERRED_NVFP4_BLOCK_SCALE` when built with `INFERRED_SM_ARCH=sm_120a`.
#ifdef INFERRED_NVFP4_BLOCK_SCALE
__device__ __forceinline__ void mma_nvfp4_inplace(
        float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2],
        unsigned sa, unsigned sb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sa), "r"(sb));
}

__global__ void bench_mma_ceiling_nvfp4(int iters, float *__restrict__ out) {
    // E2M1 0.5 (code 1) in every nibble; every UE4M3 scale 1.0 (0x38).
    const unsigned a[4] = {0x11111111u, 0x11111111u, 0x11111111u, 0x11111111u};
    const unsigned b[2] = {0x11111111u, 0x11111111u};
    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int n = 0; n < iters; ++n) {
        mma_nvfp4_inplace(d, a, b, 0x38383838u, 0x38383838u);
    }
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < 4; ++i) out[t * 4 + i] = d[i];
}

// Operands `check_mma_nvfp4` derives from a lane and an element index, mirrored
// exactly by `the_nvfp4_block_scaled_mma_follows_the_isa`. E2M1 codes 1..15, so
// every sign and magnitude appears; UE4M3 scales 0x30/0x38/0x40/0x48 (0.5, 1, 2,
// 4), permuted per lane and per chunk so that a wrong byte-to-chunk or
// thread-to-row mapping cannot cancel.
__device__ __forceinline__ unsigned fp4_check_a(int lane, int i) {
    return 1u + (unsigned)((lane * 7 + i * 3) % 15);
}

__device__ __forceinline__ unsigned fp4_check_b(int lane, int i) {
    return 1u + (unsigned)((lane * 5 + i * 11) % 15);
}

__device__ __forceinline__ unsigned fp4_check_scales(int lane, int mul, int add) {
    unsigned v = 0;
    for (int c = 0; c < 4; ++c) {
        v |= (0x30u + 8u * (unsigned)((lane * mul + c + add) % 4)) << (8 * c);
    }
    return v;
}

// One NVFP4 `mma` over a single warp, each lane storing its four accumulators
// at `lane * 4`. A test instrument: see `Cuda::check_mma_nvfp4`.
__global__ void check_mma_nvfp4(float *__restrict__ out) {
    const int lane = threadIdx.x & 31;
    unsigned a[4], b[2];
    for (int r = 0; r < 4; ++r) {
        unsigned v = 0;
        for (int i = 0; i < 8; ++i) v |= fp4_check_a(lane, r * 8 + i) << (4 * i);
        a[r] = v;
    }
    for (int r = 0; r < 2; ++r) {
        unsigned v = 0;
        for (int i = 0; i < 8; ++i) v |= fp4_check_b(lane, r * 8 + i) << (4 * i);
        b[r] = v;
    }
    float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    mma_nvfp4_inplace(d, a, b, fp4_check_scales(lane, 1, 0), fp4_check_scales(lane, 3, 1));
    for (int i = 0; i < 4; ++i) out[lane * 4 + i] = d[i];
}
#endif  // INFERRED_NVFP4_BLOCK_SCALE

}  // extern "C"
