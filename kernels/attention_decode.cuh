// attention_decode.cuh -- decode attention: attn_decode's scalar, 16-slot and 8-slot modes.
//
// Included by kernels.cu, which pulls every file into one compile unit and one
// PTX module in dependency order. Kernel names are the PTX symbols the backend
// looks up, so which file a kernel lives in changes nothing at run time.
// Kernels are in `extern "C"` blocks; templates sit outside them, since a
// template cannot have C linkage.
#pragma once

extern "C" {

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

}  // extern "C"
