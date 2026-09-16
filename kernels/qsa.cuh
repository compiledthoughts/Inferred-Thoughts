// qsa.cuh -- Qwen Sparse Attention past its budget (src/model/qwen4exp.md, QSA).
//
// Included by kernels.cu, which pulls every file into one compile unit and one
// PTX module in dependency order. Kernel names are the PTX symbols the backend
// looks up, so which file a kernel lives in changes nothing at run time.
// Nothing the 35B runs launches any of these.
#pragma once

extern "C" {

// K and V of the kept cells, gathered into a dense window: window row i is
// cache row cells[i]. Decode then runs attn_decode, unchanged, over at most
// top_k + ratio - 1 = 2,051 rows rather than the whole cache -- the "gather"
// arm of the Q2 decode fork, priced by `what_the_qsa_gather_costs` before it is
// chosen. A copy of f16 bits, so exact. One thread per window row;
// whole rows are copied sixteen bytes per load.
extern "C" __global__ void qsa_gather_kv(int n_cells, int kv_dim,
                                         const unsigned int *__restrict__ cells,
                                         const unsigned short *__restrict__ k,
                                         const unsigned short *__restrict__ v,
                                         unsigned short *__restrict__ kw,
                                         unsigned short *__restrict__ vw) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;
    const size_t src = (size_t)cells[i] * kv_dim;
    const size_t dst = (size_t)i * kv_dim;
    // Sixteen bytes per load, as stage16_rows stages K and V: a row of 512 f16
    // is 64 loads rather than 512. `kv_dim` is a multiple of 8 on every model
    // here; anything left over is copied one value at a time.
    const int wide = kv_dim & ~7;
    for (int d = 0; d < wide; d += 8)
        *(int4 *)(kw + dst + d) = *(const int4 *)(k + src + d);
    for (int d = 0; d < wide; d += 8)
        *(int4 *)(vw + dst + d) = *(const int4 *)(v + src + d);
    for (int d = wide; d < kv_dim; ++d) {
        kw[dst + d] = k[src + d];
        vw[dst + d] = v[src + d];
    }
}

// Pool, RMSNorm and rope blocks from .. from + n_new of one layer's raw f16
// indexer keys into its pooled lane, in place: ops::naive::qsa_pool, statement
// for statement. The mean is the ggml_add chain then the scale; the RMSNorm is
// the oracle's serial f64 sum of f32 squares, (x * scale) * w; the rotation
// reads a host-built f64 table with one row of `half` per block (position
// b * ratio), as rope_neox's does. --fmad=false keeps every product separate,
// so this is bit-identical. One thread per block; n_new may be 0, so decode
// launches it every pass.
extern "C" __global__ void qsa_pool(int from, int n_new, int ratio, int dim, int half,
                                    float inv, float eps,
                                    const unsigned short *__restrict__ raw,
                                    float *__restrict__ pooled,
                                    const float *__restrict__ w,
                                    const float *__restrict__ cosv,
                                    const float *__restrict__ sinv) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_new) return;
    const int b = from + i;
    const size_t first = (size_t)b * ratio;
    float *key = pooled + (size_t)b * dim;
    for (int d = 0; d < dim; ++d) {
        float acc = h2f(raw[first * dim + d]);
        for (int m = 1; m < ratio; ++m) acc += h2f(raw[(first + m) * dim + d]);
        key[d] = acc * inv;
    }
    double sum = 0.0;
    for (int d = 0; d < dim; ++d) sum += (double)(key[d] * key[d]);
    const float mean = (float)(sum / (double)dim);
    const float scale = 1.0f / sqrtf(mean + eps);
    for (int d = 0; d < dim; ++d) key[d] = key[d] * scale * w[d];
    const float *c = cosv + (size_t)b * half;
    const float *s = sinv + (size_t)b * half;
    for (int j = 0; j < half; ++j) {
        const float x0 = key[j];
        const float x1 = key[j + half];
        key[j] = x0 * c[j] - x1 * s[j];
        key[j + half] = x0 * s[j] + x1 * c[j];
    }
}

// QSA's block scores, out[t][b] = sum_h relu(q[t][h] . pooled[b]):
// ops::naive::qsa_scores -- each head's dot a serial f64 sum of f32 products,
// the heads summed in f32 in order from +0.0. One thread per (query, block), so
// nothing is reordered: bit-identical. The grid is blocks by query rows;
// n_blocks may be 0 early in a decode, which returns at once.
extern "C" __global__ void qsa_scores(int n_q, int n_blocks, int n_head, int dim,
                                      const float *__restrict__ q,
                                      const float *__restrict__ pooled,
                                      float *__restrict__ out) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (b >= n_blocks || t >= n_q) return;
    const float *k = pooled + (size_t)b * dim;
    float sum = 0.0f;
    for (int h = 0; h < n_head; ++h) {
        const float *qh = q + ((size_t)t * n_head + h) * dim;
        double dot = 0.0;
        for (int d = 0; d < dim; ++d) dot += (double)(qh[d] * k[d]);
        const float f = (float)dot;
        sum += (f > 0.0f) ? f : 0.0f;
    }
    out[(size_t)t * n_blocks + b] = sum;
}

// Each query row's cells, ascending, at `stride` per row: ops::naive::select_cells.
// While the query sees `budget` or fewer whole blocks it keeps every visible
// cell. Otherwise the best `budget` blocks, ties to the lower block, found
// without a sort: a radix select over the scores' bit patterns, eight bits at a
// time from the top. Scores are sums of ReLUs from +0.0, so never negative, and
// for those the IEEE bit pattern orders exactly as the value. That yields T, the
// budget-th best key, and how many blocks equal to T the budget still has room
// for; every block above T is kept, and the lowest-index ones equal to T. Then
// the tail, the cells after the last whole block. One thread per row, serial
// within it; exact, so the chosen cells equal the oracle's.
extern "C" __global__ void qsa_select_cells(int n_q, int n_blocks, int start_pos,
                                            int ratio, int budget, int stride,
                                            const float *__restrict__ scores,
                                            unsigned int *__restrict__ cells) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_q) return;
    const int visible = start_pos + t + 1;
    const int full = visible / ratio;
    const float *sc = scores + (size_t)t * n_blocks;
    unsigned int *out = cells + (size_t)t * stride;
    int n = 0;
    if (full <= budget) {
        for (int c = 0; c < visible; ++c) out[n++] = (unsigned int)c;
        return;
    }
    unsigned int prefix = 0, mask = 0;
    int want = budget;
    int hist[256];
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int d = 0; d < 256; ++d) hist[d] = 0;
        for (int b = 0; b < full; ++b) {
            const unsigned int key = *(const unsigned int *)&sc[b];
            if ((key & mask) == prefix) hist[(key >> shift) & 255] += 1;
        }
        int d = 255;
        for (; d > 0; --d) {
            if (hist[d] >= want) break;
            want -= hist[d];
        }
        prefix |= (unsigned int)d << shift;
        mask |= 255u << shift;
    }
    int taken = 0;
    for (int b = 0; b < full; ++b) {
        const unsigned int key = *(const unsigned int *)&sc[b];
        bool keep = key > prefix;
        if (key == prefix && taken < want) {
            keep = true;
            taken += 1;
        }
        if (!keep) continue;
        const int first = b * ratio;
        for (int m = 0; m < ratio; ++m) out[n++] = (unsigned int)(first + m);
    }
    for (int c = full * ratio; c < visible; ++c) out[n++] = (unsigned int)c;
}

}  // extern "C"
