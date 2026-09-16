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

}  // extern "C"
