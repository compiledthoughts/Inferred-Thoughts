// The two routed-FFN kernels whose memory is sized by the pick count, compiled
// once per capacity (SSD-TIER.md D18).
//
// **Included twice by `moe.cuh`, never on its own, so there is no include
// guard.** Before each include it defines:
//
//   MOE_CAP                 8 or 10: the most picks per token the kernel holds
//   MOE_TOPK_NAME           moe_topk       | moe_topk_k10
//   MOE_TOPK_SMEM           moe_topk_smem  | moe_topk_k10_smem
//   ADD_SCALED_ROWS_NAME    add_scaled_rows | add_scaled_rows_k10
//
// The capacity-8 instantiation keeps the original names, and after
// preprocessing its text is the kernel it replaced: the 35B launches the same
// source it did before this file existed. The comments on each kernel stay in
// `moe.cuh`, above the includes.
//
// Every other routed-FFN kernel uses `n_used` only as a loop bound or a
// divisor, so it needs no capacity: the grouped matmuls' tiles (`MOE_TOK`,
// `MOE_MMA_TOK`) count (token, pick) pairs, not picks.

extern "C" __global__ void MOE_TOPK_NAME(int n_expert, int n_used,
                                         const float *__restrict__ all_probs,
                                         int *__restrict__ all_ids,
                                         float *__restrict__ all_weights) {
    extern __shared__ unsigned char MOE_TOPK_SMEM[];
    float *sv = (float *)MOE_TOPK_SMEM;
    int *si = (int *)(sv + blockDim.x);

    // One block per token. Selection is independent per row -- a token's
    // experts depend only on its own probabilities -- so a prefill batch is
    // just a wider grid, and decode is `gridDim.x == 1` of the same kernel.
    const int tok = blockIdx.x;
    const float *probs = all_probs + (size_t)tok * n_expert;
    int *ids = all_ids + (size_t)tok * n_used;
    float *weights = all_weights + (size_t)tok * n_used;
    // `TopK` in `ops/cuda/ops/moe.rs`: the host refuses a launch with more
    // picks than this, since every one lands here.
    __shared__ int picked[MOE_CAP];

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

extern "C" __global__ void ADD_SCALED_ROWS_NAME(int n, int n_rows, float s0, float s1,
                                                float s2, float s3, float s4, float s5,
                                                float s6, float s7,
#if MOE_CAP == 10
                                                float s8, float s9,
#endif
                                                const float *__restrict__ rows,
                                                float *__restrict__ acc) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
#if MOE_CAP == 10
    const float sc[10] = {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9};
#else
    const float sc[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
#endif
    float a = 0.0f;
    for (int e = 0; e < n_rows; ++e) a += sc[e] * rows[(size_t)e * n + j];
    acc[j] = a;
}
