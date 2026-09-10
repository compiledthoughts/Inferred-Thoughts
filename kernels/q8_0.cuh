// q8_0.cuh -- Q8_0: the matmuls, the activation quantizer and the f16 KV write.
//
// Included by kernels.cu, which pulls every file into one compile unit and one
// PTX module in dependency order. Kernel names are the PTX symbols the backend
// looks up, so which file a kernel lives in changes nothing at run time.
// Kernels are in `extern "C"` blocks; templates sit outside them, since a
// template cannot have C linkage.
#pragma once

extern "C" {

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

}  // extern "C"
