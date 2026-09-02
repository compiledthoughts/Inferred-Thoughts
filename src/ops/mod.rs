//! Tensor primitives, behind a backend seam.
//!
//! Implementations: `naive` (scalar f32 Rust, the oracle we own), `par` (the
//! same kernels spread across threads), and, planned, `ggml` (CPU SIMD via
//! FFI, for the dense path) and `cuda` (ours, for the MoE expert path). Model
//! code is written against the [`Ops`] trait so adding a backend never touches
//! it.
//!
//! Shapes follow ggml's convention: a weight is `{ne0, ne1}` where **ne0 is the
//! contraction dimension**, so `{n_in, n_out}` maps `n_in -> n_out`, and rows of
//! length `n_in` are contiguous.

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod naive;
pub mod par;
pub mod pool;
pub mod spin;

use crate::gguf::GgmlType;

/// A weight matrix, left in whatever quantized form the file stores it.
///
/// Deliberately a borrow of the mmap rather than owned f32: materializing the
/// 9B's weights as f32 would need ~36 GB against 32 GB of RAM. See `CLAUDE.md`.
#[derive(Debug, Clone, Copy)]
pub struct Weights<'a> {
    pub data: &'a [u8],
    pub ty: GgmlType,
    /// Contraction dimension (`ne0`): the length of one row.
    pub n_in: usize,
    /// Number of rows (`ne1`): the output width.
    pub n_out: usize,
}

impl<'a> Weights<'a> {
    /// Bytes of row `j`. Rows are contiguous and each is a whole number of
    /// quantization blocks, which the GGUF parser enforces at load.
    pub fn row(&self, j: usize) -> &'a [u8] {
        let stride = self.ty.n_bytes(self.n_in as u64) as usize;
        &self.data[j * stride..(j + 1) * stride]
    }
}

/// One token's attention inputs.
///
/// A struct rather than ten positional arguments, which is how a `head_dim` and
/// an `n_head` end up swapped.
pub struct Attn<'a> {
    /// This token's queries, post-RoPE: `n_head * head_dim`.
    pub q: &'a [f32],
    /// The layer's whole key slab as f16 bits, position-major with stride
    /// `kv_dim`. Only `0..n_pos` is read.
    pub k: &'a [u16],
    /// The layer's value slab, same layout.
    pub v: &'a [u16],
    /// Distance in elements between consecutive positions.
    pub kv_dim: usize,
    /// Positions to attend over: `0..n_pos`, inclusive of this token, which is
    /// what applies the causal mask.
    pub n_pos: usize,
    pub head_dim: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    /// `1/sqrt(head_dim)`.
    pub scale: f32,
}

impl Attn<'_> {
    /// Query heads served by one key/value head.
    pub fn group(&self) -> usize {
        self.n_head / self.n_head_kv
    }
}

/// One token's inputs to the gated delta rule — GatedDeltaNet's recurrent core.
///
/// A struct for the same reason [`Attn`] is one: this has four dimensions and
/// two head counts that differ, and positional arguments are how a
/// `head_k_dim` and an `n_v_heads` end up swapped.
///
/// `alpha` and `beta` arrive **raw**, straight from their projections, and the
/// activations are applied inside the op. That keeps `softplus` and `sigmoid`
/// off the seam entirely: they act on `n_v_heads` values, which is 32 here, and
/// a seam method per elementwise function would be four more implementations
/// for arithmetic that costs nothing.
pub struct Delta<'a> {
    /// Queries, `n_k_heads * head_k_dim`, already convolved and l2-normalized.
    pub q: &'a [f32],
    /// Keys, same shape and same treatment.
    pub k: &'a [f32],
    /// Values, `n_v_heads * head_v_dim`, convolved but **not** normalized.
    pub v: &'a [f32],
    /// Raw `ssm_alpha @ x`, one per value head. `softplus(alpha + dt_bias)`
    /// scaled by `ssm_a` gives the log decay.
    pub alpha: &'a [f32],
    /// Raw `ssm_beta @ x`, one per value head. Passed through `sigmoid`.
    pub beta: &'a [f32],
    /// Per-head decay scale. Negative — it is `-A_log.exp()` upstream — which
    /// is what makes `exp(gate)` a value in `(0, 1)` rather than a blow-up.
    pub ssm_a: &'a [f32],
    /// Per-head bias added to `alpha` before `softplus`.
    pub dt_bias: &'a [f32],
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
}

impl Delta<'_> {
    /// The key/query head that value head `h` reads.
    ///
    /// **Modulo, not division.** llama.cpp reaches this two ways and both say
    /// so: the unfused path calls `ggml_repeat_4d`, which *tiles*, and the
    /// fused kernel writes `iq1 = iv1 % neq1` outright. Blocked grouping —
    /// `h / (n_v_heads / n_k_heads)`, which is what GQA does for attention in
    /// this same model — agrees only for `h = 0` and `h = 1`.
    pub fn key_head(&self, value_head: usize) -> usize {
        value_head % self.n_k_heads
    }

    /// `1/sqrt(head_k_dim)`, applied to `q` before the readout.
    pub fn scale(&self) -> f32 {
        1.0 / (self.head_k_dim as f32).sqrt()
    }

    /// Elements of recurrent state per value head: a `[key][value]` matrix.
    pub fn state_per_head(&self) -> usize {
        self.head_k_dim * self.head_v_dim
    }
}

/// Every primitive the qwen3 forward pass needs.
///
/// Methods write into caller-provided buffers so a backend never allocates on
/// the forward path.
pub trait Ops {
    /// `out = x / sqrt(mean(x^2) + eps) * weight`
    ///
    /// No `+1` on the weight — that is Gemma's variant, not Qwen's.
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]);

    /// RMSNorm applied independently to each `head_dim`-sized slice of `x`,
    /// in place. This is Qwen3's QK-norm: `weight` has length `head_dim` and is
    /// shared across heads.
    fn rms_norm_heads(&self, x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32);

    /// `out[j] = dot(w.row(j), x)` for every row, dequantizing as it goes.
    fn matmul(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]);

    /// NEOX-style rotary embedding, in place, over `n_heads` heads of
    /// `head_dim` each.
    ///
    /// NEOX pairs dimension `i` with `i + head_dim/2`, **not** with `i + 1`.
    /// `LLM_ARCH_QWEN3` sits under llama.cpp's "the pairs of head values are
    /// offset by n_rot/2" group.
    fn rope_neox(&self, x: &mut [f32], pos: usize, head_dim: usize, n_heads: usize, theta_base: f32);

    /// Numerically stable softmax, in place. Used by [`Ops::attend`]'s
    /// implementations, and by the MoE router when Stage 7 lands.
    fn softmax(&self, x: &mut [f32]);

    /// Scaled dot-product attention for one token against the cached history,
    /// all heads at once.
    ///
    /// **This is a whole-token op, not a per-head one, and that is the point.**
    /// Attention scoring is the only part of decode whose work grows with
    /// context, so it is the only part with enough work per call to pay for a
    /// parallel dispatch — `PARALLEL_THRESHOLD` in [`super::par`] records why
    /// the individual matmuls do not. Handing the backend every head at once
    /// lets it spread them across threads; handing it one head at a time would
    /// put the loop back in model code, where a backend cannot reach it.
    ///
    /// K and V arrive as **raw f16 bits**, deliberately: the seam must not
    /// depend on `KvCache`, or the ops layer would be coupled to the very type
    /// the project exists to iterate on.
    fn attend(&self, a: &Attn<'_>, out: &mut [f32]);

    /// `gate = silu(gate) * up`, in place — the SwiGLU nonlinearity.
    fn silu_mul(&self, gate: &mut [f32], up: &[f32]);

    /// L2 normalization per `head_dim`-sized slice, in place. No weight.
    ///
    /// **Not RMSNorm, despite looking like it.** `ggml_compute_forward_l2_norm_f32`
    /// scales by `1 / max(sqrt(sum(x^2)), eps)`: there is no division by `n`,
    /// and `eps` clamps the *norm* where RMSNorm adds it under the square root.
    /// Same-looking output, different function — which is why GatedDeltaNet's
    /// q and k get this and not [`Ops::rms_norm_heads`].
    fn l2_norm_heads(&self, x: &mut [f32], head_dim: usize, eps: f32);

    /// Depthwise causal conv1d over this layer's conv state and `x`, then
    /// `silu`, advancing the state.
    ///
    /// **The seam takes the state slab, not an assembled window**, for the same
    /// reason [`Ops::kv_write`] does: a layer's history belongs wherever that
    /// layer runs. A device backend keeps `state` in its own memory and never
    /// brings it home; if the model assembled the window instead, ~2 MB per
    /// layer would cross the bus every token to be handed straight back. It is
    /// also what makes a CPU/GPU layer split work without anything migrating —
    /// each backend sees only the slabs for its own layers.
    ///
    /// `state` is `[n_channels][kernel - 1]` with the **oldest sample first**,
    /// and the implementation both reads it and advances it: the window is the
    /// stored samples followed by `x`, and afterwards the state holds the last
    /// `kernel - 1` of that. That ordering is llama.cpp's — `build_conv_state`
    /// concatenates the stored state and this token along the time axis and
    /// keeps the tail — and it is the part that is easy to reverse.
    ///
    /// Depthwise means no mixing across channels: each of the 8192 channels has
    /// its own `kernel` weights and sees only its own history.
    ///
    /// The accumulator is **f32, deliberately**. `ggml_compute_forward_ssm_conv_f32`
    /// says so in a comment: "not using ggml_vec_dot_f32, because its sum is in
    /// double precision". Four taps, so nothing is lost, but the oracle has to
    /// agree with the reference rather than be better than it.
    fn ssm_conv(
        &self,
        state: &mut [f32],
        x: &[f32],
        weight: &[f32],
        kernel: usize,
        out: &mut [f32],
    );

    /// Copy one `chunk`-sized run out of every `stride` of `src`, starting at
    /// `offset`.
    ///
    /// Generic, but it exists for one shape: `qwen35`'s `attn_q` emits query
    /// and gate interleaved per head, so both are strided views of a single
    /// matmul result. Doing that split in model code would mean reading a
    /// device buffer on the host in the middle of a layer, which costs a round
    /// trip and makes the pass ungraphable.
    fn gather_chunks(
        &self,
        src: &[f32],
        chunk: usize,
        stride: usize,
        offset: usize,
        out: &mut [f32],
    );

    /// `x *= sigmoid(g)`, elementwise and in place. The sibling of
    /// [`Ops::silu_mul`], and inexact for the same reason: `expf`.
    fn sigmoid_mul(&self, x: &mut [f32], g: &[f32]);

    /// The gated delta rule for one token, every value head, state updated in
    /// place.
    ///
    /// Per value head `h`, with state `S` indexed `[value][key]` — key
    /// contiguous, matching ggml's `[S_v, S_v, H_v]` where `ne[0]` is the
    /// contraction axis:
    ///
    /// ```text
    /// g       = exp(softplus(alpha[h] + dt_bias[h]) * ssm_a[h])
    /// S      *= g                                  // scalar forget gate
    /// pred[j] = sum_i S[j][i] * k[i]               // what S currently predicts
    /// d[j]    = sigmoid(beta[h]) * (v[j] - pred[j])
    /// S[j][i] += k[i] * d[j]                       // rank-1 correction
    /// out[j]  = sum_i S[j][i] * q[i] / sqrt(head_k_dim)
    /// ```
    ///
    /// A per-head associative memory that *corrects* its stored value for the
    /// current key rather than merely accumulating it. Cost is
    /// `O(head_k_dim * head_v_dim)` per token and **independent of context
    /// length** — which is why 30 of the 35B's 40 layers need no KV cache at
    /// all, and why a 262,144-token context is plausible on this hardware.
    ///
    /// `state` is `n_v_heads * head_k_dim * head_v_dim` and is both read and
    /// written. `out` is `n_v_heads * head_v_dim`.
    fn delta_rule(&self, d: &Delta<'_>, state: &mut [f32], out: &mut [f32]);

    /// `a += b`, in place.
    fn add_assign(&self, a: &mut [f32], b: &[f32]);

    // ------------------------------------------------------- residency hints
    //
    // Three no-ops that exist for backends whose memory is not the caller's.
    //
    // A CPU backend reads and writes the very slices the model owns, so it
    // needs none of this and gets the defaults. A device backend keeps its own
    // copy, and then "the model wrote this" and "the model is about to read
    // this" stop being free facts — they are exactly the moments the two copies
    // have to agree. Naming those moments is what lets a device keep an
    // activation resident across a whole layer instead of shipping it home
    // after every operation.
    //
    // Deliberately *hints about the host*, not a buffer abstraction. Who owns
    // an activation, and whether a tensor handle should replace `&[f32]`
    // wholesale, is a larger question; this is the smallest thing that lets the
    // GPU stop round-tripping without answering it.

    /// The model wrote `buf` directly. Any device copy is now stale.
    fn host_wrote(&self, _buf: &[f32]) {}

    /// The model is about to read `buf`. Bring back whatever the device has.
    fn host_needs(&self, _buf: &mut [f32]) {}

    /// A forward pass over `n_tokens` is starting. Activation buffers are
    /// allocated per pass, so an address seen last time may be a different
    /// buffer now; anything remembered about host addresses must be dropped.
    ///
    /// `n_tokens` is passed because a backend may treat single-token decode
    /// differently from prefill — the GPU records it as a CUDA graph, which is
    /// only sound when the kernel sequence is fixed.
    fn begin_pass(&self, _n_tokens: usize) {}

    /// The pass is finished issuing work. A backend that batched it has to be
    /// told when to run it; the model must not read a result before this.
    fn end_pass(&self) {}

    /// Round `src` to f16 and write it into `slab` at `offset` elements in.
    ///
    /// This is how K and V enter the cache. It goes through the seam rather
    /// than being done by the model because *where* the cache lives follows
    /// where the layer runs: a CPU layer's history belongs in host RAM, a GPU
    /// layer's in VRAM. Passing a slab and an offset rather than a `KvCache`
    /// keeps the seam uncoupled from the type this project exists to iterate
    /// on.
    ///
    /// A device backend writes into its own copy and leaves `slab` untouched,
    /// so a host reader must go through [`Ops::host_needs`] first.
    /// The engine is starting a fresh sequence; forget any device copy of
    /// recurrent state.
    ///
    /// Needed because a device backend owns that state once it has touched it:
    /// nothing brings it home, so zeroing the host slab is invisible to it.
    /// The KV cache does not need this hint because its mirror tracks
    /// positions, and a shorter one than last time already means a reset.
    fn forget_state(&self) {}

    fn kv_write(&self, slab: &mut [u16], offset: usize, src: &[f32]) {
        for (d, &s) in slab[offset..offset + src.len()].iter_mut().zip(src) {
            *d = crate::quant::half::f32_to_f16(s);
        }
    }
}
