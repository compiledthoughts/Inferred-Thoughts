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

/// Whether a backend can run routed experts stored as `gate`, `up` and `down`.
///
/// **Checked once at load**, because a missing kernel was only recorded per call:
/// a GitHub user's Q4_K / Q5_1 quant streamed garbage while `generate` kept going.
/// The CPU backends run every format their kernels implement (`quant` and
/// `naive`'s matmul). The CUDA backend has grouped routed-expert kernels for two:
/// gate and up together as IQ4_XS or as NVFP4 (`moe_glu`), and down as either.
pub fn check_expert_types(
    gate: GgmlType,
    up: GgmlType,
    down: GgmlType,
    on_cuda: bool,
) -> crate::error::Result<()> {
    use GgmlType::*;
    let ok = if on_cuda {
        matches!((gate, up), (Iq4Xs, Iq4Xs) | (Nvfp4, Nvfp4)) && matches!(down, Iq4Xs | Nvfp4)
    } else {
        [gate, up, down]
            .iter()
            .all(|t| matches!(t, F32 | F16 | Bf16 | Q8_0 | Q5K | Q6K | Iq4Xs | Nvfp4))
    };
    if ok {
        return Ok(());
    }
    Err(crate::error::Error::UnsupportedExperts {
        found: format!("{} (gate) / {} (up) / {} (down)", gate.name(), up.name(), down.name()),
        backend: if on_cuda { "--backend cuda" } else { "the CPU backends" },
        supported: if on_cuda {
            "IQ4_XS or NVFP4 experts, gate and up of the same type"
        } else {
            "F32, F16, BF16, Q8_0, Q5_K, Q6_K, IQ4_XS and NVFP4"
        },
    })
}

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
    /// This tensor is one of many interchangeable ones drawn from a pool too
    /// large to keep resident — an MoE expert.
    ///
    /// **A fact about the tensor, not a policy.** It says only "there are
    /// thousands of these and they will not all fit", which is what lets a
    /// device backend put it in a bounded cache instead of the
    /// upload-once-keep-forever mirror. *Which* of them are resident, and when
    /// they move, stays above this type — that is the distinction the project
    /// exists to make.
    ///
    /// Set only by [`Experts::expert`]. Everything else is `false`, including
    /// the shared expert, which is one tensor per layer and belongs in the
    /// permanent mirror like any other weight.
    pub pooled: bool,
}

impl<'a> Weights<'a> {
    /// Bytes of row `j`. Rows are contiguous and each is a whole number of
    /// quantization blocks, which the GGUF parser enforces at load.
    pub fn row(&self, j: usize) -> &'a [u8] {
        let stride = self.ty.n_bytes(self.n_in as u64) as usize;
        &self.data[j * stride..(j + 1) * stride]
    }
}

/// A stack of `n_expert` weight matrices stored as one 3-D tensor.
///
/// The MoE expert tensors are `{n_in, n_out, n_expert}` in ggml order —
/// `ffn_gate_exps` on the 35B is `2048 x 512 x 256`. Expert `e` is a
/// **contiguous** run of `n_out` rows, so it can be handed to an ordinary
/// [`Weights`] with no copy and no change to the matmul.
///
/// That is the whole reason the seam does not grow a "MoE matmul": routing
/// picks 8 of 256 experts, and each pick becomes a normal matmul over a
/// borrowed sub-range. Which experts move, and when, is a *policy* question
/// that lives above this type — which is the distinction the project exists to
/// make.
#[derive(Debug, Clone, Copy)]
pub struct Experts<'a> {
    pub data: &'a [u8],
    pub ty: GgmlType,
    /// Contraction dimension of one expert.
    pub n_in: usize,
    /// Output width of one expert.
    pub n_out: usize,
    pub n_expert: usize,
    /// Per-expert second scale: `n_expert` little-endian f32, multiplied into
    /// that expert's matmul output (`ffn_*_exps.scale`, NVFP4 only, applied as
    /// llama.cpp's `build_moe_ffn` does). **Empty when the file has none**,
    /// which means 1.0.
    pub scale: &'a [u8],
}

impl<'a> Experts<'a> {
    /// Expert `e`'s second scale, 1.0 when the tensor has none.
    pub fn scale_of(&self, e: usize) -> f32 {
        match self.scale.get(e * 4..e * 4 + 4) {
            Some(b) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            None => 1.0,
        }
    }

    /// Bytes one expert occupies: `n_out` rows of `n_in` quantized elements.
    pub fn stride(&self) -> usize {
        self.ty.n_bytes(self.n_in as u64) as usize * self.n_out
    }

    /// Expert `e` as an ordinary weight matrix, borrowed from the mmap.
    ///
    /// Panics only on an out-of-range index, which is a routing bug rather than
    /// a data condition — the router cannot emit an id it was not given a
    /// column for.
    pub fn expert(&self, e: usize) -> Weights<'a> {
        debug_assert!(e < self.n_expert);
        let stride = self.stride();
        Weights {
            data: &self.data[e * stride..(e + 1) * stride],
            ty: self.ty,
            n_in: self.n_in,
            n_out: self.n_out,
            pooled: true,
        }
    }
}

/// Where a token's expert choice lives.
///
/// **The seam's answer to a fork that would otherwise be in the model.** Expert
/// selection is a *decision*, not arithmetic: it changes which weights are read
/// rather than what is computed from them. Every CPU backend makes it on the
/// host, and so did the CUDA backend until the router's probabilities could
/// stay on the card — which they must, because a CUDA graph replays a fixed
/// kernel sequence with no host participation, so a mid-pass read of a device
/// result returns the *previous* pass's contents.
///
/// Putting the variant here rather than in `moe_token` keeps the model on one
/// path: it asks [`Ops::route`] and hands the answer back to the three ops that
/// consume it, without knowing or caring which side chose.
///
/// # Contract
///
/// The default [`Ops::route`] only ever returns `Host`, so a backend that
/// overrides nothing is correct. **A backend that returns `Device` must
/// override [`Ops::matmul_experts`], [`Ops::moe_glu`] and [`Ops::moe_finish`]**,
/// because their defaults have no way to read ids that never came home. The
/// whole-model differential test is what enforces it.
#[derive(Debug, Clone)]
pub enum Route {
    /// Chosen on the host: `n_tok * n_used` expert ids, token-major, each
    /// token's in descending probability, and their normalized weights in the
    /// same order.
    Host { ids: Vec<usize>, weights: Vec<f32>, n_used: usize },
    /// Chosen on the device. The backend knows where the ids and weights are;
    /// nothing above the seam does.
    Device { n_used: usize, n_tok: usize },
}

impl Route {
    /// How many experts each token visits, whichever side chose them.
    pub fn n_used(&self) -> usize {
        match self {
            Route::Host { n_used, .. } => *n_used,
            Route::Device { n_used, .. } => *n_used,
        }
    }

    /// Tokens this route covers. One in decode; the whole batch in prefill.
    pub fn n_tok(&self) -> usize {
        match self {
            Route::Host { ids, n_used, .. } => ids.len() / (*n_used).max(1),
            Route::Device { n_tok, .. } => *n_tok,
        }
    }

    /// The chosen ids, if the host has them.
    pub fn ids(&self) -> Option<&[usize]> {
        match self {
            Route::Host { ids, .. } => Some(ids),
            Route::Device { .. } => None,
        }
    }

    /// The normalized weights, if the host has them.
    pub fn weights(&self) -> Option<&[f32]> {
        match self {
            Route::Host { weights, .. } => Some(weights),
            Route::Device { .. } => None,
        }
    }
}

/// The attention inputs for a batch of queries against one layer's history.
///
/// A struct rather than ten positional arguments, which is how a `head_dim` and
/// an `n_head` end up swapped.
///
/// **`q` may hold several consecutive query rows**, which is what makes prefill
/// one call instead of one per token. The count is derived from `q.len()` (see
/// [`Attn::n_q`]) rather than passed, so it cannot disagree with the buffer.
pub struct Attn<'a> {
    /// Queries, post-RoPE: [`Attn::n_q`] consecutive rows of `n_head *
    /// head_dim`, row `t` belonging to absolute position `n_pos - n_q + 1 + t`.
    pub q: &'a [f32],
    /// The layer's whole key slab as f16 bits, position-major with stride
    /// `kv_dim`. Only `0..n_pos` is read.
    pub k: &'a [u16],
    /// The layer's value slab, same layout.
    pub v: &'a [u16],
    /// Distance in elements between consecutive positions.
    pub kv_dim: usize,
    /// Positions the **last** query row attends over: `0..n_pos`, inclusive of
    /// itself. Earlier rows in the batch see proportionally fewer, which is
    /// what applies the causal mask — see [`Attn::n_pos_of`].
    ///
    /// Defined against the last row rather than the first so that decode, where
    /// `n_q == 1`, reads exactly as it always did: `n_pos = pos + 1`.
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

    /// Query rows in this call. Derived from the buffer, never passed.
    ///
    /// One in decode; the whole prompt in a batched prefill. Deriving it means
    /// a shape error is a failed division rather than a silently wrong mask.
    pub fn n_q(&self) -> usize {
        debug_assert_eq!(self.q.len() % (self.n_head * self.head_dim), 0);
        self.q.len() / (self.n_head * self.head_dim)
    }

    /// Positions query row `t` attends over — **the causal mask**.
    ///
    /// Row `t` is at absolute position `n_pos - n_q + t`, and attends to
    /// everything up to and including itself. At `n_q == 1` this is `n_pos`,
    /// so decode is unchanged.
    pub fn n_pos_of(&self, t: usize) -> usize {
        self.n_pos - self.n_q() + 1 + t
    }
}

/// What [`Ops::qsa_pool`] pools: blocks `from..to` of `ratio` cells of `dim`.
pub struct QsaPool<'a> {
    pub from: usize,
    pub to: usize,
    pub ratio: usize,
    pub dim: usize,
    /// `indexer.k_norm`, `dim` wide.
    pub norm: &'a [f32],
    pub eps: f32,
    pub n_rot: usize,
    pub theta: f32,
}

/// The shape of one QSA selection: this pass's query rows against its pooled
/// blocks, and the budget they are chosen under.
#[derive(Clone, Copy, Debug)]
pub struct QsaSelect {
    /// Indexer heads per query row, each `dim` wide.
    pub n_head: usize,
    pub dim: usize,
    /// Blocks pooled: every block whole at the pass's last position.
    pub n_blocks: usize,
    /// Row 0's position; rows are consecutive.
    pub start_pos: usize,
    pub ratio: usize,
    /// Whole blocks kept, `top_k / ratio`.
    pub budget: usize,
}

impl QsaSelect {
    /// Cells per row in the selection buffer: the most any query keeps,
    /// `budget * ratio` plus a tail of up to `ratio - 1`.
    pub fn stride(&self) -> usize {
        self.budget * self.ratio + self.ratio - 1
    }

    /// Cells the query at `pos` keeps: every visible cell while it can see
    /// `budget` or fewer whole blocks, otherwise `budget` blocks and its tail.
    pub fn count(&self, pos: usize) -> usize {
        let visible = pos + 1;
        if visible / self.ratio <= self.budget {
            visible
        } else {
            self.budget * self.ratio + visible % self.ratio
        }
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

/// A device backend's cumulative tier-3 counters and its VRAM, as
/// [`Ops::tier_counters`] reports them. Microsecond fields are host wall time
/// spent in that part of the fetch path; `picks_wait_us` is the host waiting on
/// the device before it can read a layer's picks, so it is GPU time.
#[derive(Debug, Clone, Copy, Default)]
pub struct TierCounters {
    /// Experts read from the model file into a VRAM slot, and their bytes.
    pub fetched: u64,
    pub fetch_bytes: u64,
    pub read_us: u64,
    pub upload_us: u64,
    pub writes_us: u64,
    pub picks_wait_us: u64,
    pub picks_copy_us: u64,
    pub prefetch_reads: u64,
    pub prefetch_used: u64,
    /// Expert reads by kernels, and how many of them crossed PCIe to the host tier.
    pub lookups: u64,
    pub host_reads: u64,
    pub slot_bytes: u64,
    /// Device memory free and total, from the driver, at the snapshot.
    pub vram_free: u64,
    pub vram_total: u64,
    /// Device copies of activation buffers held, and their bytes. They are
    /// keyed on host addresses and kept for the life of the process.
    pub mirrors: u64,
    pub mirror_bytes: u64,
    /// The expert pool does not fit in VRAM plus the host tier, so experts are
    /// fetched from the file and migration between tiers is paused.
    pub oversubscribed: bool,
    /// VRAM slots in the expert slab, each `slot_bytes`.
    pub slots: u64,
    /// Everything the backend itself accounts for on the device, by category:
    /// weights, KV (with QSA's lanes), activation copies and their Q8_0 forms,
    /// the scratch pool and the expert slab. The driver's `vram_total -
    /// vram_free` less this is what the backend does not see.
    pub resident_bytes: u64,
    pub weight_bytes: u64,
    pub kv_bytes: u64,
    pub quant_bytes: u64,
    pub pool_bytes: u64,
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

    /// Tokens in this call. Derived from the buffers, like [`Attn::n_q`].
    pub fn n_tokens(&self) -> usize {
        let per = self.n_v_heads * self.head_v_dim;
        debug_assert_eq!(self.v.len() % per, 0);
        self.v.len() / per
    }

    /// Token `t` of the batch as a single-token [`Delta`].
    ///
    /// The scan has to be applied in order, so every implementation walks the
    /// batch this way. Sharing one view function means a backend cannot get the
    /// per-token striding subtly different from the oracle's.
    pub fn row(&self, t: usize) -> Delta<'_> {
        let (kper, vper) = (self.n_k_heads * self.head_k_dim, self.n_v_heads * self.head_v_dim);
        let h = self.n_v_heads;
        Delta {
            q: &self.q[t * kper..(t + 1) * kper],
            k: &self.k[t * kper..(t + 1) * kper],
            v: &self.v[t * vper..(t + 1) * vper],
            alpha: &self.alpha[t * h..(t + 1) * h],
            beta: &self.beta[t * h..(t + 1) * h],
            // Per-head constants, shared by every token in the batch.
            ssm_a: self.ssm_a,
            dt_bias: self.dt_bias,
            head_k_dim: self.head_k_dim,
            head_v_dim: self.head_v_dim,
            n_k_heads: self.n_k_heads,
            n_v_heads: self.n_v_heads,
        }
    }
}

/// Every primitive the qwen3 forward pass needs.
///
/// Methods write into caller-provided buffers so a backend never allocates on
/// the forward path.
///
/// # The batch convention
///
/// **Every method takes a batch of consecutive tokens, and `n` is derived from
/// the buffers rather than passed.** Decode is `n == 1` of the same call, so
/// there is one code path, not two — the same property that makes
/// [`crate::model::Qwen3::forward`] serve prefill and decode and lets the cache
/// acceptance test demand bit-identical logits.
///
/// A batched buffer is **token-major**: `n` consecutive rows, each the shape
/// that single call used to take. The count comes from a division that the
/// implementation asserts is exact, so a shape error fails loudly at the seam
/// instead of becoming a silently wrong batch count.
///
/// Ops fall into three kinds, and the distinction is the whole reason this
/// works without the seam growing a parallel set of methods:
///
/// | | ops | under a batch |
/// |---|---|---|
/// | elementwise, or per fixed-size slice | [`Ops::rms_norm_heads`], [`Ops::silu_mul`], [`Ops::l2_norm_heads`], [`Ops::sigmoid_mul`], [`Ops::add_assign`], [`Ops::gather_chunks`] | **already correct.** A longer buffer is more slices; nothing to change |
/// | parallel over the batch | [`Ops::rms_norm`], [`Ops::matmul`], [`Ops::rope_neox`], [`Ops::attend`], [`Ops::kv_write`] | the win. `matmul` reads each weight row once for all `n` tokens instead of once per token |
/// | **sequential in the batch** | [`Ops::ssm_conv`], [`Ops::delta_rule`] | token `t`'s state feeds `t+1`, so these iterate. They still take the batch, so a backend may loop inside one launch, or implement a chunked parallel form, without model code changing |
///
/// A backend that simply loops over the batch is *bit-identical* to one that
/// does not, because batching changes which outputs are computed together and
/// never the order of any accumulation. That is the exactness rule in
/// `ARCHITECTURE.md` applied to a new axis, and it is why this whole change
/// needs no tolerance.
pub trait Ops {
    /// `out = x / sqrt(mean(x^2) + eps) * weight`, for each of `n` rows.
    ///
    /// `n` is `x.len() / weight.len()`: the row length *is* the weight length,
    /// so a batch is self-describing. Each row is normalized independently —
    /// the mean is per row, never across the batch.
    ///
    /// No `+1` on the weight — that is Gemma's variant, not Qwen's.
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]);

    /// RMSNorm applied independently to each `head_dim`-sized slice of `x`,
    /// in place. This is Qwen3's QK-norm: `weight` has length `head_dim` and is
    /// shared across heads.
    fn rms_norm_heads(&self, x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32);

    /// `out[t][j] = dot(w.row(j), x[t])` for every weight row and every token,
    /// dequantizing as it goes.
    ///
    /// **This is where batching pays.** `n` is `x.len() / w.n_in`, and the
    /// weight is read once for the whole batch instead of once per token: a
    /// 512-token prefill moves the model's bytes once, not 512 times. That is
    /// the difference between prefill costing what a GEMM costs and costing
    /// what generating the prompt would.
    ///
    /// Bit-exactness is free here and worth saying why: each `(t, j)` output is
    /// a complete dot product of the same weight row in the same order. Batching
    /// changes which outputs are computed together, never how any one of them
    /// accumulates. For Q8_0 the activation row is quantized once per token and
    /// shared across all weight rows, exactly as `ggml_compute_forward_mul_mat`
    /// does — so a batched call quantizes `n` rows and reuses each across the
    /// weight, which is the same arithmetic the per-token loop performed.
    fn matmul(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]);

    /// NEOX-style rotary embedding, in place, over `n_heads` heads of
    /// `head_dim` each.
    ///
    /// NEOX pairs dimension `i` with `i + head_dim/2`, **not** with `i + 1`.
    /// `LLM_ARCH_QWEN3` sits under llama.cpp's "the pairs of head values are
    /// offset by n_rot/2" group.
    /// `n_rot` is how many of each `head_dim` actually rotate; the rest pass
    /// through. `qwen3` rotates the whole head, `qwen35` rotates 64 of 256.
    ///
    /// It is a parameter rather than a loop in model code because the loop had
    /// to hand the seam a *sub-slice per head*, and a device backend keys its
    /// mirrors on the host address of a slice — so every head after the first
    /// looked like an unmirrored buffer and was uploaded from a stale host
    /// copy. It also cost 160 launches a token where one will do.
    ///
    /// **`pos` is the position of row 0**, and rows are consecutive: row `t`
    /// rotates at `pos + t`. `n` is `x.len() / (head_dim * n_heads)`. Positions
    /// in a batch are always consecutive in both phases — prefill is the prompt
    /// from `start_pos`, decode is one row — so this needs no position array.
    ///
    /// Using the batch index instead of the absolute position is the classic KV
    /// cache bug: invisible during a prefill from zero, where the two agree, and
    /// wrong for every token decoded after.
    fn rope_neox(
        &self,
        x: &mut [f32],
        pos: usize,
        head_dim: usize,
        n_rot: usize,
        n_heads: usize,
        theta_base: f32,
    );

    /// Numerically stable softmax, in place. Used by [`Ops::attend`]'s
    /// implementations, and by the MoE router when Stage 7 lands.
    /// Softmax over each row of `x`, where a row is `row` elements.
    ///
    /// **`row` is passed rather than derived, and it is the one place the batch
    /// convention cannot reach.** Every other op recovers its count from a
    /// second buffer — `matmul` from `x.len() / w.n_in`, `rms_norm` from
    /// `x.len() / weight.len()` — but softmax has a single buffer, so nothing
    /// in the arguments distinguishes one row of `n` from `n` rows of one. The
    /// caller knows: for the MoE router it is `n_expert`.
    ///
    /// Decode is `row == x.len()`, one row, exactly as before.
    fn softmax(&self, x: &mut [f32], row: usize);

    /// Scaled dot-product attention for a batch of queries against the cached
    /// history, all heads and all rows at once.
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
    ///
    /// A batch carries its own causal mask: row `t` attends over
    /// [`Attn::n_pos_of`] positions, so early rows of a prefill do strictly
    /// less work than late ones. The rows a prefill batch attends to include
    /// rows this same call wrote, which is why model code publishes the whole
    /// batch to the cache before attending.
    fn attend(&self, a: &Attn<'_>, out: &mut [f32]);

    /// `gate = silu(gate) * up`, in place — the SwiGLU nonlinearity.
    fn silu_mul(&self, gate: &mut [f32], up: &[f32]);

    /// `buf *= s`, in place: a weight's per-tensor second scale, applied after
    /// its matmul as llama.cpp's `build_lora_mm` does with `ggml_mul`. NVFP4
    /// tensors carry one; nothing else in the target models does.
    ///
    /// **A device backend must override this**: the default writes the host
    /// copy, which is stale whenever the device holds the current one.
    fn scale(&self, buf: &mut [f32], s: f32) {
        for v in buf.iter_mut() {
            *v *= s;
        }
    }

    /// L2 normalization per `head_dim`-sized slice, in place. No weight.
    ///
    /// **Not RMSNorm, despite looking like it.** `ggml_compute_forward_l2_norm_f32`
    /// scales by `1 / max(sqrt(sum(x^2)), eps)`: there is no division by `n`,
    /// and `eps` clamps the *norm* where RMSNorm adds it under the square root.
    /// Same-looking output, different function — which is why GatedDeltaNet's
    /// q and k get this and not [`Ops::rms_norm_heads`].
    fn l2_norm_heads(&self, x: &mut [f32], head_dim: usize, eps: f32);

    /// Depthwise causal conv1d over this layer's conv state and `x`, then
    /// `silu`, advancing the state — for each of `n` tokens in turn.
    ///
    /// **Sequential in the batch.** Token `t` convolves over a window that
    /// token `t-1` just advanced, so this iterates where [`Ops::matmul`]
    /// parallelizes. It still takes the whole batch rather than being called
    /// per token, so a backend can run the scan inside a single launch instead
    /// of paying launch overhead `n` times. `n` is `x.len() / n_channels`,
    /// where `n_channels` is `weight.len() / kernel`.
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

    /// The mean of `n_stream` slices of `nd` floats, taken every `n_stream * nd`
    /// from `src`, into `out`.
    ///
    /// **A fusion, not a new operation.** `qwen4exp`'s hyper-connection mixer
    /// averaged its streams as four [`Ops::gather_chunks`], three
    /// [`Ops::add_assign`] and one [`Ops::scale`] — eight launches to average
    /// four strided slices, twice a layer, 768 a token on the 125B. Those two
    /// kernels were 9.2% and 5.8% of its GPU time and the mixer is ~62% and
    /// ~74% of them. Traffic falls from ~19 x `nd` to 5 x `nd`.
    ///
    /// **Bit-identical to the sequence it replaces, by construction.** The
    /// reference order is stream 0, + 1, + 2, ... then x `1/n_stream`, every
    /// output element accumulates independently of every other, and this keeps
    /// that order per element. So it needs no tolerance and no exact-path
    /// switch; a backend that reassociated the sum would, and must not.
    ///
    /// `inv` is passed rather than derived: the sequence this replaces ends in
    /// [`Ops::scale`], a *multiply*, and `x * (1.0 / 3.0)` is not `x / 3.0` in
    /// f32. Deriving it here would break the bit-exactness above for any
    /// `n_stream` that is not a power of two.
    ///
    /// `src` is `[n_tok][n_stream][nd]` and `out` is `[n_tok][nd]`.
    ///
    /// **The default is a host loop, not the unfused sequence** — the arithmetic
    /// of that sequence, run on host memory. Right for the CPU backends. Wrong
    /// for anything that keeps activations on a device: it reads and writes the
    /// host copies and tells the device nothing, so a later device op reads a
    /// stale `out`. A backend or wrapper over a device must override this, as
    /// `Cuda` does. (The sequence itself cannot be the default: it needs a
    /// scratch buffer, and a per-call allocation is a new address for every
    /// address-keyed mirror.) Found when a test wrapper forwarded every op but
    /// this one.
    fn mean_streams(&self, src: &[f32], n_stream: usize, nd: usize, inv: f32, out: &mut [f32]) {
        for (i, o) in out.iter_mut().enumerate() {
            let (t, j) = (i / nd, i % nd);
            let base = t * n_stream * nd + j;
            let mut acc = src[base];
            for s in 1..n_stream {
                acc += src[base + s * nd];
            }
            *o = acc * inv;
        }
    }

    /// The exact dual of [`Ops::gather_chunks`]: write contiguous `src` back
    /// into `dst` at `offset`, every `stride`.
    ///
    /// Exists because the MoE FFN is a **per-token loop inside a batched
    /// pass** — routing differs per token, so its scratch is single-token and
    /// each token's result has to land in row `t` of the layer's output. The
    /// obvious `dst[at..at + nd].copy_from_slice(&acc)` is wrong in two ways at
    /// once on a device backend: it reads a host buffer the device wrote, and
    /// it writes a host address the device has never mirrored. That is
    /// `CLAUDE.md`'s sub-slice rule, and this is the fourth place it applies.
    ///
    /// The default is the host copy, which is correct for every CPU backend.
    /// **A device backend must override it.**
    fn scatter_chunks(
        &self,
        src: &[f32],
        chunk: usize,
        stride: usize,
        offset: usize,
        dst: &mut [f32],
    ) {
        for (i, block) in src.chunks_exact(chunk).enumerate() {
            let at = offset + i * stride;
            dst[at..at + chunk].copy_from_slice(block);
        }
    }

    /// `x *= sigmoid(g)`, elementwise and in place. The sibling of
    /// [`Ops::silu_mul`], and inexact for the same reason: `expf`.
    fn sigmoid_mul(&self, x: &mut [f32], g: &[f32]);

    /// The gated delta rule for a batch of tokens, every value head, state
    /// updated in place.
    ///
    /// **Sequential in the batch, and the one op with no batched form.** Token
    /// `t`'s rank-1 correction is token `t+1`'s stored state, so the tokens
    /// must be applied in order. llama.cpp has a separate chunked algorithm
    /// that recovers parallelism; this signature is what lets that arrive as a
    /// backend change rather than a model rewrite, because the seam already
    /// hands over the whole batch. `n` is `d.v.len() / (n_v_heads *
    /// head_v_dim)`.
    ///
    /// The cost of iterating here is small in the place it matters: within a
    /// GatedDeltaNet layer everything carrying real bytes — the fused `attn_qkv`
    /// and `attn_gate`, `ssm_out`, and the whole FFN — is a matmul and batches.
    /// This scan touches only the state.
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

    // ---- qwen4exp: hyper-connections and PLE (src/model/qwen4exp.md) --------
    //
    // Scalar defaults, transcribed from the ggml ops `qwen4exp.cpp` builds, so
    // every CPU backend inherits the oracle's arithmetic. **A device backend must
    // override each**: the defaults write host copies. Nothing the 35B runs calls
    // them.

    /// `x[row][j] *= w[j]`, where a row is `w.len()`: ggml's `ggml_mul` by a
    /// broadcast weight. A hyper-connection norm's weight spans all `n_stream`
    /// copies (`hc_dim`), so it is applied here after a per-stream RMSNorm with a
    /// unit weight — exact, since ggml's RMSNorm is `(x · scale) · w` too.
    fn mul_rows(&self, x: &mut [f32], w: &[f32]) {
        debug_assert_eq!(x.len() % w.len(), 0);
        for row in x.chunks_exact_mut(w.len()) {
            for (v, &wj) in row.iter_mut().zip(w) {
                *v *= wj;
            }
        }
    }

    /// `x = silu(x)`, in place: `ggml_silu_f32`, `x / (1 + exp(-x))`.
    fn silu(&self, x: &mut [f32]) {
        for v in x.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
    }

    /// `x = sigmoid(x)`, in place: `ggml_vec_sigmoid_f32`, `1 / (1 + exp(-x))`.
    fn sigmoid(&self, x: &mut [f32]) {
        for v in x.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
    }

    /// `out[t][s][j] = h[t][j] * w[t][s]`: one row per token broadcast over
    /// `n_stream` copies and scaled per copy — `ggml_mul(ggml_repeat_4d(h), w)`.
    /// A hyper-connection write is this then [`Ops::add_assign`]; the PLE gate's
    /// value is this alone. `n` is `w.len() / n_stream`, a row `h.len() / n`.
    fn mul_streams(&self, out: &mut [f32], h: &[f32], w: &[f32], n_stream: usize) {
        let n = w.len() / n_stream;
        let nd = h.len() / n.max(1);
        debug_assert_eq!(out.len(), n * n_stream * nd);
        for t in 0..n {
            for s in 0..n_stream {
                let ws = w[t * n_stream + s];
                let o = &mut out[(t * n_stream + s) * nd..(t * n_stream + s + 1) * nd];
                for (v, &hj) in o.iter_mut().zip(&h[t * nd..(t + 1) * nd]) {
                    *v = hj * ws;
                }
            }
        }
    }

    /// `out[i] = Σ_j a[i][j] · b[i][j]` over rows of `width`: `ggml_mul` then
    /// `ggml_sum_rows`, whose sum is `ggml_float` — f64, narrowed at the end —
    /// with the products taken in f32 first.
    fn row_dot(&self, a: &[f32], b: &[f32], width: usize, out: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(out.len() * width, a.len());
        for (i, o) in out.iter_mut().enumerate() {
            let mut sum = 0.0f64;
            for j in 0..width {
                sum += f64::from(a[i * width + j] * b[i * width + j]);
            }
            *o = sum as f32;
        }
    }

    /// PLE's gate, in place: `sigmoid(sgn(s) · sqrt(max(|s|, 1e-6)))`, as the chain
    /// `ggml_abs`, `ggml_clamp`, `ggml_sqrt`, `ggml_sgn`, `ggml_mul`, `ggml_sigmoid`
    /// in `build_ple` (`qwen4exp.cpp:1219-1220`). `sgn(0) = 0`.
    fn signed_sqrt_sigmoid(&self, s: &mut [f32]) {
        for v in s.iter_mut() {
            let mag = v.abs().max(1e-6).sqrt();
            let sgn = if *v > 0.0 {
                1.0
            } else if *v < 0.0 {
                -1.0
            } else {
                0.0
            };
            let g = sgn * mag;
            *v = 1.0 / (1.0 + (-g).exp());
        }
    }

    /// Depthwise causal conv1d, **dilated**, with no activation, advancing its
    /// state — PLE's conv (`build_ple`, `qwen4exp.cpp:1235-1276`).
    ///
    /// `out[t][c] = Σ_k weight[c][k] · x_pad[c][hist + t − (kernel−1−k)·dilation]`,
    /// the taps summed in `k` order in f32 (a chain of `ggml_add`), where `x_pad`
    /// is the state followed by `x`. `hist = (kernel−1)·dilation`; `state` is
    /// `[channels][hist]`, oldest first, and afterwards holds the last `hist`
    /// samples of `x_pad`. `weight` is `[channels][kernel]`. `x` is token-major.
    fn dilated_conv(
        &self,
        state: &mut [f32],
        x: &[f32],
        weight: &[f32],
        kernel: usize,
        dilation: usize,
        out: &mut [f32],
    ) {
        let channels = weight.len() / kernel;
        let hist = (kernel - 1) * dilation;
        let n = x.len() / channels;
        debug_assert_eq!(state.len(), channels * hist);
        debug_assert_eq!(out.len(), x.len());
        let mut pad = vec![0.0f32; hist + n];
        for c in 0..channels {
            pad[..hist].copy_from_slice(&state[c * hist..(c + 1) * hist]);
            for t in 0..n {
                pad[hist + t] = x[t * channels + c];
            }
            for t in 0..n {
                let mut acc = pad[hist + t - (kernel - 1) * dilation] * weight[c * kernel];
                for k in 1..kernel {
                    acc += pad[hist + t - (kernel - 1 - k) * dilation] * weight[c * kernel + k];
                }
                out[t * channels + c] = acc;
            }
            state[c * hist..(c + 1) * hist].copy_from_slice(&pad[n..]);
        }
    }

    /// **A hint, not work**: the next MoE layer will route through `next_router`,
    /// and `x` — this layer's router input, one token — predicts its choice well
    /// enough to start reading its likely experts now (SSD-TIER.md D20: the next
    /// router's top-16 on this input recalls 81% of its picks on the 125B). `k`
    /// is how many to guess.
    ///
    /// Changes no value the model computes. A backend with nothing to prefetch
    /// ignores it, which is the default; CUDA acts only while its expert pool is
    /// oversubscribed.
    fn prefetch_hint(&self, _next_router: &Weights<'_>, _x: &[f32], _k: usize) {}

    // ---- qwen4exp: Qwen Sparse Attention past its budget (qwen4exp.md, QSA) ----
    //
    // Scalar defaults, the oracle for the model's reference selection (decided
    // 16-09). A device backend must override each or refuse it.

    /// Pool blocks `p.from..p.to` of one layer's raw indexer keys into its
    /// pooled-key lane: for block `b`, the mean of cells `b*ratio ..
    /// (b+1)*ratio` as `((k0 + k1) + k2) + k3` times `1/ratio` — `ggml_add`'s
    /// chain then `ggml_scale` (`qwen4exp.cpp:609-617`) — then RMSNorm with
    /// `p.norm`, then RoPE at position `b*ratio`.
    ///
    /// `raw` is the layer's whole f16 lane, `pooled` its whole pooled lane,
    /// block-major; only blocks `from..to` are written. The arithmetic is the
    /// oracle's serial RMSNorm and `rope_neox`'s f64 table, on every backend,
    /// because the scores only rank blocks and an ulp can reorder a near-tie.
    fn qsa_pool(&self, raw: &[u16], pooled: &mut [f32], p: &QsaPool<'_>) {
        naive::qsa_pool(raw, pooled, p);
    }

    /// Score and select, per query row of this pass: `scores[t][b] = Σ_h
    /// relu(q[t][h] · pooled[b])` over `s.n_blocks` blocks (`build_qsa_top_k`'s
    /// `mul_mat`, `relu` and per-head sum, `qwen4exp.cpp:627-643`), then row `t`'s
    /// cells into `cells[t * s.stride()..]`, ascending — [`naive::select_cells`],
    /// the model's reference rule. Row `t` gets [`QsaSelect::count`] cells; the
    /// rest of its stride is left as it was.
    ///
    /// The dot accumulates in f64, as ggml's scalar `vec_dot_f32` does; the heads
    /// are summed in f32 in order, as its chain of `ggml_add` does.
    fn qsa_select(&self, q: &[f32], pooled: &[f32], s: &QsaSelect, scores: &mut [f32], cells: &mut [u32]) {
        naive::qsa_scores(q, pooled, s, scores);
        naive::qsa_select_cells(scores, s, cells);
    }

    /// Attention restricted, per query row, to chosen cells: QSA past its budget.
    ///
    /// Row `t` attends to `cells[t * stride ..][.. count]`, ascending positions,
    /// where `count` is [`qsa_cell_count`] at the row's position — known on the
    /// host, so nothing is read back. Everything else about `a` is
    /// [`Ops::attend`]'s. **Given every cell for every row it is `attend`, to the
    /// bit**, on the oracle.
    fn attend_sparse(&self, a: &Attn<'_>, cells: &[u32], sel: &QsaSelect, out: &mut [f32]) {
        debug_assert_eq!(out.len(), a.n_q() * a.n_head * a.head_dim);
        let per_kv = a.group() * a.head_dim;
        let stride = sel.stride();
        let mut sc = naive::Scratch::for_attn(a);
        for t in 0..a.n_q() {
            let count = sel.count(a.n_pos_of(t) - 1);
            let row_cells = &cells[t * stride..t * stride + count];
            let row = &mut out[t * a.n_head * a.head_dim..][..a.n_head * a.head_dim];
            for (h_kv, chunk) in row.chunks_mut(per_kv).enumerate() {
                naive::attend_kv_head_cells(a, t, h_kv, row_cells, chunk, &mut sc);
            }
        }
    }

    /// Two matmuls over the **same** activation, issued together.
    ///
    /// `out_a` and `out_b` are separate buffers, each `n_tok` rows of that
    /// weight's own width. Nothing is concatenated above this seam.
    ///
    /// **Because the cost of a small F32 matmul is reading `x`, not producing
    /// outputs.** Measured on the 35B: 39.3 us at `n_out` 1, 41.8 at 32, 43.3
    /// at 256 — twelve times the output for 10% more time, because every thread
    /// walks the same 2048-element activation and that walk is 81% of the call.
    /// Two matmuls over one activation therefore pay for it twice, and this
    /// model does that twice per layer:
    ///
    /// | pair | shapes | reads |
    /// |---|---|---|
    /// | `ssm_alpha` + `ssm_beta` | `{2048,32}` each | `normed` |
    /// | `ffn_gate_inp` + `..._shexp` | `{2048,256}` + `{2048,1}` | the MoE input |
    ///
    /// 133 calls a token, 5.51 ms of a 34 ms decode. Merging the launches
    /// removes half of them.
    ///
    /// **Two destinations rather than one concatenated output**, which is the
    /// decision that keeps this local. A combined buffer would reach [`Delta`]
    /// (separate `alpha`/`beta` slices), [`Ops::softmax`] (a 256-wide row that
    /// would have to skip a 257th element) and every backend's `delta_rule`.
    /// Separate outputs reach none of them.
    ///
    /// The default is the two calls it replaces, so **every CPU backend is
    /// unchanged and bit-identical by definition**. A device backend overrides
    /// it to merge the launch; the arithmetic is untouched either way, since
    /// each output is still one accumulation over the same values in the same
    /// order.
    fn matmul_pair(
        &self,
        a: &Weights<'_>,
        b: &Weights<'_>,
        x: &[f32],
        out_a: &mut [f32],
        out_b: &mut [f32],
    ) {
        self.matmul(a, x, out_a);
        self.matmul(b, x, out_b);
    }

    /// One matmul per routed expert, **issued together**.
    ///
    /// `out` is `picks.len()` consecutive rows of `w.n_out`. `x` is either one
    /// row of `w.n_in`, shared by every expert — the gate and up projections —
    /// or `picks.len()` rows, one each — the down projection. Which it is is
    /// derived from `x.len()`, the same convention the batch axis uses, so the
    /// count cannot disagree with the buffer.
    ///
    /// **This reverses a documented decision.** [`Experts`] says the seam does
    /// not grow a MoE matmul, because each pick is an ordinary matmul over a
    /// borrowed sub-range. That was right about the arithmetic and wrong about
    /// the machine: measured on the 35B, issuing the eight experts separately
    /// costs **18.03 ms/token** across the whole expert stage against **4.46**
    /// grouped — and not from launch overhead, which is ~2 us against ~9 us of
    /// device time, but from occupancy. A `{2048, 512}` matmul cannot fill 36
    /// SMs. The seam had no way to say "these eight go together", so a backend
    /// had no way to fix it.
    ///
    /// *Which* experts, and where their bytes live, still lives above this —
    /// `picks` arrives already chosen.
    ///
    /// The default is the loop it replaces. **It sub-slices `out`**, which is
    /// safe only for a backend that addresses memory by value; a device backend
    /// must override it, and every device backend must anyway or it gains
    /// nothing.
    fn matmul_experts(&self, w: &Experts<'_>, route: &Route, x: &[f32], out: &mut [f32]) {
        let Some(picks) = route.ids() else {
            // Unreachable through the default `route`, which never returns
            // `Device`. See `Route`'s contract: a backend that does must
            // override this.
            debug_assert!(false, "matmul_experts default given a device route");
            return;
        };
        // Which row of `x` each pick reads, derived from the buffer exactly as
        // the batch count is everywhere else. Three shapes, and only the middle
        // one is new:
        //
        //   rows == picks       one intermediate per pick — the `down` half
        //   rows == n_tok       one activation per token  — the `gate`/`up` half
        //   rows == 1           a single shared row       — decode, either half
        //
        // At `n_tok == 1` the middle and last coincide, which is why this was
        // correct with two cases for as long as the model routed one token at a
        // time, and silently wrong the moment it did not: every token in a
        // batch would have read token 0's activation.
        let n_used = route.n_used().max(1);
        let rows = x.len() / w.n_in;
        for (i, &e) in picks.iter().enumerate() {
            let r = if rows == picks.len() {
                i
            } else if rows > 1 {
                i / n_used
            } else {
                0
            };
            let xi = &x[r * w.n_in..(r + 1) * w.n_in];
            let row = &mut out[i * w.n_out..(i + 1) * w.n_out];
            self.matmul(&w.expert(e), xi, row);
            // NVFP4's per-expert second scale, on this pick's output before
            // anything reads it (`build_moe_ffn`).
            if !w.scale.is_empty() {
                self.scale(row, w.scale_of(e));
            }
        }
    }

    /// The routed FFN's gated half: `out[e] = silu(gate[e] . x) * (up[e] . x)`,
    /// for every pick, in one call.
    ///
    /// Three seam calls collapsed into one — two [`Ops::matmul_experts`] and a
    /// [`Ops::silu_mul`] — because on this platform the unit of cost is the
    /// launch, not the arithmetic. Measured at 5.7 of a layer's 38 launches.
    /// It also saves a round trip through device memory: `gate` and `up` for
    /// the same output element can be combined while both are still in
    /// registers, so only the result is stored.
    ///
    /// `out` is `picks.len()` rows of `gate.n_out`; `x` is the one activation
    /// every expert reads.
    ///
    /// The default is the three calls it replaces, so no CPU backend changes by
    /// a bit and the oracle is untouched.
    fn moe_glu(
        &self,
        gate: &Experts<'_>,
        up: &Experts<'_>,
        route: &Route,
        x: &[f32],
        out: &mut [f32],
        scratch: &mut [f32],
    ) {
        self.matmul_experts(gate, route, x, out);
        self.matmul_experts(up, route, x, scratch);
        self.silu_mul(out, scratch);
    }

    /// The routed FFN's whole tail: weighted expert sum, the shared expert's
    /// sigmoid gate, and the write back into row `at` of the layer's output.
    ///
    /// Replaces [`Ops::add_scaled_rows`], [`Ops::add_scaled_sigmoid`] and
    /// [`Ops::scatter_chunks`] — three launches per layer to produce one
    /// vector, when the per-launch cost on the target platform is **20.7 us,
    /// measured**. It also drops the `moe_acc` staging buffer entirely.
    ///
    /// `logit` is a whole buffer with an index rather than a one-element slice,
    /// because a sub-slice is a host address a device backend has never
    /// mirrored — `CLAUDE.md`'s rule, which has now caused four bugs.
    ///
    /// Order is the oracle's: experts summed ascending from zero, shared expert
    /// added last.
    #[allow(clippy::too_many_arguments)]
    fn moe_finish(
        &self,
        out: &mut [f32],
        at: usize,
        n: usize,
        rows: &[f32],
        route: &Route,
        shared: &[f32],
        logit: &[f32],
        logit_at: usize,
    ) {
        let Some(scales) = route.weights() else {
            // As `matmul_experts`: unreachable through the default `route`.
            debug_assert!(false, "moe_finish default given a device route");
            return;
        };
        // One output row per token, `n_tok` of them starting at `at`. Decode is
        // `n_tok == 1`, which is byte for byte the loop this replaced.
        //
        // Each token takes its own `n_used` expert rows, its own shared-expert
        // row and its own gate logit — three separate indexings that a
        // single-token version had no way to get wrong, because every one of
        // them was zero.
        let n_used = route.n_used().max(1);
        let n_tok = scales.len() / n_used;
        for t in 0..n_tok {
            let g = 1.0 / (1.0 + (-logit[logit_at + t]).exp());
            let w = &scales[t * n_used..(t + 1) * n_used];
            for j in 0..n {
                let mut v = 0.0f32;
                for (e, &s) in w.iter().enumerate() {
                    v += s * rows[(t * n_used + e) * n + j];
                }
                v += shared[t * n + j] * g;
                out[at + t * n + j] = v;
            }
        }
    }

    /// Choose this token's experts from the router's probabilities.
    ///
    /// **The rule, in one place.** It used to live inline in `qwen35::moe_token`
    /// and is transcribed from `llm_graph_context::build_moe_ffn`: `n_used`
    /// rounds, each scanning ascending and taking a new best only on a strict
    /// `>` so a tie resolves to the lower id; then the chosen probabilities
    /// summed **in selection order** and each divided by
    /// `max(sum, 6.103515625e-5)`, f16's smallest normal.
    ///
    /// `probs` is the softmax over *all* experts, before selection — the
    /// distinction `moe_token`'s doc calls out as easy to get wrong, since
    /// `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT` is the variant that
    /// softmaxes the top-k instead, and qwen35moe does not use it.
    ///
    /// The default brings the probabilities home and decides here, which is
    /// what every CPU backend wants and what the CUDA backend does until its
    /// expert pointer table is complete. A device backend overrides it to keep
    /// the decision on the card, which is what lets the pass be a CUDA graph.
    fn route(&self, probs: &mut [f32], n_expert: usize, n_used: usize) -> Route {
        self.host_needs(probs);
        let n_tok = probs.len() / n_expert.max(1);
        let mut ids: Vec<usize> = Vec::with_capacity(n_tok * n_used);
        let mut weights: Vec<f32> = Vec::with_capacity(n_tok * n_used);
        for t in 0..n_tok {
            let p = &probs[t * n_expert..(t + 1) * n_expert];
            let base = ids.len();
            for _ in 0..n_used {
                let mut best = usize::MAX;
                for e in 0..n_expert {
                    if ids[base..].contains(&e) {
                        continue;
                    }
                    if best == usize::MAX || p[e] > p[best] {
                        best = e;
                    }
                }
                ids.push(best);
            }
            let sum: f32 = ids[base..].iter().map(|&e| p[e]).sum();
            let denom = sum.max(6.103_515_625e-5);
            weights.extend(ids[base..].iter().map(|&e| p[e] / denom));
        }
        Route::Host { ids, weights, n_used }
    }

    /// `acc[j] = sum over e of `scales[e] * rows[e * acc.len() + j]`, summed in
    /// ascending `e`.
    ///
    /// The MoE accumulation, replacing one [`Ops::add_scaled`] per expert over
    /// an accumulator that started at zero. **Bit-identical to that loop**: the
    /// sum is parallel over `j`, which the oracle already treats as
    /// independent, and serial and ascending over `e`, which it does not.
    ///
    /// `acc` is written, not accumulated into — the zero-fill it replaces is
    /// the first term of the same sum.
    fn add_scaled_rows(&self, acc: &mut [f32], rows: &[f32], scales: &[f32]) {
        let n = acc.len();
        debug_assert_eq!(rows.len(), scales.len() * n);
        for (j, a) in acc.iter_mut().enumerate() {
            let mut v = 0.0f32;
            for (e, &s) in scales.iter().enumerate() {
                v += s * rows[e * n + j];
            }
            *a = v;
        }
    }

    /// `acc += b * sigmoid(logit[0])` — the shared expert's gate.
    ///
    /// **Exists so the logit never comes home.** It is a matmul result, so on a
    /// device backend reading it costs a full pipeline drain, and `qwen35moe`
    /// would pay one per layer per token — 40 of them, in a backend whose whole
    /// design is to touch the bus five times.
    ///
    /// The default computes the sigmoid on the host and defers to
    /// [`Ops::add_scaled`], which is exactly what the model used to do inline,
    /// so no CPU backend changes by a bit.
    fn add_scaled_sigmoid(&self, acc: &mut [f32], b: &[f32], logit: &[f32]) {
        let s = 1.0 / (1.0 + (-logit[0]).exp());
        self.add_scaled(acc, b, s);
    }

    /// `a += b * scale`, in place — the MoE expert accumulation.
    ///
    /// Separate from [`Ops::add_assign`] because a routed expert's output is
    /// weighted by its router probability before it joins the sum, and doing
    /// the scale as its own pass would read and write `b` an extra time for
    /// every one of the 8 experts a token visits.
    ///
    /// The default is the scalar loop, which is correct for every CPU backend.
    /// **A device backend must override it**, or it will read a host buffer the
    /// device owns.
    fn add_scaled(&self, a: &mut [f32], b: &[f32], scale: f32) {
        debug_assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            a[i] += b[i] * scale;
        }
    }

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
    /// Print whatever this backend can say about the work it has just done.
    ///
    /// **A hint about the backend, like the three residency hints above**, and
    /// for the same reason: `serve` is generic over `Ops` while the interesting
    /// counters — kernel launches, bus crossings, expert residency — belong to
    /// one backend. Threading a concrete type through the server to reach them
    /// would couple it to CUDA; a no-op default does not.
    ///
    /// Called once per turn. Implementations should print nothing unless asked
    /// to, since this runs in a server's hot path.
    fn device_report(&self) {}

    /// The most bytes of activations a pass can hold, declared by the engine so a
    /// device backend can budget memory for them before the first pass. A no-op
    /// where activations live in host memory anyway.
    fn reserve_activations(&self, _bytes: usize) {}

    /// Cumulative tier-3 and residency counters, for a per-turn breakdown.
    ///
    /// `None` on every CPU backend, which has no tiers. `serve` differences two
    /// snapshots to say where one turn's time went, which its total prefill
    /// time alone could not: on 06-10 a `serve` turn took ~50 s for a prompt
    /// `generate` prefilled in ~28 s on the same binary and card.
    fn tier_counters(&self) -> Option<TierCounters> {
        None
    }

    /// Which code paths this backend will actually run, for a startup banner.
    ///
    /// **A hint like [`Ops::device_report`], and for the same reason**: `serve`
    /// is generic over `Ops` while the paths worth naming belong to one backend.
    ///
    /// It exists because a configuration nobody can see is a configuration
    /// nobody can attribute a measurement to. `INFERRED_ATTN_MMA` and its
    /// siblings are read in the backend's constructor precisely so they reach
    /// every entry point -- and that makes them invisible at the call site, so
    /// a session can run a different kernel than the person reading its numbers
    /// believes. This is the other half of that fix: say so, once, at startup.
    ///
    /// Returns `(label, value)` pairs. Empty by default, so no CPU backend has
    /// to care.
    fn config_report(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }

    /// A one-time cost this backend charged to the forward pass, and its name.
    ///
    /// **Because a phase timer measures a wall clock, not a phase.** The CUDA
    /// backend places all 30,720 experts on first sight of each tensor, which
    /// happens *inside the first prefill* — 16.3 GB of mmap read, ~4.5 GiB
    /// page-locked and ~22,400 host-to-device copies. A 19-token prompt
    /// therefore reported `prefill 19 tok 25205.3 ms 0.8 tok/s`, which reads as
    /// a throughput and is nothing of the kind: ~24 s of it happens once and
    /// ~1.2 s is nineteen tokens of arithmetic.
    ///
    /// Eleventh instrument here to report an unlabelled basis, and the second
    /// where the *denominator* rather than the number was the defect. The fix
    /// is not to hide the cost — it is real time the user waited — but to name
    /// it, which is what [`crate::Profile::phases`] now does.
    ///
    /// Returning the label with the duration rather than beside it is
    /// deliberate: a duration with no name is exactly the failure this exists
    /// to correct, so the type makes one impossible.
    ///
    /// `None` for every CPU backend, which allocates nothing lazily — so the
    /// 0.6B and 9B lines print exactly as they did before.
    fn setup_cost(&self) -> Option<(u64, &'static str)> {
        None
    }

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
    /// Already batched, and needed no change to become so: the cache is
    /// position-major and a batch occupies consecutive positions, so `n` rows
    /// are one contiguous run of `n * kv_dim` elements at `start_pos * kv_dim`.
    /// A prefill therefore publishes its whole batch in one call per layer.
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

    /// Bring a device-owned recurrent state slab home.
    ///
    /// The dual of [`Ops::forget_state`], and the other half of what a
    /// checkpoint needs: `forget_state` says "the host slab is authoritative
    /// again", this says "make the host slab authoritative".
    ///
    /// A no-op on every CPU backend, which writes the caller's slab directly
    /// and has nothing to fetch. On CUDA the device copy is the authoritative
    /// one after first touch — a GatedDeltaNet layer's state is written by
    /// kernels and deliberately never comes home on the forward path — so
    /// without this a checkpoint would save whatever the host slab held before
    /// the sequence started, which is zeros. Silently: the restore would
    /// succeed and the model would continue from an empty memory.
    ///
    /// Called per layer slice, because that is the granularity the device
    /// keys its state mirrors on.
    fn read_state(&self, _host: &mut [f32]) {}

    fn kv_write(&self, slab: &mut [u16], offset: usize, src: &[f32]) {
        for (d, &s) in slab[offset..offset + src.len()].iter_mut().zip(src) {
            *d = crate::quant::half::f32_to_f16(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::check_expert_types;
    use crate::gguf::GgmlType::*;

    #[test]
    fn the_cuda_backend_refuses_routed_experts_it_has_no_kernel_for() {
        // The two formats the grouped kernels run, as the models ship them.
        assert!(check_expert_types(Nvfp4, Nvfp4, Nvfp4, true).is_ok());
        assert!(check_expert_types(Iq4Xs, Iq4Xs, Iq4Xs, true).is_ok());
        // The GitHub report: a Q4_K / Q5_1 quant.
        assert!(check_expert_types(Q4K, Q4K, Q5_1, true).is_err());
        assert!(check_expert_types(Q5_1, Q5_1, Q5_1, true).is_err());
        // `moe_glu` needs gate and up of one type.
        assert!(check_expert_types(Iq4Xs, Nvfp4, Iq4Xs, true).is_err());
        // Formats the CPU runs but the CUDA expert path does not.
        assert!(check_expert_types(Q8_0, Q8_0, Q8_0, true).is_err());
    }

    #[test]
    fn the_cpu_backends_run_every_format_their_kernels_implement() {
        for t in [F32, F16, Bf16, Q8_0, Q5K, Q6K, Iq4Xs, Nvfp4] {
            assert!(check_expert_types(t, t, t, false).is_ok(), "{}", t.name());
        }
        assert!(check_expert_types(Q4K, Q4K, Q4K, false).is_err());
        assert!(check_expert_types(Q8_0, Q8_0, Q5_1, false).is_err());
    }

    #[test]
    fn a_refusal_names_what_it_found_and_what_runs() {
        let msg = match check_expert_types(Q4K, Q4K, Q5_1, true) {
            Err(e) => e.to_string(),
            Ok(()) => String::new(),
        };
        assert!(msg.contains("q4_K") || msg.contains("Q4_K"), "{msg}");
        assert!(msg.contains("IQ4_XS or NVFP4"), "{msg}");
    }
}
