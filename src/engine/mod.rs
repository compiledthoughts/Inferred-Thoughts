//! Owns the model, the KV cache and the profiler, and drives prefill/decode.
//!
//! The split this type exists to make is **prefill vs decode**, not input vs
//! output. Prefill runs the whole prompt in one pass and is compute-bound;
//! decode runs one token against a growing cache and is memory-bound. Most
//! optimizations help exactly one of them, so measuring them together measures
//! nothing — which is why [`crate::profile::Profile`] counts them separately
//! and why they are separate methods here rather than one `generate` blob.

use std::time::{Duration, Instant};

use crate::cache::{KvCache, RecurrentState};
use crate::error::{Error, Result};
use crate::model::Model;
use crate::ops::Ops;
use crate::profile::{Ctx, Profile};

/// Why generation stopped.
///
/// The engine knows this exactly; before this existed the CLI inferred `[eos]`
/// from `produced.len() < max_tokens`, which reports a filled context as an
/// end-of-sequence stop. Two very different situations — one is the model
/// finishing, the other is us running out of room — so they get distinct names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model emitted its end-of-sequence token.
    Eos,
    /// Hit the caller's `max_new` budget.
    MaxTokens,
    /// The KV cache has no room for another position.
    ContextFull,
}

impl StopReason {
    pub fn label(self) -> &'static str {
        match self {
            StopReason::Eos => "eos",
            StopReason::MaxTokens => "token limit",
            StopReason::ContextFull => "context full",
        }
    }
}

pub struct Engine<'a, O: Ops> {
    pub model: Model<'a>,
    pub ops: O,
    cache: KvCache,
    /// Present only for architectures that have any. `qwen3` is `None`, and
    /// the allocation is skipped rather than sized to zero, so a model that
    /// needs state and does not get it fails loudly instead of scanning an
    /// empty slice.
    recurrent: Option<RecurrentState>,
    /// Largest number of prompt tokens handed to one forward pass.
    ///
    /// See [`Engine::set_max_batch`]. This exists because a batch's cost is not
    /// only time.
    max_batch: usize,
    pub prof: Profile,
}

/// Prompt tokens per forward pass, unless the caller says otherwise.
///
/// **This is a VRAM bound, not a speed knob**, and it was added after a batched
/// prefill quietly cost 4.5 GiB on a long session. A device backend keeps a
/// mirror of every activation buffer, and those buffers are sized by the batch:
/// on the 9B they come to **405 KiB per token** — 376 of `Scratch`, of which
/// the `n_ff` gate/up pair alone is 96, plus 29 of Q8_0 activation mirrors.
/// The mirrors are deliberately *not* freed between passes (`Ops::begin_pass`
/// invalidates in place, because freeing ~280 buffers per token was measured to
/// cost more than it saved), so a single 10,000-token prefill sizes them at
/// ~4 GiB and holds it for the life of the process.
///
/// That reasoning was sound while a buffer was one token wide. Batching made a
/// buffer up to `n` tokens wide without revisiting it, and on the 35B — where
/// VRAM is the entire constraint and 4.5 GiB is about ten layers of residency —
/// the trade runs backwards.
///
/// 512 rather than the whole prompt because the speedup saturates far below it:
/// the GPU's reuse is per-warp over `MM_TOK` = 4 tokens, and the rest is L2
/// hits, neither of which needs thousands of rows. It caps activations at
/// ~0.20 GiB on the 9B. llama.cpp draws the same line for the same reason
/// (`n_batch` 2048, `n_ubatch` 512).
///
/// Chunking is safe to the bit: `split_prefill_equals_single_prefill` asserts a
/// prefill split in two produces logits identical to one pass, which holds
/// because a chunk is just a prefill at a later `start_pos`.
pub const DEFAULT_MAX_BATCH: usize = 512;

/// A point a sequence can be returned to.
///
/// **What has to be saved is the recurrent state, and only that.** A KV cache
/// is a log: truncating it to `pos` leaves exactly the state that prefix would
/// have produced, so the ten attention layers of the 35B need nothing stored.
/// A GatedDeltaNet layer's state is not a log but one matrix that has absorbed
/// every token with no record of how to remove one, so the thirty recurrent
/// layers need a copy.
///
/// That asymmetry is what makes this cheap, and *fixed size*: 84 MiB on the 35B
/// whatever the depth, against the 640 MiB of KV held at 24k positions.
/// llama.cpp's context checkpoints are the same mechanism for the same reason —
/// they exist for recurrent and hybrid models, because a pure-attention model
/// needs only a truncation.
///
/// **`qwen4exp` carries one more piece of state**, held by the model rather than
/// in the recurrent slabs: PLE's n-gram window and conv history, in `ple`.
#[derive(Clone)]
pub struct Checkpoint {
    pos: usize,
    conv: Vec<f32>,
    ssm: Vec<f32>,
    ple: Option<crate::model::qwen4exp::PleSnapshot>,
}

impl Checkpoint {
    /// Position this checkpoint stands at: the number of tokens already
    /// consumed when it was taken.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Bytes held, so a caller can bound how many it keeps.
    pub fn bytes(&self) -> usize {
        (self.conv.len() + self.ssm.len()) * std::mem::size_of::<f32>()
            + self.ple.as_ref().map_or(0, |p| p.bytes())
    }
}

impl<'a, O: Ops> Engine<'a, O> {
    pub fn new(model: impl Into<Model<'a>>, ops: O, n_ctx: usize, detail: bool) -> Self {
        let model = model.into();
        // Slabs for the layers that actually attend, which on a hybrid
        // architecture is a fraction of them.
        let mut cache = KvCache::new(model.n_kv_layer(), model.kv_dim(), n_ctx);
        if model.index_dim() > 0 {
            cache = cache.with_index(model.index_dim());
        }
        let recurrent = model
            .recurrent_dims()
            .map(|(n, conv, ssm)| RecurrentState::new(n, conv, ssm));
        let mut prof = Profile::new(detail);
        prof.weight_bytes = model.weight_bytes_per_pass();
        Self {
            model,
            ops,
            cache,
            recurrent,
            max_batch: DEFAULT_MAX_BATCH,
            prof,
        }
    }

    /// Cap the prompt tokens handed to one forward pass. See
    /// [`DEFAULT_MAX_BATCH`] for why this is a memory bound rather than a
    /// tuning knob. Zero is rejected by clamping, so a caller cannot stall the
    /// engine with it.
    pub fn set_max_batch(&mut self, n: usize) {
        self.max_batch = n.max(1);
    }

    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Absolute position the next token will occupy.
    pub fn pos(&self) -> usize {
        self.cache.len()
    }

    pub fn n_ctx(&self) -> usize {
        self.cache.n_ctx()
    }

    pub fn kv_capacity_bytes(&self) -> u64 {
        self.cache.capacity_bytes()
    }

    /// Bytes of recurrent state held, zero for architectures without any.
    ///
    /// Reported separately from the KV cache because the two scale completely
    /// differently: this is constant in context and that grows with it, which
    /// is the property the whole hybrid architecture exists for.
    pub fn recurrent_capacity_bytes(&self) -> u64 {
        self.recurrent.as_ref().map_or(0, |r| r.capacity_bytes())
    }

    /// Drop everything after `len` positions, keeping the prefix usable.
    ///
    /// Returns whether it could. **A recurrent architecture cannot rewind.** A
    /// KV cache is a log — truncating it to a common prefix leaves exactly the
    /// state that prefix would have produced. A GatedDeltaNet layer's state is
    /// not a log but a single matrix that has absorbed every token, with no
    /// record of how to remove one. `HANDOFF.md` §3: "the recurrent state
    /// cannot be partially reused — it only survives via checkpoints."
    ///
    /// So this succeeds on `qwen3` and refuses on `qwen35`, and the caller
    /// restarts instead. Checkpointing the state periodically would make the
    /// refusal cheaper; it is not built.
    pub fn rewind(&mut self, len: usize) -> bool {
        if self.recurrent.is_some() || len > self.cache.len() {
            return false;
        }
        self.cache.commit(len);
        true
    }

    /// Capture the current position and recurrent state.
    ///
    /// `None` for an architecture with no recurrent state, where
    /// [`Engine::rewind`] already does the job for free and a checkpoint would
    /// be a copy of nothing.
    ///
    /// **Reads the state back through the seam first, and that is the part
    /// that would fail silently.** On CUDA the device copy is authoritative
    /// after first touch — the state is written by kernels and deliberately
    /// never comes home on the forward path — so a checkpoint taken without
    /// [`Ops::read_state`] would save the zeros the host slab still held. The
    /// restore would then succeed and the model would continue from an empty
    /// memory, producing fluent text with no recollection of the conversation.
    pub fn checkpoint(&mut self) -> Option<Checkpoint> {
        let r = self.recurrent.as_mut()?;
        // Per layer slice, because that is the granularity the device keys its
        // state mirrors on.
        for il in 0..r.n_layer() {
            self.ops.read_state(r.conv_mut(il));
            self.ops.read_state(r.ssm_mut(il));
        }
        let (conv, ssm) = r.slabs();
        let (conv, ssm) = (conv.to_vec(), ssm.to_vec());
        let ple = match &self.model {
            Model::Qwen4Exp(m) => m.ple_snapshot(&self.ops),
            _ => None,
        };
        Some(Checkpoint { pos: self.cache.len(), conv, ssm, ple })
    }

    /// Return to a checkpoint: truncate the KV log and reload the state.
    ///
    /// The KV needs no stored copy — positions after `pos` are simply no longer
    /// part of the sequence, and the next prefill overwrites them.
    pub fn restore(&mut self, c: &Checkpoint) -> Result<()> {
        if c.pos > self.cache.n_ctx() {
            return Err(Error::InconsistentArchitecture {
                what: "checkpoint",
                detail: format!("stands at {} positions, context is {}", c.pos, self.cache.n_ctx()),
            });
        }
        self.cache.commit(c.pos);
        if let Some(r) = self.recurrent.as_mut() {
            r.load(&c.conv, &c.ssm)?;
            if let (Model::Qwen4Exp(m), Some(p)) = (&self.model, &c.ple) {
                m.ple_restore(p)?;
            }
            // The device owns the authoritative copy once it has touched it, so
            // writing the host slabs is invisible without this. After both halves.
            self.ops.forget_state();
        }
        Ok(())
    }

    /// Drop the cached history. The profile is kept, so a run that resets
    /// between prompts still reports totals across all of them.
    pub fn reset(&mut self) {
        self.cache.reset();
        if let Some(r) = self.recurrent.as_mut() {
            r.reset();
            // A device backend owns this state once it has touched it, so
            // zeroing the host slab is invisible to it without being told.
            self.ops.forget_state();
        }
    }

    /// One forward pass appended at the current position, timed but not yet
    /// attributed to a phase.
    fn run(&mut self, tokens: &[u32]) -> Result<(Vec<f32>, Duration)> {
        let start = self.cache.len();
        let mut noop = |_: &str, _: usize, _: &[f32]| {};
        let mut ctx = Ctx::new(&mut noop, &mut self.prof);
        let t0 = Instant::now();
        let logits = self.model.forward(
            &self.ops,
            tokens,
            start,
            &mut self.cache,
            self.recurrent.as_mut(),
            &mut ctx,
        )?;
        Ok((logits, t0.elapsed()))
    }

    /// Process a whole prompt, in chunks of at most [`Engine::max_batch`].
    /// Returns logits for its last token.
    ///
    /// Chunking is a **memory** decision, not a speed one — see
    /// [`DEFAULT_MAX_BATCH`]. It costs nothing numerically: each chunk is an
    /// ordinary prefill at a later `start_pos`, which is the property
    /// `split_prefill_equals_single_prefill` pins down.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let (logits, dt) = self.prefill_chunked(tokens)?;
        self.prof.add_prefill(tokens.len(), dt);
        Ok(logits)
    }

    /// The chunk loop, shared by [`Engine::prefill`] and [`Engine::generate`]
    /// so neither can acquire its own batching policy.
    fn prefill_chunked(&mut self, tokens: &[u32]) -> Result<(Vec<f32>, Duration)> {
        let mut logits = Vec::new();
        let mut total = Duration::ZERO;
        for chunk in tokens.chunks(self.max_batch) {
            let (l, dt) = self.run(chunk)?;
            logits = l;
            total += dt;
        }
        Ok((logits, total))
    }

    /// Process one token against the cached history.
    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>> {
        let (logits, dt) = self.run(&[token])?;
        self.prof.add_decode(dt);
        Ok(logits)
    }

    /// Greedy generation. `on_token` is called for each new token as it is
    /// produced, so a caller can stream output without waiting for the run.
    ///
    /// The argmax comes from [`Profile::record_token`] rather than a separate
    /// pass: it already scans the logits for the top two, and doing it twice
    /// would cost a second sweep of 150k values per token for no reason.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: Option<u32>,
        mut on_token: impl FnMut(u32),
    ) -> Result<(Vec<u32>, StopReason)> {
        self.prof.reserve(self.model.n_layer(), max_new);

        let (mut logits, dt) = self.prefill_chunked(prompt)?;
        self.prof.add_prefill(prompt.len(), dt);
        let mut elapsed = dt;

        let mut produced = Vec::with_capacity(max_new);
        let mut why = StopReason::MaxTokens;
        for _ in 0..max_new {
            let pos = self.cache.len();
            let next = self.prof.record_token(pos, &logits, elapsed);
            if Some(next) == eos {
                why = StopReason::Eos;
                break;
            }
            produced.push(next);
            on_token(next);
            if produced.len() == max_new {
                why = StopReason::MaxTokens;
                break;
            }
            if pos + 1 >= self.cache.n_ctx() {
                why = StopReason::ContextFull;
                break;
            }
            let (l, dt) = self.run(&[next])?;
            self.prof.add_decode(dt);
            logits = l;
            elapsed = dt;
        }
        Ok((produced, why))
    }
}
