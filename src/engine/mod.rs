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
use crate::error::Result;
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
    pub prof: Profile,
}

impl<'a, O: Ops> Engine<'a, O> {
    pub fn new(model: impl Into<Model<'a>>, ops: O, n_ctx: usize, detail: bool) -> Self {
        let model = model.into();
        // Slabs for the layers that actually attend, which on a hybrid
        // architecture is a fraction of them.
        let cache = KvCache::new(model.n_kv_layer(), model.kv_dim(), n_ctx);
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
            prof,
        }
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

    /// Drop the cached history. The profile is kept, so a run that resets
    /// between prompts still reports totals across all of them.
    pub fn reset(&mut self) {
        self.cache.reset();
        if let Some(r) = self.recurrent.as_mut() {
            r.reset();
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

    /// Process a whole prompt in one pass. Returns logits for its last token.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let (logits, dt) = self.run(tokens)?;
        self.prof.add_prefill(tokens.len(), dt);
        Ok(logits)
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

        let (mut logits, dt) = self.run(prompt)?;
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
