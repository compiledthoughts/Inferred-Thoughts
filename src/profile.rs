//! The microscope: measurement that must not perturb what it measures.
//!
//! Two tiers, because they cost different amounts:
//!
//! * **Tier 1 — always collected.** Phase wall times (two `Instant::now()` per
//!   `forward` call), and the top-2 logits of each generated token. Both are
//!   many orders of magnitude below a token's cost, so there is no flag
//!   guarding collection; `--profile` only controls whether the report prints.
//! * **Tier 2 — `--profile-detail`.** Per-layer attention/FFN timing. Costs a
//!   pair of timestamps per layer half, ~20 ns each, against layers that take
//!   milliseconds.
//!
//! Three rules the design turns on:
//!
//! 1. **Events are plain data.** No `String`, no formatting, no allocation on
//!    the hot path — [`LayerEvent`] is `Copy` and 12 bytes, so recording one is
//!    a store into a pre-reserved `Vec`. All naming and formatting happens in
//!    [`Profile::report`], after the run.
//! 2. **Buffer, do not drain.** A channel send costs more than the push it
//!    replaces, and a drain thread would compete for memory bandwidth — the
//!    exact resource this engine exists to measure. The token boundary is
//!    already a serialized, ~100 ms event, which is live enough for anything a
//!    human reads.
//! 3. **Derive bytes, do not count them.** Weight traffic is a function of the
//!    weights' types and shapes, so a per-`matmul` counter would recompute a
//!    constant while contending across threads. [`Profile::weight_bytes`] is
//!    filled once from the model. When the expert cache arrives and misses
//!    become dynamic, the counter belongs *there* — at ~1.75 MiB per fetch,
//!    nowhere near a hot path.
//!
//! Deliberately absent: a logit lens. Projecting each layer's residual through
//! the LM head costs a full `n_embd x n_vocab` matmul per layer — roughly 29x a
//! token on the 0.6B. Layer-by-layer comparison against `llama-eval-callback`
//! already localizes numeric bugs (it is what found the RMSNorm accumulator),
//! so the lens stays unbuilt until that fails.

use std::time::{Duration, Instant};

/// Which half of a layer an event covers.
///
/// Attention and FFN are split because they scale differently: attention grows
/// with context length, the FFN does not. That difference is precisely what a
/// working KV cache should make visible.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Part {
    Attn = 0,
    Ffn = 1,
}

impl Part {
    pub fn label(self) -> &'static str {
        match self {
            Part::Attn => "attn",
            Part::Ffn => "ffn",
        }
    }
}

/// One timed layer half. `Copy`, no owned data — see rule 1 above.
#[derive(Clone, Copy, Debug)]
pub struct LayerEvent {
    /// Which `forward` call this belongs to; 0 is the prefill.
    pub step: u32,
    pub layer: u16,
    pub part: Part,
    pub ns: u32,
}

/// Everything known about one generated token.
///
/// `top1`/`top2` are here because the *margin* between them decides how to read
/// a disagreement with `llama-cli`. `CLAUDE.md` establishes that logit drift
/// against llama.cpp reaches ~1% by the final layer; a token whose top-2 logits
/// sit inside that band was always a coin flip and is not evidence of a bug. A
/// token that flips with a wide margin is.
#[derive(Clone, Copy, Debug)]
pub struct TokenRecord {
    pub pos: u32,
    pub id: u32,
    pub top2_id: u32,
    pub top1_logit: f32,
    pub top2_logit: f32,
    pub ns: u64,
}

impl TokenRecord {
    /// Gap between best and second-best logit, relative to the winner's
    /// magnitude — comparable to the ~1e-2 drift figure in `CLAUDE.md`.
    pub fn margin(&self) -> f32 {
        let denom = self.top1_logit.abs().max(1e-6);
        (self.top1_logit - self.top2_logit) / denom
    }
}

/// Counters and event buffers for one run.
///
/// Owned by the engine and threaded into the forward pass inside [`Ctx`].
#[derive(Debug, Default)]
pub struct Profile {
    /// Enables tier 2. Tier 1 is collected unconditionally.
    pub detail: bool,

    pub prefill_tokens: u64,
    pub prefill_ns: u64,
    pub decode_tokens: u64,
    pub decode_ns: u64,

    /// A one-time backend cost included in `prefill_ns`, and its name.
    ///
    /// Set from [`crate::Ops::setup_cost`] after the run, because the cost is
    /// incurred *inside* the first forward pass rather than around it: the CUDA
    /// backend places every expert on first sight of its tensor. `prefill_ns`
    /// is therefore correct as a wall clock and misleading as a rate, and
    /// [`Profile::phases`] uses this to report both honestly.
    ///
    /// `None` on every CPU backend, which is what keeps the 0.6B and 9B output
    /// byte-for-byte what it was.
    pub setup: Option<(u64, &'static str)>,

    /// Bytes of quantized weight read by one full forward pass, derived from
    /// the model's tensor types and shapes rather than counted.
    pub weight_bytes: u64,
    /// Bytes of KV cache written, and read back by attention.
    pub kv_write_bytes: u64,
    pub kv_read_bytes: u64,

    pub tokens: Vec<TokenRecord>,
    pub layers: Vec<LayerEvent>,

    step: u32,
}

impl Profile {
    pub fn new(detail: bool) -> Self {
        Self {
            detail,
            ..Default::default()
        }
    }

    /// Pre-reserve both buffers so no allocation happens mid-token.
    pub fn reserve(&mut self, n_layer: usize, max_tokens: usize) {
        self.tokens.reserve(max_tokens);
        if self.detail {
            // Two halves per layer per forward call, plus the prefill.
            self.layers.reserve(n_layer * 2 * (max_tokens + 1));
        }
    }

    /// Start of a `forward` call. Returns the step index the caller passes back
    /// to [`Profile::layer_end`]; step 0 is the prefill.
    pub fn begin_step(&mut self) -> u32 {
        let s = self.step;
        self.step += 1;
        s
    }

    /// Tier 2 only: `None` when detail is off, so the matching
    /// [`Profile::layer_end`] is a predictable branch and no timestamp is taken.
    #[inline]
    pub fn layer_begin(&self) -> Option<Instant> {
        if self.detail { Some(Instant::now()) } else { None }
    }

    #[inline]
    pub fn layer_end(&mut self, started: Option<Instant>, step: u32, layer: usize, part: Part) {
        if let Some(t0) = started {
            self.layers.push(LayerEvent {
                step,
                layer: layer as u16,
                part,
                ns: t0.elapsed().as_nanos().min(u32::MAX as u128) as u32,
            });
        }
    }

    pub fn add_prefill(&mut self, tokens: usize, elapsed: Duration) {
        self.prefill_tokens += tokens as u64;
        self.prefill_ns += elapsed.as_nanos() as u64;
    }

    pub fn add_decode(&mut self, elapsed: Duration) {
        self.decode_tokens += 1;
        self.decode_ns += elapsed.as_nanos() as u64;
    }

    /// Record the top two logits without sorting 150k values: one pass keeping
    /// the best and second-best. Tie-breaking must match
    /// [`crate::model::Qwen3::argmax`] — strictly greater, so ties keep the
    /// lower index.
    pub fn record_token(&mut self, pos: usize, logits: &[f32], elapsed: Duration) -> u32 {
        debug_assert!(logits.len() >= 2);
        let (mut i1, mut i2) = (0usize, usize::MAX);
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > logits[i1] {
                i2 = i1;
                i1 = i;
            } else if i2 == usize::MAX || v > logits[i2] {
                i2 = i;
            }
        }
        let rec = TokenRecord {
            pos: pos as u32,
            id: i1 as u32,
            top2_id: i2 as u32,
            top1_logit: logits[i1],
            top2_logit: logits[i2],
            ns: elapsed.as_nanos() as u64,
        };
        self.tokens.push(rec);
        rec.id
    }

    /// Tokens whose top-2 gap falls inside the logit drift characterized in
    /// `CLAUDE.md` — the ones that could legitimately differ from `llama-cli`
    /// without anything being wrong.
    pub fn at_risk(&self, threshold: f32) -> Vec<&TokenRecord> {
        self.tokens
            .iter()
            .filter(|t| t.margin() < threshold)
            .collect()
    }

    /// The two phases, one line each.
    ///
    /// Separated from [`Profile::report`] because these are the numbers every
    /// run wants and the rest are the numbers an investigation wants. Printing
    /// them from one place is what keeps the bare summary and `--profile` from
    /// disagreeing about the same run.
    ///
    /// The two are worth reading apart rather than as one average. Prefill is a
    /// batch over the whole prompt and decode is one token against the cache,
    /// so they differ by an order of magnitude on the same model — and a single
    /// blended tok/s over a short prompt and a long generation reports neither.
    pub fn phases(&self, out: &mut dyn std::io::Write) -> std::io::Result<()> {
        let ms = |ns: u64| ns as f64 / 1e6;

        // **The one-time cost comes out before the rate is computed**, and is
        // printed on its own line rather than folded away. It is real time the
        // user waited, so hiding it would trade one wrong number for another;
        // what was wrong was dividing it by the prompt length and calling the
        // result throughput.
        if let Some((setup_ns, label)) = self.setup {
            writeln!(
                out,
                "setup    {:>6}      {:>9.1} ms                    one-time {label}, inside the first pass",
                "",
                ms(setup_ns),
            )?;
        }
        if self.prefill_tokens > 0 {
            let setup_ns = self.setup.map_or(0, |(ns, _)| ns);
            // Saturating, and it says so if it saturated. A setup cost larger
            // than the pass that contained it means the two clocks disagree
            // about what they measured, which is worth a line rather than a
            // plausible-looking rate — the failure this whole change exists to
            // stop.
            let net_ns = self.prefill_ns.saturating_sub(setup_ns);
            if setup_ns > self.prefill_ns {
                writeln!(
                    out,
                    "prefill  {:>6} tok  {:>9.1} ms   NO RATE: {label_ns:.1} ms of setup exceeds the                      {:.1} ms pass that contained it",
                    self.prefill_tokens,
                    ms(self.prefill_ns),
                    ms(self.prefill_ns),
                    label_ns = ms(setup_ns),
                )?;
            } else {
                let s = net_ns as f64 / 1e9;
                writeln!(
                    out,
                    "prefill  {:>6} tok  {:>9.1} ms  {:>8.1} tok/s{}",
                    self.prefill_tokens,
                    ms(net_ns),
                    self.prefill_tokens as f64 / s.max(1e-9),
                    if setup_ns > 0 { "  net of setup" } else { "" },
                )?;
            }
        }
        if self.decode_tokens > 0 {
            let s = self.decode_ns as f64 / 1e9;
            writeln!(
                out,
                "decode   {:>6} tok  {:>9.1} ms  {:>8.2} tok/s  {:>7.1} ms/tok",
                self.decode_tokens,
                ms(self.decode_ns),
                self.decode_tokens as f64 / s.max(1e-9),
                ms(self.decode_ns) / self.decode_tokens as f64,
            )?;
        }
        Ok(())
    }

    /// Everything beyond the phases. Never called on the hot path.
    ///
    /// Deliberately does *not* repeat [`Profile::phases`]. Those are printed on
    /// every run, so a caller that also asks for this has already shown them,
    /// and printing them again here would put them after the wall-clock line
    /// that summarizes them.
    pub fn report(&self, out: &mut dyn std::io::Write) -> std::io::Result<()> {
        let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);

        writeln!(out, "--- profile ---")?;

        // The thesis is about bytes, so time alone measures only the symptom.
        // One decode step reads every weight exactly once, which makes the
        // implied bandwidth directly comparable to the 448 GB/s and 28.6 GB/s
        // figures in CLAUDE.md.
        writeln!(
            out,
            "weights  {:.3} GiB per forward pass",
            gib(self.weight_bytes)
        )?;
        if self.decode_tokens > 0 {
            let secs = self.decode_ns as f64 / 1e9 / self.decode_tokens as f64;
            let per_token =
                self.weight_bytes as f64 + self.kv_read_bytes as f64 / self.decode_tokens as f64;
            writeln!(
                out,
                "         {:.1} GB/s effective during decode",
                per_token / secs / 1e9
            )?;
        }
        writeln!(
            out,
            "kv       {:.1} MiB written, {:.1} MiB read back",
            self.kv_write_bytes as f64 / 1048576.0,
            self.kv_read_bytes as f64 / 1048576.0,
        )?;

        if !self.tokens.is_empty() {
            let mut margins: Vec<f32> = self.tokens.iter().map(|t| t.margin()).collect();
            margins.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let at_risk = self.at_risk(0.01).len();
            writeln!(
                out,
                "\nmargin   min {:.4}  median {:.4}  ({} of {} tokens inside the 1% drift band)",
                margins[0],
                margins[margins.len() / 2],
                at_risk,
                self.tokens.len(),
            )?;
            if at_risk > 0 {
                writeln!(
                    out,
                    "         those may legitimately differ from llama-cli; see CLAUDE.md"
                )?;
            }
        }

        if self.detail && !self.layers.is_empty() {
            self.report_layers(out)?;
        }
        Ok(())
    }

    /// Per-layer totals, split by half. Prefill (step 0) is separated from
    /// decode because they are compute- and memory-bound respectively.
    fn report_layers(&self, out: &mut dyn std::io::Write) -> std::io::Result<()> {
        let n_layer = self.layers.iter().map(|e| e.layer as usize).max().unwrap_or(0) + 1;
        let mut prefill = vec![[0u64; 2]; n_layer];
        let mut decode = vec![[0u64; 2]; n_layer];
        for e in &self.layers {
            let bucket = if e.step == 0 { &mut prefill } else { &mut decode };
            bucket[e.layer as usize][e.part as usize] += e.ns as u64;
        }

        writeln!(out, "\nper-layer totals (ms)")?;
        writeln!(
            out,
            "{:>5}  {:>10}  {:>10}  {:>10}  {:>10}",
            "layer", "pre/attn", "pre/ffn", "dec/attn", "dec/ffn"
        )?;
        for il in 0..n_layer {
            writeln!(
                out,
                "{il:>5}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}",
                prefill[il][0] as f64 / 1e6,
                prefill[il][1] as f64 / 1e6,
                decode[il][0] as f64 / 1e6,
                decode[il][1] as f64 / 1e6,
            )?;
        }
        let sum = |v: &[[u64; 2]], p: usize| v.iter().map(|l| l[p]).sum::<u64>() as f64 / 1e6;
        writeln!(
            out,
            "{:>5}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}",
            "all",
            sum(&prefill, 0),
            sum(&prefill, 1),
            sum(&decode, 0),
            sum(&decode, 1),
        )
    }

    /// Machine-readable form, for diffing two runs against each other — which
    /// is how the overhead of `--profile-detail` gets verified rather than
    /// asserted.
    pub fn to_json(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::with_capacity(256 + self.tokens.len() * 96);
        s.push_str("{\n");
        let _ = writeln!(s, "  \"prefill_tokens\": {},", self.prefill_tokens);
        let _ = writeln!(s, "  \"prefill_ns\": {},", self.prefill_ns);
        let _ = writeln!(s, "  \"decode_tokens\": {},", self.decode_tokens);
        let _ = writeln!(s, "  \"decode_ns\": {},", self.decode_ns);
        let _ = writeln!(s, "  \"weight_bytes\": {},", self.weight_bytes);
        let _ = writeln!(s, "  \"kv_write_bytes\": {},", self.kv_write_bytes);
        let _ = writeln!(s, "  \"kv_read_bytes\": {},", self.kv_read_bytes);
        s.push_str("  \"tokens\": [\n");
        for (i, t) in self.tokens.iter().enumerate() {
            let comma = if i + 1 == self.tokens.len() { "" } else { "," };
            let _ = writeln!(
                s,
                "    {{\"pos\": {}, \"id\": {}, \"top2_id\": {}, \"top1\": {:e}, \"top2\": {:e}, \"ns\": {}}}{comma}",
                t.pos, t.id, t.top2_id, t.top1_logit, t.top2_logit, t.ns
            );
        }
        s.push_str("  ],\n  \"layers\": [\n");
        for (i, e) in self.layers.iter().enumerate() {
            let comma = if i + 1 == self.layers.len() { "" } else { "," };
            let _ = writeln!(
                s,
                "    {{\"step\": {}, \"layer\": {}, \"part\": \"{}\", \"ns\": {}}}{comma}",
                e.step,
                e.layer,
                e.part.label(),
                e.ns
            );
        }
        s.push_str("  ]\n}\n");
        s
    }
}

/// What the forward pass is handed for observation: the Stage 4 tensor tracer
/// and the profiler, bundled so adding a third does not change every signature.
pub struct Ctx<'a> {
    tracer: &'a mut dyn FnMut(&str, usize, &[f32]),
    pub prof: &'a mut Profile,
}

impl<'a> Ctx<'a> {
    pub fn new(tracer: &'a mut dyn FnMut(&str, usize, &[f32]), prof: &'a mut Profile) -> Self {
        Self { tracer, prof }
    }

    /// Hand a whole intermediate tensor to the tracer, as
    /// `llama-eval-callback` prints it. Only `inferred trace` installs a real
    /// tracer; everywhere else this is an indirect call to an empty closure.
    #[inline]
    pub fn trace(&mut self, name: &str, layer: usize, t: &[f32]) {
        (self.tracer)(name, layer, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_top_two_and_agrees_with_argmax() {
        let mut p = Profile::new(false);
        let logits = [1.0f32, 5.0, 3.0, -2.0];
        let id = p.record_token(0, &logits, Duration::from_millis(1));
        assert_eq!(id, 1);
        let rec = p.tokens[0];
        assert_eq!(rec.top2_id, 2);
        assert_eq!(rec.top1_logit, 5.0);
        assert_eq!(rec.top2_logit, 3.0);
    }

    #[test]
    fn top_two_handles_a_leading_maximum() {
        // The scan starts at index 1, so a winner at index 0 exercises the
        // branch that seeds the runner-up.
        let mut p = Profile::new(false);
        assert_eq!(p.record_token(0, &[9.0, 1.0, 2.0], Duration::ZERO), 0);
        assert_eq!(p.tokens[0].top2_id, 2);
    }

    #[test]
    fn margin_flags_tokens_inside_the_drift_band() {
        let mut p = Profile::new(false);
        p.record_token(0, &[10.0, 9.999], Duration::ZERO); // 1e-4 apart
        p.record_token(1, &[10.0, 1.0], Duration::ZERO); // far apart
        assert_eq!(p.at_risk(0.01).len(), 1);
        assert!(p.tokens[0].margin() < 0.01);
        assert!(p.tokens[1].margin() > 0.01);
    }

    #[test]
    fn detail_off_records_no_layer_events() {
        let mut p = Profile::new(false);
        let step = p.begin_step();
        let t = p.layer_begin();
        assert!(t.is_none());
        p.layer_end(t, step, 0, Part::Attn);
        assert!(p.layers.is_empty());
    }

    #[test]
    fn detail_on_records_one_event_per_half() {
        let mut p = Profile::new(true);
        let step = p.begin_step();
        for il in 0..3 {
            let t = p.layer_begin();
            p.layer_end(t, step, il, Part::Attn);
            let t = p.layer_begin();
            p.layer_end(t, step, il, Part::Ffn);
        }
        assert_eq!(p.layers.len(), 6);
        assert_eq!(p.layers[0].part, Part::Attn);
        assert_eq!(p.layers[1].part, Part::Ffn);
        assert!(p.layers.iter().all(|e| e.step == 0));
    }

    /// The event must stay small and trivially copyable, or "recording is a
    /// store into a reserved buffer" stops being true.
    #[test]
    fn layer_event_is_pod_sized() {
        assert!(std::mem::size_of::<LayerEvent>() <= 16);
        assert!(std::mem::size_of::<TokenRecord>() <= 32);
    }
}

/// Nanoseconds this process has actually spent **on a CPU**, or `None` where
/// the kernel does not report it.
///
/// **Built to answer one question the wall clock cannot**: of a 60 ms token
/// with only ~24 ms of measured kernel time, is the host *working* or
/// *waiting*? Those have opposite fixes — fewer host operations versus faster
/// kernels — and three hypotheses about that gap have already been wrong.
///
/// `/proc/self/schedstat` field 0 is time on-CPU in nanoseconds, which is
/// exactly the quantity wanted and needs no libc dependency. Verified to count:
/// a 500 ms busy loop reports 497 ms.
///
/// **It measures spinning as work**, which matters here: a CUDA context created
/// with `CU_CTX_SCHED_AUTO` busy-waits on synchronization, so a host blocked in
/// `cuMemcpyDtoH` still burns a core. `--cuda-blocking` exists to take that
/// confound off the table — with it, on-CPU time is host work and nothing else.
pub fn cpu_time_ns() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/schedstat").ok()?;
    s.split_whitespace().next()?.parse().ok()
}
