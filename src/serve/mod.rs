//! An OpenAI-compatible HTTP endpoint, so an existing chat UI can drive the
//! engine.
//!
//! **This exists as an instrument as much as a convenience.** The deepest run
//! in this project before it was about 1800 tokens. A chat session exercises
//! things nothing else has: context growth over many turns, whether the caches
//! survive a turn boundary, whether `forget_state` actually does what it claims,
//! and hours of uptime. The 35B's whole story rests on a 262,144-token context
//! that has never been approached.
//!
//! # Why hand-rolled HTTP
//!
//! The engine is synchronous and holds one session, so a local server handles
//! one request at a time no matter what is underneath it. `TcpListener` plus a
//! blocking read is a few hundred lines; an async runtime would roughly triple
//! the dependency tree to provide concurrency that would immediately serialize
//! on the engine's mutex anyway. JSON is `serde_json` because parsing arbitrary
//! client payloads is where hand-rolling actually breaks.
//!
//! # The session, and the constraint GatedDeltaNet imposes
//!
//! A chat client re-sends the whole conversation every turn. Re-prefilling all
//! of it each time is O(n^2), and prefill used to run one token at a time,
//! so a long chat would become unusable. Instead the engine keeps its state and
//! only the *new* tokens are prefilled.
//!
//! That works only while the conversation grows by appending, which is exactly
//! what `ChatMl::wrap_turns` guarantees by rendering turns in order and opening
//! the assistant turn last. When the new prompt is not an extension of what has
//! been processed — an edit, a regeneration, a branch, a different client — the
//! session resets and re-runs from the beginning.
//!
//! **The reset is not laziness, it is forced.** A KV cache can be truncated to
//! a common prefix and reused, but a GatedDeltaNet layer's recurrent state
//! cannot: it is a single matrix that has absorbed every token, with no record
//! of how to undo one. `HANDOFF.md` §3 says so — "the recurrent state cannot be
//! partially reused — it only survives via checkpoints." Checkpointing it is
//! possible (state is ~2 MB per layer) and is the obvious future work; until
//! then a rewind costs a full re-run.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::engine::{Checkpoint, Engine};
use crate::error::{Error, Result};
use crate::ops::Ops;
use crate::tok::Tokenizer;
use crate::tok::chat::ChatMl;

/// Tokens between checkpoints, and the prefill slice size.
///
/// Sets the worst-case re-prefill after a divergence: land anywhere inside a
/// window and the tokens back to its start are re-run. At the 35B's measured
/// ~19 ms/token at 24k depth, 2,048 is ~39 s of worst case against the ~8
/// minutes a full restart costs there.
const CHECKPOINT_EVERY: usize = 2048;

/// The closest checkpoints are ever spaced, however small the context: each one
/// copies the whole recurrent state (~150 MiB on Qwen3.8-Flash-Next) to the host.
const CHECKPOINT_MIN_SPACING: usize = 256;

/// Tokens between checkpoints for a context of `n_ctx`: the eight return points
/// spread across it, between [`CHECKPOINT_MIN_SPACING`] and [`CHECKPOINT_EVERY`].
///
/// **A fixed 2,048 never checkpointed Qwen3.8-Flash-Next at all**: it refuses
/// contexts past 2,051 cells, so every diverging turn re-prefilled the whole
/// conversation from token zero. At `--ctx 2048` this is 256, so a divergence
/// re-runs at most 256 tokens. The 35B at `--ctx 32768` still gets 2,048.
fn checkpoint_spacing(n_ctx: usize) -> usize {
    (n_ctx / MAX_CHECKPOINTS).clamp(CHECKPOINT_MIN_SPACING, CHECKPOINT_EVERY)
}

/// Return points kept at once.
///
/// One is the whole model's recurrent state — 84 MiB on the 35B, and **fixed
/// whatever the context depth**, because the KV half of a rewind needs no copy
/// at all. Eight is 672 MB of host memory, which on a 16 GB WSL guest already
/// holding 4.49 GiB of pinned expert tier is affordable and not free.
const MAX_CHECKPOINTS: usize = 8;

/// What the CLI hands the server.
pub struct ServeOpts {
    pub port: u16,
    /// Advertised through `/v1/models` and echoed in responses. A client that
    /// asks for a different one still gets this: there is one model loaded.
    pub model_id: String,
    pub max_tokens: usize,
    /// Print each request's body and the prompt it renders to.
    pub verbose: bool,
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    messages: Vec<Message>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
}

#[derive(Deserialize)]
struct Message {
    #[serde(default = "user_role")]
    role: String,
    /// Either a string or the array-of-parts form some clients send. Both are
    /// flattened to text by [`content_text`].
    #[serde(default)]
    content: Value,
}

fn user_role() -> String {
    "user".to_string()
}

/// Flatten a message's content to plain text.
///
/// OpenAI clients send either a bare string or `[{"type":"text","text":...}]`.
/// Anything else — an image part, say — is dropped rather than refused, because
/// a UI that sends one should still get a usable answer to the text.
fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// The profiler's running totals, so a request can be reported on its own.
///
/// [`crate::profile::Profile`] accumulates for the life of the process, which is
/// what the CLI wants and not what a server does — a chat wants to know what
/// *this turn* cost. Differencing a snapshot is enough and keeps the profiler
/// free of a second notion of "session".
#[derive(Clone, Copy)]
struct Mark {
    prefill_tokens: u64,
    prefill_ns: u64,
    decode_tokens: u64,
    decode_ns: u64,
    /// One-time backend setup, cumulative — see [`Ops::setup_cost`].
    ///
    /// **Differenced like the rest, because it is not a per-turn cost and the
    /// turn that pays it is the one that looks slow.** The CUDA backend places
    /// every expert on first sight of its tensor, which happens inside the
    /// first forward pass, so turn one's prefill contains ~25 s of placement on
    /// the 35B. Reporting that as `prefill N tok ... tok/s` is the same wrong
    /// basis `Profile::phases` was fixed for, in the one place a user actually
    /// watches it happen.
    setup_ns: u64,
    setup_label: &'static str,
}

impl Mark {
    fn take<O: Ops>(e: &Engine<'_, O>) -> Self {
        let (setup_ns, setup_label) = e.ops.setup_cost().unwrap_or((0, "setup"));
        Self {
            prefill_tokens: e.prof.prefill_tokens,
            prefill_ns: e.prof.prefill_ns,
            decode_tokens: e.prof.decode_tokens,
            decode_ns: e.prof.decode_ns,
            setup_ns,
            setup_label,
        }
    }
}

/// What this turn cost, as JSON, so a **client** sees the decomposition the
/// terminal prints instead of inferring it from wall time.
///
/// This exists because inferring it does not work, and that was measured the
/// hard way on 09-09: a 12.2 s turn returning 120 tokens reads as 10 tok/s of
/// decode, and was actually 8.7 s of re-prefill after a checkpoint restore plus
/// 3.4 s of decode at 35 tok/s. Every quantity needed to tell those apart was
/// already computed and went only to stderr.
///
/// `prompt_*` and `predicted_*` follow llama.cpp's `timings` object so tooling
/// that reads theirs reads ours. The rest is ours: `setup_ms` is the one-time
/// backend cost that lands inside whichever turn pays it, and `cache_*` say how
/// much of the conversation was reused rather than re-run -- the field that
/// would have made the above self-evident.
fn timings_json<O: Ops>(engine: &Engine<'_, O>, before: Mark, how: Resume) -> Value {
    let now = Mark::take(engine);
    let setup_ns = now.setup_ns.saturating_sub(before.setup_ns);
    let prompt_n = now.prefill_tokens - before.prefill_tokens;
    let prompt_ns = (now.prefill_ns - before.prefill_ns).saturating_sub(setup_ns);
    let predicted_n = now.decode_tokens - before.decode_tokens;
    let predicted_ns = now.decode_ns - before.decode_ns;
    let per_s = |n: u64, ns: u64| if ns == 0 { 0.0 } else { n as f64 / (ns as f64 / 1e9) };
    json!({
        "prompt_n": prompt_n,
        "prompt_ms": prompt_ns as f64 / 1e6,
        "prompt_per_second": per_s(prompt_n, prompt_ns),
        "predicted_n": predicted_n,
        "predicted_ms": predicted_ns as f64 / 1e6,
        "predicted_per_second": per_s(predicted_n, predicted_ns),
        "setup_ms": setup_ns as f64 / 1e6,
        "cache_reuse": how.at(),
        "cache_label": how.label(),
        "position": engine.pos(),
        "n_ctx": engine.n_ctx(),
    })
}

/// Print what this turn cost, in the shape `inferred generate` prints.
fn report<O: Ops>(engine: &Engine<'_, O>, before: Mark) {
    let now = Mark::take(engine);
    let line = |name: &str, tok: u64, ns: u64| {
        if tok == 0 {
            return;
        }
        let ms = ns as f64 / 1e6;
        eprintln!(
            "  {name:<9}{tok:>6} tok  {ms:>9.1} ms  {:>8.2} tok/s  {:>7.1} ms/tok",
            tok as f64 / (ns as f64 / 1e9).max(1e-9),
            ms / tok as f64,
        );
    };
    // Comes out before the rate, and is printed rather than hidden: it is real
    // time the user waited. What was wrong was dividing it by the prompt length
    // and calling the result throughput.
    let setup_ns = now.setup_ns.saturating_sub(before.setup_ns);
    let prefill_ns = now.prefill_ns - before.prefill_ns;
    if setup_ns > 0 {
        eprintln!(
            "  {:<9}{:>6}      {:>9.1} ms                    one-time {}, inside this turn",
            "setup",
            "",
            setup_ns as f64 / 1e6,
            now.setup_label,
        );
    }
    line(
        "prefill",
        now.prefill_tokens - before.prefill_tokens,
        prefill_ns.saturating_sub(setup_ns),
    );
    line(
        "decode",
        now.decode_tokens - before.decode_tokens,
        now.decode_ns - before.decode_ns,
    );
    // Whatever the backend can say about itself. A no-op on every CPU
    // backend; on CUDA, and only under `--profile-device`, the launch and
    // residency counters for the turn just finished.
    engine.ops.device_report();
    let kv = engine.kv_capacity_bytes() as f64 / 1048576.0;
    let rs = engine.recurrent_capacity_bytes() as f64 / 1048576.0;
    eprintln!(
        "  kv       {:>6} / {} positions  {kv:.0} MiB{}",
        engine.pos(),
        engine.n_ctx(),
        if rs > 0.0 {
            format!(", recurrent {rs:.0} MiB")
        } else {
            String::new()
        }
    );
}

/// One loaded model, its engine, and the conversation it has consumed.
///
/// The state that matters is `rendered`: the exact text the engine has been
/// fed. A turn is cheap when the new conversation *starts with* that text,
/// because then only the difference has to be tokenized and run.
///
/// **Matching on text rather than on tokens is the point.** The obvious design
/// re-tokenizes the whole conversation each turn and looks for a common token
/// prefix. It was built that way first and it barely helped: the model emits
/// tokens, the client sends the text back, and re-tokenizing splits differently
/// at the seam — a generated newline after a prompt that already ends in one
/// merges into a single token, and every token after it shifts. Measured, a
/// second turn re-ran 51 of 62 tokens that way. Comparing text sidesteps the
/// re-tokenization entirely.
/// How a turn picked up where the last one left off.
///
/// Four outcomes with very different costs, and the log has to say which:
/// a restart at 24k positions is about eight minutes, a restore is the tokens
/// back to the last checkpoint, and a continuation is free.
#[derive(Clone, Copy)]
enum Resume {
    /// The text prefix matched: a pure continuation, nothing re-run.
    Continued(usize),
    /// The KV log was truncated to the divergence. Exact, and free — only
    /// possible with no recurrent state to rewind.
    Rewound(usize),
    /// Returned to the newest checkpoint at or before the divergence.
    Restored(usize),
    /// No checkpoint was early enough. The whole conversation runs again.
    Restarted,
}

impl Resume {
    fn at(self) -> usize {
        match self {
            Resume::Continued(n) | Resume::Rewound(n) | Resume::Restored(n) => n,
            Resume::Restarted => 0,
        }
    }

    fn label(self) -> String {
        match self {
            Resume::Continued(n) => format!("continued at {n}"),
            Resume::Rewound(n) => format!("rewound to {n}"),
            Resume::Restored(n) => format!("restored from checkpoint at {n}"),
            Resume::Restarted => "restarted from zero".to_string(),
        }
    }
}

struct Session<'a, O: Ops> {
    engine: Engine<'a, O>,
    tk: Tokenizer,
    chat: ChatMl,
    /// Conversation text the engine has consumed, including its own output.
    rendered: String,
    /// Tokens behind `rendered`, for reporting.
    consumed: usize,
    /// The token ids the engine consumed, in order.
    ///
    /// # BROKEN as a prefix, and measured to be
    ///
    /// This is a **concatenation of independent tokenizations** — turn one's
    /// full encode, then each later turn's new text encoded in isolation, then
    /// the ids the model generated. `advance` compares it against a *single*
    /// encode of the whole conversation, and the two disagree at the first turn
    /// boundary, for exactly the reason recorded on this struct: a generated
    /// newline merges with the newline the next prompt opens with.
    ///
    /// So the common prefix collapses to the end of turn one however late the
    /// real divergence is. Measured on a ten-turn session at 8,013 positions
    /// with checkpoints at ~2410, ~4823 and ~7231: editing turn seven should
    /// have restored from the checkpoint at 4823 and instead restarted from
    /// zero, because the scan stopped at ~805.
    ///
    /// **The fix is to tokenize the whole conversation every turn** and keep
    /// this as that tokenization, so the engine's positions correspond to a
    /// single encode by construction. Re-tokenizing 24k tokens costs tens of
    /// milliseconds against a prefill of ~21 ms *per token*, so the cost the
    /// text fast path exists to avoid is not worth what it breaks. The boundary
    /// effect then shows up only where new text joins — a handful of tokens at
    /// the end — instead of severing the prefix at turn one.
    tokens: Vec<u32>,
    /// Return points, oldest first. Empty for an architecture with no
    /// recurrent state, where `Engine::rewind` reaches any position for free.
    checkpoints: Vec<Checkpoint>,
    /// Position of the newest checkpoint, so spacing is measured across turns
    /// rather than within one.
    ///
    /// **Without this the ladder stays empty in the workload it exists for.**
    /// The first version took a checkpoint only between slices of a single
    /// prefill, so a turn adding fewer than `CHECKPOINT_EVERY` tokens took
    /// none — which is every ordinary turn. Measured on a real Cline session:
    /// seven turns grew the conversation to 16,050 positions with zero
    /// checkpoints, and turn 8 diverged and re-ran all 15,275 tokens, 170.9 s.
    last_ckpt: usize,
}

impl<O: Ops> Session<'_, O> {
    /// Bring the engine up to `want`, reusing what it has already consumed.
    ///
    /// Returns the last token's logits, how it resumed, and how many tokens
    /// actually had to run.
    ///
    /// **The path taken, not a position to infer it from.** The first version
    /// returned the position and the caller worked out the label with
    /// arithmetic, which cannot tell a continuation from a checkpoint restore —
    /// both resume at a nonzero position and both then run the remainder. It
    /// printed `continued at 4096` for a restore, hiding the mechanism this
    /// exists for on the very run that first exercised it.
    fn advance(&mut self, want: &str) -> Result<(Vec<f32>, Resume, usize)> {
        // **The text prefix stays the fast path, and that is deliberate.**
        // Comparing tokens instead was tried and rejected for a measured
        // reason, recorded on `Session`: the model emits a newline, the client
        // sends it back, and re-tokenizing merges it with the newline the next
        // prompt opens with, shifting every token after it. A second turn
        // re-ran 51 of 62 tokens that way. A pure continuation must not pay
        // that, so it never re-tokenizes the prefix at all.
        //
        // Tokens are for the case the text prefix cannot express: divergence.
        // The two must not drift: `return_to` truncates `tokens` to a position
        // taken from the engine, so a mismatch would silently misalign the
        // common-prefix scan against what was actually consumed.
        debug_assert_eq!(self.tokens.len(), self.consumed, "token log and position disagree");

        let (at_how, tokens) = if !self.rendered.is_empty() && want.starts_with(self.rendered.as_str())
        {
            let text = &want[self.rendered.len()..];
            (Resume::Continued(self.consumed), self.tk.encode(text, false, true))
        } else {
            // An edit, a branch, a condensed history, or a client that rewrote
            // its system prompt. Before checkpoints this restarted from token
            // zero, which at 24k positions and the measured 52 tok/s is about
            // eight minutes.
            let want_tokens = self.tk.encode(want, true, true);
            let common = want_tokens
                .iter()
                .zip(&self.tokens)
                .take_while(|(a, b)| a == b)
                .count();
            // **A cancelled turn leaves us holding more than the client sent.**
            // `absorb` records the partial answer, so a client that cancels and
            // resends the same prompt asks for a conversation we already have
            // in full: `common` reaches the end of `want` while `self.tokens`
            // runs on past it. That is a regeneration, not an empty request.
            //
            // Rewinding one token gives the pass something to run and returns
            // the last position's logits, which is what generation needs. It
            // was previously a 400, "the conversation added no new text" —
            // found by a test that cancelled and then resent.
            let target = if common == want_tokens.len() {
                common.saturating_sub(1)
            } else {
                common
            };
            let how = self.return_to(target)?;
            (how, want_tokens[how.at()..].to_vec())
        };

        if tokens.is_empty() {
            return Err(Error::InconsistentArchitecture {
                what: "chat request",
                detail: "the conversation added no new text".to_string(),
            });
        }

        let at = at_how.at();
        let need = at + tokens.len();
        if need > self.engine.n_ctx() {
            return Err(Error::InconsistentArchitecture {
                what: "context",
                detail: format!(
                    "needs {need} tokens ({at} held + {} new) but the context is {}. Restart with --ctx {} or larger",
                    tokens.len(),
                    self.engine.n_ctx(),
                    need.next_power_of_two(),
                ),
            });
        }

        // Prefilled in slices so a checkpoint can be taken between them. Each
        // slice is an ordinary prefill at a later `start_pos` — the engine
        // already chunks internally for memory — so this cannot change a bit,
        // which `split_prefill_equals_single_prefill` pins down.
        let mut logits = Vec::new();
        let mut done = 0usize;
        let spacing = checkpoint_spacing(self.engine.n_ctx());
        while done < tokens.len() {
            let take = spacing.min(tokens.len() - done);
            logits = self.engine.prefill(&tokens[done..done + take])?;
            self.tokens.extend_from_slice(&tokens[done..done + take]);
            done += take;
            self.consumed += take;
            // **Spacing measured from the last checkpoint, not from the start
            // of this turn.** The previous condition was `done < tokens.len()`,
            // which takes a checkpoint only between slices — so a turn shorter
            // than a slice took none, and an incrementally growing conversation
            // accumulated nothing to return to. Checkpointing the final slice
            // is not "a copy of where we already are": the next turn continues
            // past it, which is precisely what makes it a return point for the
            // divergence after that.
            if self.consumed - self.last_ckpt >= spacing {
                self.take_checkpoint();
            }
        }
        self.rendered = want.to_string();
        Ok((logits, at_how, tokens.len()))
    }

    /// Put the engine back at or before `common`, and say where it landed.
    ///
    /// Two mechanisms, tried in order, because they cost differently:
    ///
    ///   attention layers only   `rewind` truncates the KV log to any position,
    ///                           free, exact — `qwen3` never needs a checkpoint
    ///   recurrent state         only a saved copy will do, so the best
    ///                           available is the newest checkpoint at or
    ///                           before the divergence
    ///
    /// Falls back to a full reset when no checkpoint is early enough, which is
    /// the old behaviour and is still correct.
    fn return_to(&mut self, common: usize) -> Result<Resume> {
        if self.engine.rewind(common) {
            self.tokens.truncate(common);
            self.consumed = common;
            self.last_ckpt = self.last_ckpt.min(common);
            return Ok(Resume::Rewound(common));
        }
        if let Some(i) = self
            .checkpoints
            .iter()
            .rposition(|c| c.pos() <= common && c.pos() > 0)
        {
            let c = self.checkpoints[i].clone();
            self.engine.restore(&c)?;
            self.checkpoints.truncate(i + 1);
            self.tokens.truncate(c.pos());
            self.consumed = c.pos();
            self.last_ckpt = c.pos();
            return Ok(Resume::Restored(c.pos()));
        }
        self.engine.reset();
        self.checkpoints.clear();
        self.tokens.clear();
        self.consumed = 0;
        self.last_ckpt = 0;
        Ok(Resume::Restarted)
    }

    /// Save a return point, keeping at most [`MAX_CHECKPOINTS`].
    ///
    /// When full, every second one is dropped, keeping the newest. The ladder
    /// then spans the whole conversation at half the resolution rather than
    /// covering only its start or only its end — a divergence can land
    /// anywhere: early when a client rewrites its system prompt, late when it
    /// appends a tool result, in the middle when it condenses history.
    ///
    /// Repeated thinning leaves recent points dense and old ones sparse, which
    /// is the right bias: the cost of landing between two checkpoints is the
    /// tokens back to the earlier one, and re-running old tokens is no cheaper
    /// than re-running recent ones.
    fn take_checkpoint(&mut self) {
        let Some(c) = self.engine.checkpoint() else {
            // No recurrent state: `rewind` reaches any position for free and a
            // checkpoint would copy nothing.
            return;
        };
        self.last_ckpt = self.consumed;
        self.checkpoints.push(c);
        if self.checkpoints.len() > MAX_CHECKPOINTS {
            let mut keep = Vec::with_capacity(MAX_CHECKPOINTS);
            for (i, c) in self.checkpoints.drain(..).enumerate() {
                if i % 2 == 1 || i + 1 == MAX_CHECKPOINTS + 1 {
                    keep.push(c);
                }
            }
            self.checkpoints = keep;
        }
    }

    /// Record what the model produced, so the next turn sees it as a prefix.
    ///
    /// The turn-ending marker is deliberately *not* added: generation stops
    /// before consuming it, so the engine has not seen it, and the next
    /// request's rendering supplies it as part of the new text.
    fn absorb(&mut self, text: &str, ids: &[u32]) {
        self.rendered.push_str(text);
        self.consumed += ids.len();
        // The engine consumed these, so the next turn's common-prefix scan has
        // to see them. Without this every turn would diverge at the start of
        // the model's own previous answer.
        self.tokens.extend_from_slice(ids);
    }
}

/// Serve until the process is stopped.
pub fn serve<O: Ops>(
    engine: Engine<'_, O>,
    tk: Tokenizer,
    chat: ChatMl,
    opts: ServeOpts,
) -> Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", opts.port)).map_err(|source| Error::Io {
        path: format!("127.0.0.1:{}", opts.port),
        source,
    })?;
    let base = format!("http://127.0.0.1:{}", opts.port);
    eprintln!("serving {} (ctx {})", opts.model_id, engine.n_ctx());
    eprintln!("  base url   {base}          <- most clients want this");
    eprintln!("  or         {base}/v1       <- if the client adds /chat/completions itself");
    eprintln!("  either works; the router matches on the path suffix");
    // What this process will actually run. A number taken from a server is only
    // attributable if the arms that produced it are on the record beside it.
    for (label, value) in engine.ops.config_report() {
        eprintln!("  {label:<9}{value}");
    }

    let mut session = Session {
        engine,
        tk,
        chat,
        rendered: String::new(),
        tokens: Vec::new(),
        checkpoints: Vec::new(),
        last_ckpt: 0,
        consumed: 0,
    };

    // One connection at a time. The engine holds a single session, so
    // concurrency here would only queue somewhere less obvious.
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept failed: {e}");
                continue;
            }
        };
        if let Err(e) = handle(&mut session, stream, &opts) {
            eprintln!("request failed: {e}");
        }
    }
    Ok(())
}

/// The text `serve --warmup` prefills once before the first request.
#[cfg(feature = "cuda")]
const WARMUP_TEXT: &str = include_str!("warmup.txt");

/// What the start-up warm-up did.
#[cfg(feature = "cuda")]
pub struct Warmup {
    /// Tokens prefilled, chat markers included.
    pub tokens: usize,
    /// Wall time of that prefill, one-time expert placement included.
    pub prefill_s: f64,
    /// Experts the re-placement moved between tiers.
    pub swaps: usize,
    /// Wall time of the re-placement.
    pub replace_s: f64,
}

/// Prefill a short built-in text once, re-place the expert pool by what it
/// read, and leave the engine empty for the first request.
///
/// **Why a warm-up at all.** Every expert must be addressable before the router
/// can name one, so `ExpertCache::table` places the pool eagerly, in the order
/// tensors are first seen: VRAM fills with the early layers and the late ones
/// land in the host tier, busy or not. Migration corrects that during a
/// session, but at most 200 swaps per 64 tokens, so the first requests run on a
/// placement that used no information. This spends a few seconds of start-up,
/// once, to begin from one that did; migration still follows the session.
///
/// The text is rendered as a chat turn, so it routes through the same markers a
/// request does, and it is long enough for tens of reads per expert. A few dozen
/// tokens would give a handful, which is the noise `ExpertCache::migrate`
/// records failing to rank.
///
/// Nothing of it survives but the placement and its read counts:
/// `Engine::reset` clears the KV cache and the recurrent state, and the session
/// is built after this returns, so its prefix matching never sees these tokens.
/// `the_serve_warmup_leaves_the_logits_bit_identical` holds that to the bit.
#[cfg(feature = "cuda")]
pub fn warm_up_experts(
    engine: &mut Engine<'_, &crate::Cuda>,
    tk: &Tokenizer,
    chat: &ChatMl,
) -> Result<Warmup> {
    let tokens = tk.encode(&chat.wrap(WARMUP_TEXT), true, true);
    if tokens.len() >= engine.n_ctx() {
        return Err(Error::InconsistentArchitecture {
            what: "warmup",
            detail: format!(
                "the warm-up text is {} tokens and the context is {}; raise --ctx",
                tokens.len(),
                engine.n_ctx()
            ),
        });
    }
    let t0 = std::time::Instant::now();
    engine.prefill(&tokens)?;
    let prefill_s = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let swaps = engine.ops.replace_experts_by_counts()?;
    let replace_s = t1.elapsed().as_secs_f64();
    engine.reset();
    if let Some(e) = engine.ops.take_error() {
        return Err(e);
    }
    Ok(Warmup {
        tokens: tokens.len(),
        prefill_s,
        swaps,
        replace_s,
    })
}

fn handle<O: Ops>(
    session: &mut Session<'_, O>,
    mut stream: TcpStream,
    opts: &ServeOpts,
) -> Result<()> {
    let (method, path, body) = match read_request(&mut stream) {
        Ok(r) => r,
        // A UI opens speculative connections it never writes to; that is not
        // worth a log line.
        Err(_) => return Ok(()),
    };

    // Clients disagree about where the base URL ends. Some want
    // `http://host:port` and append `/v1/chat/completions`; others want
    // `http://host:port/v1` and append `/chat/completions`. Configure one the
    // other way and the request arrives at `/v1/v1/chat/completions`, or with a
    // stray `%20` from a trailing space in a settings box.
    //
    // A local server has nothing to gain by being strict about that, so the
    // path is decoded, trimmed and matched on its suffix. The alternative is a
    // 404 whose message the user has to reverse-engineer, which is exactly how
    // this was found.
    let route = normalize_path(&path);
    if opts.verbose {
        eprintln!("--> {method} {route}  ({} bytes)", body.len());
    }

    match (method.as_str(), route.as_str()) {
        ("GET", p) if p.ends_with("/health") => {
            send_json(&mut stream, 200, &json!({"status": "ok"}))
        }
        ("GET", p) if p.ends_with("/models") => {
            let body = json!({
                "object": "list",
                "data": [{
                    "id": opts.model_id,
                    "object": "model",
                    "created": now(),
                    "owned_by": "inferred-thoughts",
                }],
            });
            send_json(&mut stream, 200, &body)
        }
        ("POST", p) if p.ends_with("/chat/completions") => {
            chat_completions(session, &mut stream, &body, opts)
        }
        ("OPTIONS", _) => send_head(&mut stream, 204, "text/plain", 0),
        _ => send_json(
            &mut stream,
            404,
            &json!({"error": {"message": format!("no route for {method} {route}")}}),
        ),
    }
}

fn chat_completions<O: Ops>(
    session: &mut Session<'_, O>,
    stream: &mut TcpStream,
    body: &[u8],
    opts: &ServeOpts,
) -> Result<()> {
    let req: ChatRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            return send_json(
                stream,
                400,
                &json!({"error": {"message": format!("could not parse request: {e}")}}),
            );
        }
    };

    let texts: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), content_text(&m.content)))
        .collect();
    let turns: Vec<(&str, &str)> = texts
        .iter()
        .map(|(r, c)| (r.as_str(), c.as_str()))
        .collect();
    let want = session.chat.wrap_turns(&turns);

    if opts.verbose {
        clipped("body    ", &String::from_utf8_lossy(body), 1200);
        for (role, content) in &turns {
            clipped(&format!("  {role:<9}"), content, 300);
        }
        // What actually reaches the tokenizer. Everything before this is
        // already in the engine, so this is the only text that costs anything.
        let new = want.strip_prefix(session.rendered.as_str()).unwrap_or(&want);
        clipped("rendered", &want, 400);
        clipped("new     ", new, 400);
    }

    let budget = req
        .max_tokens
        .or(req.max_completion_tokens)
        .unwrap_or(opts.max_tokens);

    let mark = Mark::take(&session.engine);
    // Announced before `advance`, because a long prefill is minutes of silence
    // otherwise and the count is the only clue to why.
    let approx = want.len().saturating_sub(session.rendered.len()) / 4;
    if approx > 2048 {
        eprintln!("chat: ~{approx} new tokens to prefill; this will take a while");
    }
    let (logits, how, fresh) = match session.advance(&want) {
        Ok(v) => v,
        Err(e) => {
            return send_json(stream, 400, &json!({"error": {"message": e.to_string()}}));
        }
    };
    eprintln!(
        "chat: {} turns, {fresh} new tokens ({}), budget {budget}",
        turns.len(),
        how.label(),
    );

    let r = if req.stream {
        stream_completion(session, stream, logits, budget, opts, mark, how)
    } else {
        whole_completion(session, stream, logits, budget, opts, mark, how)
    };
    // Reported even when the client hung up mid-stream: the work still
    // happened, and a disconnect is exactly when it is useful to see what it
    // cost.
    report(&session.engine, mark);
    r
}

/// Generate greedily, calling `emit` with each new piece of text.
///
/// Detokenizing incrementally rather than per token, because a token is not a
/// character: multi-byte UTF-8 and multi-token graphemes only render correctly
/// once the following token arrives.
fn generate<O: Ops>(
    session: &mut Session<'_, O>,
    mut logits: Vec<f32>,
    budget: usize,
    mut emit: impl FnMut(&str) -> Result<()>,
    mut cancelled: impl FnMut() -> bool,
) -> Result<(String, &'static str, Vec<u32>)> {
    let eos = session.tk.eos_token_id;
    let mut shown = String::new();
    let mut produced: Vec<u32> = Vec::new();
    let mut reason = "length";

    for _ in 0..budget {
        // **Every token, and it costs nothing.** A peek is about a microsecond
        // against a 25 ms token. Breaking rather than returning an error is
        // deliberate: the engine has already consumed these tokens, so the
        // caller must still absorb them or the next turn's prefix check fails
        // and re-prefills the whole conversation.
        if cancelled() {
            reason = "cancelled";
            break;
        }
        let next = argmax(&logits);
        if Some(next) == eos {
            reason = "stop";
            break;
        }
        produced.push(next);

        if let Ok(text) = session.tk.decode(&produced, false) {
            if let Some(delta) = text.strip_prefix(shown.as_str()) {
                if !delta.is_empty() {
                    // **A failed write is a gone client, not a server error.**
                    // `emit(delta)?` propagated it, so `generate` returned
                    // `Err` and the caller never reached `absorb` — leaving the
                    // engine ahead of the session's own record, which costs the
                    // *next* turn a full re-prefill. Observed: a client that
                    // hung up inside the first token stopped generation
                    // correctly and still poisoned the session.
                    //
                    // Both detections now land in the same place, so there is
                    // one exit and it always absorbs.
                    if emit(delta).is_err() {
                        reason = "cancelled";
                        break;
                    }
                    shown = text;
                }
            } else {
                shown = text;
            }
        }

        if session.engine.pos() + 1 >= session.engine.n_ctx() {
            reason = "length";
            break;
        }
        logits = session.engine.decode(next)?;
    }
    // The ids, not just how many: the next turn's common-prefix scan needs to
    // see what the engine consumed, and the model's own output is part of that.
    Ok((shown, reason, produced))
}

fn stream_completion<O: Ops>(
    session: &mut Session<'_, O>,
    stream: &mut TcpStream,
    logits: Vec<f32>,
    budget: usize,
    opts: &ServeOpts,
    mark: Mark,
    how: Resume,
) -> Result<()> {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
    write_all(stream, head.as_bytes())?;

    let id = completion_id();
    let created = now();
    let model = opts.model_id.clone();

    let first = chunk(&id, created, &model, json!({"role": "assistant"}), None);
    sse(stream, &first)?;

    // Cloned before `sink` borrows the stream mutably.
    let probe = stream.try_clone().ok();
    let mut sink = |delta: &str| -> Result<()> {
        let c = chunk(&id, created, &model, json!({"content": delta}), None);
        sse(stream, &c)
    };
    let (text, reason, ids) = generate(session, logits, budget, &mut sink, || {
        probe.as_ref().is_some_and(client_gone)
    })?;
    session.absorb(&text, &ids);
    if reason == "cancelled" {
        eprintln!("chat: client went away after {} tokens; stopped", ids.len());
        return Ok(());
    }

    // The final chunk carries the same `timings` the whole-completion body
    // does. A streaming client would otherwise have only wall time, which is
    // exactly the measurement that misleads.
    let mut last = chunk(&id, created, &model, json!({}), Some(reason));
    if let Value::Object(ref mut m) = last {
        m.insert("timings".to_string(), timings_json(&session.engine, mark, how));
    }
    sse(stream, &last)?;
    write_all(stream, b"data: [DONE]\n\n")?;
    Ok(())
}

fn whole_completion<O: Ops>(
    session: &mut Session<'_, O>,
    stream: &mut TcpStream,
    logits: Vec<f32>,
    budget: usize,
    opts: &ServeOpts,
    mark: Mark,
    how: Resume,
) -> Result<()> {
    let prompt_tokens = session.consumed;
    // **A non-streaming request writes nothing until it is finished**, so
    // without this it cannot tell a cancelled turn from a live one, and ran the
    // whole budget into a closed socket.
    let probe = stream.try_clone().ok();
    let (text, reason, ids) = generate(session, logits, budget, |_| Ok(()), || {
        probe.as_ref().is_some_and(client_gone)
    })?;
    let n = ids.len();
    session.absorb(&text, &ids);
    if reason == "cancelled" {
        eprintln!("chat: client went away after {n} tokens; stopped");
        return Ok(());
    }

    let body = json!({
        "id": completion_id(),
        "object": "chat.completion",
        "created": now(),
        "model": opts.model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": n,
            "total_tokens": prompt_tokens + n,
        },
        // Not part of the OpenAI schema, and clients that do not know the field
        // ignore it. The ones that matter here are ours.
        "timings": timings_json(&session.engine, mark, how),
    });
    send_json(stream, 200, &body)
}

fn chunk(id: &str, created: u64, model: &str, delta: Value, finish: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

// ------------------------------------------------------------------- plumbing

/// Render control characters visibly so a prompt prints on one line.
///
/// A chat prompt is mostly newlines and `<|im_*|>` markers, and the whole point
/// of printing it is to see the structure -- which a literal newline destroys.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

/// Print at most `limit` characters, saying how many were dropped.
fn clipped(label: &str, text: &str, limit: usize) {
    let shown: String = text.chars().take(limit).collect();
    let n = text.chars().count();
    if n > limit {
        eprintln!("    {label} ({n} chars, first {limit}): {}", escape(&shown));
    } else {
        eprintln!("    {label} ({n} chars): {}", escape(&shown));
    }
}

/// Decode `%XX`, drop a query string, and trim.
///
/// Deliberately lenient: this exists so a base URL pasted with a trailing space
/// or a doubled `/v1` still reaches the right handler.
fn normalize_path(raw: &str) -> String {
    let raw = raw.split('?').next().unwrap_or(raw);
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
            if let Ok(v) = u8::from_str_radix(hex, 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).trim().trim_end_matches('/').to_string()
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn completion_id() -> String {
    format!("chatcmpl-{}", now())
}

/// Read one request: method, path, and body. Only `Content-Length` bodies are
/// supported, which is what every OpenAI client sends.
fn read_request(stream: &mut TcpStream) -> Result<(String, String, Vec<u8>)> {
    let io_err = |source| Error::Io {
        path: "http".to_string(),
        source,
    };
    let mut reader = BufReader::new(stream.try_clone().map_err(io_err)?);

    let mut line = String::new();
    reader.read_line(&mut line).map_err(io_err)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    if method.is_empty() {
        return Err(Error::InconsistentArchitecture {
            what: "http",
            detail: "empty request".to_string(),
        });
    }

    let mut len = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).map_err(io_err)? == 0 {
            break;
        }
        let trimmed = header.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some(v) = trimmed.strip_prefix("Content-Length:") {
            len = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = trimmed.strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        }
    }

    let mut body = vec![0u8; len];
    if len > 0 {
        reader.read_exact(&mut body).map_err(io_err)?;
    }
    Ok((method, path, body))
}

fn write_all(stream: &mut TcpStream, bytes: &[u8]) -> Result<()> {
    stream.write_all(bytes).map_err(|source| Error::Io {
        path: "http".to_string(),
        source,
    })?;
    stream.flush().map_err(|source| Error::Io {
        path: "http".to_string(),
        source,
    })
}

/// Has the client gone away?
///
/// **Because a generation nobody is listening to still costs a GPU.** A
/// non-streaming request wrote nothing until it finished, so a cancelled Cline
/// turn ran its whole budget -- 32,000 tokens, about thirteen minutes.
/// Streaming noticed eventually, but only when a write failed, which needs the
/// peer's RST to arrive first.
///
/// A zero-length peek on a non-blocking socket separates the three cases
/// without consuming anything: `Ok(0)` is EOF, so the peer closed;
/// `WouldBlock` is an open connection with nothing pending; `Ok(n)` means bytes
/// are waiting, which is not a disconnect. Blocking mode is restored either
/// way, because the paths that do get a response still have to write it.
///
/// Takes a `try_clone` of the stream so it can run while the emit closure holds
/// the original mutably.
fn client_gone(probe: &TcpStream) -> bool {
    if probe.set_nonblocking(true).is_err() {
        return false;
    }
    let mut byte = [0u8; 1];
    // **Four cases, and the first version only handled one of them.** It tested
    // `Ok(0)` alone, which is a clean FIN. An aborted fetch frequently ends in
    // a reset instead, and a reset surfaces as `Err(ConnectionReset)` — which
    // the old code read as "still connected" and generation carried on.
    //
    // FreeToken's log settles that the signal is really there:
    // "[FrontendAPI] WARNING Aborting request for user 11" one second after the
    // user clicked cancel in the same client. So a missed cancellation is ours.
    let gone = match probe.peek(&mut byte) {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => !matches!(
            e.kind(),
            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::Interrupted
        ),
    };
    // Restoring this is not optional: `try_clone` dups the descriptor, and
    // O_NONBLOCK lives on the shared open file description, so leaving it set
    // would make the *response* writes non-blocking too.
    let _ = probe.set_nonblocking(false);
    gone
}

fn sse(stream: &mut TcpStream, value: &Value) -> Result<()> {
    write_all(stream, format!("data: {value}\n\n").as_bytes())
}

fn send_head(stream: &mut TcpStream, status: u16, ctype: &str, len: usize) -> Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {}\r\n\
         Content-Type: {ctype}\r\n\
         Content-Length: {len}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Headers: *\r\n\
         Connection: close\r\n\r\n",
        if status == 200 { "OK" } else { "" }
    );
    write_all(stream, head.as_bytes())
}

fn send_json(stream: &mut TcpStream, status: u16, value: &Value) -> Result<()> {
    let body = value.to_string();
    send_head(stream, status, "application/json", body.len())?;
    write_all(stream, body.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 35B's long contexts keep the spacing they had; Qwen3.8-Flash-Next's
    /// 2,048-token context gets checkpoints at all; nothing goes below the floor.
    #[test]
    fn checkpoint_spacing_follows_the_context() {
        assert_eq!(checkpoint_spacing(32_768), 2_048);
        assert_eq!(checkpoint_spacing(128_000), 2_048);
        assert_eq!(checkpoint_spacing(4_096), 512);
        assert_eq!(checkpoint_spacing(2_048), 256);
        assert_eq!(checkpoint_spacing(512), 256);
    }
}
