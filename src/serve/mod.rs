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

mod tools;
use tools::ToolCall;

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
    /// Default reply budget when a request sets none; 0 is "until the context
    /// is full".
    pub max_tokens: usize,
    /// `--think`: the default for the template's `enable_thinking`, when set.
    pub think: Option<bool>,
    /// `--reasoning-effort`: the default for the template's `reasoning_effort`.
    pub reasoning_effort: Option<String>,
    /// Print each request's body and the prompt it renders to.
    pub verbose: bool,
}

#[derive(Deserialize)]
struct ChatRequest {
    /// Kept as JSON, not a struct: the model's own template reads each message,
    /// and fields a struct would drop — `tool_calls`, `tool_call_id`,
    /// `reasoning_content` — are exactly the ones it renders. See
    /// [`template_message`].
    #[serde(default)]
    messages: Vec<Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    /// OpenAI's function definitions, rendered into the prompt by the template.
    #[serde(default)]
    tools: Option<Value>,
    /// `"none"` leaves the tools out; any other choice is left to the model.
    #[serde(default)]
    tool_choice: Option<Value>,
    /// Passed to the template, which decides what it means (Qwen3.8: `xhigh`,
    /// `medium`, `low`). A value the template refuses falls back to its default
    /// rather than failing the turn.
    #[serde(default)]
    reasoning_effort: Option<String>,
    /// The template's own switches, as vLLM, SGLang and llama.cpp's server
    /// accept them: `{"enable_thinking": false}` turns thinking off for this
    /// request. Overrides `--think` and `--reasoning-effort`.
    #[serde(default)]
    chat_template_kwargs: Option<serde_json::Map<String, Value>>,
}

/// The template switches for one request: the server's defaults, then the
/// request's `reasoning_effort`, then its `chat_template_kwargs`, each
/// overriding the last.
fn template_kwargs(req: &ChatRequest, opts: &ServeOpts) -> serde_json::Map<String, Value> {
    let mut kw = serde_json::Map::new();
    if let Some(on) = opts.think {
        kw.insert("enable_thinking".to_string(), json!(on));
    }
    if let Some(e) = &opts.reasoning_effort {
        kw.insert("reasoning_effort".to_string(), json!(e));
    }
    if let Some(e) = &req.reasoning_effort {
        kw.insert("reasoning_effort".to_string(), json!(e));
    }
    if let Some(extra) = &req.chat_template_kwargs {
        for (k, v) in extra {
            kw.insert(k.clone(), v.clone());
        }
    }
    kw
}

/// Requests handled since start-up, for the turn header.
static TURNS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The line that opens each turn in the terminal: a rule, so one turn's
/// output never runs into the next.
fn turn_header(n: usize, route: &str, bytes: usize) -> String {
    let head = format!("━━ turn {n} · POST {route} · {} ", size(bytes));
    let pad = 80usize.saturating_sub(head.chars().count());
    format!("{head}{}", "━".repeat(pad))
}

/// `28761` → `28,761`.
fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A byte count for a human: `512 B`, `12.3 KB`, `1.2 MB`.
fn size(bytes: usize) -> String {
    match bytes {
        b if b < 1024 => format!("{b} B"),
        b if b < 1024 * 1024 => format!("{:.1} KB", b as f64 / 1024.0),
        b => format!("{:.1} MB", b as f64 / 1048576.0),
    }
}

/// Refuse a request: to the client as a 400, **and in the terminal**, where it
/// used to go unsaid — a turn that overflowed the context left the log
/// silent after "continued at …" while the client got the reason.
fn refuse(stream: &mut TcpStream, message: &str) -> Result<()> {
    eprintln!("  error     400 → client: {message}");
    send_json(stream, 400, &json!({"error": {"message": message}}))
}

/// What to stream after a token, given the text already sent and the half's
/// text decoded so far.
#[derive(Debug, PartialEq)]
enum Delta<'a> {
    /// New text to send.
    Emit(&'a str),
    /// The text ends inside a multi-byte character; wait for the next token.
    Wait,
    /// The decoded text no longer extends what was sent; adopt it silently.
    Resync,
}

/// **A character split across tokens is held back, not sent as `�`.** The
/// tokenizer decodes lossily, so an emoji's first token alone ends in U+FFFD.
/// Sending that and resyncing when the next token completed it left clients
/// with a `�` in place of every such character — Cline sent them back in its
/// history (27-09).
fn next_delta<'a>(sent: &str, text: &'a str) -> Delta<'a> {
    if text.ends_with('\u{FFFD}') {
        return Delta::Wait;
    }
    match text.strip_prefix(sent) {
        Some(d) => Delta::Emit(d),
        None => Delta::Resync,
    }
}

/// One OpenAI message as the chat template expects it.
///
/// Content is flattened to text by [`content_text`], as before. A tool call's
/// `arguments` arrive as a JSON *string*, and the template iterates them with
/// `|items`, so they are parsed into an object here; an unparseable one is
/// passed through and the template's own error names it.
fn template_message(m: &Value) -> Value {
    let mut out = serde_json::Map::new();
    let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
    out.insert("role".to_string(), json!(role));
    out.insert(
        "content".to_string(),
        json!(content_text(m.get("content").unwrap_or(&Value::Null))),
    );
    if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
        let calls: Vec<Value> = calls
            .iter()
            .map(|c| {
                let mut c = c.clone();
                if let Some(Value::String(s)) = c.pointer("/function/arguments") {
                    if let Ok(parsed @ Value::Object(_)) = serde_json::from_str::<Value>(s) {
                        c["function"]["arguments"] = parsed;
                    }
                }
                c
            })
            .collect();
        out.insert("tool_calls".to_string(), Value::Array(calls));
    }
    for key in ["tool_call_id", "name", "reasoning_content"] {
        if let Some(v) = m.get(key) {
            out.insert(key.to_string(), v.clone());
        }
    }
    Value::Object(out)
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
        "  kv       {:>6} / {} positions ({}%)  {kv:.0} MiB{}",
        engine.pos(),
        engine.n_ctx(),
        engine.pos() * 100 / engine.n_ctx().max(1),
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
    fn advance(&mut self, want: &str, mut cancelled: impl FnMut() -> bool) -> Result<Advanced> {
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

        // `base` is what the engine holds before this turn's new tokens, as text
        // (a continuation) or as ids (after a return), so a prefill cut short by a
        // cancel can still say which prefix of `want` it has consumed.
        let (at_how, tokens, base) = if !self.rendered.is_empty() && want.starts_with(self.rendered.as_str())
        {
            let text = &want[self.rendered.len()..];
            let tokens = self.tk.encode(text, false, true);
            (Resume::Continued(self.consumed), tokens, Base::Text(self.rendered.clone()))
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
            if !self.rendered.is_empty() {
                eprintln!(
                    "  diverged  the request left the held text at {}; tokens agree for {common} of {}, {}",
                    divergence(&self.rendered, want),
                    want_tokens.len(),
                    how.label(),
                );
            }
            let base = Base::Ids(want_tokens[..how.at()].to_vec());
            (how, want_tokens[how.at()..].to_vec(), base)
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

        // Prefilled in slices so a checkpoint can be taken between them, and so
        // a cancel is heard between them. Each slice is an ordinary prefill at a
        // later `start_pos` — the engine already chunks at `max_batch` — so this
        // cannot change a bit, which `split_prefill_equals_single_prefill` pins
        // down.
        //
        // **No wider than a batch, and the client checked before each.** A
        // prefill used to be one uninterruptible call per checkpoint spacing,
        // and on Qwen3.8-Flash-Next a 3,879-token Cline turn is 100 s of it: a
        // cancel there was heard only when decode began (17-09). A batch is
        // ~15 s on that model and a fraction of a second on the 35B.
        let mut logits = Vec::new();
        let mut done = 0usize;
        let spacing = checkpoint_spacing(self.engine.n_ctx());
        let slice = spacing.min(self.engine.max_batch());
        // **The prompt's end checkpoint goes after its last special token**, not
        // after its last token, when ordinary tokens follow. A special token is
        // atomic, so the next request re-tokenizes to the same ids through it;
        // an ordinary one can merge with what the next request appends. The
        // Qwen3.6 and Qwen3.8 templates end the prompt with `<think>\n`, and the
        // next request renders that turn as `<think>\n\n</think>`: the two
        // newlines become one token, the ids agree only to `<think>`, and a
        // checkpoint one token later is unusable — measured, a tool round-trip
        // restarted from zero (27-09).
        let stable = tokens
            .iter()
            .rposition(|&t| self.tk.is_special(t))
            .map(|i| i + 1)
            .filter(|&s| s > 0 && s < tokens.len());
        while done < tokens.len() {
            if cancelled() {
                self.rendered = self.consumed_text(want, &base, &tokens[..done]);
                return Ok(Advanced::Cancelled { done, of: tokens.len() });
            }
            let mut take = slice.min(tokens.len() - done);
            // Stop the slice at the stable point so the checkpoint lands on it.
            // A split prefill is bit-identical to a whole one
            // (`split_prefill_equals_single_prefill`).
            if let Some(s) = stable {
                if done < s && s < done + take {
                    take = s - done;
                }
            }
            logits = self.engine.prefill(&tokens[done..done + take])?;
            self.tokens.extend_from_slice(&tokens[done..done + take]);
            done += take;
            self.consumed += take;
            prefill_progress(done, tokens.len());
            // **Spacing measured from the last checkpoint, not from the start
            // of this turn.** The previous condition was `done < tokens.len()`,
            // which takes a checkpoint only between slices — so a turn shorter
            // than a slice took none, and an incrementally growing conversation
            // accumulated nothing to return to. Checkpointing the final slice
            // is not "a copy of where we already are": the next turn continues
            // past it, which is precisely what makes it a return point for the
            // divergence after that.
            if self.consumed - self.last_ckpt >= spacing || Some(done) == stable {
                self.take_checkpoint();
            }
        }
        // **And one at the end of every prompt**, however short the turn. A
        // client that sends the model's last reply back changed diverges inside
        // that reply, just past this point; with checkpoints only every spacing,
        // a conversation shorter than one spacing had nowhere to return to but
        // zero, and Cline restarted every turn of a ~1,100-token session (17-09).
        // The ladder is still capped at `MAX_CHECKPOINTS`. When the prompt ends
        // in ordinary tokens, the stable point above already took it.
        if stable.is_none() && self.consumed > self.last_ckpt {
            self.take_checkpoint();
        }
        self.rendered = want.to_string();
        Ok(Advanced::Ready(logits, at_how, tokens.len()))
    }

    /// The prefix of `want` the engine holds after a prefill stopped `done`
    /// tokens in, for `rendered`, or empty when it cannot be said exactly.
    ///
    /// Decoding a prefix of an encoding gives back a prefix of the text on this
    /// byte-level tokenizer, specials rendered — but it is checked, not assumed:
    /// a cut inside a multi-byte character, or a BOS the text does not carry,
    /// fails the check. An empty `rendered` sends the next request down the
    /// token path, which finds the same position more slowly; a wrong one would
    /// skip text the engine never saw.
    fn consumed_text(&self, want: &str, base: &Base, done: &[u32]) -> String {
        let text = match base {
            Base::Text(t) => self.tk.decode(done, true).ok().map(|d| format!("{t}{d}")),
            Base::Ids(ids) => {
                let mut all = ids.clone();
                all.extend_from_slice(done);
                self.tk.decode(&all, true).ok()
            }
        };
        match text {
            Some(t) if want.starts_with(t.as_str()) => t,
            _ => String::new(),
        }
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
        // **Already there**: a prefill cancelled part-way, then resent, has
        // exactly this as its common prefix. Restoring a checkpoint instead would
        // re-run up to a spacing of tokens for nothing.
        if common == self.consumed && common == self.engine.pos() {
            return Ok(Resume::Continued(common));
        }
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

/// Where `sent` stops agreeing with `held`: the character offset and a few
/// characters of each side from there, escaped, for the log.
fn divergence(held: &str, sent: &str) -> String {
    let at = held
        .char_indices()
        .zip(sent.chars())
        .find(|((_, a), b)| a != b)
        .map(|((i, _), _)| i)
        .unwrap_or_else(|| held.len().min(sent.len()));
    let window = |t: &str| -> String { escape(&t[at.min(t.len())..].chars().take(48).collect::<String>()) };
    format!(
        "char {} of {} held / {} sent — held \"{}\", sent \"{}\"",
        held[..at].chars().count(),
        held.chars().count(),
        sent.chars().count(),
        window(held),
        window(sent),
    )
}

/// What [`Session::advance`] did.
enum Advanced {
    /// Prefilled: the last position's logits, where it resumed from, and how
    /// many new tokens it ran.
    Ready(Vec<f32>, Resume, usize),
    /// The client went away after `done` of `of` new tokens. The engine and the
    /// session agree on what was consumed; nothing is owed to the client.
    Cancelled { done: usize, of: usize },
}

/// What the engine held before a turn's new tokens.
enum Base {
    /// A continuation: the session's text.
    Text(String),
    /// After a return: the ids of the new request's own encoding, up to there.
    Ids(Vec<u32>),
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
    eprintln!("  chat here  {base}/         <- open it in a browser");
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
    // A chat turn opens with its own header; the rest get a line under -v.
    if opts.verbose && !route.ends_with("/chat/completions") {
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
            chat_completions(session, &mut stream, p, &body, opts)
        }
        // The chat page, served by the engine itself.
        //
        // **Embedded, not read from disk.** A binary that needs a file beside it
        // is a binary someone can install wrong, and the whole point of this
        // engine is one download plus a driver. `include_str!` costs ~12 KB.
        // `normalize_path` strips the trailing slash, so the root arrives as "".
        ("GET", "" | "/index.html" | "/ui" | "/chat") => {
            send_head(&mut stream, 200, "text/html; charset=utf-8", UI.len())?;
            write_all(&mut stream, UI.as_bytes())
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
    route: &str,
    body: &[u8],
    opts: &ServeOpts,
) -> Result<()> {
    let n = TURNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    eprintln!("\n{}", turn_header(n, route, body.len()));

    let req: ChatRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => return refuse(stream, &format!("could not parse request: {e}")),
    };

    // Tools reach the model only through its own template, and only when the
    // client offers some and has not said `tool_choice: "none"`.
    let tools = match (&req.tools, req.tool_choice.as_ref().and_then(Value::as_str)) {
        (_, Some("none")) => None,
        (Some(Value::Array(a)), _) if !a.is_empty() => req.tools.as_ref(),
        _ => None,
    };
    let messages = Value::Array(req.messages.iter().map(template_message).collect());
    let mut kwargs = template_kwargs(&req, opts);
    let effort = kwargs.get("reasoning_effort").and_then(Value::as_str).map(str::to_string);
    let (want, effort_note) = match session.chat.render_with(&messages, tools, Some(&kwargs)) {
        Ok(w) => (w, effort.clone()),
        // An effort this template refuses: its own default, said in the log.
        Err(_) if effort.is_some() => {
            kwargs.remove("reasoning_effort");
            match session.chat.render_with(&messages, tools, Some(&kwargs)) {
                Ok(w) => (w, effort.map(|e| format!("{e} (not this template's; its default used)"))),
                Err(e) => return refuse(stream, &e.to_string()),
            }
        }
        Err(e) => return refuse(stream, &e.to_string()),
    };
    // 0 — the default — asks for nothing: the context's room is the limit.
    let budget = req
        .max_tokens
        .or(req.max_completion_tokens)
        .unwrap_or(opts.max_tokens);
    // Both Qwen3.6 and Qwen3.8 templates open the reply inside the thinking
    // block, so the model's first tokens are reasoning and it emits only the
    // closing marker. With `enable_thinking` false they close an empty block
    // instead, and the reply is answer from its first token.
    let thinking = want.ends_with("<think>\n");
    let mut parts = vec![format!("{} messages", req.messages.len())];
    if let Some(t) = tools.and_then(Value::as_array) {
        parts.push(format!("{} tools", t.len()));
    }
    parts.push(if budget == 0 {
        "max_tokens: until the context is full".to_string()
    } else {
        format!("max_tokens {}", thousands(budget))
    });
    if kwargs.get("enable_thinking") == Some(&json!(false)) {
        parts.push("thinking off".to_string());
    } else if let Some(e) = &effort_note {
        parts.push(format!("reasoning {e}"));
    }
    eprintln!("  request   {}", parts.join(" · "));

    if opts.verbose {
        clipped("body    ", &String::from_utf8_lossy(body), 1200);
        for m in messages.as_array().into_iter().flatten() {
            let role = m["role"].as_str().unwrap_or("?");
            clipped(&format!("  {role:<9}"), m["content"].as_str().unwrap_or(""), 300);
        }
        // What actually reaches the tokenizer. Everything before this is
        // already in the engine, so this is the only text that costs anything.
        let new = want.strip_prefix(session.rendered.as_str()).unwrap_or(&want);
        clipped("rendered", &want, 400);
        clipped("new     ", new, 400);
    }

    let mark = Mark::take(&session.engine);
    // Announced before `advance`, because a long prefill is minutes of silence
    // otherwise and the count is the only clue to why.
    let approx = want.len().saturating_sub(session.rendered.len()) / 4;
    if approx > 2048 {
        eprintln!("  prefill   ~{} new tokens; this will take a while", thousands(approx));
    }
    // Probed between prefill slices, as `generate` probes between tokens.
    let probe = stream.try_clone().ok();
    let advanced = session.advance(&want, || probe.as_ref().is_some_and(client_gone));
    let (logits, how, fresh) = match advanced {
        Ok(Advanced::Ready(logits, how, fresh)) => (logits, how, fresh),
        Ok(Advanced::Cancelled { done, of }) => {
            eprintln!("  cancelled client went away during prefill after {done} of {of} tokens");
            report(&session.engine, mark);
            return Ok(());
        }
        Err(e) => return refuse(stream, &e.to_string()),
    };

    // **The reply can use only the context that is left**, whatever the client
    // asked for: generation stops at the edge either way, and printing the
    // request's `max_tokens` as the budget hid that a nearly full context was
    // about to cut a reply off (Cline asks for 32,000 at 28,761 of 32,096).
    let (pos, n_ctx) = (session.engine.pos(), session.engine.n_ctx());
    let room = n_ctx.saturating_sub(pos + 1);
    let budget = if budget == 0 { room } else { budget.min(room) };
    let used = pos * 100 / n_ctx.max(1);
    eprintln!(
        "  context   {} · {} new · {} of {} positions ({used}%) · reply room {}{}",
        how.label(),
        thousands(fresh),
        thousands(pos),
        thousands(n_ctx),
        thousands(room),
        if room < 1024 {
            "\n            ⚠ nearly full: raise --ctx, or let the client compact the conversation"
        } else {
            ""
        },
    );

    let turn = Gen { budget, tools, thinking };
    let r = if req.stream {
        stream_completion(session, stream, logits, turn, opts, mark, how)
    } else {
        whole_completion(session, stream, logits, turn, opts, mark, how)
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
///
/// Returns the reply **as the client sees it**, with the thinking markers, and
/// the same reply **without** them. The second is what [`Session::absorb`]
/// records: `rendered` is compared against the conversation a client sends
/// back, and clients strip the thinking from their history, so keeping the
/// markers there would diverge the prefix and re-prefill every turn.
fn generate<O: Ops>(
    session: &mut Session<'_, O>,
    mut logits: Vec<f32>,
    turn: Gen<'_>,
    mut emit: impl FnMut(Out<'_>) -> Result<()>,
    mut cancelled: impl FnMut() -> bool,
) -> Result<Reply> {
    let Gen { budget, tools, thinking } = turn;
    let eos = session.tk.eos_token_id;
    let mut reason = "length";
    // **A tool call is its own part, collected and parsed whole.** The model
    // writes it between its `<tool_call>` and `</tool_call>` tokens in the
    // format its template taught it (`tools::parse`); the client gets it as
    // OpenAI's `tool_calls`, never as text. Only when the request offered
    // tools: without them the markers are dropped as control tokens, as before.
    let call_open = tools.and(session.tk.special_id("<tool_call>"));
    let call_close = session.tk.special_id("</tool_call>");
    let mut in_call = false;
    let mut call_ids: Vec<u32> = Vec::new();
    // **A call streams as it is written** (`tools::partial_args`): its name once
    // complete, then its arguments JSON as each part becomes certain. A 177B
    // writing a file through a tool call is minutes of generation, and a call
    // sent only whole left the client showing nothing for all of it (27-09).
    let mut call_named = false;
    let mut call_sent = String::new();
    let mut calls: Vec<ToolCall> = Vec::new();
    // A call that does not parse is shown as text rather than lost.
    let mut unparsed = String::new();
    // **Reasoning is a field of its own, not tagged text.** A reasoning model
    // emits `<think>…</think>` around its reasoning and both are control tokens.
    // Dropping them left the client one undifferentiated string and Cline showed
    // the thinking as the reply; keeping them in `content` only moved the problem,
    // since Cline renders the tags literally rather than folding them. So the two
    // halves are streamed apart — `reasoning_content` and `content`, as DeepSeek's
    // and vLLM's OpenAI-compatible servers do — and the markers themselves never
    // go on the wire.
    //
    // A model without the markers yields `None` twice, `open` never matches, and
    // every token is answer, exactly as before.
    let open = session.tk.special_id("<think>");
    let close = session.tk.special_id("</think>");
    let mut part = if thinking { Part::Reasoning } else { Part::Content };
    // Each half is decoded from its own ids: a token is not a character, so the
    // text only renders once the following token arrives, and the two halves
    // must not interleave while that settles.
    let mut ids: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
    let mut shown: [String; 2] = [String::new(), String::new()];
    let mut produced: Vec<u32> = Vec::new();

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

        if call_open.is_some() && Some(next) == call_open {
            in_call = true;
            call_ids.clear();
            call_named = false;
            call_sent.clear();
            logits = session.engine.decode(next)?;
            continue;
        }
        if in_call {
            let index = calls.len();
            let mut gone = false;
            if Some(next) == call_close {
                in_call = false;
                let text = session.tk.decode(&call_ids, false).unwrap_or_default();
                match tools::parse(&text, tools) {
                    Some(call) => {
                        // Whatever the stream has not sent yet: all of it for a
                        // JSON-format call, the held-back tail for an XML one.
                        let whole = Value::Object(call.arguments.clone()).to_string();
                        if !call_named {
                            gone |= emit(Out::CallStart(index, &call.name)).is_err();
                        }
                        match whole.strip_prefix(call_sent.as_str()) {
                            Some(rest) if !rest.is_empty() => {
                                gone |= emit(Out::CallArgs(index, rest)).is_err();
                            }
                            Some(_) => {}
                            None => eprintln!(
                                "  warning   tool call {index} streamed arguments that its whole parse does not extend"
                            ),
                        }
                        calls.push(call);
                    }
                    None => {
                        if call_named {
                            eprintln!("  warning   tool call {index} was streamed but does not parse whole");
                        }
                        let raw = format!("<tool_call>{text}</tool_call>");
                        gone |= emit(Out::Text(&raw, Part::Content)).is_err();
                        unparsed.push_str(&raw);
                    }
                }
            } else {
                call_ids.push(next);
                // Send what the call's text so far makes certain.
                if let Ok(text) = session.tk.decode(&call_ids, false) {
                    if !text.ends_with('\u{FFFD}') {
                        if let (Some(name), prefix) = tools::partial_args(&text, tools) {
                            if !call_named {
                                gone |= emit(Out::CallStart(index, &name)).is_err();
                                call_named = true;
                            }
                            if let Some(d) = prefix.strip_prefix(call_sent.as_str()) {
                                if !d.is_empty() {
                                    gone |= emit(Out::CallArgs(index, d)).is_err();
                                    call_sent = prefix;
                                }
                            }
                        }
                    }
                }
            }
            if gone {
                reason = "cancelled";
                break;
            }
            if session.engine.pos() + 1 >= session.engine.n_ctx() {
                reason = "length";
                break;
            }
            logits = session.engine.decode(next)?;
            continue;
        }

        // The markers switch halves and are never rendered themselves.
        if Some(next) == open {
            part = Part::Reasoning;
            logits = session.engine.decode(next)?;
            continue;
        }
        if Some(next) == close {
            part = Part::Content;
            logits = session.engine.decode(next)?;
            continue;
        }
        let half = part as usize;
        ids[half].push(next);

        if let Ok(text) = session.tk.decode(&ids[half], false) {
            match next_delta(&shown[half], &text) {
                // Half a character: send nothing until it is whole.
                Delta::Wait => {}
                Delta::Resync => shown[half] = text.clone(),
                Delta::Emit(delta) if !delta.is_empty() => {
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
                    if emit(Out::Text(delta, part)).is_err() {
                        reason = "cancelled";
                        break;
                    }
                    shown[half] = text.clone();
                }
                Delta::Emit(_) => {}
            }
        }

        if session.engine.pos() + 1 >= session.engine.n_ctx() {
            reason = "length";
            break;
        }
        logits = session.engine.decode(next)?;
    }
    // The whole halves as decoded, for a non-streaming reply: what was streamed,
    // plus a character still held back when generation stopped.
    for h in 0..2 {
        if let Ok(t) = session.tk.decode(&ids[h], false) {
            shown[h] = t;
        }
    }
    // The ids, not just how many: the next turn's common-prefix scan needs to
    // see what the engine consumed, and the model's own output is part of that.
    // A call cut off by the budget or the context is shown as the text it got to.
    if in_call {
        let text = session.tk.decode(&call_ids, false).unwrap_or_default();
        unparsed.push_str(&format!("<tool_call>{text}"));
    }
    let [mut content, reasoning] = shown;
    content.push_str(&unparsed);
    if !calls.is_empty() && reason == "stop" {
        reason = "tool_calls";
    }
    let plain = session
        .tk
        .decode(&produced, false)
        .unwrap_or_else(|_| format!("{reasoning}{content}"));
    Ok(Reply { content, reasoning, plain, reason, ids: produced, calls })
}

/// What one turn generates with.
#[derive(Clone, Copy)]
struct Gen<'a> {
    budget: usize,
    /// The request's `tools`, when offered: turns on tool-call parsing and
    /// types the parsed arguments.
    tools: Option<&'a Value>,
    /// The prompt ends inside the thinking block, so generation starts there.
    thinking: bool,
}

/// One piece of a turn, as `generate` hands it to the transport.
enum Out<'a> {
    /// Text for one half.
    Text(&'a str, Part),
    /// Tool call `index` has begun, and this is its function's name.
    CallStart(usize, &'a str),
    /// More of tool call `index`'s `arguments` JSON, to append to what was sent.
    CallArgs(usize, &'a str),
}

/// OpenAI's id for the `index`-th call of a turn created at `created`.
fn call_id(created: u64, index: usize) -> String {
    format!("call_{created}_{index}")
}

/// Which half of a reasoning model's turn a piece of text belongs to. The
/// discriminants index `generate`'s per-half buffers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
    Content = 0,
    Reasoning = 1,
}

/// One finished turn, split the way a client needs it.
struct Reply {
    /// The answer: OpenAI's `content`.
    content: String,
    /// The reasoning between the markers: `reasoning_content`, empty for a model
    /// that does not think.
    reasoning: String,
    /// Both halves with no markers, in the order the engine produced them. What
    /// [`Session::absorb`] records, because `rendered` is matched against the
    /// history a client sends back.
    plain: String,
    reason: &'static str,
    ids: Vec<u32>,
    /// Tool calls, in the order the model made them.
    calls: Vec<ToolCall>,
}

fn stream_completion<O: Ops>(
    session: &mut Session<'_, O>,
    stream: &mut TcpStream,
    logits: Vec<f32>,
    turn: Gen<'_>,
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
    let mut sink = |out: Out<'_>| -> Result<()> {
        let delta = match out {
            Out::Text(text, Part::Reasoning) => json!({ "reasoning_content": text }),
            Out::Text(text, Part::Content) => json!({ "content": text }),
            // OpenAI's streamed call: the first delta names it, the rest append
            // to its `arguments` string.
            Out::CallStart(index, name) => json!({ "tool_calls": [{
                "index": index,
                "id": call_id(created, index),
                "type": "function",
                "function": {"name": name, "arguments": ""},
            }] }),
            Out::CallArgs(index, more) => json!({ "tool_calls": [{
                "index": index,
                "function": {"arguments": more},
            }] }),
        };
        let c = chunk(&id, created, &model, delta, None);
        sse(stream, &c)
    };
    let Reply { plain, reason, ids, .. } = generate(session, logits, turn, &mut sink, || {
        probe.as_ref().is_some_and(client_gone)
    })?;
    session.absorb(&plain, &ids);
    if reason == "cancelled" {
        eprintln!("  cancelled client went away after {} tokens", ids.len());
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
    turn: Gen<'_>,
    opts: &ServeOpts,
    mark: Mark,
    how: Resume,
) -> Result<()> {
    let prompt_tokens = session.consumed;
    // **A non-streaming request writes nothing until it is finished**, so
    // without this it cannot tell a cancelled turn from a live one, and ran the
    // whole budget into a closed socket.
    let probe = stream.try_clone().ok();
    let Reply { content, reasoning, plain, reason, ids, calls } =
        generate(session, logits, turn, |_| Ok(()), || {
            probe.as_ref().is_some_and(client_gone)
        })?;
    let n = ids.len();
    session.absorb(&plain, &ids);
    if reason == "cancelled" {
        eprintln!("  cancelled client went away after {n} tokens");
        return Ok(());
    }

    let created = now();
    // `reasoning_content` beside `content`, DeepSeek's field and the one
    // OpenAI-compatible clients look for; omitted when the model does not think,
    // rather than sent empty. With tool calls, an empty `content` is `null`, as
    // OpenAI sends it.
    let mut message = json!({"role": "assistant", "content": content});
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    if !calls.is_empty() {
        // The template puts a blank line between the thinking and the call; a
        // reply that is only a call has no content, as OpenAI reports it.
        if content.trim().is_empty() {
            message["content"] = Value::Null;
        }
        message["tool_calls"] = Value::Array(
            calls.iter().enumerate().map(|(i, c)| c.to_openai(&call_id(created, i))).collect(),
        );
    }
    let body = json!({
        "id": completion_id(),
        "object": "chat.completion",
        "created": created,
        "model": opts.model_id,
        "choices": [{
            "index": 0,
            "message": message,
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

/// The chat page. One file: markup, style and the client, no framework and no
/// build step, talking to this same server's `/v1/chat/completions`.
const UI: &str = include_str!("../../ui/index.html");

/// The wordmark, once, before a server loads its model.
///
/// Solid blocks rather than outlines, and only when stderr is a terminal: a log
/// file or a pipe keeps the machine-readable shape every script here greps.
pub fn banner() {
    use std::io::IsTerminal;
    if !std::io::stderr().is_terminal() {
        return;
    }
    crate::platform::init_console();
    eprintln!();
    for line in [
        "  ████ █  █ ████ ████ ███  ███  ████ ███ ",
        "   ██  ██ █ █    █    █  █ █  █ █    █  █",
        "   ██  █ ██ ███  ███  ███  ███  ███  █  █",
        "   ██  █  █ █    █    █ █  █ █  █    █  █",
        "  ████ █  █ █    ████ █  █ █  █ ████ ███ ",
        "",
        "  ████ █  █  ██  █  █  ███ █  █ ████  ███",
        "   ██  █  █ █  █ █  █ █    █  █  ██  █   ",
        "   ██  ████ █  █ █  █ █ ██ ████  ██   ██ ",
        "   ██  █  █ █  █ █  █ █  █ █  █  ██     █",
        "   ██  █  █  ██   ██   ███ █  █  ██  ███ ",
    ] {
        eprintln!("{line}");
    }
    eprintln!("                        by compiledthoughts.dev\n");
}

/// One line, rewritten in place, while a long prompt prefills.
///
/// **Because the wait is otherwise silent.** A 3,879-token turn on
/// Qwen3.8-Flash-Next is ~100 s of prefill during which the terminal shows
/// nothing, and a user cannot tell a slow model from a hung one.
///
/// Only when stderr is a terminal, so a log file or a pipe keeps the shape
/// `inferred generate` prints, and only past one slice, so short turns — every
/// turn on the small models — stay quiet.
fn prefill_progress(done: usize, total: usize) {
    use std::io::IsTerminal;
    if total <= 512 || !std::io::stderr().is_terminal() {
        return;
    }
    let pct = done as f64 * 100.0 / total as f64;
    // `\r` and no newline: the line is replaced, not appended. The final slice
    // clears it, since the turn's own report follows immediately.
    if done >= total {
        eprint!("\r{:80}\r", "");
    } else {
        let bar = progress_bar(done, total, 30);
        eprint!("\r  prefill {bar} {done:>6} / {total} tokens  {pct:>5.1}%");
    }
    let _ = std::io::Write::flush(&mut std::io::stderr());
}

/// `[=========>          ]`: `width` cells between the brackets, filled in
/// proportion to `done / total`, with `>` marking the leading edge until the
/// bar is full. ASCII only, so every console draws it the same.
fn progress_bar(done: usize, total: usize, width: usize) -> String {
    let filled = if total == 0 { width } else { (done.min(total) * width) / total };
    let mut bar = String::with_capacity(width + 2);
    bar.push('[');
    for i in 0..width {
        bar.push(if i + 1 < filled || (i + 1 == filled && filled == width) {
            '='
        } else if i + 1 == filled {
            '>'
        } else {
            ' '
        });
    }
    bar.push(']');
    bar
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

    #[test]
    fn numbers_and_sizes_read_as_a_person_writes_them() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1000), "1,000");
        assert_eq!(thousands(28761), "28,761");
        assert_eq!(thousands(1234567), "1,234,567");
        assert_eq!(size(512), "512 B");
        assert_eq!(size(105341), "102.9 KB");
        assert_eq!(size(3 * 1048576), "3.0 MB");
    }

    #[test]
    fn the_turn_header_is_one_80_column_rule() {
        let h = turn_header(34, "/v1/chat/completions", 105341);
        assert!(h.starts_with("━━ turn 34 · POST /v1/chat/completions · 102.9 KB ━"), "{h}");
        assert_eq!(h.chars().count(), 80);
    }

    #[test]
    fn a_split_character_is_held_back_then_sent_whole() {
        // An emoji's first token decodes to a replacement character.
        assert_eq!(next_delta("Hi ", "Hi \u{FFFD}"), Delta::Wait);
        // The next token completes it: sent whole, never as `�`.
        assert_eq!(next_delta("Hi ", "Hi 😀"), Delta::Emit("😀"));
        assert_eq!(next_delta("Hi 😀", "Hi 😀 there"), Delta::Emit(" there"));
        // Text that no longer extends what was sent is adopted, not streamed.
        assert_eq!(next_delta("abc", "abd"), Delta::Resync);
    }

    #[test]
    fn the_prefill_bar_fills_in_proportion() {
        assert_eq!(progress_bar(0, 5312, 10), "[          ]");
        assert_eq!(progress_bar(1024, 5312, 10), "[>         ]");
        assert_eq!(progress_bar(2656, 5312, 10), "[====>     ]");
        assert_eq!(progress_bar(5312, 5312, 10), "[==========]");
        assert_eq!(progress_bar(9999, 5312, 10), "[==========]");
        assert_eq!(progress_bar(0, 0, 4), "[====]");
    }

    /// **A cancel during prefill is heard between slices, and the resent request
    /// continues where it stopped** (17-09). The 0.2B test model on `Naive`, a
    /// ~1,400-token turn at `--ctx 4096` (spacing 512, batch 512): cancelled
    /// after two slices, the session holds exactly 1,024 of its tokens and the
    /// matching prefix of its text; the same request again continues at 1,024
    /// and finishes where an uncancelled session does. Where the rest tokenizes
    /// as it did in one piece, the logits are the uncancelled ones to the bit.
    use crate::gguf::GgufFile;
    use crate::ops::naive::Naive;

    /// The 0.2B test model, from `INFERRED_MODEL_DIR` or `~/models`.
    fn tiny_model() -> Option<GgufFile> {
        let name = "Qwen3.8-Flash-Next-0.2B-A0.2B-NVFP4exp.gguf";
        let path = std::env::var("INFERRED_MODEL_DIR")
            .ok()
            .map(std::path::PathBuf::from)
            .into_iter()
            .chain(std::env::var("HOME").ok().map(|h| std::path::Path::new(&h).join("models")))
            .map(|d| d.join(name))
            .find(|p| p.exists());
        match path {
            Some(p) => Some(GgufFile::open(&p).expect("open")),
            None => {
                println!("SKIPPED: no {name}");
                None
            }
        }
    }

    /// A fresh `serve` session on `Naive`, as `serve` builds one.
    fn tiny_session(f: &GgufFile, n_ctx: usize) -> Session<'_, Naive> {
        let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
        let chat = ChatMl::detect(&tk, &f.metadata).expect("chatml");
        Session {
            engine: Engine::new(crate::model::Model::load(f).expect("load"), Naive, n_ctx, false),
            tk,
            chat,
            rendered: String::new(),
            tokens: Vec::new(),
            checkpoints: Vec::new(),
            last_ckpt: 0,
            consumed: 0,
        }
    }

    #[test]
    #[ignore = "needs the 0.2B test model's NVFP4-expert GGUF in ~/models or INFERRED_MODEL_DIR"]
    fn a_cancel_during_prefill_stops_between_slices_and_resumes() {
        let Some(f) = tiny_model() else { return };
        let readme = include_str!("../../README.md");
        let cut = readme.char_indices().map(|(i, _)| i).take_while(|&i| i <= 5000).last().unwrap_or(0);
        let session = || tiny_session(&f, 4096);
        let mut plain = session();
        let want = plain.chat.wrap(&readme[..cut]);

        let Advanced::Ready(whole, _, n) = plain.advance(&want, || false).expect("uncancelled") else {
            panic!("an uncancelled prefill reported a cancel");
        };
        assert!(n > 1024 + 256, "the turn is too short to cut after two slices: {n} tokens");

        let mut s = session();
        let mut checks = 0;
        let r = s.advance(&want, || {
            checks += 1;
            checks == 3
        });
        let Ok(Advanced::Cancelled { done, of }) = r else {
            panic!("the third check should have cancelled");
        };
        assert_eq!((done, of), (1024, n));
        assert_eq!((s.consumed, s.engine.pos(), s.tokens.len()), (1024, 1024, 1024));
        assert!(!s.rendered.is_empty() && want.starts_with(s.rendered.as_str()), "the text prefix was lost");

        let Advanced::Ready(resumed, how, fresh) = s.advance(&want, || false).expect("resent") else {
            panic!("the resent request reported a cancel");
        };
        println!("  resumed: {} , {fresh} new tokens, ends at {}", how.label(), s.engine.pos());
        assert!(matches!(how, Resume::Continued(1024)), "resumed as {}", how.label());
        assert_eq!(s.consumed, s.engine.pos());
        let same_split = s.tokens == plain.tokens;
        println!("  the rest tokenized as in one piece: {same_split}");
        if same_split {
            assert!(
                whole.iter().zip(&resumed).all(|(a, b)| a.to_bits() == b.to_bits()),
                "same tokens, same slices, different logits"
            );
        }
    }

    /// **A client that sends the last reply back changed resumes at the end of the
    /// prompt, not from zero** (17-09). A turn far shorter than one checkpoint
    /// spacing (`--ctx 8192`, 1,024), eight generated tokens, then the same
    /// conversation with a different assistant message and a new question: the
    /// only return point is the checkpoint every prompt now ends with. Before it,
    /// Cline restarted every turn of a ~1,100-token session.
    ///
    /// **Where exactly: after the prompt's last special token** (27-09), here
    /// `<|im_start|>` two tokens before the end of `…assistant\n`. The ordinary
    /// tokens after it can merge with what the next request appends — on the
    /// Qwen3.6/3.8 templates a trailing `<think>\n` did, and a tool round-trip
    /// restarted from zero — so the checkpoint stops short of them, at the cost
    /// of re-running those few tokens.
    #[test]
    #[ignore = "needs the 0.2B test model's NVFP4-expert GGUF in ~/models or INFERRED_MODEL_DIR"]
    fn a_changed_reply_returns_to_the_end_of_the_prompt() {
        let Some(f) = tiny_model() else { return };
        let mut s = tiny_session(&f, 8192);
        let user = "According to all known laws of aviation, there is no way a bee should be able to fly.";
        let first = s.chat.wrap_turns(&[("user", user)]);
        let Advanced::Ready(logits, _, n) = s.advance(&first, || false).expect("first turn") else {
            panic!("the first turn reported a cancel");
        };
        assert!(n < checkpoint_spacing(8192), "the turn must be shorter than a spacing: {n}");
        // `plain` is what a session records: both halves, no markers.
        let turn = Gen { budget: 8, tools: None, thinking: false };
        let r = generate(&mut s, logits, turn, |_| Ok(()), || false).expect("generate");
        let (text, ids) = (r.content.clone(), r.ids.clone());
        s.absorb(&r.plain, &r.ids);

        let second = s.chat.wrap_turns(&[
            ("user", user),
            ("assistant", "Something else entirely."),
            ("user", "And then?"),
        ]);
        let Advanced::Ready(_, how, fresh) = s.advance(&second, || false).expect("second turn") else {
            panic!("the second turn reported a cancel");
        };
        println!("  first turn {n} tokens, generated {:?}; second turn {}, {fresh} new", text, how.label());
        let prompt = s.tk.encode(&first, true, true);
        let stable = prompt
            .iter()
            .rposition(|&t| s.tk.is_special(t))
            .map(|i| i + 1)
            .filter(|&p| p < prompt.len())
            .unwrap_or(n);
        assert!(stable < n, "the prompt ends in ordinary tokens after its last special one");
        assert!(
            matches!(how, Resume::Restored(p) if p == stable),
            "resumed as {}, not at the last special token ({stable} of {n})",
            how.label()
        );
    }

    /// The two halves of a turn leave `generate` separated, and no marker or
    /// scaffolding token reaches either.
    ///
    /// **What it guards.** Cline shows `reasoning_content` as thinking and
    /// `content` as the reply. Dropping the markers left it showing the model's
    /// reasoning as the answer; putting them in `content` left it printing the
    /// tags literally. `plain`, which the session records, has to stay the whole
    /// reply either way, or the next turn re-prefills.
    #[test]
    #[ignore = "needs the 0.2B test model's NVFP4-expert GGUF in ~/models or INFERRED_MODEL_DIR"]
    fn the_halves_of_a_turn_are_separated_and_carry_no_markers() {
        let Some(f) = tiny_model() else { return };
        let mut s = tiny_session(&f, 2048);
        let prompt = s.chat.wrap_turns(&[("user", "What is 2+2?")]);
        let Advanced::Ready(logits, _, _) = s.advance(&prompt, || false).expect("prefill") else {
            panic!("the turn reported a cancel");
        };
        let mut streamed: Vec<(Part, String)> = Vec::new();
        let r = generate(
            &mut s,
            logits,
            Gen { budget: 24, tools: None, thinking: false },
            |out| {
                if let Out::Text(delta, part) = out {
                    streamed.push((part, delta.to_string()));
                }
                Ok(())
            },
            || false,
        )
        .expect("generate");

        for text in [&r.content, &r.reasoning, &r.plain] {
            assert!(!text.contains("<think>"), "a marker reached the client: {text:?}");
            assert!(!text.contains("</think>"), "a marker reached the client: {text:?}");
            assert!(!text.contains("<|im_end|>"), "scaffolding reached the client: {text:?}");
        }
        // The session records both halves, so the next turn's prefix scan sees
        // everything the engine consumed.
        assert_eq!(r.plain, format!("{}{}", r.reasoning, r.content), "plain must be both halves");
        // Every streamed piece belongs to the half it was tagged with.
        let (mut c, mut t) = (String::new(), String::new());
        for (part, delta) in &streamed {
            match part {
                Part::Content => c.push_str(delta),
                Part::Reasoning => t.push_str(delta),
            }
        }
        assert_eq!(c, r.content, "streamed content must equal the whole one");
        assert_eq!(t, r.reasoning, "streamed reasoning must equal the whole one");
        println!("  reasoning {} chars, content {} chars", r.reasoning.len(), r.content.len());
    }

    #[test]
    fn divergence_names_the_first_differing_character() {
        let d = divergence("hello world", "hello there");
        assert!(d.starts_with("char 6 of 11 held / 11 sent"), "{d}");
        assert!(d.contains("\"world\"") && d.contains("\"there\""), "{d}");
        let d = divergence("abc", "abcdef");
        assert!(d.starts_with("char 3 of 3 held / 6 sent"), "{d}");
        let d = divergence("héllo", "hélp");
        assert!(d.starts_with("char 3 of 5 held / 4 sent"), "{d}");
    }

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
