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
//! of it each time is O(n^2), and on `qwen35` prefill runs one token at a time,
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

use crate::engine::Engine;
use crate::error::{Error, Result};
use crate::ops::Ops;
use crate::tok::Tokenizer;
use crate::tok::chat::ChatMl;

/// What the CLI hands the server.
pub struct ServeOpts {
    pub port: u16,
    /// Advertised through `/v1/models` and echoed in responses. A client that
    /// asks for a different one still gets this: there is one model loaded.
    pub model_id: String,
    pub max_tokens: usize,
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
}

impl Mark {
    fn take<O: Ops>(e: &Engine<'_, O>) -> Self {
        Self {
            prefill_tokens: e.prof.prefill_tokens,
            prefill_ns: e.prof.prefill_ns,
            decode_tokens: e.prof.decode_tokens,
            decode_ns: e.prof.decode_ns,
        }
    }
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
    line(
        "prefill",
        now.prefill_tokens - before.prefill_tokens,
        now.prefill_ns - before.prefill_ns,
    );
    line(
        "decode",
        now.decode_tokens - before.decode_tokens,
        now.decode_ns - before.decode_ns,
    );
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
struct Session<'a, O: Ops> {
    engine: Engine<'a, O>,
    tk: Tokenizer,
    chat: ChatMl,
    /// Conversation text the engine has consumed, including its own output.
    rendered: String,
    /// Tokens behind `rendered`, for reporting.
    consumed: usize,
}

impl<O: Ops> Session<'_, O> {
    /// Bring the engine up to `want`, reusing what it has already consumed.
    ///
    /// Returns the last token's logits, whether the session continued, and how
    /// many tokens actually had to run.
    fn advance(&mut self, want: &str) -> Result<(Vec<f32>, bool, usize)> {
        let reused = !self.rendered.is_empty() && want.starts_with(self.rendered.as_str());
        let text = if reused {
            &want[self.rendered.len()..]
        } else {
            // An edit, a branch, or a different client. Nothing here tries to
            // rewind: a KV cache could be truncated to a common prefix, but a
            // GatedDeltaNet layer's state is one matrix that has absorbed every
            // token with no record of how to remove one. Restarting is the only
            // correct move for `qwen35`, and doing the same for `qwen3` keeps
            // one code path.
            self.engine.reset();
            self.consumed = 0;
            want
        };

        // BOS belongs to the start of a sequence, so a continuation must not
        // add one.
        let tokens = self.tk.encode(text, !reused, true);
        if tokens.is_empty() {
            return Err(Error::InconsistentArchitecture {
                what: "chat request",
                detail: "the conversation added no new text".to_string(),
            });
        }
        let logits = self.engine.prefill(&tokens)?;
        self.consumed += tokens.len();
        self.rendered = want.to_string();
        Ok((logits, reused, tokens.len()))
    }

    /// Record what the model produced, so the next turn sees it as a prefix.
    ///
    /// The turn-ending marker is deliberately *not* added: generation stops
    /// before consuming it, so the engine has not seen it, and the next
    /// request's rendering supplies it as part of the new text.
    fn absorb(&mut self, text: &str, tokens: usize) {
        self.rendered.push_str(text);
        self.consumed += tokens;
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

    let mut session = Session {
        engine,
        tk,
        chat,
        rendered: String::new(),
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

    let budget = req
        .max_tokens
        .or(req.max_completion_tokens)
        .unwrap_or(opts.max_tokens);

    let mark = Mark::take(&session.engine);
    let (logits, reused, fresh) = match session.advance(&want) {
        Ok(v) => v,
        Err(e) => {
            return send_json(stream, 400, &json!({"error": {"message": e.to_string()}}));
        }
    };
    eprintln!(
        "chat: {} turns, {fresh} new tokens ({}), budget {budget}",
        turns.len(),
        if reused { "continued" } else { "restarted" },
    );

    let r = if req.stream {
        stream_completion(session, stream, logits, budget, opts)
    } else {
        whole_completion(session, stream, logits, budget, opts)
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
) -> Result<(String, &'static str, usize)> {
    let eos = session.tk.eos_token_id;
    let mut shown = String::new();
    let mut produced: Vec<u32> = Vec::new();
    let mut reason = "length";

    for _ in 0..budget {
        let next = argmax(&logits);
        if Some(next) == eos {
            reason = "stop";
            break;
        }
        produced.push(next);

        if let Ok(text) = session.tk.decode(&produced, false) {
            if let Some(delta) = text.strip_prefix(shown.as_str()) {
                if !delta.is_empty() {
                    emit(delta)?;
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
    let n = produced.len();
    Ok((shown, reason, n))
}

fn stream_completion<O: Ops>(
    session: &mut Session<'_, O>,
    stream: &mut TcpStream,
    logits: Vec<f32>,
    budget: usize,
    opts: &ServeOpts,
) -> Result<()> {
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
    write_all(stream, head.as_bytes())?;

    let id = completion_id();
    let created = now();
    let model = opts.model_id.clone();

    let first = chunk(&id, created, &model, json!({"role": "assistant"}), None);
    sse(stream, &first)?;

    let mut sink = |delta: &str| -> Result<()> {
        let c = chunk(&id, created, &model, json!({"content": delta}), None);
        sse(stream, &c)
    };
    let (text, reason, n) = generate(session, logits, budget, &mut sink)?;
    session.absorb(&text, n);

    let last = chunk(&id, created, &model, json!({}), Some(reason));
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
) -> Result<()> {
    let prompt_tokens = session.consumed;
    let (text, reason, n) = generate(session, logits, budget, |_| Ok(()))?;
    session.absorb(&text, n);

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
