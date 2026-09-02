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

/// One loaded model, its engine, and how much of the conversation it has seen.
struct Session<'a, O: Ops> {
    engine: Engine<'a, O>,
    tk: Tokenizer,
    chat: ChatMl,
    /// Tokens the engine has already consumed, in order. The engine's caches
    /// are valid exactly for this prefix.
    processed: Vec<u32>,
}

impl<O: Ops> Session<'_, O> {
    /// Bring the engine up to `tokens`, reusing what it has already seen.
    ///
    /// Returns the logits for the last token. The whole point is the first
    /// branch: a chat that only ever appends pays for the new turn, not for the
    /// conversation.
    fn advance(&mut self, tokens: &[u32]) -> Result<(Vec<f32>, bool)> {
        // Longest common prefix, not an exact one. Requiring the whole history
        // to match looks right and almost never holds: the model generates a
        // token, the client sends the *text* back, and re-tokenizing it can
        // split differently at the seam. A first generated newline, after a
        // prompt that already ends in one, is enough: the pair merges into a
        // single token on the way back in and every later token shifts.
        let lcp = self
            .processed
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count();

        let reused = if lcp == self.processed.len() && !self.processed.is_empty() {
            true
        } else if self.engine.rewind(lcp) {
            // Attention only: the cache is a log, so truncating it to the
            // common prefix leaves exactly the state that prefix produced.
            self.processed.truncate(lcp);
            lcp > 0
        } else {
            // Recurrent: nothing to truncate, so start over.
            self.engine.reset();
            self.processed.clear();
            false
        };
        let fresh = &tokens[self.processed.len()..];
        if fresh.is_empty() {
            return Err(Error::InconsistentArchitecture {
                what: "chat request",
                detail: "prompt is not longer than what has already been processed".to_string(),
            });
        }
        let logits = self.engine.prefill(fresh)?;
        self.processed.extend_from_slice(fresh);
        Ok((logits, reused))
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
    eprintln!(
        "serving {} on http://127.0.0.1:{}/v1  (ctx {})",
        opts.model_id,
        opts.port,
        engine.n_ctx()
    );

    let mut session = Session {
        engine,
        tk,
        chat,
        processed: Vec::new(),
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

    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => send_json(&mut stream, 200, &json!({"status": "ok"})),
        ("GET", "/v1/models") => {
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
        ("POST", "/v1/chat/completions") => chat_completions(session, &mut stream, &body, opts),
        ("OPTIONS", _) => send_head(&mut stream, 204, "text/plain", 0),
        _ => send_json(
            &mut stream,
            404,
            &json!({"error": {"message": format!("no route for {method} {path}")}}),
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
    let prompt = session.chat.wrap_turns(&turns);
    let tokens = session.tk.encode(&prompt, true, true);

    let budget = req
        .max_tokens
        .or(req.max_completion_tokens)
        .unwrap_or(opts.max_tokens);

    let already = session.processed.len();
    let (logits, reused) = match session.advance(&tokens) {
        Ok(v) => v,
        Err(e) => {
            return send_json(stream, 400, &json!({"error": {"message": e.to_string()}}));
        }
    };
    // What `advance` actually had to run, which is the number worth watching:
    // on a continued session it is one turn, on a restart the whole history.
    let new_tokens = if reused { tokens.len() - already } else { tokens.len() };
    eprintln!(
        "chat: {} prompt tokens, {} prefilled ({}), budget {budget}",
        tokens.len(),
        new_tokens,
        if reused { "continued" } else { "restarted" },
    );

    if req.stream {
        stream_completion(session, stream, logits, budget, opts)
    } else {
        whole_completion(session, stream, logits, budget, opts)
    }
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
) -> Result<(String, &'static str)> {
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
        session.processed.push(next);

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
    Ok((shown, reason))
}

fn stream_completion<O: Ops>(
    session: &mut Session<'_, O>,
    stream: &mut TcpStream,
    logits: Vec<f32>,
    budget: usize,
    opts: &ServeOpts,
) -> Result<()> {
    let head = "HTTP/1.1 200 OK\r\n\
         Content-Type: text/event-stream\r\n\
         Cache-Control: no-cache\r\n\
         Connection: close\r\n\
         Access-Control-Allow-Origin: *\r\n\r\n";
    write_all(stream, head.as_bytes())?;

    let id = completion_id();
    let created = now();
    let model = opts.model_id.clone();

    // The first chunk carries the role, as the OpenAI stream does.
    let first = chunk(&id, created, &model, json!({"role": "assistant"}), None);
    sse(stream, &first)?;

    let mut sink = |delta: &str| -> Result<()> {
        let c = chunk(&id, created, &model, json!({"content": delta}), None);
        sse(stream, &c)
    };
    let (_, reason) = generate(session, logits, budget, &mut sink)?;

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
    let prompt_tokens = session.processed.len();
    let (text, reason) = generate(session, logits, budget, |_| Ok(()))?;
    let completion_tokens = session.processed.len() - prompt_tokens;

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
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
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
