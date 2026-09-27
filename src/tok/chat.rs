//! Chat turn structure, detected from the model rather than assumed.
//!
//! An instruct model fed a raw prompt runs in completion mode: it never enters
//! the assistant turn, so it never emits the turn-ending token the engine stops
//! on, and it degenerates into continuing the prompt as prose. Wrapping the
//! prompt is therefore not a convenience — it is the difference between a
//! usable answer and a repetition loop.
//!
//! Two renderers, for two jobs.
//!
//! - **[`ChatMl::wrap`] and [`ChatMl::wrap_turns`]** recognize *one* shape —
//!   ChatML, the `<|im_start|>role\n…<|im_end|>` structure — for `generate
//!   --chat`, the `serve` warm-up and the tests that pin prompts to the token.
//!   Unchanged, so every number measured through them stays comparable.
//! - **[`ChatMl::render`]** runs the model's own `tokenizer.chat_template`, as
//!   written, for `serve`. Tools, tool calls, tool results and the thinking
//!   block are all spelled by the template, and a model's template *is* its
//!   specification: re-typing it in Rust would drift from what the model was
//!   trained on (the hand-written ChatML path never opened the reply with
//!   `<think>\n`, which both Qwen3.6 and Qwen3.8 templates do).
//!
//! The engine is [minijinja], Jinja by Jinja's author, configured as Hugging
//! Face `transformers` configures Jinja for chat templates: `trim_blocks`,
//! `lstrip_blocks`, Python's string methods, `raise_exception`, and a `tojson`
//! with Python's separators. An earlier version of this file refused a template
//! engine as too much surface area to write; borrowing a mature one is a
//! different trade.
//!
//! Nothing here is hardcoded per model. The marker spellings must be present as
//! real tokens in the file's own vocabulary, and the file's own chat template
//! must actually use them; both are checked, and either failing is an error.
//! That is what `CLAUDE.md`'s "never invent format constants" asks for: the
//! constants are read from the model, and a model that disagrees fails loudly.

use std::sync::Arc;

use super::Tokenizer;
use crate::error::{Error, Result};
use crate::gguf::Metadata;

/// The ChatML markers, confirmed to exist in a specific model, and the model's
/// own chat template, compiled.
#[derive(Debug, Clone)]
pub struct ChatMl {
    start: String,
    end: String,
    /// `None` when the template failed to compile; [`ChatMl::render`] then
    /// returns the reason instead of guessing a format.
    template: Option<Arc<minijinja::Environment<'static>>>,
    template_error: Option<String>,
}

/// GGUF key holding the Jinja chat template.
const TEMPLATE_KEY: &str = "tokenizer.chat_template";

/// The name the compiled template is registered under; any name would do.
const TEMPLATE_NAME: &str = "chat";

/// Compile a chat template the way `transformers` sets up Jinja for one
/// (`utils/chat_template_utils.py`): blocks trimmed, Python's string methods
/// available, `raise_exception` defined, and `tojson` as Python's `json.dumps`.
fn compile(source: &str) -> std::result::Result<minijinja::Environment<'static>, String> {
    let mut env = minijinja::Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_filter("tojson", tojson);
    env.add_function("raise_exception", raise_exception);
    env.add_template_owned(TEMPLATE_NAME, source.to_string())
        .map_err(|e| format!("{e:#}"))?;
    Ok(env)
}

fn template_error(e: minijinja::Error) -> Error {
    Error::UnsupportedChatTemplate { detail: format!("{e:#}") }
}

/// `raise_exception(message)`: the templates' own validation, surfaced as an
/// error rather than rendered.
fn raise_exception(message: String) -> std::result::Result<minijinja::Value, minijinja::Error> {
    Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, message))
}

/// `tojson` as `transformers` defines it: `json.dumps(x, ensure_ascii=False)`,
/// so `", "` and `": "` between items and keys in insertion order.
///
/// **Not minijinja's built-in**, which is compact (`","`, `":"`) and escapes
/// HTML characters. The tools block is rendered with this filter, and the model
/// was trained on Python's spacing; a compact line is a different prompt.
fn tojson(value: minijinja::Value) -> std::result::Result<minijinja::Value, minijinja::Error> {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PythonFormatter);
    serde::Serialize::serialize(&value, &mut ser)
        .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::BadSerialization, e.to_string()))?;
    let text = String::from_utf8(out)
        .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::BadSerialization, e.to_string()))?;
    Ok(minijinja::Value::from_safe_string(text))
}

/// `json.dumps`'s default separators: `", "` and `": "`.
struct PythonFormatter;

impl serde_json::ser::Formatter for PythonFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

impl ChatMl {
    /// Confirm this model speaks ChatML, or say why not.
    ///
    /// Two independent checks, because either alone can be satisfied by
    /// accident: the markers must be real tokens in the vocabulary (so the
    /// wrapped prompt tokenizes to single tokens rather than literal text), and
    /// the model's own template must reference them (so we are not imposing
    /// ChatML on a Llama- or Gemma-style model that happens to carry the
    /// tokens).
    pub fn detect(tk: &Tokenizer, md: &Metadata) -> Result<Self> {
        let (start, end) = ("<|im_start|>", "<|im_end|>");

        for marker in [start, end] {
            if tk.special_id(marker).is_none() {
                return Err(Error::UnsupportedChatTemplate {
                    detail: format!("{marker} is not a special token in this vocabulary"),
                });
            }
        }

        let template = md.get_string(TEMPLATE_KEY).map_err(|_| Error::UnsupportedChatTemplate {
            detail: format!("model has no {TEMPLATE_KEY}; pass a pre-formatted prompt instead"),
        })?;
        if !template.contains(start) {
            return Err(Error::UnsupportedChatTemplate {
                detail: format!(
                    "{TEMPLATE_KEY} does not use {start}; only the ChatML shape is implemented"
                ),
            });
        }

        let (template, template_error) = match compile(template) {
            Ok(env) => (Some(Arc::new(env)), None),
            Err(e) => (None, Some(e)),
        };
        Ok(Self {
            start: start.to_string(),
            end: end.to_string(),
            template,
            template_error,
        })
    }

    /// Render a conversation with the model's own chat template, and open the
    /// assistant turn.
    ///
    /// `messages` and `tools` are OpenAI-shaped JSON, passed to the template as
    /// the variables of the same names, with `add_generation_prompt` set. Tool
    /// calls in assistant messages must carry `arguments` as an object: the
    /// template iterates them with `|items`. [`crate::serve`] converts the JSON
    /// string OpenAI clients send.
    pub fn render(&self, messages: &serde_json::Value, tools: Option<&serde_json::Value>) -> Result<String> {
        self.render_with(messages, tools, None)
    }

    /// [`ChatMl::render`], plus the template's own switches — what OpenAI-style
    /// servers call `chat_template_kwargs`: `enable_thinking` (Qwen3.6 and
    /// Qwen3.8 pre-fill an empty thinking block when it is false),
    /// `reasoning_effort` (Qwen3.8: `xhigh`, the default, `medium` or `low`;
    /// any other value is refused by name), `preserve_thinking`. A template
    /// ignores the ones it does not read. `messages`, `tools` and
    /// `add_generation_prompt` are set here and cannot be overridden.
    pub fn render_with(
        &self,
        messages: &serde_json::Value,
        tools: Option<&serde_json::Value>,
        kwargs: Option<&serde_json::Map<String, serde_json::Value>>,
    ) -> Result<String> {
        let env = self.template.as_ref().ok_or_else(|| Error::UnsupportedChatTemplate {
            detail: format!(
                "the model's chat template does not compile: {}",
                self.template_error.as_deref().unwrap_or("unknown")
            ),
        })?;
        let tmpl = env.get_template(TEMPLATE_NAME).map_err(template_error)?;
        let mut ctx = kwargs.cloned().unwrap_or_default();
        ctx.insert("messages".to_string(), messages.clone());
        match tools {
            Some(t) => {
                ctx.insert("tools".to_string(), t.clone());
            }
            None => {
                ctx.remove("tools");
            }
        }
        ctx.insert("add_generation_prompt".to_string(), serde_json::Value::Bool(true));
        tmpl.render(minijinja::Value::from_serialize(&ctx))
            .map_err(template_error)
    }

    /// One user turn, followed by the opening of the assistant turn so the
    /// model continues *as* the assistant.
    ///
    /// The trailing newline after `assistant` is load-bearing: without it the
    /// prompt is one token shorter and the model produces a different answer.
    /// That is not a quirk of ours — it is why hand-writing this in a shell is
    /// fragile, since `$(...)` strips trailing newlines.
    pub fn wrap(&self, user: &str) -> String {
        self.wrap_turns(&[("user", user)])
    }

    /// Render a whole conversation and open the assistant turn.
    ///
    /// Roles pass through as the caller gives them, so `system`, `user` and
    /// `assistant` all work. Turns render in order and the assistant turn is
    /// opened at the end, which is what makes this rendering a *prefix* of the
    /// next request's rendering — the property the server depends on to
    /// continue a session rather than re-run it.
    pub fn wrap_turns(&self, turns: &[(&str, &str)]) -> String {
        let (s, e) = (&self.start, &self.end);
        let mut out = String::new();
        for (role, content) in turns {
            out.push_str(&format!("{s}{role}\n{content}{e}\n"));
        }
        out.push_str(&format!("{s}assistant\n"));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact byte layout matters, so assert it rather than eyeballing it.
    fn chatml() -> ChatMl {
        ChatMl {
            start: "<|im_start|>".to_string(),
            end: "<|im_end|>".to_string(),
            template: None,
            template_error: None,
        }
    }

    /// A `ChatMl` whose template is `source`, for testing the engine setup
    /// without a model file.
    fn with_template(source: &str) -> ChatMl {
        ChatMl {
            template: Some(Arc::new(compile(source).expect("compiles"))),
            ..chatml()
        }
    }

    #[test]
    fn tojson_uses_pythons_separators_and_keeps_key_order() {
        // `json.dumps({"b": 1, "a": [1, "x"], "u": "é\n"}, ensure_ascii=False)`
        let c = with_template("{{ tools[0] | tojson }}");
        let tools = serde_json::json!([{"b": 1, "a": [1, "x"], "u": "é\n"}]);
        assert_eq!(
            c.render(&serde_json::json!([]), Some(&tools)).unwrap(),
            r#"{"b": 1, "a": [1, "x"], "u": "é\n"}"#
        );
    }

    #[test]
    fn blocks_are_trimmed_and_python_string_methods_work() {
        // trim_blocks + lstrip_blocks: a block tag's own line leaves nothing.
        let c = with_template(
            "{%- for m in messages %}\n  {% if m.content.startswith('x') %}\n[{{ m.content.split(':')[1].rstrip('!') }}]\n  {% endif %}\n{%- endfor %}",
        );
        let msgs = serde_json::json!([{"role": "user", "content": "x:hi!"}, {"role": "user", "content": "y"}]);
        assert_eq!(c.render(&msgs, None).unwrap(), "[hi]\n");
    }

    #[test]
    fn template_switches_reach_the_template_only_when_given() {
        let c = with_template(
            "[{{ reasoning_effort|default('xhigh') }}|{{ 'off' if enable_thinking is defined and enable_thinking is false else 'on' }}|{{ messages|length }}]",
        );
        let m = serde_json::json!([{"role": "user", "content": "hi"}]);
        let kw = serde_json::json!({"reasoning_effort": "medium", "enable_thinking": false});
        assert_eq!(c.render_with(&m, None, kw.as_object()).unwrap(), "[medium|off|1]");
        assert_eq!(c.render_with(&m, None, None).unwrap(), "[xhigh|on|1]");
        assert_eq!(c.render(&m, None).unwrap(), "[xhigh|on|1]");
        // A switch cannot replace the conversation itself.
        let sneaky = serde_json::json!({"messages": []});
        assert_eq!(c.render_with(&m, None, sneaky.as_object()).unwrap(), "[xhigh|on|1]");
    }

    #[test]
    fn raise_exception_is_an_error_not_text() {
        let c = with_template("{{ raise_exception('No messages provided.') }}");
        let e = c.render(&serde_json::json!([]), None).unwrap_err().to_string();
        assert!(e.contains("No messages provided."), "{e}");
    }

    #[test]
    fn an_uncompilable_template_is_refused_by_name() {
        let c = ChatMl { template_error: Some("unexpected end".to_string()), ..chatml() };
        let e = c.render(&serde_json::json!([]), None).unwrap_err().to_string();
        assert!(e.contains("does not compile") && e.contains("unexpected end"), "{e}");
    }

    #[test]
    fn wraps_a_user_turn_and_opens_the_assistant_turn() {
        assert_eq!(
            chatml().wrap("hi"),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn keeps_the_trailing_newline() {
        // Dropping this is the 15-vs-16-token bug that changes the answer.
        assert!(chatml().wrap("hi").ends_with("assistant\n"));
    }

    #[test]
    fn user_text_is_inserted_verbatim() {
        // No trimming, no escaping: whatever the user typed is what the model
        // sees between the markers.
        let w = chatml().wrap("  two  spaces\nand a newline  ");
        assert!(w.contains("  two  spaces\nand a newline  "));
    }
}
