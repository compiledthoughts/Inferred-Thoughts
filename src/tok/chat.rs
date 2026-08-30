//! Chat turn structure, detected from the model rather than assumed.
//!
//! An instruct model fed a raw prompt runs in completion mode: it never enters
//! the assistant turn, so it never emits the turn-ending token the engine stops
//! on, and it degenerates into continuing the prompt as prose. Wrapping the
//! prompt is therefore not a convenience — it is the difference between a
//! usable answer and a repetition loop.
//!
//! **This is not a Jinja interpreter.** `tokenizer.chat_template` is 4100
//! characters of Jinja on Qwen3, and interpreting it properly means a template
//! engine and a lot of surface area for something that is not this project's
//! thesis. Instead we recognize *one* shape — ChatML, the
//! `<|im_start|>role\n…<|im_end|>` structure — and refuse anything else with a
//! named error rather than guessing.
//!
//! Nothing here is hardcoded per model. The marker spellings must be present as
//! real tokens in the file's own vocabulary, and the file's own chat template
//! must actually use them; both are checked, and either failing is an error.
//! That is what `CLAUDE.md`'s "never invent format constants" asks for: the
//! constants are read from the model, and a model that disagrees fails loudly.

use super::Tokenizer;
use crate::error::{Error, Result};
use crate::gguf::Metadata;

/// The ChatML markers, confirmed to exist in a specific model.
#[derive(Debug, Clone)]
pub struct ChatMl {
    start: String,
    end: String,
}

/// GGUF key holding the Jinja chat template.
const TEMPLATE_KEY: &str = "tokenizer.chat_template";

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

        Ok(Self {
            start: start.to_string(),
            end: end.to_string(),
        })
    }

    /// One user turn, followed by the opening of the assistant turn so the
    /// model continues *as* the assistant.
    ///
    /// The trailing newline after `assistant` is load-bearing: without it the
    /// prompt is one token shorter and the model produces a different answer.
    /// That is not a quirk of ours — it is why hand-writing this in a shell is
    /// fragile, since `$(...)` strips trailing newlines.
    pub fn wrap(&self, user: &str) -> String {
        let (s, e) = (&self.start, &self.end);
        format!("{s}user\n{user}{e}\n{s}assistant\n")
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
        }
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
