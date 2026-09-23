//! BPE tokenizer built entirely from GGUF metadata.
//!
//! Nothing here is hardcoded per model: the vocabulary, merges, special token
//! ids and pre-tokenizer choice all come from the file. The only per-model
//! knowledge is which pre-tokenizer names we implement, and an unknown one is
//! an error rather than a silent fallback to a default that would tokenize
//! subtly differently.
//!
//! Flags for our two models, read from llama.cpp's `llama-vocab.cpp`:
//! `qwen2` and `qwen35` both set `clean_spaces = false`, `ignore_merges =
//! false`, `add_space_prefix = false` and `byte_encode = true`.

pub mod chat;
mod bpe;
mod split;
mod unicode;
// Generated: carries the full flag set from the reference even though the
// splitters only consult four of the categories.
#[rustfmt::skip]
#[allow(dead_code)]
mod unicode_data;

pub use split::PreType;

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::gguf::Metadata;
use bpe::ByteEncoder;

/// Token type codes from GGUF (`llama_token_type` in the reference).
mod token_type {
    pub const CONTROL: i32 = 3;
    pub const USER_DEFINED: i32 = 4;
}

pub struct Tokenizer {
    tokens: Vec<String>,
    token_types: Vec<i32>,
    /// Keyed by raw bytes rather than `String` so a lookup can mirror C++'s
    /// `std::string` semantics exactly, including non-UTF-8 byte fallbacks.
    token_to_id: HashMap<Vec<u8>, u32>,
    ranks: HashMap<(Vec<u8>, Vec<u8>), u32>,
    /// Special tokens, longest first, so matching is greedy.
    specials: Vec<(Vec<u8>, u32)>,
    byte_encoder: ByteEncoder,
    pre: PreType,

    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub padding_token_id: Option<u32>,
    pub add_bos_token: bool,
    pub add_eos_token: bool,
}

impl Tokenizer {
    pub fn from_metadata(md: &Metadata) -> Result<Self> {
        let model = md.get_string("tokenizer.ggml.model")?;
        if model != "gpt2" {
            return Err(Error::UnsupportedTokenizerModel {
                model: model.to_string(),
            });
        }

        let pre_name = md.get_string("tokenizer.ggml.pre")?;
        let pre = match pre_name {
            "qwen2" => PreType::Qwen2,
            "qwen35" => PreType::Qwen35,
            other => {
                return Err(Error::UnsupportedPreTokenizer {
                    pre: other.to_string(),
                    supported: "qwen2, qwen35",
                });
            }
        };

        let tokens: Vec<String> = md.get_string_array("tokenizer.ggml.tokens")?.to_vec();

        let token_types: Vec<i32> = match md.get_array("tokenizer.ggml.token_type")? {
            crate::gguf::Array::I32(v) => v.clone(),
            crate::gguf::Array::U32(v) => v.iter().map(|&x| x as i32).collect(),
            other => {
                return Err(Error::TypeMismatch {
                    key: "tokenizer.ggml.token_type".to_string(),
                    expected: "arr[i32]",
                    actual: other.elem_type_name(),
                });
            }
        };
        if token_types.len() != tokens.len() {
            return Err(Error::TokenTypeLengthMismatch {
                n_types: token_types.len(),
                n_tokens: tokens.len(),
            });
        }

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (id, t) in tokens.iter().enumerate() {
            // On a duplicate, the reference keeps the first id.
            token_to_id.entry(t.as_bytes().to_vec()).or_insert(id as u32);
        }

        // Merges are "left right"; the reference searches for the separator
        // from index 1, so a merge whose left side is itself a space still
        // splits correctly.
        let merge_list = md.get_string_array("tokenizer.ggml.merges")?;
        let mut ranks = HashMap::with_capacity(merge_list.len());
        for (i, m) in merge_list.iter().enumerate() {
            // Byte search, not a `str` search: merge sides are byte-encoded, so
            // a merge like "Ġ Ġ" starts with a two-byte codepoint and byte
            // index 1 is not a character boundary. A space is never a UTF-8
            // continuation byte, so the split itself always lands cleanly.
            let bytes = m.as_bytes();
            let pos = bytes
                .get(1..)
                .and_then(|rest| rest.iter().position(|&b| b == b' '))
                .map(|p| p + 1)
                .ok_or_else(|| Error::BadMerge {
                    index: i,
                    text: m.clone(),
                })?;
            let left = bytes[..pos].to_vec();
            let right = bytes[pos + 1..].to_vec();
            ranks.entry((left, right)).or_insert(i as u32);
        }

        let mut specials: Vec<(Vec<u8>, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                matches!(
                    token_types[*i],
                    token_type::CONTROL | token_type::USER_DEFINED
                )
            })
            .map(|(i, t)| (t.as_bytes().to_vec(), i as u32))
            .collect();
        // Longest first so that a token which is a prefix of another cannot
        // shadow it during the scan.
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

        Ok(Self {
            tokens,
            token_types,
            token_to_id,
            ranks,
            specials,
            byte_encoder: ByteEncoder::new(),
            pre,
            bos_token_id: md.get_u32("tokenizer.ggml.bos_token_id").ok(),
            eos_token_id: md.get_u32("tokenizer.ggml.eos_token_id").ok(),
            padding_token_id: md.get_u32("tokenizer.ggml.padding_token_id").ok(),
            // Absent means false for BPE in the reference.
            add_bos_token: md.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(false),
            add_eos_token: md.get_bool("tokenizer.ggml.add_eos_token").unwrap_or(false),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn pre_type(&self) -> PreType {
        self.pre
    }

    pub fn token_text(&self, id: u32) -> Result<&str> {
        self.tokens
            .get(id as usize)
            .map(|s| s.as_str())
            .ok_or(Error::TokenOutOfRange {
                id,
                vocab_size: self.tokens.len(),
            })
    }

    /// Id of a special token by its literal spelling, e.g. `<|im_start|>`.
    ///
    /// The reverse of [`Tokenizer::token_text`] restricted to special tokens,
    /// so callers can ask the *model* whether it knows a marker instead of
    /// assuming an id. See [`chat::ChatMl::detect`].
    pub fn special_id(&self, text: &str) -> Option<u32> {
        self.specials
            .iter()
            .find(|(spelling, _)| spelling.as_slice() == text.as_bytes())
            .map(|(_, id)| *id)
    }

    pub fn is_special(&self, id: u32) -> bool {
        matches!(
            self.token_types.get(id as usize).copied(),
            Some(token_type::CONTROL) | Some(token_type::USER_DEFINED)
        )
    }

    /// Encode text to token ids.
    ///
    /// `add_special` prepends/appends BOS/EOS per the file's own
    /// `add_bos_token` / `add_eos_token` flags. `parse_special` makes literal
    /// special-token text such as `<|im_start|>` encode as that single token
    /// rather than as ordinary characters.
    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();

        if add_special && self.add_bos_token {
            if let Some(bos) = self.bos_token_id {
                out.push(bos);
            }
        }

        if parse_special {
            for fragment in self.partition_specials(text) {
                match fragment {
                    Fragment::Special(id) => out.push(id),
                    Fragment::Text(s) => self.encode_raw(s, &mut out),
                }
            }
        } else {
            self.encode_raw(text, &mut out);
        }

        if add_special && self.add_eos_token {
            if let Some(eos) = self.eos_token_id {
                out.push(eos);
            }
        }

        out
    }

    /// Pre-tokenize, byte-encode, merge, and look up -- the path for text with
    /// no special tokens in it.
    fn encode_raw(&self, text: &str, out: &mut Vec<u32>) {
        for word in split::split_str(text, self.pre) {
            let encoded = self.byte_encoder.encode(word.as_bytes());
            let bytes = encoded.as_bytes();

            for (a, b) in bpe::merge_word(bytes, &self.ranks) {
                let piece = &bytes[a..b];
                match self.token_to_id.get(piece) {
                    Some(&id) => out.push(id),
                    None => {
                        // Byte fallback, as in the reference: emit whatever
                        // single-byte tokens exist and drop the rest rather
                        // than inventing an id.
                        for &byte in piece {
                            if let Some(&id) = self.token_to_id.get(&[byte][..]) {
                                out.push(id);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Split text around literal special-token spellings, greedily and longest
    /// first.
    fn partition_specials<'a>(&self, text: &'a str) -> Vec<Fragment<'a>> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut start = 0usize;
        let mut i = 0usize;

        while i < bytes.len() {
            // UTF-8 is self-synchronizing, so a valid token can only match on a
            // character boundary; checking anyway keeps the slicing below from
            // ever panicking if that reasoning is wrong.
            let hit = if text.is_char_boundary(i) {
                self.specials
                    .iter()
                    .find(|(tok, _)| bytes[i..].starts_with(tok))
            } else {
                None
            };

            match hit {
                Some((tok, id)) => {
                    if i > start {
                        out.push(Fragment::Text(&text[start..i]));
                    }
                    out.push(Fragment::Special(*id));
                    i += tok.len();
                    start = i;
                }
                None => i += 1,
            }
        }

        if start < bytes.len() {
            out.push(Fragment::Text(&text[start..]));
        }
        out
    }

    /// Decode token ids back to text.
    ///
    /// `render_special` controls whether control tokens appear literally; with
    /// it off they are skipped, which is what a chat UI wants.
    pub fn decode(&self, ids: &[u32], render_special: bool) -> Result<String> {
        self.decode_keeping(ids, &|id| render_special || !self.is_special(id))
    }

    /// Decode, rendering a special token only when `keep` says so.
    ///
    /// **Why the middle ground exists.** A reasoning model's turn is
    /// `<think>…</think>` then the answer, and both markers are control tokens.
    /// Rendering every special leaks `<|im_end|>` and the chat scaffolding into
    /// the reply; rendering none hands a client one undifferentiated string, so
    /// it cannot tell reasoning from answer — which is what `serve` did, and
    /// what left Cline showing the model's thinking as its reply.
    pub fn decode_keeping(&self, ids: &[u32], keep: &dyn Fn(u32) -> bool) -> Result<String> {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let text = self.token_text(id)?;
            if self.is_special(id) {
                if keep(id) {
                    bytes.extend_from_slice(text.as_bytes());
                }
                continue;
            }
            match self.byte_encoder.decode(text) {
                Some(raw) => bytes.extend_from_slice(&raw),
                // A token outside the byte alphabet is emitted as-is rather
                // than dropped.
                None => bytes.extend_from_slice(text.as_bytes()),
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

enum Fragment<'a> {
    Text(&'a str),
    Special(u32),
}
