//! `--chat` must reproduce the hand-written ChatML prompt exactly.
//!
//! This exists because the hand-written form is fragile in a way that fails
//! silently: `$(printf ...)` in a shell strips the trailing newline, giving a
//! 15-token prompt instead of 16 and a completely different answer. The whole
//! point of the flag is to make that impossible, so the test pins the token
//! sequence rather than the string.

use inferred_thoughts::tok::chat::ChatMl;
use inferred_thoughts::{GgufFile, Tokenizer};

mod common;
use common::model_or_skip;

const PROMPT: &str = "List the capitals of 10 countries";

/// The literal prompt that was verified by hand against `llama-tokenize`,
/// trailing newline included.
const HAND_WRITTEN: &str =
    "<|im_start|>user\nList the capitals of 10 countries<|im_end|>\n<|im_start|>assistant\n";

#[test]
#[ignore = "loads the real model; run with --release -- --ignored"]
fn chat_wrapping_matches_the_hand_written_prompt() {
    model_or_skip!(path);
    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");

    let wrapped = ChatMl::detect(&tk, &f.metadata)
        .expect("Qwen3 is a ChatML model")
        .wrap(PROMPT);
    assert_eq!(wrapped, HAND_WRITTEN);

    // Both must tokenize to the same 16 ids -- llama-tokenize agrees on this
    // sequence, and the markers must become single special tokens rather than
    // literal text.
    let ours = tk.encode(&wrapped, true, true);
    let theirs = tk.encode(HAND_WRITTEN, true, true);
    assert_eq!(ours, theirs);
    assert_eq!(ours.len(), 16, "got {ours:?}");
    assert_eq!(tk.special_id("<|im_start|>"), Some(ours[0]));
}

/// Dropping the trailing newline is the exact failure the flag prevents, so
/// assert the two really are distinguishable.
#[test]
#[ignore = "loads the real model; run with --release -- --ignored"]
fn a_missing_trailing_newline_is_a_different_prompt() {
    model_or_skip!(path);
    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");

    let full = tk.encode(HAND_WRITTEN, true, true);
    let trimmed = tk.encode(HAND_WRITTEN.trim_end_matches('\n'), true, true);
    assert_eq!(full.len(), 16);
    assert_eq!(trimmed.len(), 15);
    assert_ne!(full, trimmed);
}

/// The markers are looked up in the model's own vocabulary, not assumed.
#[test]
#[ignore = "loads the real model; run with --release -- --ignored"]
fn markers_come_from_the_vocabulary() {
    model_or_skip!(path);
    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");

    assert_eq!(tk.special_id("<|im_start|>"), Some(151644));
    assert_eq!(tk.special_id("<|im_end|>"), Some(151645));
    assert_eq!(tk.special_id("<think>"), Some(151667));
    assert_eq!(tk.special_id("</think>"), Some(151668));
    assert_eq!(tk.special_id("<|not_a_real_token|>"), None);
}
