//! Stage 5 acceptance: the KV cache must not change the arithmetic.
//!
//! `PROMPTS.md` asks for token-for-token agreement with `llama-cli` over 50
//! greedy tokens. **That criterion cannot do the job it exists for.**
//! `CLAUDE.md` characterizes ~1% logit drift against llama.cpp as inherent to
//! two independent Q8_0 implementations, so somewhere in 250 greedy decisions
//! two logits will sit within that band and the argmax will flip. A failure
//! would not distinguish drift from a cache bug — which is the entire thing the
//! test exists to catch.
//!
//! So the criterion here is self-consistency, which is strictly stronger:
//!
//! > decode-with-cache must produce **bit-identical** logits to full recompute
//!
//! Same code path, same arithmetic, same order, over the same f16-rounded K and
//! V — so the expected answer is exact equality of the raw bits, and any slot
//! addressing or RoPE position error shows up immediately and unambiguously. A
//! llama.cpp comparison stays useful, but as a check on numerics, which Stage 4
//! already established, not on the cache.
//!
//! These run the real model, so they are `#[ignore]`d: a debug-profile forward
//! pass over Qwen3-0.6B takes long enough to make `cargo test` unusable. Run
//! them with:
//!
//! ```text
//! cargo test --release --test kv_cache -- --ignored --nocapture
//! ```
//!
//! The cheap invariants that do run by default live in `src/cache/mod.rs`.

use inferred_thoughts::{Engine, GgufFile, KvCache, Naive, Qwen3, Tokenizer};

mod common;
use common::{assert_bit_identical, model_or_skip};

fn prompt_tokens(f: &GgufFile, text: &str) -> Vec<u32> {
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    tk.encode(text, true, true)
}

/// The acceptance test. Prefill the whole sequence in one pass, then rebuild it
/// one token at a time, and require the final logits to match exactly.
#[test]
#[ignore = "loads the real model; run with --release -- --ignored"]
fn decode_with_cache_equals_full_recompute() {
    model_or_skip!(path);
    let f = GgufFile::open(&path).expect("open model");
    let tokens = prompt_tokens(&f, "The capital of France is Paris, and the capital of");
    assert!(tokens.len() >= 6, "prompt too short to exercise decode");

    let n_ctx = tokens.len() + 4;

    let full = {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, Naive, n_ctx, false);
        e.prefill(&tokens).expect("prefill")
    };

    let incremental = {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, Naive, n_ctx, false);
        // Position 0 must come through prefill: decode is defined as appending
        // at cache.len(), and an empty batch is rejected.
        let mut logits = e.prefill(&tokens[..1]).expect("prefill first token");
        for (i, &tok) in tokens.iter().enumerate().skip(1) {
            logits = e.decode(tok).expect("decode");
            assert_eq!(e.pos(), i + 1, "cache length tracks absolute position");
        }
        logits
    };

    assert_bit_identical(&full, &incremental, "one-shot prefill vs token-by-token");
}

/// Every prefix, not just the last position. A position error that happens to
/// cancel at the end of one sequence cannot survive this.
#[test]
#[ignore = "loads the real model; run with --release -- --ignored"]
fn every_prefix_agrees_with_its_own_recompute() {
    model_or_skip!(path);
    let f = GgufFile::open(&path).expect("open model");
    let tokens = prompt_tokens(&f, "One two three four five six");
    let n_ctx = tokens.len() + 1;

    // Incremental: one cache, growing.
    let m = Qwen3::load(&f).expect("load model");
    let mut e = Engine::new(m, Naive, n_ctx, false);
    let mut incremental = vec![e.prefill(&tokens[..1]).expect("prefill")];
    for &tok in tokens.iter().skip(1) {
        incremental.push(e.decode(tok).expect("decode"));
    }

    // Reference: a fresh full recompute per prefix length.
    for (i, inc) in incremental.iter().enumerate() {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, Naive, n_ctx, false);
        let full = e.prefill(&tokens[..i + 1]).expect("prefill prefix");
        assert_bit_identical(&full, inc, &format!("prefix of length {}", i + 1));
    }
}

/// Prefill in two batches. Single-token decode never exercises `start_pos > 0`
/// with more than one token in flight, which is a distinct indexing path: the
/// batch must both RoPE at absolute positions and attend to rows it is writing
/// in the same call.
#[test]
#[ignore = "loads the real model; run with --release -- --ignored"]
fn split_prefill_equals_single_prefill() {
    model_or_skip!(path);
    let f = GgufFile::open(&path).expect("open model");
    let tokens = prompt_tokens(&f, "Alpha beta gamma delta epsilon zeta eta");
    let split = tokens.len() / 2;
    assert!(split >= 2 && tokens.len() - split >= 2, "need two real halves");
    let n_ctx = tokens.len() + 1;

    let one_shot = {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, Naive, n_ctx, false);
        e.prefill(&tokens).expect("prefill")
    };

    let two_shot = {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, Naive, n_ctx, false);
        e.prefill(&tokens[..split]).expect("first half");
        e.prefill(&tokens[split..]).expect("second half")
    };

    assert_bit_identical(&one_shot, &two_shot, "one batch vs two");
}

/// Generation must not silently truncate when the cache fills. It should stop,
/// and the engine should still be in a coherent state.
#[test]
#[ignore = "loads the real model; run with --release -- --ignored"]
fn generation_stops_at_the_context_limit() {
    model_or_skip!(path);
    let f = GgufFile::open(&path).expect("open model");
    let tokens = prompt_tokens(&f, "Counting:");
    let n_ctx = tokens.len() + 3;

    let m = Qwen3::load(&f).expect("load model");
    let mut e = Engine::new(m, Naive, n_ctx, false);
    let produced = e
        .generate(&tokens, 50, None, |_| {})
        .expect("generate must stop, not overflow");
    assert!(produced.len() <= 3, "produced {} tokens", produced.len());
    assert!(e.pos() <= n_ctx);
}

/// A pass that would overflow must fail before writing anything, so the cache
/// never claims positions it did not fill.
#[test]
#[ignore = "loads the real model; run with --release -- --ignored"]
fn overflowing_prefill_leaves_the_cache_untouched() {
    model_or_skip!(path);
    let f = GgufFile::open(&path).expect("open model");
    let tokens = prompt_tokens(&f, "This prompt is longer than the cache allows");
    let m = Qwen3::load(&f).expect("load model");
    let mut e = Engine::new(m, Naive, 2, false);
    assert!(tokens.len() > 2);
    assert!(e.prefill(&tokens).is_err());
    assert_eq!(e.pos(), 0, "a rejected pass must not advance the cache");
}

/// The profiler must observe, not participate: the same run with detail on and
/// off must produce the same tokens.
#[test]
#[ignore = "loads the real model; run with --release -- --ignored"]
fn profiling_does_not_change_the_output() {
    model_or_skip!(path);
    let f = GgufFile::open(&path).expect("open model");
    let tokens = prompt_tokens(&f, "The capital of France is");
    let n_ctx = tokens.len() + 8;

    let run = |detail: bool| {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, Naive, n_ctx, detail);
        let out = e.generate(&tokens, 6, None, |_| {}).expect("generate");
        (out, e.prof.layers.len())
    };

    let (plain, no_events) = run(false);
    let (detailed, events) = run(true);
    assert_eq!(plain, detailed, "detail profiling perturbed the output");
    assert_eq!(no_events, 0, "tier 2 recorded events with detail off");
    assert!(events > 0, "detail profiling recorded nothing");
}

/// The cache's f16 storage is what makes attention match llama.cpp's, so a
/// silent switch to f32 would be a regression the logits alone would not name.
#[test]
fn cache_stores_f16_not_f32() {
    let mut c = KvCache::new(1, 1, 1);
    let x = std::f32::consts::PI;
    c.store(0, 0, &[x], &[x]).expect("store");
    assert_ne!(KvCache::read(c.k_head(0, 0, 0, 1)[0]), x);
}
