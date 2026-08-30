//! The differential test between backends.
//!
//! `CLAUDE.md` calls for "three implementations behind one `ops` seam, with a
//! differential test asserting they agree." `par` is the second, so this is
//! that test — and because `par` only changes *which thread* computes an output
//! row, never how one is computed, "agree" means **identical bits**, not a
//! tolerance. The narrow unit-level version lives in `src/ops/par.rs`; this is
//! the whole model, end to end, where an indexing slip in the chunked dispatch
//! would surface as a handful of wrong rows deep in some layer.
//!
//! Run with:
//!
//! ```text
//! cargo test --release --test backends -- --ignored --nocapture
//! ```

use inferred_thoughts::{Engine, GgufFile, Naive, Par, Qwen3, Tokenizer};

mod common;
use common::{assert_bit_identical, model_or_skip};

/// A full generation on each backend must produce the same tokens *and* the
/// same logits. Tokens alone would be too weak: an argmax can survive a wrong
/// logit, and this test exists to catch the wrong logit.
#[test]
#[ignore = "loads the real model; run with --release -- --ignored"]
fn par_reproduces_naive_bit_for_bit() {
    model_or_skip!(path);
    Par::init(4);

    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let tokens = tk.encode("The capital of France is", true, true);
    let n_ctx = tokens.len() + 12;

    // Prefill and decode are separate code paths through matmul -- prefill
    // calls it once per token per weight, decode once per weight -- so both get
    // compared, not just the final answer.
    let serial = {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, Naive, n_ctx, false);
        let prefill = e.prefill(&tokens).expect("prefill");
        let decode = e
            .decode(inferred_thoughts::Qwen3::argmax(&prefill))
            .expect("decode");
        (prefill, decode)
    };

    let parallel = {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, Par, n_ctx, false);
        let prefill = e.prefill(&tokens).expect("prefill");
        let decode = e
            .decode(inferred_thoughts::Qwen3::argmax(&prefill))
            .expect("decode");
        (prefill, decode)
    };

    assert_bit_identical(&serial.0, &parallel.0, "prefill logits");
    assert_bit_identical(&serial.1, &parallel.1, "decode logits");
}

/// Longer run, comparing the generated text itself. This is the check that
/// would notice a race the single-pass comparison happened to miss.
#[test]
#[ignore = "loads the real model; run with --release -- --ignored"]
fn par_generates_the_same_text_as_naive() {
    model_or_skip!(path);
    Par::init(4);

    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let tokens = tk.encode("The capital of France is", true, true);
    let n_ctx = tokens.len() + 40;

    let m = Qwen3::load(&f).expect("load model");
    let mut e = Engine::new(m, Naive, n_ctx, false);
    let serial = e.generate(&tokens, 32, None, |_| {}).expect("generate");

    let m = Qwen3::load(&f).expect("load model");
    let mut e = Engine::new(m, Par, n_ctx, false);
    let parallel = e.generate(&tokens, 32, None, |_| {}).expect("generate");

    assert_eq!(
        serial,
        parallel,
        "backends diverged: {:?} vs {:?}",
        tk.decode(&serial, false),
        tk.decode(&parallel, false),
    );
}
