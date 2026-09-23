//! Shared helpers for tests that need a real model on disk.
//!
//! These tests cannot be self-contained the way the dequantization fixtures
//! are: the forward pass needs the whole model. So they locate one, and skip
//! loudly when it is absent rather than quietly passing.

use std::path::{Path, PathBuf};

/// The 0.6B, the smallest model the project targets. Honours
/// `INFERRED_MODEL_DIR`, matching the tokenizer suite's convention.
#[allow(dead_code)]
pub fn find_model() -> Option<PathBuf> {
    find_model_named("Qwen3-0.6B-Q8_0.gguf")
}

/// Any of the target models, by file name. `CLAUDE.md` records where each
/// lives; `INFERRED_MODEL_DIR` overrides.
#[allow(dead_code)]
pub fn find_model_named(name: &str) -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("INFERRED_MODEL_DIR") {
        roots.push(PathBuf::from(dir));
    }
    if let Ok(home) = std::env::var("HOME") {
        roots.push(Path::new(&home).join("models"));
    }
    roots.push(PathBuf::from("/mnt/d/aiprojects/models"));
    roots.into_iter().map(|r| r.join(name)).find(|p| p.exists())
}

/// Bind the model path or return, announcing the skip on stdout — `--nocapture`
/// is part of the documented invocation, so a silent no-op is not possible.
#[allow(unused_macros)]
macro_rules! model_or_skip {
    ($path:ident) => {
        let Some($path) = crate::common::find_model() else {
            println!("SKIPPED: no Qwen3-0.6B-Q8_0.gguf found; set INFERRED_MODEL_DIR");
            return;
        };
    };
}

#[allow(unused_imports)]
pub(crate) use model_or_skip;

/// A prompt file from `measurements/`, or `None` with the skip announced.
///
/// **Read at run time on purpose.** These prompts are the project's own
/// internal text — real prose, because word salad routes to a handful of
/// experts and flatters every cache — so a published tree does not carry them.
/// An `include_str!` would make the whole test target unbuildable there;
/// `INFERRED_PROMPT_DIR` points at them when they live elsewhere.
#[allow(dead_code)]
pub fn text_or_skip(name: &str) -> Option<String> {
    let dir = std::env::var("INFERRED_PROMPT_DIR").unwrap_or_else(|_| "measurements".to_string());
    match std::fs::read_to_string(Path::new(&dir).join(name)) {
        Ok(text) => Some(text),
        Err(_) => {
            println!("SKIPPED: no {name} in {dir}; set INFERRED_PROMPT_DIR");
            None
        }
    }
}

/// Compare raw bits, not a tolerance.
///
/// `#[allow(dead_code)]`: every integration test binary compiles this module
/// separately, so a helper only some of them use looks unused in the others.
///
/// `CLAUDE.md` forbids adjusting a tolerance until a test passes, and in both
/// places this is used there is nothing to adjust: the two paths under
/// comparison run the same operations in the same order, so anything other than
/// equality is a bug.
#[allow(dead_code)]
pub fn assert_bit_identical(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: different logit counts");
    let mismatches: Vec<usize> = (0..a.len())
        .filter(|&i| a[i].to_bits() != b[i].to_bits())
        .collect();
    if !mismatches.is_empty() {
        let i = mismatches[0];
        panic!(
            "{what}: {} of {} logits differ; first at {i}: {:?} ({:#x}) vs {:?} ({:#x})",
            mismatches.len(),
            a.len(),
            a[i],
            a[i].to_bits(),
            b[i],
            b[i].to_bits(),
        );
    }
}
