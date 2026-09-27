//! `ChatMl::render` must reproduce the model's own chat template exactly as
//! Hugging Face `transformers` renders it.
//!
//! The fixtures under `tests/fixtures/chat_template/` are written by
//! `scripts/dump_chat_template_fixture.py`, which runs each GGUF's template
//! through Python's `jinja2` with `transformers`' setup. Here the same template,
//! read from the same file, goes through minijinja, and every conversation must
//! match to the byte — tools, tool calls, tool results, thinking, and the
//! templates' own refusals. A prompt that differs by a space is a different
//! prompt from the one the model was trained on.
//!
//! The 0.2B stands in for Qwen3.8-Flash-Next: the two files carry the same
//! template, byte for byte.

use inferred_thoughts::tok::chat::ChatMl;
use inferred_thoughts::{GgufFile, Tokenizer};

mod common;

fn check(fixture: &str) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chat_template");
    let body = std::fs::read_to_string(dir.join(fixture)).expect("read fixture");
    let fx: serde_json::Value = serde_json::from_str(&body).expect("parse fixture");
    let model = fx["model"].as_str().expect("model name");
    let Some(path) = common::find_model_named(model) else {
        println!("SKIPPED: no {model} found; set INFERRED_MODEL_DIR");
        return;
    };

    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let chat = ChatMl::detect(&tk, &f.metadata).expect("a ChatML model");

    let cases = fx["cases"].as_array().expect("cases");
    for case in cases {
        let name = case["name"].as_str().unwrap_or("?");
        let tools = case.get("tools").filter(|t| !t.is_null());
        let got = chat.render(&case["messages"], tools);
        match (case["expected"].as_str(), case["error"].as_str()) {
            (Some(want), _) => {
                let got = got.unwrap_or_else(|e| panic!("{model} / {name}: render failed: {e}"));
                if got != want {
                    let at = got.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(got.len().min(want.len()));
                    panic!(
                        "{model} / {name}: differs at byte {at}\n ours: {:?}\n want: {:?}",
                        &got[at.saturating_sub(40)..(at + 40).min(got.len())],
                        &want[at.saturating_sub(40)..(at + 40).min(want.len())],
                    );
                }
            }
            (None, Some(err)) => {
                let e = got.expect_err(&format!("{model} / {name}: the template should refuse this"));
                assert!(e.to_string().contains(err), "{model} / {name}: wrong refusal: {e}");
            }
            (None, None) => panic!("{model} / {name}: fixture has neither expected text nor an error"),
        }
    }
    println!("{model}: {} conversations identical to transformers' rendering", cases.len());
}

#[test]
#[ignore = "reads the model's template from the GGUF; run with --release -- --ignored"]
fn qwen36_35b_template_renders_as_transformers_does() {
    check("Qwen3.6-35B-A3B-NVFP4-Q8_0.json");
}

#[test]
#[ignore = "reads the model's template from the GGUF; run with --release -- --ignored"]
fn qwen38_flash_next_template_renders_as_transformers_does() {
    check("Qwen3.8-Flash-Next-0.2B-A0.2B-NVFP4exp.json");
}

#[test]
#[ignore = "reads the model's template from the GGUF; run with --release -- --ignored"]
fn qwen3_0_6b_template_renders_as_transformers_does() {
    check("Qwen3-0.6B-Q8_0.json");
}

#[test]
#[ignore = "reads the model's template from the GGUF; run with --release -- --ignored"]
fn qwen35_9b_template_renders_as_transformers_does() {
    check("Qwen3.5-9B-Q8_0.json");
}
