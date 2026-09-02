//! The `qwen35` config, read from the real files.
//!
//! `src/model/qwen35.rs` derives every GatedDeltaNet dimension from the `ssm.*`
//! metadata rather than hardcoding it. That derivation is only worth anything
//! if it agrees with the tensors actually in the file, which is what this
//! checks — for both models, since the 9B and the 35B share the recurrent
//! hyperparameters and differ only in width and in having an MoE FFN.

use inferred_thoughts::GgufFile;
use inferred_thoughts::model::qwen35::Config;

mod common;

/// Assert a tensor exists with exactly this shape, naming both on failure.
fn shape(f: &GgufFile, name: &str) -> Vec<u64> {
    f.tensor(name)
        .unwrap_or_else(|| panic!("{name} is missing"))
        .dims
        .clone()
}

/// Decoding must advance the cache position.
///
/// This is a regression test for a bug that was invisible everywhere it was
/// looked for. `qwen35::forward` did not call `KvCache::commit`, so
/// `KvCache::len` never moved and `Engine::run` began every decode step at
/// position 0: each token overwrote KV slot 0, attended only to itself, and
/// was rotated at position 0.
///
/// Nothing caught it. A single `forward` call gets its positions right
/// internally, so the layer-by-layer trace against `llama-eval-callback`
/// matched to 1e-8 at layer 0 and the tensor comparison looked clean. And 24 of
/// the 32 layers are GatedDeltaNet, whose recurrent state advances correctly
/// regardless of `pos`, so the model went on emitting plausible English for a
/// dozen tokens before collapsing -- which reads as numerical drift rather than
/// a positional bug. It was mistaken for exactly that.
///
/// Asserting the position directly is cheap and would have named it at once.
#[test]
#[ignore = "loads the real 9B; run with --release -- --ignored"]
fn decoding_advances_the_cache_position() {
    use inferred_thoughts::{Engine, Model, Spin, Tokenizer};

    let Some(path) = common::find_model_named("Qwen3.5-9B-Q8_0.gguf") else {
        println!("SKIPPED: no Qwen3.5-9B-Q8_0.gguf found");
        return;
    };
    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let m = Model::load(&f).expect("load model");
    let mut engine = Engine::new(m, Spin::new(8), 512, false);

    let prompt = tk.encode("The capital of France is", true, true);
    assert_eq!(engine.pos(), 0, "a fresh engine is at position 0");

    engine.prefill(&prompt).expect("prefill");
    assert_eq!(
        engine.pos(),
        prompt.len(),
        "prefill must leave the cache holding every prompt token"
    );

    for step in 1..=3 {
        engine.decode(7).expect("decode");
        assert_eq!(
            engine.pos(),
            prompt.len() + step,
            "each decode step must claim the next position"
        );
    }

    engine.reset();
    assert_eq!(engine.pos(), 0, "reset returns to the start");
}

#[test]
#[ignore = "loads the real 9B; run with --release -- --ignored"]
fn config_matches_the_9b_tensor_shapes() {
    let Some(path) = common::find_model_named("Qwen3.5-9B-Q8_0.gguf") else {
        println!("SKIPPED: no Qwen3.5-9B-Q8_0.gguf found");
        return;
    };
    let f = GgufFile::open(&path).expect("open model");
    let c = Config::from_gguf(&f).expect("qwen35 config");

    assert_eq!(c.n_layer, 32);
    assert_eq!(c.n_embd, 4096);
    assert_eq!(c.head_dim, 256);
    assert_eq!(c.n_rot, 64, "partial RoPE: 64 of 256");
    assert_eq!(c.rope_sections, [11, 11, 10, 0]);

    // Layer 0 is recurrent, layer 3 is full attention.
    assert!(c.is_recurrent(0) && c.is_recurrent(1) && c.is_recurrent(2));
    assert!(!c.is_recurrent(3));

    // Every derived dimension, checked against the tensor that embodies it.
    let n = c.n_embd as u64;
    assert_eq!(shape(&f, "blk.0.ssm_conv1d.weight"), vec![c.ssm_d_conv as u64, c.conv_dim() as u64]);
    assert_eq!(shape(&f, "blk.0.attn_qkv.weight"), vec![n, c.conv_dim() as u64]);
    assert_eq!(shape(&f, "blk.0.attn_gate.weight"), vec![n, c.value_dim() as u64]);
    assert_eq!(shape(&f, "blk.0.ssm_alpha.weight"), vec![n, c.n_v_heads() as u64]);
    assert_eq!(shape(&f, "blk.0.ssm_beta.weight"), vec![n, c.n_v_heads() as u64]);
    assert_eq!(shape(&f, "blk.0.ssm_a"), vec![c.n_v_heads() as u64]);
    assert_eq!(shape(&f, "blk.0.ssm_dt.bias"), vec![c.n_v_heads() as u64]);
    assert_eq!(shape(&f, "blk.0.ssm_norm.weight"), vec![c.head_v_dim() as u64]);
    assert_eq!(shape(&f, "blk.0.ssm_out.weight"), vec![c.value_dim() as u64, n]);

    // The attention layer's fused query-and-gate projection.
    assert_eq!(shape(&f, "blk.3.attn_q.weight"), vec![n, c.q_gate_dim() as u64]);
    assert_eq!(shape(&f, "blk.3.attn_k.weight"), vec![n, c.kv_dim() as u64]);
    assert_eq!(shape(&f, "blk.3.attn_v.weight"), vec![n, c.kv_dim() as u64]);

    // A recurrent layer has no attn_q, and an attention layer has no ssm_out.
    assert!(f.tensor("blk.0.attn_q.weight").is_none());
    assert!(f.tensor("blk.3.ssm_out.weight").is_none());
}

/// The 35B is `qwen35moe`, so `Config::from_gguf` refuses it by name rather
/// than silently misreading it. Its recurrent dimensions are identical, which
/// is what makes the 9B a valid stepping stone.
#[test]
#[ignore = "loads the real 35B; run with --release -- --ignored"]
fn the_35b_is_refused_by_name_but_shares_the_dimensions() {
    let Some(path) = common::find_model_named("Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf") else {
        println!("SKIPPED: no 35B found");
        return;
    };
    let f = GgufFile::open(&path).expect("open model");
    let err = Config::from_gguf(&f).expect_err("qwen35moe is not qwen35");
    assert!(format!("{err}").contains("qwen35moe"), "{err}");

    // Same conv and state geometry as the 9B, different width.
    assert_eq!(shape(&f, "blk.0.ssm_conv1d.weight"), vec![4, 8192]);
    assert_eq!(shape(&f, "blk.0.ssm_norm.weight"), vec![128]);
    assert_eq!(shape(&f, "blk.0.attn_qkv.weight"), vec![2048, 8192]);
    assert!(f.tensor("blk.0.ffn_gate_exps.weight").is_some(), "MoE");
}
