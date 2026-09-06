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

/// **Batched prefill must evolve the recurrent state exactly as one token at a
/// time does.** The acceptance criterion for batching `qwen35`.
///
/// `qwen3`'s equivalent lives in `tests/kv_cache.rs`, and for that architecture
/// a batch only had to get the KV cache and the causal mask right. Here there
/// is a second, harder thing to get right: 24 of the 32 layers are
/// GatedDeltaNet, and their state is a running matrix that token `t` updates
/// for token `t+1`. Batching those layers means iterating the scan *inside* the
/// seam instead of outside it, and an off-by-one in that loop -- a token's
/// alpha read from the wrong row, a state advanced twice, a conv window taken
/// before rather than after the update -- would still produce fluent-looking
/// text. This is what makes such a slip a failure instead of a mystery.
///
/// Bit-identical, not a tolerance: the batched path runs the same arithmetic in
/// the same order on the same values, so anything else is a bug rather than
/// drift.
#[test]
#[ignore = "loads the real 9B; run with --release -- --ignored"]
fn batched_prefill_equals_token_by_token() {
    use inferred_thoughts::{Engine, Model, Naive, Tokenizer};

    let Some(path) = common::find_model_named("Qwen3.5-9B-Q8_0.gguf") else {
        println!("SKIPPED: no Qwen3.5-9B-Q8_0.gguf found");
        return;
    };
    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let tokens = tk.encode("The capital of France is Paris, and the capital of", true, true);
    assert!(tokens.len() >= 6, "prompt too short to exercise the scan");
    let n_ctx = tokens.len() + 4;

    // The whole prompt in one pass.
    let batched = {
        let m = Model::load(&f).expect("load model");
        let mut e = Engine::new(m, Naive, n_ctx, false);
        e.prefill(&tokens).expect("prefill")
    };

    // The same prompt one token at a time. A recurrent architecture cannot
    // rewind, so this needs its own engine rather than a reset mid-run.
    let stepwise = {
        let m = Model::load(&f).expect("load model");
        let mut e = Engine::new(m, Naive, n_ctx, false);
        let mut logits = e.prefill(&tokens[..1]).expect("prefill first token");
        for (i, &tok) in tokens.iter().enumerate().skip(1) {
            logits = e.decode(tok).expect("decode");
            assert_eq!(e.pos(), i + 1, "cache length tracks absolute position");
        }
        logits
    };

    assert_eq!(batched.len(), stepwise.len(), "same vocabulary");
    let differing = batched
        .iter()
        .zip(&stepwise)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing, 0,
        "{differing} of {} logits differ between a batched prefill and the same \
         tokens one at a time; the batch is not reproducing the scan",
        batched.len()
    );
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
fn the_35b_config_reads_and_the_moe_half_appears() {
    let Some(path) = common::find_model_named("Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf") else {
        println!("SKIPPED: no 35B found");
        return;
    };
    let f = GgufFile::open(&path).expect("open model");
    let c = Config::from_gguf(&f).expect("qwen35moe config");

    // The dimensions that differ from the 9B, read from the file rather than
    // from notes -- an earlier version of CLAUDE.md had several of these wrong.
    assert_eq!(c.n_layer, 40, "40 blocks; the 41st is MTP and lives elsewhere");
    assert_eq!(c.n_embd, 2048);
    assert_eq!(c.n_head, 16);
    assert_eq!(c.n_head_kv, 2);
    assert_eq!(c.head_dim, 256);
    assert_eq!(c.n_vocab, 248320);
    assert_eq!(c.full_attention_interval, 4);

    // Same conv and state geometry as the 9B, different width.
    assert_eq!(shape(&f, "blk.0.ssm_conv1d.weight"), vec![4, 8192]);
    assert_eq!(shape(&f, "blk.0.ssm_norm.weight"), vec![128]);
    assert_eq!(shape(&f, "blk.0.attn_qkv.weight"), vec![2048, 8192]);

    let m = c.moe.expect("qwen35moe must carry the expert config");
    assert_eq!((m.n_expert, m.n_expert_used), (256, 8));
    assert_eq!((m.expert_ff, m.shared_ff), (512, 512));

    // `n_ff` describes one expert on the routed variant; there is no dense
    // feed_forward_length key in the file at all.
    assert_eq!(c.n_ff, 512);

    // 30 GatedDeltaNet, 10 attention, at i % 4 == 3.
    let attn = (0..c.n_layer).filter(|&i| !c.is_recurrent(i)).count();
    assert_eq!(attn, 10, "10 of 40 blocks attend");
    assert!(c.is_recurrent(0) && !c.is_recurrent(3), "attention at i % 4 == 3");
}

/// The whole 35B stack loads, and the expert tensors address correctly.
///
/// **The expert tensors are the one genuinely new shape**, and the two
/// orientations are transposes of each other -- `ffn_gate_exps` is
/// `{2048, 512, 256}` while `ffn_down_exps` is `{512, 2048, 256}`. Swapping
/// them would produce plausible numbers rather than an error, so the check is
/// that each expert's borrowed sub-range lands exactly where the file says.
#[test]
#[ignore = "loads the real 35B; run with --release -- --ignored"]
fn the_35b_loads_and_experts_address_contiguously() {
    use inferred_thoughts::Model;

    let Some(path) = common::find_model_named("Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf") else {
        println!("SKIPPED: no 35B found");
        return;
    };
    let f = GgufFile::open(&path).expect("open model");
    let m = Model::load(&f).expect("the 35B must load");
    assert_eq!(m.arch(), "qwen35moe");
    assert_eq!(m.n_layer(), 40);
    assert_eq!(m.n_kv_layer(), 10, "only the attention blocks need a KV slab");

    // Expert addressing, checked against the file's own byte counts.
    for (name, n_in, n_out) in [
        ("blk.0.ffn_gate_exps.weight", 2048usize, 512usize),
        ("blk.0.ffn_up_exps.weight", 2048, 512),
        ("blk.0.ffn_down_exps.weight", 512, 2048),
    ] {
        let info = f.tensor(name).unwrap_or_else(|| panic!("{name} missing"));
        assert_eq!(info.dims, vec![n_in as u64, n_out as u64, 256]);
        let total = f.tensor_bytes(info).len();
        assert_eq!(total % 256, 0, "{name} must divide into 256 experts");
        // IQ4_XS is 4.25 bpw, so one expert is n_out rows of n_in elements.
        // block_iq4_xs in ggml-common.h is
        //   { ggml_half d; uint16_t scales_h; uint8_t scales_l[QK_K/64];
        //     uint8_t qs[QK_K/2]; }
        // = 2 + 2 + 4 + 128 = 136 bytes per QK_K = 256 elements.
        assert_eq!(
            total / 256,
            n_out * (n_in / 256) * 136,
            "{name}: per-expert byte count"
        );
    }

    // A token reads 8 of 256 routed experts plus the shared one, so the
    // per-pass figure must be far below the 17.51 GiB the file occupies.
    let per_pass = m.weight_bytes_per_pass() as f64 / (1u64 << 30) as f64;
    assert!(
        (1.0..4.0).contains(&per_pass),
        "a token should read ~2 GiB, not {per_pass:.2} GiB -- storage counted as traffic?"
    );
    println!("  35B reads {per_pass:.2} GiB per token");
}

/// The 35B produces coherent text — the acceptance test for the MoE path.
///
/// **Deliberately a text assertion rather than a numeric one.** The routing
/// rule has four places where a plausible misreading still yields fluent
/// output: a softmax of the top 8 instead of the top 8 of the softmax, an
/// unnormalized weight vector, a missing shared expert, or the shared expert
/// added without its sigmoid gate. Each of those degrades the model rather than
/// breaking it, so "it ran" proves nothing and only the content does.
///
/// `llama-eval-callback` remains the sharp instrument for localizing a numeric
/// bug; this is the cheap standing check that the whole path is assembled.
#[test]
#[ignore = "loads the real 35B; run with --release -- --ignored"]
fn the_35b_generates_coherent_text() {
    use inferred_thoughts::{Engine, Model, Spin, Tokenizer};

    let Some(path) = common::find_model_named("Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf") else {
        println!("SKIPPED: no 35B found");
        return;
    };
    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let m = Model::load(&f).expect("load the 35B");
    let mut e = Engine::new(m, Spin::new(8), 128, false);

    let prompt = tk.encode("The capital of France is", true, true);
    let (out, _) = e.generate(&prompt, 6, None, |_| {}).expect("generate");
    let text = tk.decode(&out, false).expect("decode");
    println!("  35B says: {text:?}");

    // Routing that is wrong but self-consistent still produces English, so the
    // check is for the fact the model actually knows.
    assert!(
        text.to_lowercase().contains("paris"),
        "the 35B answered {text:?}; a routing error degrades fluency last and \
         factual recall first, so suspect the top-k, the weight normalization, \
         or the shared expert's gate"
    );
}

/// **The 35B's routed FFN, batched against one token at a time — bit for bit.**
///
/// This is the instrument the MoE path did not have, and its absence is a
/// specific gap rather than a general one. `moe_token` runs *inside* a batched
/// pass: routing differs per token, so the FFN is a per-token loop lifting one
/// row out of the batch and putting one back. `CLAUDE.md` names that as "the
/// one place the batch convention does not reach", and every existing 35B test
/// runs at `n_tok = 1`, where the distinction cannot appear:
///
/// * `the_35b_generates_coherent_text` decodes six tokens and greps for
///   "paris". A token that read *another* token's experts would still answer
///   Paris, because at `n_tok = 1` there is no other token.
/// * `device_topk_reproduces_the_host_selection` now covers a batch, but only
///   the selection. Nothing checks that the chosen experts are then applied to
///   the row they were chosen for.
///
/// That is the exact shape of defect the warp attention kernel shipped with
/// last session: correct at op level, wrong in the model, invisible to both
/// existing harnesses. Writing this *before* the batched MoE lands means it
/// passes trivially today, which is the point — a test written afterwards
/// proves much less, because a green light then cannot distinguish "the change
/// is right" from "the test cannot see the change".
///
/// Bit-identical rather than a tolerance: batching redistributes work without
/// altering any accumulation order, so anything else is a bug. `Spin` rather
/// than `Naive` because a 35B forward pass on one scalar thread is not a test
/// anyone will run, and the differential tests already pin `Spin` to `Naive`.
#[test]
#[ignore = "loads the real 35B; run with --release -- --ignored"]
fn batched_moe_prefill_equals_token_by_token() {
    use inferred_thoughts::{Engine, Model, Spin, Tokenizer};

    let Some(path) = common::find_model_named("Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf") else {
        println!("SKIPPED: no 35B found");
        return;
    };
    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let tokens = tk.encode("The capital of France is Paris, and the capital of", true, true);
    // Enough tokens that a row-indexing slip lands somewhere other than row 0.
    assert!(tokens.len() >= 6, "prompt too short to exercise the batch");
    let n_ctx = tokens.len() + 4;

    let batched = {
        let m = Model::load(&f).expect("load the 35B");
        let mut e = Engine::new(m, Spin::new(8), n_ctx, false);
        e.prefill(&tokens).expect("batched prefill")
    };

    // Its own engine: 30 of 40 layers are GatedDeltaNet and recurrent state
    // cannot be rewound, so a reset mid-run would not be the same experiment.
    let stepwise = {
        let m = Model::load(&f).expect("load the 35B");
        let mut e = Engine::new(m, Spin::new(8), n_ctx, false);
        let mut logits = e.prefill(&tokens[..1]).expect("prefill first token");
        for &tok in tokens.iter().skip(1) {
            logits = e.decode(tok).expect("decode");
        }
        logits
    };

    assert_eq!(batched.len(), stepwise.len(), "same vocabulary");
    let differing = batched
        .iter()
        .zip(&stepwise)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing, 0,
        "{differing} of {} logits differ between a batched prefill and the same tokens \
         one at a time. The routed FFN is the only part of this model that indexes a \
         row explicitly — suspect `moe_token`'s gather of `s.normed` at `at`, its \
         scatter into `s.ffn_out`, or a route resolved for one token and applied to \
         another.",
        batched.len()
    );
}
