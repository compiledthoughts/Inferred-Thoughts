//! `qwen4exp` loads from the real files, with every setting the file holds.
//!
//! Step 1's gate (`src/model/qwen4exp.md`, "Status"): both GGUFs load, each
//! setting equals what `inferred inspect` and llama.cpp's `gguf_dump.py` print for
//! the file, and every tensor is either mapped or unread on purpose. The literals
//! below are copied from those dumps (15-09-2026), not from a model card.

use inferred_thoughts::model::qwen4exp::Config;
use inferred_thoughts::{GgufFile, Model};

mod common;

const TINY: &str = "Qwen3.8-Flash-Next-0.2B-A0.2B-NVFP4exp.gguf";
const FULL: &str = "Qwen3.8-Flash-Next-NVFP4-Q8_0.gguf";

fn open(name: &str) -> Option<GgufFile> {
    let Some(path) = common::find_model_named(name) else {
        println!("SKIPPED: no {name} found; set INFERRED_MODEL_DIR");
        return None;
    };
    Some(GgufFile::open(&path).expect("open the GGUF"))
}

/// The settings both files share: everything that is not a size.
fn assert_shared(c: &Config) {
    assert_eq!(c.head_dim, 256);
    assert_eq!(c.n_head_kv, 2);
    assert_eq!(c.n_rot, 64);
    assert_eq!(c.rope_sections, [11, 11, 10, 0]);
    assert_eq!(c.rope_theta, 1.0e7);
    assert_eq!(c.n_vocab, 248_320);
    assert_eq!((c.ssm_d_conv, c.ssm_d_state, c.ssm_dt_rank, c.ssm_n_group), (4, 128, 48, 16));
    assert_eq!(c.ssm_d_inner, 6144);
    assert_eq!((c.hc.n_stream, c.hc.low_rank), (4, 320));
    assert_eq!((c.indexer.n_head, c.indexer.head_dim, c.indexer.top_k), (4, 128, 2048));
    for il in 0..c.n_layer {
        let qsa = (il + 1) % 4 == 0;
        assert_eq!(c.recurrent[il], !qsa, "layer {il}");
        assert_eq!(c.compress_ratios[il], if qsa { 4 } else { 0 }, "layer {il}");
    }
    let p = c.ple.as_ref().expect("both files carry a PLE layer");
    assert_eq!((p.layer, p.ngram_size, p.heads_per_ngram, p.conv_kernel), (1, 3, 8, 4));
    assert_eq!(p.n_heads(), 16);
    assert_eq!((p.eos_token_id, p.image_token_id), (248_044, Some(248_056)));
    // Exact: these reach 2.4e13, past any float, and a rounded one is a different hash.
    assert_eq!(p.layer_multipliers, vec![23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]);
}

#[test]
#[ignore = "needs the 0.2B test model's NVFP4-expert GGUF; run with -- --ignored"]
fn the_0_2b_test_model_loads_with_every_setting_from_the_file() {
    let Some(f) = open(TINY) else { return };
    let m = Model::load(&f).expect("load the 0.2B test model");
    let Model::Qwen4Exp(q) = &m else { panic!("loaded as {}, not qwen4exp", m.arch()) };
    let c = &q.cfg;

    assert_shared(c);
    assert_eq!((c.n_layer, c.n_embd, c.n_head), (4, 256, 8));
    assert_eq!(
        (c.moe.n_expert, c.moe.n_expert_used, c.moe.expert_ff, c.moe.shared_ff),
        (8, 4, 256, 256)
    );
    let p = c.ple.as_ref().expect("PLE");
    assert_eq!(p.head_dim, 16);
    assert_eq!(&p.head_offsets[..4], &[0, 2053, 4116, 6185]);
    assert_eq!(&p.head_vocab_sizes[..4], &[2053, 2063, 2069, 2081]);
    assert_eq!(m.n_kv_layer(), 1);

    // No NVFP4 activation scales in this file: llama-quantize writes none.
    let (on_purpose, unknown) = q.unmapped(&f);
    assert!(unknown.is_empty(), "tensors the loader does not know: {unknown:?}");
    assert!(on_purpose.is_empty(), "unexpected unread tensors: {on_purpose:?}");
    assert_eq!(q.n_mapped(), f.tensors.len());
    println!("0.2B: {} tensors mapped; {c:?}", q.n_mapped());
}

#[test]
#[ignore = "needs the 125B GGUF (119 GiB, mapped not read); run with -- --ignored"]
fn the_125b_loads_with_every_setting_from_the_file() {
    let Some(f) = open(FULL) else { return };
    let m = Model::load(&f).expect("load the 125B");
    let Model::Qwen4Exp(q) = &m else { panic!("loaded as {}, not qwen4exp", m.arch()) };
    let c = &q.cfg;

    assert_shared(c);
    assert_eq!((c.n_layer, c.n_embd, c.n_head), (48, 2560, 24));
    assert_eq!(
        (c.moe.n_expert, c.moe.n_expert_used, c.moe.expert_ff, c.moe.shared_ff),
        (512, 10, 640, 640)
    );
    let p = c.ple.as_ref().expect("PLE");
    assert_eq!(p.head_dim, 160);
    assert_eq!(&p.head_offsets[..4], &[0, 20_000_003, 40_000_026, 60_000_059]);
    assert_eq!(&p.head_vocab_sizes[..4], &[20_000_003, 20_000_023, 20_000_033, 20_000_047]);
    assert!(p.min_rows() <= 320_001_536, "the table has 320,001,536 rows");
    assert_eq!(m.n_kv_layer(), 12);

    // The NVFP4 expert tensors carry per-expert `.input_scale`s, 3 per layer, which
    // no path reads; everything else is mapped.
    let (on_purpose, unknown) = q.unmapped(&f);
    assert!(unknown.is_empty(), "tensors the loader does not know: {unknown:?}");
    assert_eq!(on_purpose.len(), 48 * 3, "input_scale tensors: {on_purpose:?}");
    assert_eq!(q.n_mapped() + on_purpose.len(), f.tensors.len());
    println!(
        "125B: {} of {} tensors mapped, {} input_scale unread; {:.1} MiB of weights a decode token",
        q.n_mapped(),
        f.tensors.len(),
        on_purpose.len(),
        m.weight_bytes_per_pass() as f64 / 1048576.0
    );
}

/// **The whole forward pass against llama.cpp, end to end** (`src/model/qwen4exp.md`,
/// step 3): the 0.2B test model's NVFP4-expert copy, greedy on the `naive` oracle,
/// reproduces exactly the 40 tokens llama.cpp's CPU build generated from the same
/// file and prompt — its memorized text. Every block kind runs: GDN, attention,
/// hyper-connections, MoE with the shared expert, and PLE, whose n-gram window and
/// conv history carry across the 39 decode passes.
///
/// The per-tensor comparison is `scripts/compare_eval_callback.py` against
/// `llama-eval-callback`; this is the check a regression cannot pass by accident.
#[test]
#[ignore = "needs the 0.2B test model's NVFP4-expert GGUF; run with -- --ignored"]
fn the_0_2b_test_model_reproduces_llama_cpps_greedy_text_on_naive() {
    let Some(f) = open(TINY) else { return };
    let tk = inferred_thoughts::Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let m = Model::load(&f).expect("load");
    let tokens = tk.encode("According to all known laws", true, true);
    assert_eq!(tokens, vec![10865, 310, 660, 3750, 6657], "llama.cpp's tokens for the prompt");
    let mut e = inferred_thoughts::Engine::new(m, inferred_thoughts::Naive, 64, false);
    let (produced, _) = e.generate(&tokens, 40, None, |_| {}).expect("generate");
    let text = tk.decode(&produced, false).expect("decode");
    // `llama-completion -p 'According to all known laws' -n 40 --temp 0 --top-k 1`
    // on the CPU-only build at 3057bb66c, 15-09-2026.
    let want = " of aviation, there is no way a bee should be able to fly. Its wings are too \
                small to get its fat little body off the ground. The bee, of course, flies \
                anyway because bees";
    assert_eq!(text, want);
}
