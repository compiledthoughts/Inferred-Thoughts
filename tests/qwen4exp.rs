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
    assert_eq!(text, LLAMA_CPP_40);
}

/// `llama-completion -p 'According to all known laws' -n 40 --temp 0 --top-k 1` on
/// the CPU-only build at 3057bb66c, 15-09-2026, from the 0.2B NVFP4-expert copy.
const LLAMA_CPP_40: &str = " of aviation, there is no way a bee should be able to fly. Its wings are \
                            too small to get its fat little body off the ground. The bee, of course, \
                            flies anyway because bees";

/// Prefill `prompt[..split]`, checkpoint, continue with the rest; then return to
/// the checkpoint, wander off down `detour`, return again, and replay the rest.
/// Returns the logits of the first continuation and of the replay.
///
/// The detour is the point: without it a restore that forgot PLE's state would
/// still find the window and conv history where the first continuation left
/// them — wrong, but by coincidence only as wrong as a refusal. After a detour
/// they hold another history entirely.
fn replay_through_a_checkpoint<O: inferred_thoughts::ops::Ops>(
    e: &mut inferred_thoughts::Engine<'_, O>,
    prompt: &[u32],
    split: usize,
    detour: &[u32],
) -> (Vec<f32>, Vec<f32>) {
    e.prefill(&prompt[..split]).expect("prefill the prefix");
    let cp = e.checkpoint().expect("qwen4exp has recurrent state, so a checkpoint must exist");
    assert_eq!(cp.pos(), split, "a checkpoint stands at the position it was taken");
    let first = e.prefill(&prompt[split..]).expect("continue");
    e.restore(&cp).expect("restore for the detour");
    e.prefill(detour).expect("the detour");
    e.restore(&cp).expect("restore for the replay");
    assert_eq!(e.pos(), split, "restoring returns the engine to the checkpoint's position");
    let second = e.prefill(&prompt[split..]).expect("replay");
    (first, second)
}

fn assert_same_bits(a: &[f32], b: &[f32], what: &str) {
    let differing = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    assert_eq!(differing, 0, "{what}: {differing} of {} logits differ", a.len());
}

/// **A checkpoint carries PLE's state: `serve` can return to one mid-conversation**
/// (`src/model/qwen4exp.md`, PLE). On the `naive` oracle: the continuation replayed
/// after a restore — past a detour — is bit-identical to the first, and a prefill
/// split around the checkpoint equals one shot, since chunking is exact.
///
/// Before `PleSnapshot`, every restore here was refused ("PLE history across a
/// rewind"), which is what limited `serve` to a single turn.
#[test]
#[ignore = "needs the 0.2B test model's NVFP4-expert GGUF; run with -- --ignored"]
fn the_0_2b_restores_a_checkpoint_with_its_ple_state_on_naive() {
    use inferred_thoughts::{Engine, Naive};
    let Some(f) = open(TINY) else { return };
    let tk = inferred_thoughts::Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let prompt = tk.encode("According to all known laws of aviation, there is no way a bee should be able", true, true);
    let detour = tk.encode(" The quick brown fox jumps over", false, false);
    let split = prompt.len() / 2;
    assert!(split >= 4 && !detour.is_empty(), "prompt and detour too short to test anything");

    let one_shot = {
        let mut e = Engine::new(Model::load(&f).expect("load"), Naive, 64, false);
        e.prefill(&prompt).expect("one-shot prefill")
    };
    let mut e = Engine::new(Model::load(&f).expect("load"), Naive, 64, false);
    let (first, second) = replay_through_a_checkpoint(&mut e, &prompt, split, &detour);
    assert_same_bits(&first, &one_shot, "a prefill split at the checkpoint against one shot");
    assert_same_bits(&second, &first, "the continuation replayed after a restore");
}

/// **The same on CUDA**, where PLE's conv history lives on the device: the replay
/// equals the first continuation only if the checkpoint read that history back
/// (`Ops::read_state`) and the restore made the device reload it (`forget_state`).
/// Either one missing restores into the detour's history, or a stale one.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an sm_120 device and the 0.2B test model's NVFP4-expert GGUF"]
fn the_0_2b_restores_a_checkpoint_with_its_ple_state_on_the_gpu() {
    use inferred_thoughts::{Cuda, Engine};
    let Some(f) = open(TINY) else { return };
    let tk = inferred_thoughts::Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let prompt = tk.encode("According to all known laws of aviation, there is no way a bee should be able", true, true);
    let detour = tk.encode(" The quick brown fox jumps over", false, false);
    let split = prompt.len() / 2;

    let gpu = Cuda::new(0).expect("cuda device");
    let mut e = Engine::new(Model::load(&f).expect("load"), &gpu, 64, false);
    let (first, second) = replay_through_a_checkpoint(&mut e, &prompt, split, &detour);
    if let Some(err) = gpu.take_error() {
        panic!("a CUDA op reported an error: {err}");
    }
    assert_same_bits(&second, &first, "the continuation replayed after a restore, on the GPU");
}

/// **Past the budget the indexer really does cut cells** — and below it, nothing
/// changes. Prefills the 0.2B just past 2,051 cells and compares the logits with
/// the same run forced dense (`INFERRED_QSA_DENSE=1`): they must differ, or the
/// sparse path is not engaging and every other QSA check would pass vacuously.
///
/// Run on its own process because the switch is read from the environment.
#[test]
#[ignore = "needs the 0.2B test model's NVFP4-expert GGUF; run with -- --ignored"]
fn qsa_selection_changes_the_answer_past_the_budget() {
    use inferred_thoughts::{Engine, Naive};
    let Some(f) = open(TINY) else { return };
    // 2,060 tokens: past 2,051, so the last few queries select; short enough to
    // prefill on the oracle in a few seconds.
    let tokens: Vec<u32> = (0..2060u32).map(|i| 1000 + (i * 7919) % 50_000).collect();
    let run = || {
        let mut e = Engine::new(Model::load(&f).expect("load"), Naive, tokens.len() + 4, false);
        e.prefill(&tokens).expect("prefill")
    };

    let sparse = run();
    // SAFETY: single-threaded test, and the variable is read inside `run` below.
    unsafe { std::env::set_var("INFERRED_QSA_DENSE", "1") };
    let dense = run();
    unsafe { std::env::remove_var("INFERRED_QSA_DENSE") };

    let differing = sparse.iter().zip(&dense).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert!(
        differing > 0,
        "sparse and dense gave identical logits at {} cells, so the selection never engaged",
        tokens.len()
    );
    println!("  {differing} of {} logits differ between the selection and dense", sparse.len());
}

/// Greedy, ties to the lower id.
#[cfg(feature = "cuda")]
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// The exact kernels on the GPU, the rest on the CPU: the bisection instrument of
/// step 4's part 2, and of QSA Q2's.
#[cfg(feature = "cuda")]
mod exact_only {
    use inferred_thoughts::ops::{Attn, Delta, Experts, Ops, QsaPool, QsaSelect, Route, Weights};
    use inferred_thoughts::{Cuda, Naive};

    /// The exact kernels on the GPU, the rest on the CPU. A bisection instrument,
    /// not a backend: every GPU result is pulled home for the next CPU op, and
    /// every CPU write is announced so a later GPU op re-uploads it.
    ///
    /// The second field is a host staging buffer for `moe_glu`, reserved once so its
    /// address — which the device keys its mirror on — never changes.
    pub struct ExactOnly<'a>(pub &'a Cuda, pub std::cell::RefCell<Vec<f32>>);
    impl Ops for ExactOnly<'_> {
        // ---- on the GPU, exact
        fn rms_norm(&self, x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
            self.0.rms_norm(x, w, eps, out);
            self.0.host_needs(out);
        }
        fn matmul(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) {
            self.0.matmul(w, x, out);
            self.0.host_needs(out);
        }
        fn matmul_pair(&self, a: &Weights<'_>, b: &Weights<'_>, x: &[f32], oa: &mut [f32], ob: &mut [f32]) {
            self.0.matmul_pair(a, b, x, oa, ob);
            self.0.host_needs(oa);
            self.0.host_needs(ob);
        }
        fn matmul_experts(&self, w: &Experts<'_>, route: &Route, x: &[f32], out: &mut [f32]) {
            self.0.matmul_experts(w, route, x, out);
            self.0.host_needs(out);
        }
        fn rms_norm_heads(&self, x: &mut [f32], w: &[f32], head_dim: usize, eps: f32) {
            self.0.rms_norm_heads(x, w, head_dim, eps);
            self.0.host_needs(x);
        }
        fn l2_norm_heads(&self, x: &mut [f32], head_dim: usize, eps: f32) {
            self.0.l2_norm_heads(x, head_dim, eps);
            self.0.host_needs(x);
        }
        fn rope_neox(&self, x: &mut [f32], pos: usize, hd: usize, n_rot: usize, nh: usize, theta: f32) {
            self.0.rope_neox(x, pos, hd, n_rot, nh, theta);
            self.0.host_needs(x);
        }
        fn add_assign(&self, a: &mut [f32], b: &[f32]) {
            self.0.add_assign(a, b);
            self.0.host_needs(a);
        }
        fn scale(&self, buf: &mut [f32], s: f32) {
            self.0.scale(buf, s);
            self.0.host_needs(buf);
        }
        fn gather_chunks(&self, src: &[f32], chunk: usize, stride: usize, offset: usize, out: &mut [f32]) {
            self.0.gather_chunks(src, chunk, stride, offset, out);
            self.0.host_needs(out);
        }
        fn scatter_chunks(&self, src: &[f32], chunk: usize, stride: usize, offset: usize, dst: &mut [f32]) {
            self.0.scatter_chunks(src, chunk, stride, offset, dst);
            self.0.host_needs(dst);
        }
        fn mul_rows(&self, x: &mut [f32], w: &[f32]) {
            self.0.mul_rows(x, w);
            self.0.host_needs(x);
        }
        fn mul_streams(&self, out: &mut [f32], h: &[f32], w: &[f32], n_stream: usize) {
            self.0.mul_streams(out, h, w, n_stream);
            self.0.host_needs(out);
        }
        fn row_dot(&self, a: &[f32], b: &[f32], width: usize, out: &mut [f32]) {
            self.0.row_dot(a, b, width, out);
            self.0.host_needs(out);
        }
        fn dilated_conv(
            &self,
            state: &mut [f32],
            x: &[f32],
            weight: &[f32],
            kernel: usize,
            dilation: usize,
            out: &mut [f32],
        ) {
            self.0.dilated_conv(state, x, weight, kernel, dilation, out);
            self.0.host_needs(out);
        }

        // ---- on the CPU: expf, or outside the exact set
        fn softmax(&self, x: &mut [f32], row: usize) {
            Naive.softmax(x, row);
            self.0.host_wrote(x);
        }
        fn silu_mul(&self, gate: &mut [f32], up: &[f32]) {
            Naive.silu_mul(gate, up);
            self.0.host_wrote(gate);
        }
        fn sigmoid_mul(&self, x: &mut [f32], g: &[f32]) {
            Naive.sigmoid_mul(x, g);
            self.0.host_wrote(x);
        }
        fn silu(&self, x: &mut [f32]) {
            Naive.silu(x);
            self.0.host_wrote(x);
        }
        fn sigmoid(&self, x: &mut [f32]) {
            Naive.sigmoid(x);
            self.0.host_wrote(x);
        }
        fn signed_sqrt_sigmoid(&self, s: &mut [f32]) {
            Naive.signed_sqrt_sigmoid(s);
            self.0.host_wrote(s);
        }
        #[allow(clippy::too_many_arguments)]
        fn moe_finish(
            &self,
            out: &mut [f32],
            at: usize,
            n: usize,
            rows: &[f32],
            route: &Route,
            shared: &[f32],
            logit: &[f32],
            logit_at: usize,
        ) {
            Naive.moe_finish(out, at, n, rows, route, shared, logit, logit_at);
            self.0.host_wrote(out);
        }
        fn ssm_conv(&self, state: &mut [f32], x: &[f32], weight: &[f32], kernel: usize, out: &mut [f32]) {
            Naive.ssm_conv(state, x, weight, kernel, out);
            self.0.host_wrote(out);
        }
        fn delta_rule(&self, d: &Delta<'_>, state: &mut [f32], out: &mut [f32]) {
            Naive.delta_rule(d, state, out);
            self.0.host_wrote(out);
        }
        fn kv_write(&self, slab: &mut [u16], offset: usize, src: &[f32]) {
            Naive.kv_write(slab, offset, src);
        }
        fn attend(&self, a: &Attn<'_>, out: &mut [f32]) {
            Naive.attend(a, out);
            self.0.host_wrote(out);
        }
        /// The gated half split at its SiLU: gate and up as GPU matmuls, the SiLU on
        /// the CPU. CUDA's NVFP4 `matmul_experts` takes only one row per (token,
        /// pick) — gate and up normally go through its fused `moe_glu` — so each
        /// token's row is repeated per pick here. Every output is still the same
        /// weight row against the same activation values.
        fn moe_glu(
            &self,
            gate: &Experts<'_>,
            up: &Experts<'_>,
            route: &Route,
            x: &[f32],
            out: &mut [f32],
            scratch: &mut [f32],
        ) {
            let (n_used, n_in) = (route.n_used(), gate.n_in);
            let mut rep = self.1.borrow_mut();
            rep.clear();
            for p in 0..(x.len() / n_in) * n_used {
                let r = p / n_used;
                rep.extend_from_slice(&x[r * n_in..(r + 1) * n_in]);
            }
            assert!(rep.capacity() == 1 << 20, "the staging buffer moved");
            self.0.host_wrote(&rep);
            self.matmul_experts(gate, route, &rep, out);
            self.matmul_experts(up, route, &rep, scratch);
            self.silu_mul(out, scratch);
        }
        // `route` keeps its trait default: chosen on the host from the CPU softmax.


        // ---- QSA: pooling and selection on the GPU, where they are exact; the
        // lanes and the cells stay there. Attention on the CPU, over the device's
        // cells read back (`Cuda::read_cells`, the debug readback).
        fn qsa_pool(&self, raw: &[u16], pooled: &mut [f32], p: &QsaPool<'_>) {
            self.0.qsa_pool(raw, pooled, p);
        }
        fn qsa_select(&self, q: &[f32], pooled: &[f32], sel: &QsaSelect, scores: &mut [f32], cells: &mut [u32]) {
            self.0.qsa_select(q, pooled, sel, scores, cells);
        }
        fn attend_sparse(&self, a: &Attn<'_>, cells: &[u32], sel: &QsaSelect, out: &mut [f32]) {
            let device = self.0.read_cells(cells).expect("the device's selection");
            Naive.attend_sparse(a, &device, sel, out);
            self.0.host_wrote(out);
        }

        // ---- residency, forwarded
        fn host_wrote(&self, buf: &[f32]) {
            self.0.host_wrote(buf)
        }
        fn host_needs(&self, buf: &mut [f32]) {
            self.0.host_needs(buf)
        }
        fn begin_pass(&self, n_tokens: usize) {
            self.0.begin_pass(n_tokens)
        }
        fn end_pass(&self) {
            self.0.end_pass()
        }
        fn forget_state(&self) {
            self.0.forget_state()
        }
    }
}

/// **Step 4, part 2: with the `expf` ops on the CPU, the GPU reproduces the oracle
/// bit for bit through the whole 0.2B** (`src/model/qwen4exp.md`, step 4).
///
/// Everything that is exact on the device runs there — every matmul (Q8_0, the F32
/// router, the NVFP4 experts against a Q8_0 activation), the serial RMSNorm, RoPE,
/// the gathers and adds, and qwen4exp's `mul_rows`, `mul_streams`, `row_dot` and
/// `dilated_conv` with its history held on the device. Everything that calls
/// `expf`, or is otherwise outside the exact set, runs on `Naive`: softmax and
/// routing, every SiLU and sigmoid (fused or not), PLE's gate, the GDN conv and
/// delta rule, attention and its KV writes.
///
/// Prefill and 16 decode passes, each pass's logits compared to the bit. A failure
/// here is a kernel or residency defect, not rounding; if this passes and the
/// full-GPU test fails, the cause is `expf` and FP4 activations.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an sm_120 device and the 0.2B test model's NVFP4-expert GGUF"]
fn the_0_2b_on_the_gpu_is_bit_identical_with_the_expf_ops_on_the_cpu() {
    use exact_only::ExactOnly;
    use inferred_thoughts::{Cuda, Engine, Naive};

    let Some(f) = open(TINY) else { return };
    let tk = inferred_thoughts::Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let tokens = tk.encode("According to all known laws", true, true);
    let steps = 16;
    let n_ctx = tokens.len() + steps + 4;

    // The oracle, and the tokens both runs are fed.
    let (fed, cpu) = {
        let mut e = Engine::new(Model::load(&f).expect("load"), Naive, n_ctx, false);
        let mut all = vec![e.prefill(&tokens).expect("cpu prefill")];
        let mut fed = Vec::new();
        for _ in 0..steps {
            let t = argmax(all.last().expect("logits"));
            fed.push(t);
            all.push(e.decode(t).expect("cpu decode"));
        }
        (fed, all)
    };

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    gpu.rms_serial(true);
    gpu.nvfp4_fp4(false);
    let mixed = {
        let mut e = Engine::new(
            Model::load(&f).expect("load"),
            ExactOnly(&gpu, std::cell::RefCell::new(Vec::with_capacity(1 << 20))),
            n_ctx,
            false,
        );
        let mut all = vec![e.prefill(&tokens).expect("mixed prefill")];
        for &t in &fed {
            all.push(e.decode(t).expect("mixed decode"));
        }
        all
    };
    if let Some(err) = gpu.take_error() {
        panic!("a CUDA op reported an error: {err}");
    }

    for (pass, (a, b)) in cpu.iter().zip(&mixed).enumerate() {
        let differing = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        let worst = a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        println!("  pass {pass:>2}: {differing} of {} logits differ, worst {worst:e}", a.len());
        assert_eq!(
            differing, 0,
            "pass {pass}: the exact kernels are not exact through the whole model; worst {worst:e}"
        );
    }
}

/// **Step 4, part 3: the 0.2B on the GPU at its defaults reproduces llama.cpp's 40
/// greedy tokens word for word** — FP4 x FP4 NVFP4 experts, tensor-core attention,
/// tree RMSNorm, device routing and CUDA graphs on decode. The departures from the
/// oracle are real (`expf`, FP4 activations); the text agreeing across 40 tokens is
/// the evidence they stay below what changes a greedy choice here.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an sm_120 device and the 0.2B test model's NVFP4-expert GGUF"]
fn the_0_2b_test_model_reproduces_llama_cpps_greedy_text_on_the_gpu() {
    let Some(f) = open(TINY) else { return };
    let tk = inferred_thoughts::Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let tokens = tk.encode("According to all known laws", true, true);
    let gpu = inferred_thoughts::Cuda::new(0).expect("cuda device");
    let mut e = inferred_thoughts::Engine::new(Model::load(&f).expect("load"), &gpu, 64, false);
    let (produced, _) = e.generate(&tokens, 40, None, |_| {}).expect("generate");
    if let Some(err) = gpu.take_error() {
        panic!("a CUDA op reported an error: {err}");
    }
    println!("  graphs off for a mid-pass read: {}", gpu.graphs_off_for_mid_pass_read());
    let text = tk.decode(&produced, false).expect("decode");
    assert_eq!(text, LLAMA_CPP_40);
}

/// Token ids for a synthetic prompt: deterministic, spread over the vocabulary.
fn synthetic(n: usize, seed: u32) -> Vec<u32> {
    (0..n as u32).map(|i| 1000 + (i * 7919 + seed * 104_729) % 50_000).collect()
}

/// **QSA Q2, part 2: past the budget, with the `expf` ops and attention on the
/// CPU, the GPU reproduces the oracle bit for bit** (`src/model/qwen4exp.md`,
/// "QSA on CUDA").
///
/// Step 4's instrument, with QSA's pooling and selection on the device and the
/// attention on `Naive` over the cells the device chose (read back). 2,060 prompt
/// tokens — the last prefill chunk selects — then 16 decode passes, each of which
/// selects, every logit compared to the bit. Since pooling, scores and selection
/// are exact, this holds only if the device keeps the oracle's cells in every
/// query of every pass, through the pooled lanes and their watermark.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an sm_120 device and the 0.2B test model's NVFP4-expert GGUF"]
fn the_0_2b_past_the_budget_is_bit_identical_with_the_expf_ops_on_the_cpu() {
    use exact_only::ExactOnly;
    use inferred_thoughts::{Cuda, Engine, Naive};

    let Some(f) = open(TINY) else { return };
    let tokens = synthetic(2060, 0);
    let steps = 16;
    let n_ctx = tokens.len() + steps + 4;

    let (fed, cpu) = {
        let mut e = Engine::new(Model::load(&f).expect("load"), Naive, n_ctx, false);
        let mut all = vec![e.prefill(&tokens).expect("cpu prefill")];
        let mut fed = Vec::new();
        for _ in 0..steps {
            let t = argmax(all.last().expect("logits"));
            fed.push(t);
            all.push(e.decode(t).expect("cpu decode"));
        }
        (fed, all)
    };

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    gpu.rms_serial(true);
    gpu.nvfp4_fp4(false);
    let mixed = {
        let mut e = Engine::new(
            Model::load(&f).expect("load"),
            ExactOnly(&gpu, std::cell::RefCell::new(Vec::with_capacity(1 << 20))),
            n_ctx,
            false,
        );
        let mut all = vec![e.prefill(&tokens).expect("mixed prefill")];
        for &t in &fed {
            all.push(e.decode(t).expect("mixed decode"));
        }
        all
    };
    if let Some(err) = gpu.take_error() {
        panic!("a CUDA op reported an error: {err}");
    }
    for (pass, (a, b)) in cpu.iter().zip(&mixed).enumerate() {
        let differing = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        println!("  pass {pass:>2}: {differing} of {} logits differ", a.len());
        assert_eq!(differing, 0, "pass {pass}: past the budget the device's selection is not the oracle's");
    }
}

/// **The pooled-key watermark survives a checkpoint round trip on the GPU**, past
/// the budget: prefill to 2,100, checkpoint, continue to 2,400 (pooling blocks up
/// to 600), return and take a 300-token detour, return again and replay.
///
/// **The detour is the pass that can go wrong.** It rewrites cells 2,100 on, so
/// blocks 525 on must be pooled again from its keys; a watermark left at 600
/// would select against the first continuation's pooled keys. So the detour is
/// compared with the same detour taken by a fresh engine that never saw the
/// continuation. The replay alone cannot tell: stale keys from the first
/// continuation are the right ones for its replay (checked: with the watermark
/// never lowered, the replay still matched and the detour did not).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an sm_120 device and the 0.2B test model's NVFP4-expert GGUF"]
fn the_0_2b_restores_a_checkpoint_past_the_budget_on_the_gpu() {
    use inferred_thoughts::{Cuda, Engine};
    let Some(f) = open(TINY) else { return };
    let (prompt, detour, split) = (synthetic(2400, 0), synthetic(300, 7), 2100);
    let gpu = Cuda::new(0).expect("cuda device");
    let mut e = Engine::new(Model::load(&f).expect("load"), &gpu, 2464, false);
    e.prefill(&prompt[..split]).expect("prefix");
    let cp = e.checkpoint().expect("a checkpoint");
    let first = e.prefill(&prompt[split..]).expect("continue");
    e.restore(&cp).expect("restore for the detour");
    let detour_after = e.prefill(&detour).expect("the detour");
    e.restore(&cp).expect("restore for the replay");
    let second = e.prefill(&prompt[split..]).expect("replay");

    // Held alongside `e`, so no buffer the device keys on is recycled.
    let mut fresh = Engine::new(Model::load(&f).expect("load"), &gpu, 2464, false);
    fresh.prefill(&prompt[..split]).expect("fresh prefix");
    let detour_fresh = fresh.prefill(&detour).expect("fresh detour");
    if let Some(err) = gpu.take_error() {
        panic!("a CUDA op reported an error: {err}");
    }
    assert_same_bits(&detour_after, &detour_fresh, "a detour after a restore against a fresh one, past the budget");
    assert_same_bits(&second, &first, "the continuation replayed after a restore, past the budget");
}

/// **Decode selects at every depth once QSA is reachable**, and below the budget
/// the selection keeps every cell: with a 4,096-cell context, the 0.2B's decode
/// runs the sparse path — gather, then attention over the window — from its first
/// token, under CUDA graphs, and must still print llama.cpp's 40 tokens.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an sm_120 device and the 0.2B test model's NVFP4-expert GGUF"]
fn the_0_2b_reproduces_llama_cpps_greedy_text_through_the_sparse_path_on_the_gpu() {
    let Some(f) = open(TINY) else { return };
    let tk = inferred_thoughts::Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let tokens = tk.encode("According to all known laws", true, true);
    let gpu = inferred_thoughts::Cuda::new(0).expect("cuda device");
    let mut e = inferred_thoughts::Engine::new(Model::load(&f).expect("load"), &gpu, 4096, false);
    let (produced, _) = e.generate(&tokens, 40, None, |_| {}).expect("generate");
    if let Some(err) = gpu.take_error() {
        panic!("a CUDA op reported an error: {err}");
    }
    assert!(!gpu.graphs_off_for_mid_pass_read(), "the sparse path must not read the device mid-pass");
    let text = tk.decode(&produced, false).expect("decode");
    assert_eq!(text, LLAMA_CPP_40);
}

/// **`serve`'s sequence, past the budget, is exact on the GPU**: in one engine, a
/// 2,301-token first turn prefilled in 1,024-token slices with a checkpoint after
/// each (as `serve` does at `--ctx 8192`), then 16 decode steps; a 25-token
/// continuation and 16 more; a restore to the checkpoint at 2,048 and a different
/// 428-token tail, 16 steps; a reset and a short unrelated turn; then the restored
/// conversation again from zero, in the same slices. The restored turn's logits
/// must equal the fresh one's in every step.
///
/// The 125B's server gave different replies for these two (16-09); this separates
/// the engine's own restore path from anything only the oversubscribed model has.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs an sm_120 device and the 0.2B test model's NVFP4-expert GGUF"]
fn the_0_2b_serve_sequence_past_the_budget_restores_exactly_on_the_gpu() {
    use inferred_thoughts::{Cuda, Engine};
    let Some(f) = open(TINY) else { return };
    let gpu = Cuda::new(0).expect("cuda device");
    let mut e = Engine::new(Model::load(&f).expect("load"), &gpu, 8192, false);
    let slice = 1024;
    let prefill_in_slices = |e: &mut Engine<'_, &Cuda>, toks: &[u32], cps: &mut Vec<_>| {
        let mut last = Vec::new();
        for chunk in toks.chunks(slice) {
            last = e.prefill(chunk).expect("slice");
            if chunk.len() == slice {
                cps.push(e.checkpoint().expect("checkpoint"));
            }
        }
        last
    };
    let decode_steps = |e: &mut Engine<'_, &Cuda>, first: Vec<f32>, n: usize| {
        let mut all = vec![first];
        for _ in 0..n {
            let t = argmax(all.last().expect("logits"));
            all.push(e.decode(t).expect("decode"));
        }
        all
    };

    let turn1 = synthetic(2301, 0);
    let mut cps = Vec::new();
    let l = prefill_in_slices(&mut e, &turn1, &mut cps);
    decode_steps(&mut e, l, 16);
    let l = e.prefill(&synthetic(25, 3)).expect("continuation");
    decode_steps(&mut e, l, 16);

    // The edited turn: turn 1's first 2,048 tokens, then a different tail.
    let mut edited = turn1[..2048].to_vec();
    edited.extend(synthetic(428, 5));
    let cp = cps.iter().rev().find(|c| c.pos() <= 2048).expect("a checkpoint at or before 2,048").clone();
    assert_eq!(cp.pos(), 2048);
    e.restore(&cp).expect("restore");
    let l = e.prefill(&edited[2048..]).expect("restored tail");
    let restored = decode_steps(&mut e, l, 16);

    e.reset();
    let l = e.prefill(&synthetic(14, 9)).expect("elsewhere");
    decode_steps(&mut e, l, 4);

    e.reset();
    let mut ignored = Vec::new();
    let l = prefill_in_slices(&mut e, &edited, &mut ignored);
    let fresh = decode_steps(&mut e, l, 16);
    if let Some(err) = gpu.take_error() {
        panic!("a CUDA op reported an error: {err}");
    }
    for (step, (a, b)) in restored.iter().zip(&fresh).enumerate() {
        let differing = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        println!("  step {step:>2}: {differing} of {} logits differ", a.len());
        assert_eq!(differing, 0, "step {step}: the restored turn is not the fresh one");
    }
}
