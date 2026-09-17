//! Per-op differential test: the CUDA backend against the `naive` oracle.
//!
//! This runs *before* the whole-model comparison for a reason. A wrong token
//! deep in a generation tells you the GPU disagrees; it does not tell you which
//! of eight kernels is responsible. Each op here is driven with synthetic data
//! and compared on its own, so a failure names the kernel.
//!
//! Two classes of result are expected, and the test encodes the difference:
//!
//! * **Bit-identical** — `matmul`, `rope_neox`, `add_assign`. Integer and f32
//!   arithmetic in the oracle's order, with `--fmad=false` preventing
//!   contraction. Anything less is a bug.
//! * **Close** — `softmax`, `silu_mul`, `attend`. These call `expf`, and CUDA's
//!   is not obliged to match glibc's to the last bit. For `softmax` and
//!   `silu_mul` the tolerance is a few ulp: nothing else differs, so anything
//!   larger means the accumulation order is wrong rather than the library.
//!
//!   `rms_norm` and `rms_norm_heads` reduce the sum of squares as a tree,
//!   which f64 addition makes order-dependent. `rms_tolerance` derives what
//!   that is worth, and `rms_serial_restores_bit_equality` shows the flag
//!   buying the exactness back.
//!
//!   `attend` is the other op that reorders. It is flash-decoding, which
//!   accumulates per chunk of the KV sequence and combines, so it cannot
//!   reproduce a single serial pass. Its tolerance is **derived from the
//!   decomposition** — see `attend_tolerance` — rather than set to whatever
//!   passes, which is what `CLAUDE.md` asks for when a tolerance is
//!   unavoidable.
//!
//! Each op is bracketed by `begin_pass` and `host_needs`, which is the
//! residency contract the seam now carries: a device backend leaves its result
//! on the card and the caller says when it wants it on the host. Without the
//! `host_needs` these comparisons read an untouched buffer — which is exactly
//! how this test caught the first version of that change.
//!
//! ```text
//! cargo test --release --features cuda --test cuda_ops -- --ignored \
//!   --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` is not optional. Every test here builds its own `Cuda`
//! against the same device, and run concurrently they fault intermittently with
//! `CUDA_ERROR_ILLEGAL_ADDRESS` — which is sticky, so the first test to fault
//! takes the others down with it and the panic names an innocent line. This
//! predates the diagnostics below (it reproduces on a clean checkout, roughly
//! one run in two) and is recorded here rather than fixed because the cause is
//! in how these tests share a context, not in any kernel. `CLAUDE.md` already
//! documents the serial invocation; this header did not.

#![cfg(feature = "cuda")]

use inferred_thoughts::ops::{Attn, Delta, Experts, Ops, Weights};
use inferred_thoughts::quant::half::f32_to_f16;
use inferred_thoughts::{Cuda, Engine, GgufFile, Naive, Qwen3, Tokenizer};

/// **This card's tensor-core ceilings, measured rather than quoted.**
///
/// The memory side has had a number since the beginning — 448 GB/s, and every
/// bandwidth claim in `BENCHMARKS.md` is a fraction of it. The compute side had
/// none, so "this kernel is slow" was never quantified against anything, and a
/// whole session went into traffic experiments on kernels that turned out to be
/// using ~1% of the arithmetic.
///
/// Both figures come from back-to-back `mma` instructions on eight independent
/// accumulators, no memory in the loop, 1728 warps to fill all 48 slots on each
/// of the 36 SMs. `BENCHMARKS.md`'s 09-09 (ceilings) entry has the method.
///
/// For scale: the FP32 CUDA cores are 24.0 TFLOP/s, so int8 tensor cores are
/// **8x** the scalar path. That ratio is why the one architectural change that
/// ever paid here was moving IQ4_XS onto `mma`.
const MMA_S8_PEAK_TOPS: f64 = 197.4;
const MMA_F16_PEAK_TFLOPS: f64 = 50.7;

mod common;

/// Deterministic values in [-1, 1). A fixed sequence so a failure is
/// reproducible and a regression is comparable to the run that found it.
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / 8388608.0 - 1.0
        })
        .collect()
}

/// Pack rows of f32 into Q8_0 exactly as the file stores them: per 32 values,
/// an f16 scale then 32 signed bytes.
fn q8_0(values: &[f32], n_in: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for row in values.chunks_exact(n_in) {
        for block in row.chunks_exact(32) {
            let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
            for &v in block {
                out.push(((v * id).round() as i8) as u8);
            }
        }
    }
    out
}

/// Largest absolute difference, and how many elements differ at all.
fn compare(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut worst = 0.0f32;
    let mut differing = 0;
    for (x, y) in a.iter().zip(b) {
        if x.to_bits() != y.to_bits() {
            differing += 1;
            worst = worst.max((x - y).abs());
        }
    }
    (worst, differing)
}

fn exact(name: &str, cpu: &[f32], gpu: &[f32]) {
    let (worst, differing) = compare(cpu, gpu);
    println!("  {name:<16} {differing:>6} of {:<6} differ   worst {worst:e}", cpu.len());
    assert_eq!(
        differing, 0,
        "{name} must be bit-identical to the oracle; {differing} of {} elements differ, \
         worst {worst:e}. This op is pure f32/integer arithmetic in a fixed order, so a \
         difference is a defect, not rounding.",
        cpu.len()
    );
}

fn close(name: &str, cpu: &[f32], gpu: &[f32], tol: f32) {
    let (worst, differing) = compare(cpu, gpu);
    println!("  {name:<16} {differing:>6} of {:<6} differ   worst {worst:e}", cpu.len());
    assert!(
        worst <= tol,
        "{name} differs by {worst:e}, over the {tol:e} allowed for an expf disagreement. \
         That is too large to be the library; suspect the accumulation order."
    );
}

/// The error a tree reduction is allowed in `rms_norm`, derived from where it
/// actually lands — which is not where you would first look.
///
/// The reorder moves the f64 sum by at most `n * 2^-53` relative: ~1.1e-13 at
/// n = 1024, and 1.5e-16 measured. **That is not what this covers.** `mean` is
/// then cast to f32, whose values are spaced 2^-23 = 1.19e-7 apart, about a
/// million times coarser. Two sums differing by 1e-13 round to the *same* f32
/// unless a rounding boundary falls between them — and when they do, `scale` is
/// identical bits and so is every output element.
///
/// So the difference is bimodal: exactly zero almost always, and about one f32
/// ulp of `scale` when a boundary is straddled, which is roughly once in 1e9
/// calls at the measured 1.5e-16. The tolerance has to cover the second case,
/// so it is set by the **f32 cast**, not by the reorder — and is therefore ~1e6
/// times larger than the reorder alone would suggest. Expect `0 of n differ` in
/// practice; the allowance is for the rare call that straddles.
fn rms_tolerance(reference: &[f32]) -> f32 {
    let magnitude = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    f32::EPSILON * magnitude.max(1.0)
}

/// The error flash-decoding is allowed, derived from how it decomposes.
///
/// The kernel splits the KV sequence into chunks of 128. Within a chunk the max
/// is a tree, which is exact because max never rounds, and the sum is a tree,
/// so ~log2(128) = 7 roundings. The combine across chunks is serial in one
/// thread, so ~`n_split` roundings. Add a few for the rescaling and `expf`.
///
/// Output is a convex combination of the values, so the absolute error scales
/// with their magnitude rather than with anything larger.
///
/// This is a *bound*, not a fitted constant: the measured error sits well
/// under it, and it is written this way so that a real regression — an
/// indexing slip, a missed rescale — exceeds it rather than hiding beneath a
/// number chosen after the fact.
fn attend_tolerance(n_pos: usize, reference: &[f32]) -> f32 {
    let n_split = n_pos.div_ceil(128) as f32;
    let roundings = 7.0 + n_split + 4.0;
    let magnitude = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    roundings * f32::EPSILON * magnitude.max(1.0)
}

#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn every_op_agrees_with_the_oracle() {
    let gpu = Cuda::new(0).expect("cuda device");
    // One op at a time, read straight back. A graph defers the whole pass to
    // `end_pass`, so it cannot serve this shape.
    gpu.use_graphs(false);
    println!("device {} sm_{}{}", gpu.name(), gpu.capability().0, gpu.capability().1);

    let n = 1024usize;
    let eps = 1e-6f32;

    // --- rms_norm ------------------------------------------------------
    {
        let x = noise(n, 1);
        let w = noise(n, 2);
        let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
        Naive.rms_norm(&x, &w, eps, &mut a);
        gpu.begin_pass(1);
        gpu.rms_norm(&x, &w, eps, &mut b);
        gpu.host_needs(&mut b);
        close("rms_norm", &a, &b, rms_tolerance(&a));
    }

    // --- rms_norm_heads ------------------------------------------------
    {
        let head_dim = 128;
        let mut a = noise(n, 3);
        let mut b = a.clone();
        let w = noise(head_dim, 4);
        Naive.rms_norm_heads(&mut a, &w, head_dim, eps);
        gpu.begin_pass(1);
        gpu.rms_norm_heads(&mut b, &w, head_dim, eps);
        gpu.host_needs(&mut b);
        close("rms_norm_heads", &a, &b, rms_tolerance(&a));
    }

    // --- matmul, Q8_0 --------------------------------------------------
    {
        let (n_in, n_out) = (1024usize, 512usize);
        let raw = noise(n_in * n_out, 5);
        let packed = q8_0(&raw, n_in);
        let w = Weights {
            data: &packed,
            ty: inferred_thoughts::gguf::GgmlType::Q8_0,
            n_in,
            n_out,
            pooled: false,
        };
        let x = noise(n_in, 6);
        let (mut a, mut b) = (vec![0.0; n_out], vec![0.0; n_out]);
        Naive.matmul(&w, &x, &mut a);
        gpu.begin_pass(1);
        gpu.matmul(&w, &x, &mut b);
        gpu.host_needs(&mut b);
        exact("matmul_q8_0", &a, &b);
    }

    // --- rope_neox -----------------------------------------------------
    {
        let (head_dim, n_heads) = (128usize, 8usize);
        let mut a = noise(head_dim * n_heads, 7);
        let mut b = a.clone();
        Naive.rope_neox(&mut a, 37, head_dim, head_dim, n_heads, 1.0e6);
        gpu.begin_pass(1);
        gpu.rope_neox(&mut b, 37, head_dim, head_dim, n_heads, 1.0e6);
        gpu.host_needs(&mut b);
        exact("rope_neox", &a, &b);
    }

    // --- add_assign ----------------------------------------------------
    {
        let mut a = noise(n, 8);
        let mut b = a.clone();
        let other = noise(n, 9);
        Naive.add_assign(&mut a, &other);
        gpu.begin_pass(1);
        gpu.add_assign(&mut b, &other);
        gpu.host_needs(&mut b);
        exact("add_assign", &a, &b);
    }

    // --- softmax -------------------------------------------------------
    {
        let mut a = noise(384, 10);
        let mut b = a.clone();
        let n = a.len();
        Naive.softmax(&mut a, n);
        gpu.begin_pass(1);
        let n = b.len();
        gpu.softmax(&mut b, n);
        gpu.host_needs(&mut b);
        close("softmax", &a, &b, 1e-7);
    }

    // --- silu_mul ------------------------------------------------------
    {
        let mut a = noise(n, 11);
        let mut b = a.clone();
        let up = noise(n, 12);
        Naive.silu_mul(&mut a, &up);
        gpu.begin_pass(1);
        gpu.silu_mul(&mut b, &up);
        gpu.host_needs(&mut b);
        close("silu_mul", &a, &b, 1e-6);
    }

    // --- attend --------------------------------------------------------
    {
        let (head_dim, n_head, n_head_kv, n_pos) = (128usize, 16usize, 8usize, 96usize);
        let kv_dim = n_head_kv * head_dim;
        let q = noise(n_head * head_dim, 13);
        let kf = noise(n_pos * kv_dim, 14);
        let vf = noise(n_pos * kv_dim, 15);
        let k: Vec<u16> = kf.iter().map(|&v| f32_to_f16(v)).collect();
        let v: Vec<u16> = vf.iter().map(|&v| f32_to_f16(v)).collect();

        let attn = Attn {
            q: &q,
            k: &k,
            v: &v,
            kv_dim,
            n_pos,
            head_dim,
            n_head,
            n_head_kv,
            scale: 1.0 / (head_dim as f32).sqrt(),
        };
        let (mut a, mut b) = (vec![0.0; n_head * head_dim], vec![0.0; n_head * head_dim]);
        Naive.attend(&attn, &mut a);
        gpu.begin_pass(1);
        gpu.attend(&attn, &mut b);
        gpu.host_needs(&mut b);
        close("attend", &a, &b, attend_tolerance(n_pos, &a));
    }

    assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");
}

/// The error an elementwise op that calls `expf` is allowed, derived.
///
/// Everything but `expf` is the oracle's arithmetic in the oracle's order, so the
/// only freedom is exp's own rounding: one f32 ulp, relative. A sigmoid `s` moves
/// by `s(1-s)` per unit of relative change in `exp(-x)`, at most a quarter, so one
/// ulp there is at most `EPSILON/4` of output; SiLU multiplies that by `|x|`. The
/// final division rounds once more, one ulp of the output. So the gap is under
/// `(|x|/4 + 1) * EPSILON` — under `2 * EPSILON * max(1, |x|)` — and this allows
/// twice that. An indexing or ordering slip is orders of magnitude larger.
fn expf_tolerance(input: &[f32]) -> f32 {
    let magnitude = input.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    4.0 * f32::EPSILON * magnitude.max(1.0)
}

/// **qwen4exp's hyper-connection and PLE ops on the GPU, against their scalar
/// trait defaults** (`src/model/qwen4exp.md`, step 4).
///
/// Four are exact and must be bit-identical: `mul_rows`, `mul_streams`, `row_dot`
/// (an f64 serial sum, as `ggml_sum_rows`) and `dilated_conv`, output **and** the
/// history it leaves behind. Three call `expf` and get [`expf_tolerance`]:
/// `silu`, `sigmoid`, `signed_sqrt_sigmoid`.
///
/// Shapes are the 0.2B's (`n_embd` 256, four streams, `hc_dim` 1024) and the
/// 125B's widths where a kernel loops over one (`row_dot` at 2560). The conv runs
/// at PLE's kernel 4 and dilation 3 — a 9-sample history — at batch sizes on both
/// sides of it, then carries its state across a second pass, which is what decode
/// does. A second geometry (kernel 3, dilation 2) keeps the indexing honest.
///
/// Every weight is built once and held for the whole test: `resident` and
/// `state_resident` key on host addresses, and a recycled address would hand a
/// case the previous one's device copy.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_qwen4exp_ops_agree_with_the_oracle() {
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);

    let (nd, n_stream, n_tok) = (256usize, 4usize, 3usize);
    let hc_dim = nd * n_stream;

    // --- mul_rows ------------------------------------------------------
    let w_rows = noise(hc_dim, 0x401);
    {
        let mut a = noise(n_tok * hc_dim, 0x402);
        let mut b = a.clone();
        Naive.mul_rows(&mut a, &w_rows);
        gpu.begin_pass(n_tok);
        gpu.mul_rows(&mut b, &w_rows);
        gpu.host_needs(&mut b);
        gpu.end_pass();
        exact("mul_rows", &a, &b);
    }

    // --- silu, sigmoid -------------------------------------------------
    // Wide inputs, so both tails and the middle of the curve are exercised.
    {
        let x: Vec<f32> = noise(4096, 0x403).iter().map(|v| v * 12.0).collect();
        let (mut a, mut b) = (x.clone(), x.clone());
        Naive.silu(&mut a);
        gpu.begin_pass(1);
        gpu.silu(&mut b);
        gpu.host_needs(&mut b);
        gpu.end_pass();
        close("silu", &a, &b, expf_tolerance(&x));

        let (mut a, mut b) = (x.clone(), x.clone());
        Naive.sigmoid(&mut a);
        gpu.begin_pass(1);
        gpu.sigmoid(&mut b);
        gpu.host_needs(&mut b);
        gpu.end_pass();
        close("sigmoid", &a, &b, expf_tolerance(&[1.0]));
    }

    // --- mul_streams ---------------------------------------------------
    {
        let h = noise(n_tok * nd, 0x404);
        let w = noise(n_tok * n_stream, 0x405);
        let (mut a, mut b) = (vec![0.0f32; n_tok * hc_dim], vec![0.0f32; n_tok * hc_dim]);
        Naive.mul_streams(&mut a, &h, &w, n_stream);
        gpu.begin_pass(n_tok);
        gpu.mul_streams(&mut b, &h, &w, n_stream);
        gpu.host_needs(&mut b);
        gpu.end_pass();
        exact("mul_streams", &a, &b);
    }

    // --- row_dot -------------------------------------------------------
    for width in [nd, 2560] {
        let rows = n_tok * n_stream;
        let x = noise(rows * width, 0x406 + width as u64);
        let y = noise(rows * width, 0x407 + width as u64);
        let (mut a, mut b) = (vec![0.0f32; rows], vec![0.0f32; rows]);
        Naive.row_dot(&x, &y, width, &mut a);
        gpu.begin_pass(n_tok);
        gpu.row_dot(&x, &y, width, &mut b);
        gpu.host_needs(&mut b);
        gpu.end_pass();
        exact(&format!("row_dot {width}"), &a, &b);
    }

    // --- signed_sqrt_sigmoid -------------------------------------------
    // Zeros of both signs, values under the 1e-6 clamp, and the range PLE's
    // scaled dot products reach.
    {
        let mut s: Vec<f32> = noise(4096, 0x408).iter().map(|v| v * 40.0).collect();
        s[..6].copy_from_slice(&[0.0, -0.0, 1e-9, -1e-9, 1e-6, -1e-6]);
        let (mut a, mut b) = (s.clone(), s.clone());
        Naive.signed_sqrt_sigmoid(&mut a);
        gpu.begin_pass(1);
        gpu.signed_sqrt_sigmoid(&mut b);
        gpu.host_needs(&mut b);
        gpu.end_pass();
        close("signed_sqrt_sig", &a, &b, expf_tolerance(&[1.0]));
    }

    // --- dilated_conv --------------------------------------------------
    let geometries = [(4usize, 3usize), (3, 2)];
    let weights: Vec<Vec<f32>> =
        geometries.iter().map(|&(k, d)| noise(hc_dim * k, 0x409 + (k * 10 + d) as u64)).collect();
    for (g, &(kernel, dilation)) in geometries.iter().enumerate() {
        let w = &weights[g];
        let hist = (kernel - 1) * dilation;
        for n in [1usize, 2, 5, hist, hist + 1, 17] {
            let seed = noise(hc_dim * hist, 0x40a + n as u64);
            let x1 = noise(n * hc_dim, 0x40b + n as u64);
            let x2 = noise(hc_dim, 0x40c + n as u64);

            // The oracle: this pass, then one decode step carrying the history.
            let mut s_cpu = seed.clone();
            let (mut o1_cpu, mut o2_cpu) = (vec![0.0f32; n * hc_dim], vec![0.0f32; hc_dim]);
            Naive.dilated_conv(&mut s_cpu, &x1, w, kernel, dilation, &mut o1_cpu);
            Naive.dilated_conv(&mut s_cpu, &x2, w, kernel, dilation, &mut o2_cpu);

            let mut s_gpu = seed.clone();
            let (mut o1_gpu, mut o2_gpu) = (vec![0.0f32; n * hc_dim], vec![0.0f32; hc_dim]);
            gpu.forget_state();
            gpu.begin_pass(n);
            gpu.dilated_conv(&mut s_gpu, &x1, w, kernel, dilation, &mut o1_gpu);
            gpu.host_needs(&mut o1_gpu);
            gpu.end_pass();
            gpu.begin_pass(1);
            gpu.dilated_conv(&mut s_gpu, &x2, w, kernel, dilation, &mut o2_gpu);
            gpu.host_needs(&mut o2_gpu);
            gpu.end_pass();
            gpu.read_state_into(&mut s_gpu).expect("read conv history");
            assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

            let tag = format!("conv k{kernel} d{dilation} n{n}");
            exact(&format!("{tag} out"), &o1_cpu, &o1_gpu);
            exact(&format!("{tag} next"), &o2_cpu, &o2_gpu);
            exact(&format!("{tag} state"), &s_cpu, &s_gpu);
        }
    }

    assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");
}

/// The whole model, GPU against CPU.
///
/// Deliberately **not** a bit-equality test, and deliberately not a loose one
/// either. The per-op test above establishes that five kernels are exact and
/// three differ only by an `expf` ulp; this one measures what those ulps become
/// after 28 layers, where every matmul re-quantizes its activation to Q8_0 and
/// a difference near a quantization boundary is amplified to a whole step.
///
/// `CLAUDE.md` measures that step at **1.07e-3 of tensor magnitude**. So the
/// question this test answers is not "are they equal" — they cannot be — but
/// "is the disagreement the size the quantization floor predicts, or larger".
#[test]
#[ignore = "needs an sm_120 device and the real model"]
fn the_model_agrees_with_the_oracle_to_the_quantization_floor() {
    common::model_or_skip!(path);
    let gpu = Cuda::new(0).expect("cuda device");

    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let tokens = tk.encode("The capital of France is", true, true);

    // **A fixed continuation, not each run's own argmax.** Twelve steps, which
    // is past the point where the GPU replays the step as a CUDA graph, so that
    // transition gets checked too.
    //
    // Driving decode with `argmax(&l)` per run made this test unable to
    // distinguish drift from a bug, which is the exact reason `CLAUDE.md`
    // replaced Stage 5's acceptance criterion: "an argmax would flip somewhere
    // in 250 greedy decisions whether or not the cache is correct". Two runs
    // that differ inside the floor eventually pick different tokens, and from
    // then on they are processing different sequences and their logits are not
    // comparable at all.
    //
    // It cost a real result. The warp attention kernel — 3x at depth, agreeing
    // with the oracle across 72 op-level comparisons — was disabled for a day
    // because this test failed by "a whole argmax" (17689 against 5429). With
    // the sequence fixed it passes at 1.69e-2 of magnitude against this same
    // 9e-2 ceiling. The kernel was never wrong; the harness could not tell.
    let steps: Vec<u32> = tk.encode(" Paris, and the capital of Japan is Tokyo, and", false, true);
    assert!(steps.len() >= 8, "need enough steps to cross the graph transition");
    let n_ctx = tokens.len() + steps.len() + 4;

    let cpu_logits = {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, Naive, n_ctx, false);
        let mut l = e.prefill(&tokens).expect("prefill");
        for &t in &steps {
            l = e.decode(t).expect("decode");
        }
        l
    };
    let gpu_logits = {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, &gpu, n_ctx, false);
        let mut l = e.prefill(&tokens).expect("prefill");
        for &t in &steps {
            l = e.decode(t).expect("decode");
        }
        assert!(gpu.graph_active(), "the graph never engaged, so this tested the eager path");
        l
    };
    assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

    let (worst, differing) = compare(&cpu_logits, &gpu_logits);
    let magnitude = cpu_logits
        .iter()
        .fold(0.0f32, |m: f32, &v: &f32| m.max(v.abs()));
    let relative = worst / magnitude;

    println!(
        "  logits          {differing:>6} of {:<6} differ   worst {worst:e}  \
         ({relative:e} of magnitude {magnitude:.3})",
        cpu_logits.len()
    );
    println!(
        "  argmax          cpu {}  gpu {}",
        Qwen3::argmax(&cpu_logits),
        Qwen3::argmax(&gpu_logits)
    );

    // `CLAUDE.md` characterizes this exactly: a one-ulp input difference flips a
    // Q8_0 quant and moves a tensor by 1.07e-3 of magnitude, and "compounded
    // over 28 layers this reaches ~1e-2 typical, ~9e-2 worst, and ~1% on the
    // final logits". The ceiling is that documented worst case, not a number
    // chosen to make this pass — and `only_the_expf_ops_diverge` below is what
    // establishes the *cause* is the three exp-dependent kernels and nothing
    // else. Without that companion test this bound would be far too loose.
    let ceiling = 9.0e-2;
    assert!(
        relative <= ceiling,
        "GPU logits differ from the oracle by {relative:e} of magnitude, over the \
         {ceiling:e} that 28 layers of quantization amplification explain. Check \
         only_the_expf_ops_diverge first: if that still passes, the defect is in one \
         of softmax, silu_mul or attend."
    );
}

/// The decisive one: **with the three `expf` kernels run on the CPU, the GPU
/// reproduces the oracle bit for bit through the whole model.**
///
/// The per-op test shows five kernels are exact on synthetic data. This shows
/// they stay exact on the real thing — 28 layers, 151,936 logits, every bit —
/// which is what turns "the other three differ by an ulp" from an assumption
/// into the *only* remaining explanation for the full-GPU divergence above.
///
/// If this fails, there is a kernel defect. If it passes and the full-GPU test
/// fails, the cause is `expf` and the magnitude is quantization amplification.
#[test]
#[ignore = "needs an sm_120 device and the real model"]
fn only_the_expf_ops_diverge() {
    common::model_or_skip!(path);
    let gpu = Cuda::new(0).expect("cuda device");
    // This mixes CPU and GPU op by op, so every GPU result is read
    // immediately. A graph batches the pass and would defer them all.
    gpu.use_graphs(false);

    /// The five exact kernels on the GPU, the three exp-dependent ones on the
    /// CPU. Not a backend anyone should run — a bisection instrument.
    ///
    /// It is also the first thing in this project to straddle the two devices,
    /// and it has to honour the residency contract to do it: every GPU result
    /// is pulled home so the following CPU op can read it, and every CPU write
    /// is announced so a later GPU op does not trust a stale copy. That is the
    /// same bookkeeping a real CPU/GPU layer split will need, at a granularity
    /// no real split would choose.
    struct ExactOnly<'a>(&'a Cuda);
    impl Ops for ExactOnly<'_> {
        fn rms_norm(&self, x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
            self.0.rms_norm(x, w, eps, out);
            self.0.host_needs(out);
        }
        fn rms_norm_heads(&self, x: &mut [f32], w: &[f32], head_dim: usize, eps: f32) {
            self.0.rms_norm_heads(x, w, head_dim, eps);
            self.0.host_needs(x);
        }
        // GatedDeltaNet is not part of what this instrument bisects: it runs
        // qwen3, which has no recurrent layers. Forwarding to the oracle keeps
        // the impl total without pretending the GPU has these kernels.
        fn l2_norm_heads(&self, x: &mut [f32], head_dim: usize, eps: f32) {
            Naive.l2_norm_heads(x, head_dim, eps);
        }
        fn ssm_conv(
            &self,
            state: &mut [f32],
            x: &[f32],
            weight: &[f32],
            kernel: usize,
            out: &mut [f32],
        ) {
            Naive.ssm_conv(state, x, weight, kernel, out);
        }
        fn gather_chunks(
            &self,
            src: &[f32],
            chunk: usize,
            stride: usize,
            offset: usize,
            out: &mut [f32],
        ) {
            Naive.gather_chunks(src, chunk, stride, offset, out);
        }
        fn sigmoid_mul(&self, x: &mut [f32], g: &[f32]) {
            Naive.sigmoid_mul(x, g);
        }
        fn delta_rule(&self, d: &Delta<'_>, state: &mut [f32], out: &mut [f32]) {
            Naive.delta_rule(d, state, out);
        }
        fn matmul(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) {
            self.0.matmul(w, x, out);
            self.0.host_needs(out);
        }
        fn rope_neox(
            &self,
            x: &mut [f32],
            p: usize,
            hd: usize,
            n_rot: usize,
            nh: usize,
            theta: f32,
        ) {
            self.0.rope_neox(x, p, hd, n_rot, nh, theta);
            self.0.host_needs(x);
        }
        fn add_assign(&self, a: &mut [f32], b: &[f32]) {
            self.0.add_assign(a, b);
            self.0.host_needs(a);
        }

        fn softmax(&self, x: &mut [f32], row: usize) {
            Naive.softmax(x, row);
            self.0.host_wrote(x);
        }
        fn silu_mul(&self, gate: &mut [f32], up: &[f32]) {
            Naive.silu_mul(gate, up);
            self.0.host_wrote(gate);
        }
        fn attend(&self, a: &Attn<'_>, out: &mut [f32]) {
            Naive.attend(a, out);
            self.0.host_wrote(out);
        }

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
    }

    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let tokens = tk.encode("The capital of France is", true, true);

    // Prefill *and* sixteen decode steps. Prefill alone exercised only ~285
    // rms_norm calls, which is thin evidence for a kernel whose block
    // reduction is allowed to depart from the oracle's summation order — the
    // argument is that the f32 cast of `mean` absorbs the difference, and a
    // boundary case would show up as a rare flip rather than a systematic one.
    // More calls, and decode's changing activations, make that a real test.
    let steps = 16;

    let cpu = {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, Naive, tokens.len() + steps + 4, false);
        let mut all = e.prefill(&tokens).expect("prefill");
        for _ in 0..steps {
            all = e.decode(Qwen3::argmax(&all)).expect("decode");
        }
        all
    };
    let mixed = {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, ExactOnly(&gpu), tokens.len() + steps + 4, false);
        let mut all = e.prefill(&tokens).expect("prefill");
        for _ in 0..steps {
            all = e.decode(Qwen3::argmax(&all)).expect("decode");
        }
        all
    };
    assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

    let (worst, differing) = compare(&cpu, &mixed);
    println!(
        "  exact-only      {differing:>6} of {:<6} differ   worst {worst:e}",
        cpu.len()
    );
    assert_eq!(
        differing, 0,
        "the five kernels that claim bit-exactness are not exact through the whole \
         model: {differing} of {} logits differ, worst {worst:e}. The per-op test \
         passing means the defect needs real shapes or real data to show.",
        cpu.len()
    );
}

/// What one `Ops` call costs before any arithmetic happens.
///
/// The per-layer profile shows decode time tracking *op count* rather than
/// bytes moved — the FFN moves far more weight than the attention projections
/// and costs half as much. That points at fixed per-call overhead rather than
/// at any kernel, and this measures it: a trivial launch, then the same launch
/// wrapped in the upload/download the seam forces on every op.
#[test]
#[ignore = "needs an sm_120 device; a measurement, not an assertion"]
fn per_op_round_trip_cost() {
    use inferred_thoughts::ops::cuda::DeviceBuffer;
    use std::time::Instant;

    let gpu = Cuda::new(0).expect("cuda device");
    let n = 1024usize;
    let host = noise(n, 21);
    let x = DeviceBuffer::from_slice(&host).expect("alloc x");
    let y = DeviceBuffer::from_slice(&host).expect("alloc y");
    let mut back = vec![0.0f32; n];
    let reps = 2000;

    // Warm the module and the JIT before timing anything.
    gpu.saxpy(1.0, &x, &y, n).expect("warmup");

    let bench = |label: &str, f: &mut dyn FnMut()| {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
        println!("  {label:<22} {us:>7.1} us");
        us
    };

    let launch = bench("launch + sync", &mut || {
        gpu.saxpy(1.0, &x, &y, n).expect("saxpy");
    });
    let up = bench("4 KiB h2d", &mut || {
        x.write(&host).expect("h2d");
    });
    let down = bench("4 KiB d2h", &mut || {
        y.read(&mut back).expect("d2h");
    });
    let round_trip = bench("full round trip", &mut || {
        x.write(&host).expect("h2d");
        gpu.saxpy(1.0, &x, &y, n).expect("saxpy");
        y.read(&mut back).expect("d2h");
    });

    println!(
        "  -> launch {:.0}%, h2d {:.0}%, d2h {:.0}% of a round trip",
        100.0 * launch / round_trip,
        100.0 * up / round_trip,
        100.0 * down / round_trip
    );
    println!(
        "  a decode step runs ~478 ops, so the floor is {:.1} ms/token \
         against the CPU's 16.0",
        round_trip * 478.0 / 1000.0
    );
}

/// The three GatedDeltaNet primitives against the oracle.
///
/// Split from `every_op_agrees_with_the_oracle` because these carry state: the
/// conv window and the SSM matrix are read *and written*, so a failure can be
/// in what came back or in what was left behind, and the test has to check
/// both. A kernel that computes the right output and corrupts the state would
/// pass any check of the output alone, and would then look like drift.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_gdn_ops_agree_with_the_oracle() {
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);

    // The 9B's real shapes.
    let (hk, hv, nk, nv) = (128usize, 128, 16, 32);
    let (kdim, vdim) = (hk * nk, hv * nv);
    let eps = 1e-6f32;

    // --- l2_norm_heads --------------------------------------------------
    {
        let mut a = noise(kdim, 21);
        let mut b = a.clone();
        Naive.l2_norm_heads(&mut a, hk, eps);
        gpu.begin_pass(1);
        gpu.l2_norm_heads(&mut b, hk, eps);
        gpu.host_needs(&mut b);
        exact("l2_norm_heads", &a, &b);
    }

    // --- ssm_conv, output and the state it leaves ------------------------
    {
        let kernel = 4usize;
        let n = 1024usize;
        let x = noise(n, 22);
        let w = noise(n * kernel, 23);
        let mut sa = noise(n * (kernel - 1), 24);
        let mut sb = sa.clone();
        let (mut oa, mut ob) = (vec![0.0; n], vec![0.0; n]);

        Naive.ssm_conv(&mut sa, &x, &w, kernel, &mut oa);
        gpu.begin_pass(1);
        gpu.ssm_conv(&mut sb, &x, &w, kernel, &mut ob);
        gpu.host_needs(&mut ob);
        close("ssm_conv", &oa, &ob, 6.0 * f32::EPSILON);

        // The state is device-owned after the call, so read it back the same
        // way the model would have to.
        gpu.read_state_into(&mut sb).expect("read state back");
        exact("ssm_conv state", &sa, &sb);
    }

    // --- delta_rule ------------------------------------------------------
    {
        let q = noise(kdim, 25);
        let k = noise(kdim, 26);
        let v = noise(vdim, 27);
        let alpha = noise(nv, 28);
        let beta = noise(nv, 29);
        // ssm_a is -exp(A_log) upstream, so it is negative and the gate lands
        // inside (0, 1). A positive value here would make the state explode and
        // the test would pass on garbage.
        let ssm_a: Vec<f32> = noise(nv, 30).iter().map(|v| -v.abs()).collect();
        let dt = noise(nv, 31);

        let d = Delta {
            q: &q, k: &k, v: &v,
            alpha: &alpha, beta: &beta, ssm_a: &ssm_a, dt_bias: &dt,
            head_k_dim: hk, head_v_dim: hv, n_k_heads: nk, n_v_heads: nv,
        };
        let mut sa = noise(nv * hk * hv, 32);
        let mut sb = sa.clone();
        let (mut oa, mut ob) = (vec![0.0; vdim], vec![0.0; vdim]);

        Naive.delta_rule(&d, &mut sa, &mut oa);
        gpu.begin_pass(1);
        gpu.delta_rule(&d, &mut sb, &mut ob);
        gpu.host_needs(&mut ob);

        // Tolerance is the expf class, as for softmax and silu_mul: the gate
        // and beta go through expf and logf, which CUDA is not obliged to round
        // as glibc does. Everything after that is the oracle's own order --
        // one thread per value row, summing the key axis ascending -- so a
        // larger error means the decomposition is wrong, not the library.
        close("delta_rule", &oa, &ob, 40.0 * f32::EPSILON);
        gpu.read_state_into(&mut sb).expect("read state back");
        close("delta_rule state", &sa, &sb, 40.0 * f32::EPSILON);
    }
    assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");
}

/// `--rms-serial` buys back exactly what the tree gave up.
///
/// The point of keeping the serial kernel is that determinism is hard to
/// recover once it is gone, so the claim "the flag restores bit-equality with
/// the oracle" has to be tested rather than asserted in a comment. It also
/// pins the default: if the serial path were ever quietly made a tree too, the
/// first half of this fails.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn rms_serial_restores_bit_equality() {
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);

    let n = 1024usize;
    let head_dim = 128usize;
    let eps = 1e-6f32;
    let x = noise(n, 11);
    let w = noise(n, 12);
    let wh = noise(head_dim, 13);

    let mut want = vec![0.0; n];
    Naive.rms_norm(&x, &w, eps, &mut want);
    let mut want_heads = x.clone();
    Naive.rms_norm_heads(&mut want_heads, &wh, head_dim, eps);

    for serial in [true, false] {
        gpu.rms_serial(serial);
        let label = if serial { "serial" } else { "tree" };

        let mut got = vec![0.0; n];
        gpu.begin_pass(1);
        gpu.rms_norm(&x, &w, eps, &mut got);
        gpu.host_needs(&mut got);

        let mut got_heads = x.clone();
        gpu.begin_pass(1);
        gpu.rms_norm_heads(&mut got_heads, &wh, head_dim, eps);
        gpu.host_needs(&mut got_heads);

        if serial {
            exact(&format!("rms_norm {label}"), &want, &got);
            exact(&format!("rms_norm_heads {label}"), &want_heads, &got_heads);
        } else {
            close(&format!("rms_norm {label}"), &want, &got, rms_tolerance(&want));
            close(
                &format!("rms_norm_heads {label}"),
                &want_heads,
                &got_heads,
                rms_tolerance(&want_heads),
            );
        }
    }
    assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");
}

/// Where does `matmul_q8_0_warp` actually lose its bandwidth?
///
/// It is 66.5% of device time on the 9B and moves ~235 GB/s against the card's
/// 448. Two candidates call for opposite fixes, so they are separated rather
/// than guessed at — the same discipline `why_is_the_rms_reduction_slow` used,
/// which found that three plausible explanations were all wrong.
///
/// * `tree` keeps the loads and replaces the serial cross-block sum with a warp
///   reduction. Inexact, so it can never ship as-is; it prices the tail.
/// * `u16` keeps the tail and halves the load count. `34*b` from an aligned
///   base is even but never a multiple of four, so two bytes is the widest load
///   the on-disk layout allows without repacking.
///
/// Shapes are the 9B's, because the serial tail grows with `n_in`: 128 blocks
/// for the attention projections, 384 for `ffn_down`. The 0.6B's 32 would hide
/// it.
///
/// A measurement, not an assertion.
#[test]
#[ignore = "needs an sm_120 device; a measurement, not an assertion"]
fn where_does_the_matmul_lose_its_bandwidth() {
    let gpu = Cuda::new(0).expect("cuda device");
    let reps = 200;

    // (n_in, n_out, what it is in the 9B)
    let shapes = [
        (4096usize, 4096usize, "attn_out  4096x4096"),
        (4096, 12288, "ffn_up    4096x12288"),
        (12288, 4096, "ffn_down 12288x4096"),
    ];
    let median = |mut runs: Vec<f64>| -> f64 {
        runs.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a timing"));
        runs[1]
    };

    println!(
        "  {:<22} {:>8} {:>8} {:>8} {:>8} {:>8}    {:>8} {:>8}",
        "shape", "base", "tree", "u16", "packed", "pk+tree", "GB/s bas", "GB/s pk"
    );
    for (n_in, n_out, label) in shapes {
        let mut us = [0.0f64; 5];
        for (i, v) in ["bench_mm_base", "bench_mm_tree", "bench_mm_u16"].iter().enumerate() {
            us[i] = median((0..3).map(|_| gpu.bench_matmul(v, n_in, n_out, reps).expect("bench")).collect());
        }
        for (i, v) in ["bench_mm_packed", "bench_mm_packed_tree"].iter().enumerate() {
            us[3 + i] = median(
                (0..3)
                    .map(|_| gpu.bench_matmul_packed(v, n_in, n_out, reps).expect("bench"))
                    .collect(),
            );
        }
        // The packed layout moves the same bytes: 32 quants plus a 2-byte
        // scale, just in two arrays instead of interleaved.
        let bytes = (n_out * (n_in / 32) * 34) as f64;
        println!(
            "  {label:<22} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1}    {:>8.0} {:>8.0}",
            us[0], us[1], us[2], us[3], us[4],
            bytes / (us[0] * 1e-6) / 1e9,
            bytes / (us[3] * 1e-6) / 1e9
        );
    }
    assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");
}

/// Why is RMSNorm's serial f64 sum so expensive?
///
/// It is ~40% of device time and the largest single kernel in a token. The
/// obvious explanation — "f64 is slow on a consumer card" — is about
/// *throughput*, and a dependent chain is a latency problem. At ~59 us for 1024
/// adds, that is ~164 cycles each at 2.84 GHz, which is far more than an FP64
/// add should cost even here.
///
/// So this varies one thing at a time. If the cost turns out to be memory or
/// occupancy rather than FP64 latency, it is recoverable without trading any
/// exactness away, which is much the better outcome.
///
/// A measurement, not an assertion.
#[test]
#[ignore = "needs an sm_120 device; a measurement, not an assertion"]
fn why_is_the_rms_reduction_slow() {
    let gpu = Cuda::new(0).expect("cuda device");
    let reps = 4000;

    // 1024 and 2048 are n_embd of the 0.6B and of the 35B. The chain is as long
    // as the vector, so if the cost really is dependent-chain latency the
    // second column is twice the first — and that is what prices the 35B's 81
    // calls a token. 8192 is not a model dimension: it is there because the
    // fast variants sit at the launch floor at 1024 and cannot be told apart
    // until the work outgrows it.
    let sizes = [1024usize, 2048, 8192];

    // (label, kernel, needs n floats of dynamic shared)
    let cases = [
        ("serial f64, global", "bench_serial_f64_global", false),
        ("serial f32, global", "bench_serial_f32_global", false),
        ("serial f64, shared", "bench_shared_f64", true),
        ("serial f32, shared", "bench_shared_f32", true),
        ("tree f64", "bench_tree_f64", false),
    ];

    let (floor, _) = gpu
        .bench_kernel("bench_empty", 1024, 0, 256, 0, reps)
        .expect("bench kernel");
    println!("  launch floor (empty kernel) {floor:.2} us — read every number against it");
    println!();
    println!(
        "  {:<22} {:>11} {:>11} {:>11}",
        "variant", "n=1024 us", "n=2048 us", "n=8192 us"
    );
    for (label, kernel, staged) in cases {
        let mut us = [0.0f64; 3];
        for (k, n) in sizes.iter().enumerate() {
            let shared = if staged { *n } else { 0 };
            // Median of three: at these durations a single run picks up clock
            // ramp, and "no change" without a noise floor is how the shared
            // staging result was missed the first time.
            let mut runs: Vec<f64> = (0..3)
                .map(|_| {
                    gpu.bench_kernel(kernel, *n, shared, 256, 0, reps)
                        .expect("bench kernel")
                        .0
                })
                .collect();
            runs.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a timing"));
            us[k] = runs[1];
        }
        println!("  {label:<22} {:>11.2} {:>11.2} {:>11.2}", us[0], us[1], us[2]);
    }

    // The measurement that decides the question, and it is not the timing.
    //
    // A tree is faster because it breaks the dependent chain, and it is refused
    // because breaking the chain changes the answer. But "changes the answer"
    // is a property of f64 addition, not of reductions in general — so vary the
    // decomposition and see whose answer actually moves. Each variant runs at
    // four block sizes over identical input and the f64 results are compared
    // bit for bit.
    //
    // An order-free reduction reports one value across all four *by
    // construction*. That is the property the equal-bits rule needs;
    // serialness is only one way to get it, and it is the expensive way.
    let n = 1024usize;
    for (mode, name) in [
        (0u8, "benign"),
        (1, "wide, ~39 binades"),
        (2, "adversarial: 1e8 among 1.0s"),
    ] {
        // Must mirror `bench_kernel`'s input, and the oracle's `rms_scale`: the
        // square is rounded to f32 before it is widened, exactly as ggml does.
        let mut want = 0.0f64;
        for i in 0..n {
            let base = (i % 97) as f32 * 0.01 - 0.5;
            let v = match mode {
                1 => base * 2.0f32.powi((i % 35) as i32 - 17),
                2 => {
                    if i % 512 == 0 {
                        1.0e8
                    } else {
                        1.0
                    }
                }
                _ => base,
            };
            want += f64::from(v * v);
        }

        println!();
        println!("  n={n}, block sizes 32/64/128/256, {name} input");
        println!("  host serial f64 reference {want:.17e}");
        println!(
            "  {:<22} {:>12} {:>26} {:>11}",
            "variant", "orders", "value", "rel err"
        );
        for (label, kernel, staged) in cases {
            let shared = if staged { n } else { 0 };
            let mut seen: Vec<u64> = Vec::new();
            let mut value = 0.0f64;
            for threads in [32u32, 64, 128, 256] {
                let (_, got) = gpu
                    .bench_kernel(kernel, n, shared, threads, mode, 8)
                    .expect("bench kernel");
                value = got;
                if !seen.contains(&got.to_bits()) {
                    seen.push(got.to_bits());
                }
            }
            println!(
                "  {label:<22} {:>12} {value:>26.17e} {:>11.1e}",
                match seen.len() {
                    1 => "1 (exact)".to_string(),
                    k => format!("{k} differ"),
                },
                (value - want) / want
            );
        }
    }

    assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");
}

/// The host expert selection, transcribed from `qwen35::moe_token`.
///
/// Kept here rather than exported so the test compares against the rule as
/// *written in the model*, and a change to one without the other shows up as a
/// failure rather than as both agreeing on something new.
fn host_topk(all: &[f32], n_expert: usize, n_used: usize) -> (Vec<i32>, Vec<f32>) {
    let mut ids = Vec::new();
    let mut weights = Vec::new();
    for probs in all.chunks(n_expert) {
        let mut pick: Vec<(usize, f32)> = Vec::with_capacity(n_used);
        for _ in 0..n_used {
            let mut best = usize::MAX;
            for e in 0..probs.len() {
                if pick.iter().any(|(p, _)| *p == e) {
                    continue;
                }
                if best == usize::MAX || probs[e] > probs[best] {
                    best = e;
                }
            }
            pick.push((best, probs[best]));
        }
        let sum: f32 = pick.iter().map(|(_, p)| *p).sum();
        let denom = sum.max(6.103_515_625e-5);
        ids.extend(pick.iter().map(|(e, _)| *e as i32));
        weights.extend(pick.iter().map(|(_, p)| p / denom));
    }
    (ids, weights)
}

/// `add_scaled_rows` on the device equals the oracle's serial ascending sum, to the
/// bit, at every row count up to both compiled capacities (SSD-TIER.md D18): 8 or
/// fewer launch `add_scaled_rows`, 9 and 10 launch `add_scaled_rows_k10`, whose
/// two extra scalar arguments are the only difference.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn add_scaled_rows_matches_the_oracle_at_both_capacities() {
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    const N: usize = 2048;
    // Every buffer held for the whole test: the backend keys mirrors on host
    // addresses.
    let rows = noise(10 * N, 41);
    let scales = noise(10, 42);
    let mut accs: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0f32; N]).collect();
    let mut wants: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0f32; N]).collect();
    for (i, n_rows) in [3usize, 8, 9, 10].into_iter().enumerate() {
        Naive.add_scaled_rows(&mut wants[i], &rows[..n_rows * N], &scales[..n_rows]);
        gpu.begin_pass(1);
        gpu.add_scaled_rows(&mut accs[i], &rows[..n_rows * N], &scales[..n_rows]);
        gpu.host_needs(&mut accs[i]);
        if let Some(e) = gpu.take_error() {
            panic!("{n_rows} rows: driver error: {e}");
        }
        common::assert_bit_identical(&accs[i], &wants[i], &format!("add_scaled_rows, {n_rows} rows"));
    }
}

/// `moe_topk` on the device reproduces the host selection exactly, including
/// the tie rule and the weight normalization.
///
/// **This is the kernel that decides which weights are read, not what is
/// computed from them.** A wrong pick does not degrade the output the way a
/// numeric bug does — it produces fluent text from the wrong experts, at full
/// confidence, and no tolerance-based check anywhere else in this file would
/// notice. So the assertion is equality of ids and *bit* equality of weights.
///
/// The tie cases are the point. Selection scans ascending and takes a new best
/// only on a strict `>`, so equal probabilities must resolve to the lower
/// index; a device reduction that used `>=`, or that combined halves in the
/// wrong order, would differ only when two experts happen to tie — which on
/// real softmax output is rare enough to hide for a very long time.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn device_topk_reproduces_the_host_selection() {
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);

    let mut cases: Vec<(&str, Vec<f32>)> = Vec::new();

    // Ordinary router output: softmax over noise, which is what the real thing
    // produces and where ties essentially never happen.
    for seed in [1u64, 2, 3, 4, 5] {
        let mut p = noise(256, seed);
        let max = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in p.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in p.iter_mut() {
            *v /= sum;
        }
        cases.push(("softmax of noise", p));
    }

    // Every expert identical: the whole selection is decided by the tie rule,
    // so the answer must be exactly 0..n_used.
    cases.push(("all equal", vec![1.0 / 256.0; 256]));

    // Ties among the leaders only, planted away from index 0 so an
    // implementation that silently prefers low indices for the wrong reason
    // still has to get the *set* right.
    let mut tied = vec![0.001f32; 256];
    for e in [200usize, 7, 91, 199, 12, 250, 3, 44, 45, 46] {
        tied[e] = 0.05;
    }
    cases.push(("ten-way tie for eight places", tied));

    // A single dominant expert and a flat tail, which stresses the clamp:
    // seven of the eight picks contribute almost nothing to the sum.
    let mut spiked = vec![1e-12f32; 256];
    spiked[137] = 1.0;
    cases.push(("one spike, denormal tail", spiked));

    // Everything below the 6.103515625e-5 clamp, so `denom` is the clamp rather
    // than the sum and the weights do not sum to one. Getting this wrong scales
    // the whole FFN and still produces text.
    cases.push(("entirely below the clamp", vec![1e-9f32; 256]));

    // Descending, so the correct answer is 0..8 by value and by index at once,
    // and ascending, where it is the last eight in reverse.
    cases.push(("descending", (0..256).map(|i| (256 - i) as f32 / 32896.0).collect()));
    cases.push(("ascending", (0..256).map(|i| (i + 1) as f32 / 32896.0).collect()));

    // **Batched too.** A prefill routes every token in the batch at once, and
    // the kernel went from one block to one block per token; a decode-shaped
    // test would never see a token read another's probabilities.
    let batched: Vec<f32> = cases.iter().flat_map(|(_, p)| p.iter().copied()).collect();
    let (want_ids, want_w) = host_topk(&batched, 256, 8);
    let (got_ids, got_w) = gpu.moe_topk_readback(&batched, 256, 8).expect("moe_topk batch");
    assert_eq!(got_ids, want_ids, "batched over {} tokens: ids differ", cases.len());
    for (i, (g, w)) in got_w.iter().zip(want_w.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "batched weight {i}");
    }

    for (label, probs) in &cases {
        let (want_ids, want_w) = host_topk(probs, 256, 8);
        let (got_ids, got_w) = gpu.moe_topk_readback(probs, 256, 8).expect("moe_topk");
        assert_eq!(got_ids, want_ids, "{label}: expert ids differ");
        for (i, (g, w)) in got_w.iter().zip(want_w.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{label}: weight {i} is {g:e} against {w:e}; selection is exact by \
                 construction and the sum is serial, so any difference is a bug"
            );
        }
    }

    // n_used other than 8, since the kernel loops over it and the shared
    // `picked` array is sized for the maximum. 9 and 10 launch `moe_topk_k10`
    // (SSD-TIER.md D18), the same source compiled with room for 10.
    for n_used in [1usize, 2, 4, 8, 9, 10] {
        let probs = {
            let mut p = noise(256, 99);
            for v in p.iter_mut() {
                *v = v.abs() / 256.0;
            }
            p
        };
        let (want_ids, want_w) = host_topk(&probs, 256, n_used);
        let (got_ids, got_w) = gpu.moe_topk_readback(&probs, 256, n_used).expect("moe_topk");
        assert_eq!(got_ids, want_ids, "n_used {n_used}: ids differ");
        for (g, w) in got_w.iter().zip(want_w.iter()) {
            assert_eq!(g.to_bits(), w.to_bits(), "n_used {n_used}: weights differ");
        }
    }

    // **Qwen3.8-Flash-Next's own shape: top-10 of 512**, batched, with a planted
    // twelve-way tie for ten places. 512 experts span two 256-thread strides of the
    // one block, so this is also the first test where the per-thread scan wraps.
    let mut flash: Vec<f32> = Vec::new();
    for seed in [7u64, 8, 9] {
        let mut p = noise(512, seed);
        for v in p.iter_mut() {
            *v = v.abs() / 512.0;
        }
        flash.extend(p);
    }
    let mut tie = vec![0.0005f32; 512];
    for e in [511usize, 256, 255, 0, 300, 301, 12, 400, 399, 128, 129, 480] {
        tie[e] = 0.04;
    }
    flash.extend(tie);
    let (want_ids, want_w) = host_topk(&flash, 512, 10);
    let (got_ids, got_w) = gpu.moe_topk_readback(&flash, 512, 10).expect("moe_topk_k10");
    assert_eq!(got_ids, want_ids, "top-10 of 512: ids differ");
    for (i, (g, w)) in got_w.iter().zip(want_w.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "top-10 of 512: weight {i}");
    }
    assert!(gpu.moe_topk_readback(&flash, 512, 11).is_err(), "11 picks must be refused, not launched");
}

/// What `attn_flash` costs as decode context grows, at the 35B's attention
/// shape.
///
/// **A benchmark at d512 never sees this and an agentic session never leaves
/// it.** A real 22-turn Cline session reached 19,942 positions and decoded at
/// 22.55 tok/s where a 200-token run reads 29.76 — a 24% loss that has to be
/// attention, since it is the only term in the model that grows with context.
/// The 10.8 ms/token difference over ~408 MB of live KV implies ~38 GB/s
/// against a card measured at 409.6, but that is a difference of two whole-model
/// runs at different depths, which is exactly the kind of two-point attribution
/// that has been wrong three times in this repo. So: one kernel, one shape, one
/// axis.
///
/// `qwen35moe`: 16 query heads, 2 kv heads, head_dim 256, kv_dim 512, and only
/// 10 of 40 layers attend. Decode is one query row, so `n_q == 1`.
///
/// Prints rather than asserts — there is no correct answer here, only a curve,
/// and a threshold would encode today's hardware.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn what_attention_costs_as_context_grows() {
    const HEAD_DIM: usize = 256;
    const N_HEAD: usize = 16;
    const N_HEAD_KV: usize = 2;
    const KV_DIM: usize = N_HEAD_KV * HEAD_DIM;
    const LAYERS: usize = 10;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);

    // **Past d32768 as of 10-09.** Prefill and decode both bend into a steeper
    // second regime somewhere between the shallow and deep measurements, and
    // this sweep stopping at 32768 is why nothing could say where. The points
    // either side of 32768 are there to resolve a knee, not to fill the axis.
    let depths = [
        256usize, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384, 19942, 24576, 28160, 32768,
        36864, 40960, 49152, 57856, 65536,
    ];
    let max = depths.iter().copied().max().unwrap_or(0);

    // f16 bits, as the cache stores them. 65536 x 512 x 2 bytes = 64 MiB each.
    let k: Vec<u16> = (0..max * KV_DIM).map(|i| ((i * 2654435761) >> 13) as u16 & 0x3bff).collect();
    let v: Vec<u16> = (0..max * KV_DIM).map(|i| ((i * 40503) >> 11) as u16 & 0x3bff).collect();
    let q = noise(N_HEAD * HEAD_DIM, 7);
    let mut out = vec![0.0f32; N_HEAD * HEAD_DIM];

    println!("\nattn_flash, one query row, {N_HEAD}q/{N_HEAD_KV}kv x {HEAD_DIM}");
    println!(
        "  {:>7}  {:>9}  {:>9}  {:>7}  {:>9}  {:>9}  {:>11}  {:>13}",
        "n_pos", "thread", "warp", "speedup", "GB/s thr", "GB/s warp", "ms/tok warp", "us/1k pos warp"
    );
    // The previous row's depth and warp time, for the incremental slope. A knee
    // is a change in slope, and a column of totals hides one inside a trend.
    let mut prev: Option<(usize, f64)> = None;
    for d in depths {
        let a = Attn {
            q: &q,
            k: &k,
            v: &v,
            kv_dim: KV_DIM,
            n_pos: d,
            head_dim: HEAD_DIM,
            n_head: N_HEAD,
            n_head_kv: N_HEAD_KV,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        };
        // Warm the mirrors and the kernel, then time a batch with the queue
        // full — a per-call synchronize is what makes a 20 us kernel read 100.
        for _ in 0..3 {
            gpu.begin_pass(1);
            gpu.attend(&a, &mut out);
            gpu.host_needs(&mut out);
            gpu.end_pass();
        }
        const REPS: u32 = 50;
        // Both kernels, same shape, same sitting, warm-up discarded each time.
        // A/B rather than before/after, because `f32_staged` shipped a 1.65x
        // regression off a measurement of one shape taken at a different moment.
        let mut timed = |warp: bool| {
            gpu.attn_warp(Some(warp));
            for _ in 0..2 {
                gpu.begin_pass(1);
                gpu.attend(&a, &mut out);
                gpu.host_needs(&mut out);
                gpu.end_pass();
            }
            let t0 = std::time::Instant::now();
            gpu.begin_pass(1);
            for _ in 0..REPS {
                gpu.attend(&a, &mut out);
            }
            gpu.end_pass();
            gpu.host_needs(&mut out);
            t0.elapsed().as_secs_f64() * 1e6 / f64::from(REPS)
        };
        // The scalar arms are forced off the tensor cores, which decode takes
        // past its threshold by default.
        gpu.attn_decode_mma(Some(false));
        let us = timed(false);
        let warp_us = timed(true);
        gpu.attn_decode_mma(Some(true));
        let mma_us = timed(true);
        gpu.attn_decode_tile16(true);
        let mma16_us = timed(true);
        gpu.attn_decode_tile16(false);
        gpu.attn_decode_mma(None);
        gpu.attn_warp(None);

        // K and V, both f16, over the live window.
        let bytes = (d * KV_DIM * 2 * 2) as f64;
        let slope = prev.map_or(String::from("-"), |(pd, pus)| {
            format!("{:.2}", (warp_us - pus) * 1000.0 / (d - pd) as f64)
        });
        println!(
            "  {:>7}  {:>9.1}  {:>9.1}  {:>7.2}  {:>9.1}  {:>9.1}  {:>11.2}  {:>14}",
            d,
            us,
            warp_us,
            us / warp_us,
            bytes / (us * 1e3),
            bytes / (warp_us * 1e3),
            warp_us * LAYERS as f64 / 1000.0,
            slope,
        );
        prev = Some((d, warp_us));
        println!(
            "  {:>7}  tensor cores {mma_us:>9.1} us ({:.2}x warp), 16-slot tile {mma16_us:>9.1} us ({:.2}x warp)",
            "",
            warp_us / mma_us,
            warp_us / mma16_us
        );
    }
    println!("  ms/token is one call x {LAYERS} attending layers.\n");
}

/// **QSA Q2, per op: pooling, scores and the selection are bit-identical to the
/// oracle, so the device keeps exactly the oracle's cells** (`src/model/qwen4exp.md`,
/// "QSA on CUDA").
///
/// - `qsa_pool`: two lanes of 1,024 blocks, pooled on the device in two steps and
///   an empty third (as decode issues it), against one oracle call.
/// - `qsa_select`: prefill rows and decode rows past the budget, rows below it
///   (every cell kept), and a decode row with no whole block yet. **The second
///   lane starts with 2,400 zero cells**, whose 600 blocks pool to exactly zero
///   and score exactly 0, so the 512-block boundary falls inside a 600-way tie
///   and the tie rule decides which zero blocks are kept.
/// - `attend_sparse`, at the 125B's and the 0.2B's attention shapes: **equal to
///   the bit to the device's own dense attention over the same cells gathered on
///   the host**, in the default decode mode (the kept window is past 2,048, so
///   the tensor cores), which is what proves the gather and the per-row wiring;
///   and in the scalar mode, within `attend_tolerance` of the oracle.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_qsa_ops_agree_with_the_oracle() {
    use inferred_thoughts::ops::{QsaPool, QsaSelect};
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    let (dim, ih, ratio, budget, n_ctx) = (128usize, 4usize, 4usize, 512usize, 4096usize);
    let n_blocks = n_ctx / ratio;
    let norm: Vec<f32> = noise(dim, 0x9a01).iter().map(|v| 1.0 + 0.25 * v).collect();
    let (eps, n_rot, theta) = (1e-6f32, 64usize, 1.0e7f32);
    let pool = |from: usize, to: usize| QsaPool { from, to, ratio, dim, norm: &norm, eps, n_rot, theta };

    // Every buffer the device keys by address is held to the end of the test.
    let raw_noise: Vec<u16> = noise(n_ctx * dim, 0x9a02).iter().map(|&v| f32_to_f16(v)).collect();
    let mut raw_ties = raw_noise.clone();
    raw_ties[..2400 * dim].fill(0);
    let raws = [raw_noise, raw_ties];
    let mut cpu_lanes = vec![vec![0.0f32; n_blocks * dim]; 2];
    let mut gpu_lanes = vec![vec![0.0f32; n_blocks * dim]; 2];

    // --- qsa_pool ------------------------------------------------------
    for (li, raw) in raws.iter().enumerate() {
        Naive.qsa_pool(raw, &mut cpu_lanes[li], &pool(0, n_blocks));
        gpu.begin_pass(1);
        gpu.qsa_pool(raw, &mut gpu_lanes[li], &pool(0, 300));
        gpu.qsa_pool(raw, &mut gpu_lanes[li], &pool(300, n_blocks));
        gpu.qsa_pool(raw, &mut gpu_lanes[li], &pool(n_blocks, n_blocks));
        gpu.end_pass();
        if let Some(e) = gpu.take_error() {
            panic!("cuda error pooling lane {li}: {e}");
        }
        let got = gpu.read_pooled(&gpu_lanes[li]).expect("pooled readback");
        exact(&format!("qsa_pool {li}"), &cpu_lanes[li], &got);
    }
    let zero_block = cpu_lanes[1][..dim].to_vec();
    assert!(zero_block.iter().all(|&v| v.to_bits() == 0), "a zero block must pool to +0.0");

    // --- qsa_select ----------------------------------------------------
    // (lane, start_pos, rows, blocks pooled)
    let cases = [
        (0usize, 4093usize, 3usize, 1024usize),
        (1, 4093, 3, 1024),
        (0, 4095, 1, 1024),
        (1, 4095, 1, 1024),
        (0, 100, 2, 25),
        (0, 1, 1, 0),
    ];
    let sel_of = |start_pos: usize, nb: usize| QsaSelect { n_head: ih, dim, n_blocks: nb, start_pos, ratio, budget };
    struct Sel {
        q: Vec<f32>,
        cpu_scores: Vec<f32>,
        gpu_scores: Vec<f32>,
        cpu_cells: Vec<u32>,
        gpu_cells: Vec<u32>,
    }
    let stride = sel_of(0, 0).stride();
    let mut sels: Vec<Sel> = cases
        .iter()
        .enumerate()
        .map(|(ci, &(_, _, n_q, nb))| Sel {
            q: noise(n_q * ih * dim, 0x9b00 + ci as u64),
            cpu_scores: vec![0.0; n_q * nb.max(1)],
            gpu_scores: vec![0.0; n_q * nb.max(1)],
            cpu_cells: vec![0; n_q * stride],
            gpu_cells: vec![0; n_q * stride],
        })
        .collect();
    for (ci, &(lane, start_pos, n_q, nb)) in cases.iter().enumerate() {
        let sel = sel_of(start_pos, nb);
        let Sel { q, cpu_scores, gpu_scores, cpu_cells, gpu_cells } = &mut sels[ci];
        Naive.qsa_select(q, &cpu_lanes[lane], &sel, cpu_scores, cpu_cells);
        gpu.begin_pass(n_q);
        gpu.host_wrote(q);
        gpu.qsa_select(q, &gpu_lanes[lane], &sel, gpu_scores, gpu_cells);
        gpu.host_needs(gpu_scores);
        gpu.end_pass();
        if let Some(e) = gpu.take_error() {
            panic!("cuda error selecting case {ci}: {e}");
        }
        // With no whole block there are no scores, only a placeholder slot.
        if nb > 0 {
            exact(&format!("scores {ci}"), cpu_scores, gpu_scores);
        }
        let got = gpu.read_cells(gpu_cells).expect("cells readback");
        let mut kept_zero = (0usize, 0usize);
        for t in 0..n_q {
            let pos = start_pos + t;
            let count = sel.count(pos);
            let (want, have) = (&cpu_cells[t * stride..][..count], &got[t * stride..][..count]);
            assert_eq!(want, have, "case {ci} row {t}: the device kept different cells than the oracle");
            let full = (pos + 1) / ratio;
            if full > budget {
                assert!(count < pos + 1, "case {ci} row {t}: past the budget, cells must be dropped");
            } else {
                assert_eq!(count, pos + 1, "case {ci} row {t}: within the budget, every cell is kept");
            }
            if lane == 1 && full > budget {
                let row = &cpu_scores[t * nb..(t + 1) * nb];
                let kept: std::collections::HashSet<u32> = want.iter().map(|&c| c / ratio as u32).collect();
                for b in 0..full {
                    if row[b] == 0.0 {
                        if kept.contains(&(b as u32)) {
                            kept_zero.0 += 1;
                        } else {
                            kept_zero.1 += 1;
                        }
                    }
                }
            }
        }
        // The copies made above are the device's cells; hold them for attention.
        gpu_cells.copy_from_slice(&got);
        if lane == 1 && start_pos > 1000 {
            println!("  case {ci}: zero-score blocks kept {} / dropped {}", kept_zero.0, kept_zero.1);
            assert!(
                kept_zero.0 > 0 && kept_zero.1 > 0,
                "the tie lane must put the budget's boundary inside the zero-score tie"
            );
        }
    }

    // --- attend_sparse -------------------------------------------------
    // (head_dim, n_head, n_head_kv): the 125B, the 0.2B.
    let shapes = [(256usize, 24usize, 2usize), (256, 8, 2)];
    let attn_cases = [1usize, 2];
    struct Att {
        k: Vec<u16>,
        v: Vec<u16>,
        q: Vec<f32>,
        sparse: Vec<f32>,
        scalar: Vec<f32>,
        want: Vec<f32>,
        rows_q: Vec<Vec<f32>>,
        rows_k: Vec<Vec<u16>>,
        rows_v: Vec<Vec<u16>>,
        rows_out: Vec<Vec<f32>>,
    }
    let mut atts: Vec<Att> = Vec::new();
    for (si, &(hd, nh, nkv)) in shapes.iter().enumerate() {
        for &ci in &attn_cases {
            let n_q = cases[ci].2;
            let kv_dim = nkv * hd;
            atts.push(Att {
                k: noise(n_ctx * kv_dim, 0x9c00 + si as u64).iter().map(|&x| f32_to_f16(x)).collect(),
                v: noise(n_ctx * kv_dim, 0x9d00 + si as u64).iter().map(|&x| f32_to_f16(x)).collect(),
                q: noise(n_q * nh * hd, 0x9e00 + (si * 10 + ci) as u64),
                sparse: vec![0.0; n_q * nh * hd],
                scalar: vec![0.0; n_q * nh * hd],
                want: vec![0.0; n_q * nh * hd],
                rows_q: (0..n_q).map(|_| vec![0.0; nh * hd]).collect(),
                rows_k: (0..n_q).map(|_| vec![0; stride * kv_dim]).collect(),
                rows_v: (0..n_q).map(|_| vec![0; stride * kv_dim]).collect(),
                rows_out: (0..n_q).map(|_| vec![0.0; nh * hd]).collect(),
            });
        }
    }
    let mut ai = 0;
    for &(hd, nh, nkv) in &shapes {
        for &ci in &attn_cases {
            let (_, start_pos, n_q, nb) = cases[ci];
            let sel = sel_of(start_pos, nb);
            let cells = &sels[ci].gpu_cells;
            let kv_dim = nkv * hd;
            let scale = 1.0 / (hd as f32).sqrt();
            let Att { k, v, q, sparse, scalar, want, rows_q, rows_k, rows_v, rows_out } = &mut atts[ai];
            ai += 1;
            let a = Attn { q: &q[..], k: &k[..], v: &v[..], kv_dim, n_pos: start_pos + n_q, head_dim: hd, n_head: nh, n_head_kv: nkv, scale };

            gpu.begin_pass(n_q);
            gpu.host_wrote(&q[..]);
            gpu.attend_sparse(&a, cells, &sel, sparse);
            gpu.host_needs(sparse);
            gpu.end_pass();

            gpu.attn_decode_mma(Some(false));
            gpu.begin_pass(n_q);
            gpu.host_wrote(&q[..]);
            gpu.attend_sparse(&a, cells, &sel, scalar);
            gpu.host_needs(scalar);
            gpu.end_pass();
            gpu.attn_decode_mma(None);
            if let Some(e) = gpu.take_error() {
                panic!("cuda error in attend_sparse {nh}q case {ci}: {e}");
            }

            Naive.attend_sparse(&a, cells, &sel, want);
            let name = format!("sparse {nh}q c{ci}");
            close(&name, want, scalar, attend_tolerance(stride, want));

            // The same cells gathered on the host, attended densely by the device
            // one row at a time: the same kernels on the same data.
            let per_row = nh * hd;
            for t in 0..n_q {
                let count = sel.count(start_pos + t);
                for (j, &c) in cells[t * stride..][..count].iter().enumerate() {
                    let (src, dst) = (c as usize * kv_dim, j * kv_dim);
                    rows_k[t][dst..dst + kv_dim].copy_from_slice(&k[src..src + kv_dim]);
                    rows_v[t][dst..dst + kv_dim].copy_from_slice(&v[src..src + kv_dim]);
                }
                rows_q[t].copy_from_slice(&q[t * per_row..(t + 1) * per_row]);
                let ar = Attn {
                    q: &rows_q[t],
                    k: &rows_k[t],
                    v: &rows_v[t],
                    kv_dim,
                    n_pos: count,
                    head_dim: hd,
                    n_head: nh,
                    n_head_kv: nkv,
                    scale,
                };
                gpu.begin_pass(1);
                gpu.host_wrote(&rows_q[t]);
                gpu.attend(&ar, &mut rows_out[t]);
                gpu.host_needs(&mut rows_out[t]);
                gpu.end_pass();
                exact(&format!("{name} row {t}"), &rows_out[t], &sparse[t * per_row..(t + 1) * per_row]);
            }
        }
    }
    if let Some(e) = gpu.take_error() {
        panic!("cuda error: {e}");
    }
}

/// What gathering QSA's kept cells into a dense window costs: the gather arm of
/// the Q2 decode fork (`src/model/qwen4exp.md`). The mask arm is priced by
/// `what_attention_costs_as_context_grows` at full depth; the gather arm is
/// that bench's d2048 row plus this.
///
/// 2,051 cells, the most a query keeps: 512 whole blocks of 4 spread evenly
/// over the cache (the least local choice), plus a 3-cell tail. Qwen3.8's
/// attention shape, `kv_dim` 512. A few seconds; prints, does not assert
/// beyond the gather's own check that it copied the right rows.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn what_the_qsa_gather_costs() {
    const KV_DIM: usize = 512;
    const RATIO: usize = 4;
    const BLOCKS: usize = 512;
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    println!("
qsa_gather_kv, {} cells x kv_dim {KV_DIM}, K and V", BLOCKS * RATIO + 3);
    println!("  {:>7}  {:>9}  {:>9}  {:>8}", "n_pos", "device us", "issue us", "GB/s");
    for n_pos in [4096usize, 16384, 65536] {
        let k: Vec<u16> = (0..n_pos * KV_DIM).map(|i| ((i * 2654435761) >> 13) as u16 & 0x3bff).collect();
        let v: Vec<u16> = (0..n_pos * KV_DIM).map(|i| ((i * 40503) >> 11) as u16 & 0x3bff).collect();
        let full = (n_pos - 3) / RATIO;
        let mut cells: Vec<u32> = (0..BLOCKS)
            .flat_map(|b| {
                let first = b * full / BLOCKS * RATIO;
                (first..first + RATIO).map(|c| c as u32)
            })
            .collect();
        cells.extend((full * RATIO..full * RATIO + 3).map(|c| c as u32));
        let (dev, issue) = gpu.bench_qsa_gather(&k, &v, KV_DIM, &cells, 50).expect("gather bench");
        let bytes = (cells.len() * KV_DIM * 2 * 2) as f64;
        println!("  {n_pos:>7}  {dev:>9.1}  {issue:>9.1}  {:>8.1}", bytes / (dev * 1e3));
    }
    if let Some(e) = gpu.take_error() {
        panic!("cuda error: {e}");
    }
}

/// **Where attention changes regime with depth, and which buffer does it.**
///
/// Prefill at depth fits two lines rather than one (`BENCHMARKS.md`, 10-09
/// late): `1.907 + 1.038e-4 * D` shallow and `2.897 + 1.301e-4 * D` deep, and
/// decode bends the same way. Both attention sweeps stopped at d32768, so
/// nothing could say where the second regime starts or what starts it.
///
/// Two buffers on the production prefill path reach 32 MiB at exactly d32768,
/// and `CLAUDE.md` puts this card's L2 at 32 MB:
///
/// | buffer, per attending layer | bytes | 32 MiB at |
/// |---|---|---|
/// | the K window the score phase reads | `n_pos * kv_dim * 2` | 32,768 at `kv_dim` 512 |
/// | the partials `attn_flash` writes for the combine | `qgroup * n_head * n_split * head_dim * 4` | 32,768 at `qgroup` 8 |
///
/// That is a coincidence of this model's shapes, and it means a knee at 32768
/// cannot name either one. So each arm moves one crossing and leaves the other
/// where it is, with no kernel changed:
///
/// | arm | K crossing | partials crossing |
/// |---|---|---|
/// | `qgroup` 8, `n_head_kv` 2 -- production | 32,768 | 32,768 |
/// | `qgroup` 4 | 32,768 | 65,536 |
/// | `qgroup` 16 | 32,768 | 16,384 |
/// | `n_head_kv` 1 | 65,536 | 32,768 |
///
/// A knee that follows the `qgroup` arms is the partials; one that follows the
/// `n_head_kv` arm is K; one that follows neither says the working-set story is
/// wrong, which is also a result. **A hypothesis under test, not a finding** --
/// the 32 MB figure is the least established thing in it.
///
/// The slope column is in the whole-model fit's units -- ms of ten attending
/// layers per prompt token, per position of depth -- taken between this depth
/// and the previous one, so it reads directly against 1.038e-4 and 1.301e-4.
///
/// Bare of the activation bus but for one thing: `begin_pass` invalidates every
/// mirror, so the timed pass re-uploads the 12.5 MiB query once. ~0.45 ms
/// against ~20 ms calls at d4096 and far less deeper, identical on every arm.
/// The output comes home once, after the timed calls.
///
/// `the_attention_agrees_with_the_oracle_past_32k` checks every construction
/// timed here against `naive`, including the one-kv-head shape and the
/// non-production `qgroup` values, none of which the 35B runs.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn which_buffer_bends_attention_at_depth() {
    const HEAD_DIM: usize = 256;
    const N_HEAD: usize = 16;
    const N_Q: usize = 512;
    const LAYERS: usize = 10;
    const MAX_POS: usize = 65536;
    /// The FP32 CUDA cores this path runs on, from the ceiling table in
    /// `CLAUDE.md`'s 09-09 section.
    const FP32_PEAK_TFLOPS: f64 = 24.0;
    /// `CLAUDE.md`'s figure for this card, and the premise being tested.
    const L2_BYTES: usize = 32 << 20;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    // The f32 `attn_flash` path is what this is about; the tensor-core
    // kernel is the prefill default.
    gpu.set_attn_vmma(false);
    gpu.set_attn_mma(false);

    // Every slab and buffer is held for the whole test: mirrors key on the host
    // address, so a dropped buffer hands the next arm a recycled one. The
    // one-kv-head slabs are separate allocations for the same reason rather
    // than sub-slices of the two-head ones.
    let k2: Vec<u16> =
        (0..MAX_POS * 2 * HEAD_DIM).map(|i| ((i * 2654435761) >> 13) as u16 & 0x3bff).collect();
    let v2: Vec<u16> = (0..MAX_POS * 2 * HEAD_DIM).map(|i| ((i * 40503) >> 11) as u16 & 0x3bff).collect();
    let k1: Vec<u16> = k2[..MAX_POS * HEAD_DIM].to_vec();
    let v1: Vec<u16> = v2[..MAX_POS * HEAD_DIM].to_vec();
    let q = noise(N_Q * N_HEAD * HEAD_DIM, 7);
    let mut out = vec![0.0f32; N_Q * N_HEAD * HEAD_DIM];

    // (label, qgroup, n_head_kv)
    let arms: [(&str, usize, usize); 4] =
        [("g8 kv2", 8, 2), ("g4 kv2", 4, 2), ("g16 kv2", 16, 2), ("g8 kv1", 8, 1)];
    let depths = [4096usize, 8192, 16384, 24576, 32768, 40960, 49152, 57856, 65536];

    println!("\nattention at depth, prefill shape: n_q {N_Q}, {N_HEAD}q x {HEAD_DIM}, x{LAYERS} layers");
    for (label, g, kv) in arms {
        let k_at = L2_BYTES / (kv * HEAD_DIM * 2);
        let p_at = L2_BYTES / (g * N_HEAD * HEAD_DIM * 4) * 128;
        println!("  {label:<8} K window reaches 32 MiB at {k_at:>6}, partials at {p_at:>6}");
    }

    let mut best = vec![[f64::MAX; 4]; depths.len()];
    for (di, &n_pos) in depths.iter().enumerate() {
        // Arms interleaved inside each round, so drift lands on all four.
        for _ in 0..3 {
            for (ai, &(_, g, kv)) in arms.iter().enumerate() {
                let (k, v) = if kv == 2 { (&k2, &v2) } else { (&k1, &v1) };
                let a = Attn {
                    q: &q,
                    k,
                    v,
                    kv_dim: kv * HEAD_DIM,
                    n_pos,
                    head_dim: HEAD_DIM,
                    n_head: N_HEAD,
                    n_head_kv: kv,
                    scale: 1.0 / (HEAD_DIM as f32).sqrt(),
                };
                gpu.set_qgroup(g);
                // Warm-up: uploads the new KV positions and grows the pooled
                // partials to this arm's size, neither of which is attention.
                gpu.begin_pass(N_Q);
                gpu.attend(&a, &mut out);
                gpu.host_needs(&mut out);
                gpu.end_pass();

                const REPS: u32 = 2;
                let t = std::time::Instant::now();
                gpu.begin_pass(N_Q);
                for _ in 0..REPS {
                    gpu.attend(&a, &mut out);
                }
                gpu.end_pass();
                gpu.host_needs(&mut out);
                let ms = t.elapsed().as_secs_f64() * 1e3 / f64::from(REPS);
                best[di][ai] = best[di][ai].min(ms);
            }
        }
        assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error at n_pos {n_pos}");
    }
    gpu.set_qgroup(8);

    // ms of ten layers per prompt token, as the whole-model fit counts it.
    let per_tok = |ms: f64| ms * LAYERS as f64 / N_Q as f64;
    print!("\n  {:>6}", "n_pos");
    for (label, _, _) in arms {
        print!("  {label:>9} {:>10}", "slope");
    }
    println!("  {:>7}", "of fp32");
    for (di, &n_pos) in depths.iter().enumerate() {
        print!("  {n_pos:>6}");
        for ai in 0..arms.len() {
            let ms = best[di][ai];
            let slope = if di == 0 {
                String::from("-")
            } else {
                let dd = (n_pos - depths[di - 1]) as f64;
                format!("{:.3e}", (per_tok(ms) - per_tok(best[di - 1][ai])) / dd)
            };
            print!("  {ms:>7.1}ms {slope:>10}");
        }
        // Production arm against the cores it runs on. Two MACs per (query,
        // key, dim), over every head and every row's causal window.
        let n_pos_first = (n_pos + 1 - N_Q) as f64;
        let windows = N_Q as f64 * n_pos_first + (N_Q * (N_Q - 1) / 2) as f64;
        let flops = 4.0 * N_HEAD as f64 * windows * HEAD_DIM as f64;
        let tf = flops / (best[di][0] * 1e-3) / 1e12;
        println!("  {:>6.2}%", 100.0 * tf / FP32_PEAK_TFLOPS);
    }
    println!(
        "\n  slope: ms of {LAYERS} layers per prompt token, per position; the whole-model fit \
         is 1.038e-4 shallow and 1.301e-4 deep\n"
    );
}

/// **Decode attention on the tensor cores, against the oracle, inside the same
/// derived bound as prefill's.**
///
/// `attn_decode`'s tensor-core mode fills the tile's slots with the query heads
/// that share a kv head, in one row -- a different slot map from prefill's rows
/// of one query head, so it is checked on its own, at n_q 1 and forced on at
/// every depth, including ones below where it would be chosen. The bound and its
/// derivation are those of `the_tensor_core_attention_stays_inside_its_derived_bound`.
///
/// A forced mode that silently fell back to the scalar path would pass a bound
/// test, so each case also runs the scalar mode and asserts the two differ: the
/// tensor-core mode rounds Q through f16, and matching bits would mean it never
/// ran.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_tensor_core_decode_attention_stays_inside_its_derived_bound() {
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);

    // (head_dim, n_head, n_head_kv, depths): the 35B, the 0.6B, and one query
    // head per kv head so a single slot is live.
    let shapes: [(usize, usize, usize, &[usize]); 3] = [
        (256, 16, 2, &[1, 40, 129, 4096, 32769]),
        (128, 16, 8, &[1, 40, 4096]),
        (128, 16, 16, &[40, 4096]),
    ];

    // Held for the whole test: mirrors key on host addresses.
    struct Held {
        kf: Vec<f32>,
        vf: Vec<f32>,
        k: Vec<u16>,
        v: Vec<u16>,
        q: Vec<f32>,
        mma: Vec<f32>,
        mma16: Vec<f32>,
        scalar: Vec<f32>,
    }
    let mut held: Vec<Held> = shapes
        .iter()
        .enumerate()
        .map(|(si, &(head_dim, n_head, n_head_kv, depths))| {
            let kv_dim = n_head_kv * head_dim;
            let max_pos = depths.iter().copied().max().unwrap_or(0);
            let kf = noise(max_pos * kv_dim, 0x6c00 + si as u64);
            let vf = noise(max_pos * kv_dim, 0x7700 + si as u64);
            let k = kf.iter().map(|&x| f32_to_f16(x)).collect();
            let v = vf.iter().map(|&x| f32_to_f16(x)).collect();
            let q = noise(n_head * head_dim, 0x7200 + si as u64);
            Held { kf, vf, k, v, mma: vec![0.0; q.len()], mma16: vec![0.0; q.len()], scalar: vec![0.0; q.len()], q }
        })
        .collect();

    let f16_slack = 1.0 + 2f64.powi(-10);
    let norm = |r: &[f32]| r.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    let mut worst_ratio = 0.0f64;
    for (si, &(head_dim, n_head, n_head_kv, depths)) in shapes.iter().enumerate() {
        let kv_dim = n_head_kv * head_dim;
        let Held { kf, vf, k, v, q, mma, mma16, scalar } = &mut held[si];
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q2 = q.chunks_exact(head_dim).map(norm).fold(0.0, f64::max) * f16_slack;

        for &n_pos in depths {
            let a = Attn { q: &q[..], k: &k[..], v: &v[..], kv_dim, n_pos, head_dim, n_head, n_head_kv, scale };
            let mut want = vec![0.0f32; q.len()];
            Naive.attend(&a, &mut want);

            // The 8-slot tile is the default wherever it fits; the 16-slot one
            // is still reached for wider groups, so both are held to the bound.
            for (force, tile16, into) in
                [(true, false, &mut *mma), (true, true, &mut *mma16), (false, false, &mut *scalar)]
            {
                gpu.attn_decode_mma(Some(force));
                gpu.attn_decode_tile16(tile16);
                gpu.begin_pass(1);
                gpu.attend(&a, &mut into[..]);
                gpu.host_needs(&mut into[..]);
                gpu.end_pass();
            }
            gpu.attn_decode_mma(None);
            gpu.attn_decode_tile16(false);
            if let Some(e) = gpu.take_error() {
                panic!("cuda error at hd{head_dim} kv{n_head_kv} d{n_pos}: {e}");
            }

            let k2 = kf[..n_pos * kv_dim].chunks_exact(head_dim).map(norm).fold(0.0, f64::max) * f16_slack;
            let vmax =
                vf[..n_pos * kv_dim].iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs())) * f16_slack;
            let d = 2f64.powi(-11) * f64::from(scale) * q2 * k2;
            let bound = vmax * ((2.0 * d).exp_m1() + 2f64.powi(-11) + n_pos as f64 * 2f64.powi(-25))
                + f64::from(attend_tolerance(n_pos, &want));
            let worst_of = |got: &[f32]| {
                want.iter().zip(got).fold(0.0f64, |m, (x, y)| m.max((f64::from(*x) - f64::from(*y)).abs()))
            };
            let worst = worst_of(&mma[..]).max(worst_of(&mma16[..]));
            let (_, from_scalar) = compare(&scalar[..], &mma[..]);
            worst_ratio = worst_ratio.max(worst / bound);
            println!(
                "  hd{head_dim} kv{n_head_kv} decode d{n_pos:<6} worst {worst:.3e}  bound {bound:.3e}  \
                 {from_scalar} of {} differ from the scalar mode",
                want.len()
            );
            assert!(
                worst <= bound,
                "decode attention on the tensor cores is outside its derived bound at hd{head_dim} \
                 kv{n_head_kv} d{n_pos}: {worst:e} against {bound:e}. Suspect the slot map -- slot c \
                 of kv head hk is query head hk * gqa + c -- the kv offset, or a padded slot marked live."
            );
            // Not at one position: the only softmax weight is exactly 1 in both
            // modes, so both return `v[0]` exactly and equal bits are correct.
            assert!(
                n_pos == 1 || from_scalar > 0,
                "hd{head_dim} kv{n_head_kv} d{n_pos}: the forced tensor-core mode matched the scalar \
                 mode bit for bit, so it never ran"
            );
        }
    }
    println!("  worst error is {worst_ratio:.2e} of its bound");
}

/// **Decode attention changes mode under a CUDA graph without changing the
/// answer.**
///
/// `attn_decode` switches between its scalar and tensor-core modes by an
/// argument, and that is only sound if graph replay really updates the grid,
/// the shared size and the arguments in place. A replay that kept the recorded
/// ones would attend with stale values -- the word salad `attn_flash_mma` once
/// produced at decode. So the same fixed tokens are decoded with graphs on and
/// off, with the switch landing mid-decode and a 128-position chunk boundary
/// crossed after it, and the logits must be **bit-identical**: a graph changes
/// how launches are issued, never what they compute.
///
/// A third run never switches, and must differ, so the test cannot pass with
/// the tensor-core mode never having run.
#[test]
#[ignore = "needs the 0.6B and an sm_120 device"]
fn decode_attention_changes_mode_under_a_graph_without_changing_the_answer() {
    common::model_or_skip!(path);
    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");

    let tokens = tk.encode(&"The capital of France is Paris. ".repeat(16), true, true);
    let steps = tk.encode(&" and the capital of Japan is Tokyo,".repeat(4), false, true);
    let switch_at = tokens.len() + 6;
    assert!(
        tokens.len() < 128 && tokens.len() + steps.len() > 128 + 4,
        "{} prompt and {} decode tokens: the decode has to cross position 128 after the switch",
        tokens.len(),
        steps.len()
    );

    let run = |graphs: bool, from: usize| -> Vec<f32> {
        let gpu = Cuda::new(0).expect("cuda device");
        gpu.use_graphs(graphs);
        gpu.set_attn_decode_mma_from(from);
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, &gpu, tokens.len() + steps.len() + 4, false);
        let mut l = e.prefill(&tokens).expect("prefill");
        for &t in &steps {
            l = e.decode(t).expect("decode");
        }
        assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");
        l
    };

    let graphed = run(true, switch_at);
    let eager = run(false, switch_at);
    let never = run(true, usize::MAX);

    let (worst, differing) = compare(&eager, &graphed);
    let (_, from_never) = compare(&never, &graphed);
    println!(
        "  graphs on vs off   {differing} of {} logits differ, worst {worst:e}\n  \
         switched vs never  {from_never} differ",
        graphed.len()
    );
    assert_eq!(
        differing, 0,
        "decoding across the mode switch gives different logits with graphs on and off: replay \
         did not carry the new grid, shared size or arguments into a node"
    );
    assert!(from_never > 0, "the run that switched matches the one that never did, so it never switched");
}

/// **The restaged tensor-core attention kernel is bit-identical to its old
/// staging, at every shape the rewrite could get wrong.**
///
/// `attn_flash_mma_v` now stages K and V sixteen bytes per instruction and the
/// query by stride, where it divided and took a remainder for every element --
/// ~80% of the kernel at d32768. The same bytes move, so the answer must be the
/// same bits, and `dbg_attn_mma_v_0` keeps the old staging to prove it. This is
/// the exact guard; `the_tensor_core_attention_stays_inside_its_derived_bound`
/// is the one against the oracle.
///
/// The shapes are chosen for what a strided copy can get wrong: n_q that is not
/// a multiple of the 16-row tile (padding rows), windows ending mid-chunk and
/// mid-sub-tile, n_pos either side of 32768, head_dim 128 where lanes 16-31
/// stage nothing, and **a KV slab exactly n_pos long**, so the zeroed tail of
/// the last chunk lies outside the allocation and a wrong guard reads past it.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_restaged_tensor_core_attention_is_bit_identical() {
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);

    // One slab pair per kv width, held for the whole test: the backend keys its
    // KV mirror on the host address and counts uploaded *positions*, so reusing
    // one slab at a different kv_dim would upload the wrong bytes.
    let slab = |n: usize, mul: usize, shift: u32| -> Vec<u16> {
        (0..n).map(|i| ((i * mul) >> shift) as u16 & 0x3bff).collect()
    };
    // (head_dim, n_head, n_head_kv, positions in the slab)
    let shapes: [(usize, usize, usize, usize); 4] =
        [(256, 16, 2, 32769), (128, 16, 8, 4097), (128, 16, 16, 4097), (256, 16, 2, 1000)];
    let slabs: Vec<(Vec<u16>, Vec<u16>)> = shapes
        .iter()
        .map(|&(hd, _, kv, pos)| (slab(pos * kv * hd, 2654435761, 13), slab(pos * kv * hd, 40503, 11)))
        .collect();

    let rows = [2usize, 17, 33, 512];
    // One query and two outputs per (rows, head_dim): a shorter slice of a
    // longer buffer shares its address and would share its mirror.
    let held_q: Vec<Vec<f32>> = [256usize, 128]
        .iter()
        .flat_map(|&hd| rows.iter().map(move |&n| noise(n * 16 * hd, 0x7a00 + (n * hd) as u64)))
        .collect();
    let mut held_new: Vec<Vec<f32>> = held_q.iter().map(|q| vec![0.0; q.len()]).collect();
    let mut held_old: Vec<Vec<f32>> = held_q.iter().map(|q| vec![0.0; q.len()]).collect();

    let mut checked = 0usize;
    for (si, &(head_dim, n_head, n_head_kv, slab_pos)) in shapes.iter().enumerate() {
        let depths: Vec<usize> = if slab_pos == 32769 {
            vec![127, 128, 129, 1000, 4095, 32767, 32768, 32769]
        } else if slab_pos == 1000 {
            vec![1000]
        } else {
            vec![129, 1000, 4097]
        };
        let (k, v) = &slabs[si];
        for (ri, &n_q) in rows.iter().enumerate() {
            let at = if head_dim == 256 { ri } else { rows.len() + ri };
            for &n_pos in depths.iter().chain(std::iter::once(&n_q.max(2))) {
                if n_pos < n_q || n_pos > slab_pos {
                    continue;
                }
                let a = Attn {
                    q: &held_q[at],
                    k,
                    v,
                    kv_dim: n_head_kv * head_dim,
                    n_pos,
                    head_dim,
                    n_head,
                    n_head_kv,
                    scale: 1.0 / (head_dim as f32).sqrt(),
                };
                for (mask, into) in [(None, &mut held_new[at]), (Some(0), &mut held_old[at])] {
                    gpu.attn_vmma_dbg(None);
                    gpu.set_attn_vmma(true);
                    gpu.attn_vmma_dbg(mask);
                    gpu.begin_pass(n_q);
                    gpu.attend(&a, &mut into[..]);
                    gpu.host_needs(&mut into[..]);
                    gpu.end_pass();
                }
                if let Some(e) = gpu.take_error() {
                    panic!("cuda error at hd{head_dim} kv{n_head_kv} n_q {n_q} d{n_pos}: {e}");
                }
                let (worst, differing) = compare(&held_old[at], &held_new[at]);
                assert_eq!(
                    differing, 0,
                    "hd{head_dim} kv{n_head_kv} slab {slab_pos} n_q {n_q} d{n_pos}: {differing} of \
                     {} outputs differ from the old staging, worst {worst:e}. The same bytes are \
                     supposed to move. Suspect the lane-to-dimension map (lane l stages [8l, 8l+8)), \
                     the warp's position stride, the zeroed tail past n_pos_max, or the padded rows.",
                    held_new[at].len()
                );
                checked += held_new[at].len();
            }
        }
    }
    gpu.attn_vmma_dbg(None);
    gpu.set_attn_vmma(false);
    gpu.set_attn_mma(false);
    println!("  {checked} outputs compared with the old staging, all bit-identical");
}

/// **The tensor-core attention kernel against the oracle, inside a derived
/// bound.**
///
/// `attn_flash_mma_v` is a precision change, not a reordering: Q is cast to f16
/// before the score GEMM and the rescaled probabilities are cast to f16 before
/// the V GEMM. `HANDOFF.md` 08-09 settled that this is acceptable (1.3e-6 to
/// 3.4e-5 against the f32 path, and FlashAttention-3 finds FP16 attention *more*
/// accurate than a naive f32 one). This bounds it, in units of `max|v|` because
/// the output is a convex combination of values:
///
/// - **Q to f16** moves each element by at most half an f16 ulp, 2^-11 of
///   itself, so a score by at most `d = 2^-11 * scale * |q|_2 * |k|_2`
///   (Cauchy-Schwarz). Scores all moving by at most `d` move every softmax
///   weight by a factor inside `e^(+-2d)`, so the output by `(e^(2d) - 1)`.
/// - **P to f16** rounds a weight by at most 2^-11 of itself, or by half the
///   smallest subnormal step, 2^-25, once it falls below f16's normal range.
///   Over `n_pos` weights against a normaliser of at least 1 (the maximum
///   term), at most `2^-11 + n_pos * 2^-25`.
/// - The f32 accumulation both sides still do: `attend_tolerance`.
///
/// **A bound, and a loose one.** Measured error sits orders of magnitude under
/// it. It exists to catch defects that move outputs by a large fraction of
/// `max|v|` -- a wrong row, a wrong kv head, an unmasked window. It cannot see a
/// one-position slip in a deep, near-uniform softmax, which is why the staging
/// rewrite is guarded by bit-equality instead.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_tensor_core_attention_stays_inside_its_derived_bound() {
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    gpu.set_attn_vmma(true);

    const N_Q: usize = 17;
    // (head_dim, n_head, n_head_kv, depths)
    let shapes: [(usize, usize, usize, &[usize]); 2] =
        [(256, 16, 2, &[40, 300, 4096, 32769]), (128, 16, 8, &[40, 300, 4096])];

    // Every buffer for every shape is built up front and held to the end: the
    // backend keys its mirrors on host addresses, so a per-shape allocation
    // freed and recycled at a different kv_dim would be uploaded as the wrong
    // bytes -- the trap `the_warp_attention_agrees_with_the_oracle` records.
    struct Held {
        kf: Vec<f32>,
        vf: Vec<f32>,
        k: Vec<u16>,
        v: Vec<u16>,
        q: Vec<f32>,
        got: Vec<f32>,
    }
    let mut held: Vec<Held> = shapes
        .iter()
        .enumerate()
        .map(|(si, &(head_dim, n_head, n_head_kv, depths))| {
            let kv_dim = n_head_kv * head_dim;
            let max_pos = depths.iter().copied().max().unwrap_or(0);
            let kf = noise(max_pos * kv_dim, 0x6b00 + si as u64);
            let vf = noise(max_pos * kv_dim, 0x7600 + si as u64);
            let k = kf.iter().map(|&x| f32_to_f16(x)).collect();
            let v = vf.iter().map(|&x| f32_to_f16(x)).collect();
            let q = noise(N_Q * n_head * head_dim, 0x7100 + si as u64);
            let got = vec![0.0f32; q.len()];
            Held { kf, vf, k, v, q, got }
        })
        .collect();

    let mut worst_ratio = 0.0f64;
    for (si, &(head_dim, n_head, n_head_kv, depths)) in shapes.iter().enumerate() {
        let kv_dim = n_head_kv * head_dim;
        let Held { kf, vf, k, v, q, got } = &mut held[si];
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Largest query and key norms, and the largest value, allowing each the
        // f16 rounding the kernel's inputs carry.
        let f16_slack = 1.0 + 2f64.powi(-10);
        let q2 = q
            .chunks_exact(head_dim)
            .map(|r| r.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt())
            .fold(0.0, f64::max)
            * f16_slack;

        for &n_pos in depths {
            let k2 = kf[..n_pos * kv_dim]
                .chunks_exact(head_dim)
                .map(|r| r.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt())
                .fold(0.0, f64::max)
                * f16_slack;
            let vmax = vf[..n_pos * kv_dim].iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()))
                * f16_slack;

            let a = Attn { q: &q, k: &k, v: &v, kv_dim, n_pos, head_dim, n_head, n_head_kv, scale };
            let mut want = vec![0.0f32; q.len()];
            Naive.attend(&a, &mut want);

            gpu.begin_pass(N_Q);
            gpu.attend(&a, &mut got[..]);
            gpu.host_needs(&mut got[..]);
            gpu.end_pass();
            if let Some(e) = gpu.take_error() {
                panic!("cuda error at hd{head_dim} d{n_pos}: {e}");
            }

            let d = 2f64.powi(-11) * f64::from(scale) * q2 * k2;
            let bound = vmax * ((2.0 * d).exp_m1() + 2f64.powi(-11) + n_pos as f64 * 2f64.powi(-25))
                + f64::from(attend_tolerance(n_pos, &want));
            let worst = want
                .iter()
                .zip(got.iter())
                .fold(0.0f64, |m, (x, y)| m.max((f64::from(*x) - f64::from(*y)).abs()));
            worst_ratio = worst_ratio.max(worst / bound);
            println!(
                "  hd{head_dim} kv{n_head_kv} n_q {N_Q} d{n_pos:<6} worst {worst:.3e}  bound {bound:.3e}  ({:.1e} of it)",
                worst / bound
            );
            assert!(
                worst <= bound,
                "the tensor-core attention kernel is outside its derived bound at hd{head_dim} \
                 kv{n_head_kv} d{n_pos}: {worst:e} against {bound:e}. Precision alone cannot put it \
                 there; suspect the row a query tile reads, the kv head offset, the causal window \
                 of a padded row, or the sub-tile rescale."
            );
        }
    }
    gpu.set_attn_vmma(false);
    gpu.set_attn_mma(false);
    println!("  worst error is {worst_ratio:.2e} of its bound");
}

/// **What the tensor-core attention kernel is made of.**
///
/// `attn_flash_mma_v` stages K once per 16 query vectors where `attn_flash`
/// walks it once per vector, and issues a fraction of the instructions -- yet at
/// d32768 the two cost the same per (row, head, chunk) of work, ~117 ns. So the
/// kernel that should have been the fast one is stalled on something that is
/// neither load count nor instruction count, and anything built on its structure
/// would inherit the stall. This prices its pieces, the same way
/// `what_attention_is_made_of` priced `attn_flash`.
///
/// `saves` is against the real `mma_v`, and the nothing-removed copy is asserted
/// bit-identical to it before anything is timed. `attn_flash` is timed alongside
/// as the production reference.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn what_the_tensor_core_attention_is_made_of() {
    const HEAD_DIM: usize = 256;
    const N_HEAD: usize = 16;
    const N_HEAD_KV: usize = 2;
    const KV_DIM: usize = N_HEAD_KV * HEAD_DIM;
    const N_Q: usize = 512;
    const LAYERS: usize = 10;
    const MAX_POS: usize = 32768;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);

    let k: Vec<u16> = (0..MAX_POS * KV_DIM).map(|i| ((i * 2654435761) >> 13) as u16 & 0x3bff).collect();
    let v: Vec<u16> = (0..MAX_POS * KV_DIM).map(|i| ((i * 40503) >> 11) as u16 & 0x3bff).collect();
    let q = noise(N_Q * N_HEAD * HEAD_DIM, 7);
    let mut out = vec![0.0f32; N_Q * N_HEAD * HEAD_DIM];
    let mut reference = vec![0.0f32; N_Q * N_HEAD * HEAD_DIM];

    // (label, decomposed mma_v mask, tensor cores: 0 = attn_flash, 2 = mma_v)
    // The last two are not removals: they move the same bytes a faster way, so
    // they are held to bit-equality with `mma_v` like the nothing-removed copy.
    let arms: [(&str, Option<i32>, u8); 14] = [
        ("mma_v", None, 2),
        ("attn_flash (production)", None, 0),
        ("dbg copy, nothing removed", Some(0), 2),
        ("no query staging", Some(1), 2),
        ("no K staging", Some(2), 2),
        ("no score GEMM", Some(4), 2),
        ("no softmax", Some(8), 2),
        ("no V staging", Some(16), 2),
        ("no V GEMM", Some(32), 2),
        ("no K or V staging", Some(18), 2),
        ("no GEMMs", Some(36), 2),
        ("floor: all removed", Some(63), 2),
        ("16-byte K and V staging", Some(192), 2),
        ("16-byte K/V + divless query", Some(448), 2),
    ];
    let configure = |vdbg: Option<i32>, cores: u8| {
        gpu.attn_vmma_dbg(None);
        gpu.set_attn_vmma(cores == 2);
        gpu.set_attn_mma(cores >= 1);
        gpu.attn_vmma_dbg(vdbg);
    };

    for n_pos in [8192usize, 32768] {
        let a = Attn {
            q: &q,
            k: &k,
            v: &v,
            kv_dim: KV_DIM,
            n_pos,
            head_dim: HEAD_DIM,
            n_head: N_HEAD,
            n_head_kv: N_HEAD_KV,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        };
        let run = |vdbg: Option<i32>, cores: u8, into: &mut Vec<f32>| {
            configure(vdbg, cores);
            gpu.begin_pass(N_Q);
            gpu.attend(&a, into);
            gpu.host_needs(into);
            gpu.end_pass();
        };
        run(None, 2, &mut reference);
        for mask in [0, 192, 448] {
            run(Some(mask), 2, &mut out);
            let differing =
                reference.iter().zip(&out).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(
                differing, 0,
                "d{n_pos}: mask {mask} differs from mma_v in {differing} outputs; it is meant \
                 to move the same bytes, so this is a staging bug"
            );
        }
        assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error at d{n_pos}");

        let mut best = [f64::MAX; 14];
        for _ in 0..3 {
            for (ai, &(_, vdbg, cores)) in arms.iter().enumerate() {
                run(vdbg, cores, &mut out);
                const REPS: u32 = 2;
                let t = std::time::Instant::now();
                gpu.begin_pass(N_Q);
                for _ in 0..REPS {
                    gpu.attend(&a, &mut out);
                }
                gpu.end_pass();
                gpu.host_needs(&mut out);
                best[ai] = best[ai].min(t.elapsed().as_secs_f64() * 1e3 / f64::from(REPS));
            }
            assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error at d{n_pos}");
        }
        configure(None, 0);

        let base = best[0];
        println!(
            "\nattn_flash_mma_v decomposed, n_q {N_Q}, d{n_pos}: mma_v {base:.1} ms/call, \
             {:.3} ms/token over {LAYERS} layers",
            base * LAYERS as f64 / N_Q as f64
        );
        println!("  {:<30} {:>10} {:>10} {:>7}", "arm", "ms/call", "saves", "share");
        for (ai, &(label, _, _)) in arms.iter().enumerate() {
            let saves = base - best[ai];
            println!("  {label:<30} {:>10.2} {:>+10.2} {:>6.1}%", best[ai], saves, 100.0 * saves / base);
        }
    }
}

/// **What `attn_flash` is made of: each piece priced by leaving it out.**
///
/// Attention is linear in depth from d8192 to d65536 on every arm of
/// `which_buffer_bends_attention_at_depth`, at 4.6% of the fp32 cores, and the
/// one-kv-head arm reads half the bytes for the same cost -- so it is bound by
/// work, not traffic. Reading the kernel ranks its pieces by instruction count,
/// and that is exactly the kind of reading that sent two IQ4_XS attempts at
/// parts worth 2 us of 144. This prices them instead.
///
/// Each `dbg_attn_flash_*` kernel is a compile-time instance of one template
/// with one piece left out, so no arm can be quietly paying for work its branch
/// was supposed to remove. **The nothing-left-out instance is asserted
/// bit-identical to production** before anything is timed: an arm that differs
/// from the kernel it claims to decompose is measuring something else.
///
/// Read the `saves` column, not the times: it is production minus the arm, i.e.
/// what that piece costs. The pieces need not sum to the whole -- removing one
/// can change what the SMs do with the rest -- and the floor arm, with every
/// piece gone but the launch, the grid, the barriers between phases and the
/// partial stores, says how much is structure rather than arithmetic.
///
/// The last two arms are the tensor-core kernels, which rewrite the GEMMs
/// rather than removing anything. `attn_flash_mma_v` puts both on the tensor
/// cores and had never been measured; both round Q (and `mma_v` the
/// probabilities) to f16, so they carry a relative error rather than a bit check.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn what_attention_is_made_of() {
    const HEAD_DIM: usize = 256;
    const N_HEAD: usize = 16;
    const N_HEAD_KV: usize = 2;
    const KV_DIM: usize = N_HEAD_KV * HEAD_DIM;
    const N_Q: usize = 512;
    const LAYERS: usize = 10;
    const MAX_POS: usize = 32768;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);

    // Held for the whole test: mirrors key on the host address.
    let k: Vec<u16> = (0..MAX_POS * KV_DIM).map(|i| ((i * 2654435761) >> 13) as u16 & 0x3bff).collect();
    let v: Vec<u16> = (0..MAX_POS * KV_DIM).map(|i| ((i * 40503) >> 11) as u16 & 0x3bff).collect();
    let q = noise(N_Q * N_HEAD * HEAD_DIM, 7);
    let mut out = vec![0.0f32; N_Q * N_HEAD * HEAD_DIM];
    let mut reference = vec![0.0f32; N_Q * N_HEAD * HEAD_DIM];

    // (label, decomposed-kernel mask, tensor cores: 0 none, 1 score, 2 both)
    let arms: [(&str, Option<i32>, u8); 14] = [
        ("production", None, 0),
        ("dbg copy, nothing left out", Some(0), 0),
        ("no query load", Some(1), 0),
        ("no score phase", Some(2), 0),
        ("no max tree", Some(4), 0),
        ("max tree barriers only", Some(8), 0),
        ("no expf", Some(16), 0),
        ("no sum tree", Some(32), 0),
        ("no V phase", Some(64), 0),
        ("no combine lane-0 loop", Some(128), 0),
        ("no combine V sum", Some(256), 0),
        ("floor: all but barriers", Some(119 | 384), 0),
        ("mma: score GEMM", None, 1),
        ("mma_v: both GEMMs", None, 2),
    ];
    let configure = |dbg: Option<i32>, cores: u8| {
        gpu.attn_dbg(dbg);
        // `set_attn_vmma(true)` implies the score kernel, so it goes first.
        gpu.set_attn_vmma(cores == 2);
        gpu.set_attn_mma(cores >= 1);
    };

    for n_pos in [8192usize, 32768] {
        let a = Attn {
            q: &q,
            k: &k,
            v: &v,
            kv_dim: KV_DIM,
            n_pos,
            head_dim: HEAD_DIM,
            n_head: N_HEAD,
            n_head_kv: N_HEAD_KV,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        };

        // The construction check, before any time is taken.
        let run = |dbg: Option<i32>, cores: u8, into: &mut Vec<f32>| {
            configure(dbg, cores);
            gpu.begin_pass(N_Q);
            gpu.attend(&a, into);
            gpu.host_needs(into);
            gpu.end_pass();
        };
        run(None, 0, &mut reference);
        run(Some(0), 0, &mut out);
        let differing = reference.iter().zip(&out).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        assert_eq!(
            differing, 0,
            "d{n_pos}: the nothing-left-out copy differs from production in {differing} outputs, \
             so every decomposed arm would be timing a different kernel"
        );
        assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error at d{n_pos}");
        let mag = reference.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let mut rel = [0.0f32; 3];
        for cores in [1u8, 2] {
            run(None, cores, &mut out);
            let worst = reference.iter().zip(&out).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
            rel[cores as usize] = worst / mag.max(1e-30);
        }

        // Interleaved inside each round, so drift lands on every arm.
        let mut best = [f64::MAX; 14];
        for _ in 0..3 {
            for (ai, &(_, dbg, cores)) in arms.iter().enumerate() {
                run(dbg, cores, &mut out);
                const REPS: u32 = 2;
                let t = std::time::Instant::now();
                gpu.begin_pass(N_Q);
                for _ in 0..REPS {
                    gpu.attend(&a, &mut out);
                }
                gpu.end_pass();
                gpu.host_needs(&mut out);
                best[ai] = best[ai].min(t.elapsed().as_secs_f64() * 1e3 / f64::from(REPS));
            }
            assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error at d{n_pos}");
        }
        configure(None, 0);

        let prod = best[0];
        println!(
            "\nattn_flash decomposed, n_q {N_Q}, d{n_pos}: production {prod:.1} ms/call, \
             {:.3} ms/token over {LAYERS} layers",
            prod * LAYERS as f64 / N_Q as f64
        );
        println!("  {:<30} {:>10} {:>10} {:>7}", "arm", "ms/call", "saves", "share");
        for (ai, &(label, _, cores)) in arms.iter().enumerate() {
            let saves = prod - best[ai];
            let tail = if cores > 0 {
                format!("   {:.2}x, rel {:.2e}", prod / best[ai], rel[cores as usize])
            } else {
                String::new()
            };
            println!(
                "  {label:<30} {:>10.2} {:>+10.2} {:>6.1}%{tail}",
                best[ai],
                saves,
                100.0 * saves / prod
            );
        }
    }
}

/// Flash-decoding attention agrees with the oracle past d32768, on every
/// construction `which_buffer_bends_attention_at_depth` times.
///
/// **A cost bench that builds its own inputs needs a correctness check on the
/// same construction** -- `what_the_moe_ffn_costs` timed the wrong experts for
/// its whole life while the oracle test beside it used a different path.
/// `the_warp_attention_agrees_with_the_oracle` stops at d5000; this covers the
/// depths, the `qgroup` values and the one-kv-head shape the sweep uses.
///
/// The depths straddle what changes at 32768: `n_split` goes 256 -> 257 inside
/// a group of rows at 32769, and 65536 is the deepest the sweep reaches.
/// Seventeen query rows leave every `qgroup` a partial last group.
///
/// Real f16s of noise in [-1, 1) rather than the sweep's hashed bits, because
/// the derived tolerance only means something over values of sane magnitude.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_attention_agrees_with_the_oracle_past_32k() {
    const HEAD_DIM: usize = 256;
    const N_HEAD: usize = 16;
    const MAX_POS: usize = 65536;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    // The f32 `attn_flash` path is what this is about; the tensor-core
    // kernel is the prefill default.
    gpu.set_attn_vmma(false);
    gpu.set_attn_mma(false);

    let f16s = |n: usize, seed: u64| -> Vec<u16> { noise(n, seed).iter().map(|&x| f32_to_f16(x)).collect() };
    let k2 = f16s(MAX_POS * 2 * HEAD_DIM, 41);
    let v2 = f16s(MAX_POS * 2 * HEAD_DIM, 42);
    let k1 = f16s(MAX_POS * HEAD_DIM, 43);
    let v1 = f16s(MAX_POS * HEAD_DIM, 44);
    // One held query and one held output per row count: a shorter slice of a
    // longer buffer shares its address, and so would share its mirror.
    let q1 = noise(N_HEAD * HEAD_DIM, 45);
    let q17 = noise(17 * N_HEAD * HEAD_DIM, 46);
    let mut got1 = vec![0.0f32; q1.len()];
    let mut got17 = vec![0.0f32; q17.len()];

    let mut checked = 0usize;
    for n_pos in [32767usize, 32768, 32769, 65536] {
        for kv in [2usize, 1] {
            let (k, v) = if kv == 2 { (&k2, &v2) } else { (&k1, &v1) };
            for n_q in [1usize, 17] {
                let (q, got) = if n_q == 1 { (&q1, &mut got1) } else { (&q17, &mut got17) };
                let a = Attn {
                    q,
                    k,
                    v,
                    kv_dim: kv * HEAD_DIM,
                    n_pos,
                    head_dim: HEAD_DIM,
                    n_head: N_HEAD,
                    n_head_kv: kv,
                    scale: 1.0 / (HEAD_DIM as f32).sqrt(),
                };
                let mut want = vec![0.0f32; got.len()];
                Naive.attend(&a, &mut want);
                let tol = attend_tolerance(n_pos, &want);
                // Decode is a group of one whatever `qgroup` says.
                let groups: &[usize] = if n_q == 1 { &[8] } else { &[4, 8, 16] };
                for &g in groups {
                    gpu.set_qgroup(g);
                    gpu.begin_pass(n_q);
                    gpu.attend(&a, &mut got[..]);
                    gpu.host_needs(&mut got[..]);
                    gpu.end_pass();
                    close(&format!("attend d{n_pos} kv{kv} nq{n_q} g{g}"), &want, &got[..], tol);
                    checked += want.len();
                }
            }
        }
    }
    gpu.set_qgroup(8);
    if let Some(e) = gpu.take_error() {
        panic!("cuda error: {e}");
    }
    println!("  {checked} outputs within the derived tolerance past d32768");
}

/// The staged F32 matmul is bit-identical to the oracle at every shape the
/// model uses.
///
/// **The exactness here is load-bearing and was nearly given up on a false
/// premise.** `ffn_gate_inp` is F32 because the router decides *which* experts
/// run, so a one-ulp difference is not a rounding difference — it can pick a
/// different expert and change the answer categorically. `matmul_f32_t`'s
/// comment claimed its ~40 us floor could only be beaten by splitting the
/// reduction; the decomposition says the serial chain costs 1% and the loads
/// cost the rest, so staging through shared memory beats it 2.65x while thread
/// `j` still walks its own row in ascending `k`.
///
/// That argument is only as good as this test. Shapes are the real ones:
/// `ffn_gate_inp_shexp` is `{2048, 1}`, `ssm_alpha` and `ssm_beta` are
/// `{2048, 32}`, `ffn_gate_inp` is `{2048, 256}` — and 257 for the boundary,
/// which falls back to the row-per-thread kernel.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_staged_f32_matmul_is_bit_identical() {
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    let cpu = Naive;

    // Every weight buffer is held for the whole test. The backend caches its
    // transposed copy by host address, so a buffer dropped per shape can hand
    // the next shape a recycled address -- and until the cache checked sizes,
    // the previous shape's smaller upload, read past its end.
    let shapes = [(2048usize, 1usize), (2048, 32), (2048, 256), (2048, 257), (256, 8)];
    let held: Vec<Vec<u8>> = shapes
        .iter()
        .map(|&(n_in, n_out)| {
            let wf = noise(n_in * n_out, 0x51ed + n_out as u64);
            wf.iter().flat_map(|v| v.to_le_bytes()).collect()
        })
        .collect();
    for (&(n_in, n_out), bytes) in shapes.iter().zip(&held) {
        let w = Weights { data: bytes, ty: inferred_thoughts::gguf::GgmlType::F32, n_in, n_out, pooled: false };

        for n_tok in [1usize, 3] {
            let x = noise(n_in * n_tok, 0xbeef + n_tok as u64);
            let mut want = vec![0.0f32; n_out * n_tok];
            cpu.matmul(&w, &x, &mut want);

            let mut got = vec![0.0f32; n_out * n_tok];
            gpu.begin_pass(n_tok);
            gpu.host_wrote(&x);
            gpu.matmul(&w, &x, &mut got);
            gpu.host_needs(&mut got);
            gpu.end_pass();

            for (i, (g, w2)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w2.to_bits(),
                    "{{{n_in},{n_out}}} n_tok {n_tok}, row {i}: {g:e} against {w2:e}. \
                     The staged kernel accumulates serially in ascending k per row, \
                     exactly as naive::dot_row does, so any difference is a bug rather \
                     than a reordering."
                );
            }
        }
    }
    if let Some(e) = gpu.take_error() {
        panic!("cuda error: {e}");
    }
}

/// The F16 matmul is bit-identical to the oracle.
///
/// Qwen3.8-Flash-Next's 0.2B test model stores its QSA indexer projections as
/// F16 (`{256, 512}` and `{256, 128}`), and QSA's selection ranks blocks by
/// scores computed from them, so an ulp here can change which cells a query
/// sees. The weights are arbitrary f16 bit patterns rather than rounded noise,
/// so subnormals and both zeros are exercised; only the exponent-31 patterns
/// (inf and NaN), which no weight holds, are left out.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_f16_matmul_is_bit_identical() {
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    let cpu = Naive;

    // Held for the whole test: the backend caches the transposed weight by host
    // address (see `the_staged_f32_matmul_is_bit_identical`).
    let shapes = [(256usize, 512usize), (256, 128), (2560, 512), (2560, 129), (7, 1)];
    let held: Vec<Vec<u8>> = shapes
        .iter()
        .map(|&(n_in, n_out)| {
            let mut state = 0x16f0_0000u64 + n_out as u64;
            (0..n_in * n_out)
                .flat_map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let mut h = (state >> 33) as u16;
                    if h & 0x7c00 == 0x7c00 {
                        h &= !0x4000;
                    }
                    h.to_le_bytes()
                })
                .collect()
        })
        .collect();
    let mut outputs = Vec::new();
    for (&(n_in, n_out), bytes) in shapes.iter().zip(&held) {
        let w = Weights { data: bytes, ty: inferred_thoughts::gguf::GgmlType::F16, n_in, n_out, pooled: false };
        for n_tok in [1usize, 3] {
            let x = noise(n_in * n_tok, 0xf16 + n_tok as u64);
            let mut want = vec![0.0f32; n_out * n_tok];
            cpu.matmul(&w, &x, &mut want);

            let mut got = vec![0.0f32; n_out * n_tok];
            gpu.begin_pass(n_tok);
            gpu.host_wrote(&x);
            gpu.matmul(&w, &x, &mut got);
            gpu.host_needs(&mut got);
            gpu.end_pass();
            if let Some(e) = gpu.take_error() {
                panic!("cuda error at {{{n_in},{n_out}}} n_tok {n_tok}: {e}");
            }
            exact(&format!("matmul F16 {{{n_in},{n_out}}} n_tok {n_tok}"), &want, &got);
            outputs.push((x, got));
        }
    }

    // The negative case: the comparison above catches a weight read off by one
    // f16 code. The GPU gets a copy (its own address, so its own upload) in
    // which one weight is 8200 where the oracle's is 8192; that row, and only
    // that row, must differ in every token.
    let (n_in, n_out, n_tok, row) = (256usize, 512usize, 3usize, 5usize);
    let x = noise(n_in * n_tok, 0xf16f);
    let col = (0..n_in)
        .find(|&k| (0..n_tok).all(|t| x[t * n_in + k].abs() > 0.5))
        .expect("a column where every token's input is large");
    let mut good = held[0].clone();
    let at = (row * n_in + col) * 2;
    good[at..at + 2].copy_from_slice(&0x7000u16.to_le_bytes());
    let mut bad = good.clone();
    bad[at..at + 2].copy_from_slice(&0x7001u16.to_le_bytes());
    let wg = Weights { data: &good, ty: inferred_thoughts::gguf::GgmlType::F16, n_in, n_out, pooled: false };
    let wb = Weights { data: &bad, ty: inferred_thoughts::gguf::GgmlType::F16, n_in, n_out, pooled: false };
    let mut want = vec![0.0f32; n_out * n_tok];
    cpu.matmul(&wg, &x, &mut want);
    let mut got = vec![0.0f32; n_out * n_tok];
    gpu.begin_pass(n_tok);
    gpu.host_wrote(&x);
    gpu.matmul(&wb, &x, &mut got);
    gpu.host_needs(&mut got);
    gpu.end_pass();
    if let Some(e) = gpu.take_error() {
        panic!("cuda error in the negative case: {e}");
    }
    let differing: Vec<usize> = (0..n_out * n_tok).filter(|&i| want[i].to_bits() != got[i].to_bits()).collect();
    let expected: Vec<usize> = (0..n_tok).map(|t| t * n_out + row).collect();
    println!("  negative case: outputs {differing:?} differ, expected {expected:?}");
    assert_eq!(
        differing, expected,
        "one weight moved by one f16 code must change exactly its own output row in every          token; if nothing differs, the comparison cannot see a misread weight"
    );
}

/// The warp-per-position score phase agrees with the oracle at the depths it
/// actually runs at.
///
/// **`every_op_agrees_with_the_oracle` cannot cover this.** It attends over 96
/// positions, and the dispatch only reaches for `attn_flash_warp` at 768 and
/// above — so the kernel that serves every long session was untested by the
/// test whose whole job is to catch a wrong kernel.
///
/// The score phase changes the dot product's summation order: serial over
/// `head_dim` becomes eight serial terms per lane plus a five-level shuffle
/// tree. `attn_flash` was already outside the bit-exact set for `expf` and for
/// flash-decoding's per-chunk rescaling, and its tolerance is *derived* from
/// that decomposition. A tree over 32 accumulates error as O(log n) where a
/// serial walk of 256 is O(n), so this is **more** accurate, not less — which
/// is why the same `attend_tolerance` is used rather than a looser one. If that
/// reasoning is wrong this test says so.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_warp_attention_agrees_with_the_oracle() {
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    // The f32 `attn_flash` path is what this is about; the tensor-core
    // kernel is the prefill default.
    gpu.set_attn_vmma(false);
    gpu.set_attn_mma(false);

    // **One slab for every shape and depth, allocated once.**
    //
    // A fresh `Vec` per case is a trap, and it caught this test twice before it
    // caught anything else: the backend keys its KV mirror on the host address
    // and records how much of it has been uploaded, so a dropped buffer whose
    // allocation gets reused looks like the same cache with its contents
    // already on the device. Both times the failure landed on the *unchanged*
    // thread path, which is what gave the artifact away rather than a kernel.
    //
    // One live slab, sized for the widest `kv_dim`, is also the model's own
    // layout — one slab per layer, `n_pos` growing into it.
    const MAX_POS: usize = 5000;
    const MAX_KV_DIM: usize = 16 * 128;
    let q = noise(16 * 256, 21);
    let kf = noise(MAX_POS * MAX_KV_DIM, 22);
    let vf = noise(MAX_POS * MAX_KV_DIM, 23);
    let k: Vec<u16> = kf.iter().map(|&x| f32_to_f16(x)).collect();
    let v: Vec<u16> = vf.iter().map(|&x| f32_to_f16(x)).collect();

    // Both models' shapes. The 0.6B's (128, 16, 8) over a handful of positions
    // is what the whole-model differential test exercises, and covering only
    // the 35B's (256, 16, 2) at depth is how a wrong answer there reached a
    // commit.
    for &(head_dim, n_head, n_head_kv) in
        &[(256usize, 16usize, 2usize), (128, 16, 8), (128, 16, 16)]
    {
    let kv_dim = n_head_kv * head_dim;
    let q = &q[..n_head * head_dim];

    for n_pos in [1usize, 2, 5, 31, 96, 127, 128, 129, 768, 1024, 2048, 5000] {
    // **Batched query rows too.** A prefill attends several rows in one call,
    // each with its own causal window, and `attend_impl` launches them
    // separately — which is a path a decode-shaped test never reaches. The
    // whole-model differential test is a five-token prefill, so this is the
    // dimension that separates the two.
    for n_q in [1usize, 5] {
        if n_q > n_pos {
            continue;
        }
        let q = &q[..];
        let q = &q[..(n_head * head_dim).min(q.len())];
        let qb: Vec<f32> = (0..n_q).flat_map(|_| q.iter().copied()).collect();
        let attn = Attn {
            q: &qb,
            k: &k,
            v: &v,
            kv_dim,
            n_pos,
            head_dim,
            n_head,
            n_head_kv,
            scale: 1.0 / (head_dim as f32).sqrt(),
        };

        let mut want = vec![0.0; n_q * n_head * head_dim];
        Naive.attend(&attn, &mut want);
        let tol = attend_tolerance(n_pos, &want);

        for force in [Some(false), Some(true)] {
            gpu.attn_warp(force);
            let mut got = vec![0.0; n_q * n_head * head_dim];
            gpu.begin_pass(n_q);
            gpu.attend(&attn, &mut got);
            gpu.host_needs(&mut got);
            gpu.end_pass();
            let label = if force == Some(true) { "warp" } else { "thread" };
            close(
                &format!("attend {label} d{n_pos} hd{head_dim} kv{n_head_kv} nq{n_q}"),
                &want,
                &got,
                tol,
            );
        }
    }
    }
    }
    gpu.attn_warp(None);
    if let Some(e) = gpu.take_error() {
        panic!("cuda error: {e}");
    }
}

/// **The 35B's MoE forward pass, GPU against the oracle, on a batch.**
///
/// The gap this closes: `the_model_agrees_with_the_oracle_to_the_quantization_floor`
/// above runs the 0.6B, which has no experts, and the 35B's own standing check
/// runs on the CPU and greps for "paris". So nothing compared the routed FFN's
/// CUDA kernels against anything, at any batch shape — the expert matmuls, the
/// device top-k, the pointer gather and `moe_finish` were covered only by ops
/// tests on synthetic data and by a generation reading plausibly.
///
/// A batch is the shape that matters. Every one of those kernels indexes three
/// things that are all zero at `n_tok == 1`: the token's activation row, its
/// slice of the expert weights, and its shared-expert gate. `moe_glu` divides
/// `blockIdx.y` by `n_used` to recover the token, which at one token is always
/// zero however wrong the arithmetic is.
///
/// Not bit-equality, and deliberately: `Spin` and `Cuda` disagree by an `expf`
/// ulp in `softmax` and `silu_mul` — both of which the routed FFN uses, the
/// first on the router itself — and the RMSNorm tree adds its own. The bound is
/// `CLAUDE.md`'s quantization-amplification figure, the same one the 0.6B test
/// uses, and the companion evidence is that the CPU path is separately proven
/// bit-identical batched against stepwise by
/// `qwen35::batched_moe_prefill_equals_token_by_token`.
///
/// A routing error is what this is really for, and routing errors are not
/// small: picking one wrong expert of 256 changes the output categorically
/// rather than by an ulp, so the floor below is a sharp detector even though it
/// is not zero.
#[test]
#[ignore = "needs an sm_120 device and the real 35B"]
fn the_35b_moe_agrees_with_the_oracle_on_a_batch() {
    use inferred_thoughts::{Model, Spin};

    let Some(path) = common::find_model_named("Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf") else {
        println!("SKIPPED: no 35B found");
        return;
    };
    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let tokens = tk.encode(
        "The capital of France is Paris, and the capital of Japan is Tokyo, and the capital of",
        true,
        true,
    );
    assert!(tokens.len() >= 12, "prompt too short to exercise the batch");
    let n_ctx = tokens.len() + 4;

    let cpu = {
        let m = Model::load(&f).expect("load the 35B");
        let mut e = Engine::new(m, Spin::new(8), n_ctx, false);
        e.prefill(&tokens).expect("cpu prefill")
    };

    let gpu_logits = {
        let gpu = Cuda::new(0).expect("cuda device");
        let m = Model::load(&f).expect("load the 35B");
        let mut e = Engine::new(m, &gpu, n_ctx, false);
        let l = e.prefill(&tokens).expect("gpu prefill");
        assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");
        l
    };

    let (worst, differing) = compare(&cpu, &gpu_logits);
    let magnitude = cpu.iter().fold(0.0f32, |m: f32, &v: &f32| m.max(v.abs()));
    let relative = worst / magnitude;
    println!(
        "  logits          {differing:>6} of {:<6} differ   worst {worst:e}  \
         ({relative:e} of magnitude {magnitude:.3})",
        cpu.len()
    );
    println!(
        "  argmax          cpu {}  gpu {}",
        argmax(&cpu),
        argmax(&gpu_logits)
    );

    // The 35B is 40 layers against the 0.6B's 28, so the same per-layer floor
    // compounds further; the ceiling is scaled by the layer ratio rather than
    // re-fitted, so it stays derived rather than chosen to pass.
    let ceiling = 9.0e-2 * (40.0 / 28.0);
    assert!(
        relative <= ceiling,
        "GPU logits differ from the oracle by {relative:e} of magnitude, over the \
         {ceiling:e} that 40 layers of quantization amplification explain. A routing \
         error is the first thing to suspect — it changes the answer categorically \
         rather than by an ulp. Check moe_glu's token index, moe_finish's per-token \
         slices, and moe_gather_ptrs' token-major layout."
    );
    assert_eq!(
        argmax(&cpu),
        argmax(&gpu_logits),
        "the two backends disagree on the next token, which at this floor means a \
         routing or indexing defect rather than drift"
    );
}

/// `serve --warmup` must not change a single logit.
///
/// Where an expert lives changes which address a kernel reads, never what it
/// computes, so a warmed-up engine and a cold one must agree to the bit on the
/// same request, through prefill and through decode. That also proves
/// `Engine::reset` leaves nothing of the warm-up behind: a KV position or a
/// recurrent state carried over would move the logits by far more than an ulp.
///
/// **And it checks the warm-up actually moved something.** With no swaps the
/// comparison passes trivially, which is the first thing a test of a placement
/// change has to rule out.
#[test]
#[ignore = "needs an sm_120 device and the real 35B"]
fn the_serve_warmup_leaves_the_logits_bit_identical() {
    use inferred_thoughts::Model;
    use inferred_thoughts::serve::warm_up_experts;
    use inferred_thoughts::tok::chat::ChatMl;

    const DECODE: usize = 8;
    let Some(path) = common::find_model_named("Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf") else {
        println!("SKIPPED: no 35B found");
        return;
    };
    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let chat = ChatMl::detect(&tk, &f.metadata).expect("chat template");
    let tokens = tk.encode(
        &chat.wrap("Explain, in two sentences, why a CPU cache miss is expensive."),
        true,
        true,
    );
    // Room for the warm-up text as well as the request.
    let n_ctx = 4096;

    // One device context at a time: two would each size an expert slab from the
    // same free VRAM. `e` is declared after `gpu`, so it drops first.
    let run = |warm: bool| -> (Vec<Vec<f32>>, usize) {
        let gpu = Cuda::new(0).expect("cuda device");
        let m = Model::load(&f).expect("load the 35B");
        let mut e = Engine::new(m, &gpu, n_ctx, false);
        let swaps = if warm {
            let w = warm_up_experts(&mut e, &tk, &chat).expect("warm-up");
            println!(
                "  warm-up {} tokens in {:.1} s (placement included), {} swaps in {:.2} s",
                w.tokens, w.prefill_s, w.swaps, w.replace_s
            );
            w.swaps
        } else {
            0
        };
        let mut logits = vec![e.prefill(&tokens).expect("prefill")];
        for _ in 0..DECODE {
            let next = argmax(logits.last().expect("logits")) as u32;
            logits.push(e.decode(next).expect("decode"));
        }
        assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");
        (logits, swaps)
    };

    let (cold, _) = run(false);
    let (warm, swaps) = run(true);
    println!("  warm-up re-placed {swaps} experts");
    assert!(swaps > 0, "the warm-up moved no experts, so this comparison proves nothing");
    for (i, (a, b)) in cold.iter().zip(&warm).enumerate() {
        common::assert_bit_identical(a, b, &format!("logits after pass {i}"));
    }
}

/// Index of the largest logit; `Qwen3::argmax` is the 0.6B's and takes its own type.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// The same self-consistency check on CUDA, where the failure mode actually
/// lives.
///
/// `restoring_a_checkpoint_reproduces_the_continuation` in `qwen35.rs` runs on
/// `Spin`, where the recurrent state is the very slab the model writes and a
/// checkpoint cannot miss it. **On CUDA the device copy is authoritative** — the
/// state is written by kernels and never comes home on the forward path — so
/// this is the version that would actually catch a checkpoint saving zeros, or
/// a restore the device ignored because nobody invalidated its copy.
///
/// It also covers the fix that came out of the first server run: `forget_state`
/// bumps a generation rather than freeing the state buffers, so a restore
/// re-uploads into the allocation that already exists. If that ever regresses
/// to a stale pointer this fails rather than reading freed memory.
#[test]
#[ignore = "needs an sm_120 device and the real 9B"]
fn restoring_a_checkpoint_reproduces_the_continuation_on_the_device() {
    use inferred_thoughts::Model;

    let Some(path) = common::find_model_named("Qwen3.5-9B-Q8_0.gguf") else {
        println!("SKIPPED: no Qwen3.5-9B-Q8_0.gguf found");
        return;
    };
    let gpu = Cuda::new(0).expect("cuda device");
    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let tokens = tk.encode(
        "The capital of France is Paris, and the capital of Japan is Tokyo, and the capital of",
        true,
        true,
    );
    let split = tokens.len() / 2;

    let m = Model::load(&f).expect("load model");
    let mut e = Engine::new(m, &gpu, tokens.len() + 8, false);
    e.prefill(&tokens[..split]).expect("prefill the prefix");
    let cp = e.checkpoint().expect("the 9B has recurrent state");

    let first = e.prefill(&tokens[split..]).expect("continue");
    e.restore(&cp).expect("restore");
    let second = e.prefill(&tokens[split..]).expect("continue again");
    assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

    let differing = first
        .iter()
        .zip(&second)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differing, 0,
        "{differing} of {} logits differ after a device checkpoint round trip. The          recurrent state is the only thing a restore has to carry: check that          Ops::read_state brought it home before the copy, and that forget_state made          the device re-read the host slab afterwards.",
        first.len()
    );
}

/// **`matmul_pair` is bit-identical to the two matmuls it replaces.**
///
/// The merge changes which outputs share a launch and nothing else: each output
/// is still one thread walking `k` ascending over the same values in the same
/// order. So this demands equal bits against the oracle, not a tolerance —
/// which is the same argument that made batching, the warp matmul and the
/// grouped expert kernels free.
///
/// The shapes are the model's own pairs plus two boundary cases. `{2048,256}` +
/// `{2048,1}` is the router with the shared-expert gate, and it crosses the
/// 128-thread block: 257 rows is three blocks, so a thread in the last block
/// writes `out_b` while its neighbours in earlier blocks write `out_a`. That
/// split is the one thing this kernel does that `matmul_f32_t` does not, so it
/// is what the boundary rows are here to catch.
///
/// **The interleave is the other hazard.** The weight is column-major, so
/// merging along the output means `b`'s rows follow `a`'s *within every
/// super-block* — appending the two buffers instead would produce a different
/// tensor that still has the right shape and still runs. A wrong interleave
/// shows up as `out_b` being right and `out_a` wrong past row 0, or as both
/// drifting after the first super-block, which comparing every element catches
/// and comparing a norm would not.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_paired_f32_matmul_is_bit_identical() {
    use inferred_thoughts::gguf::GgmlType;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    let cpu = Naive;

    // (n_in, n_a, n_b): the model's two pairs, then a block boundary and a
    // case where `b` is wider than `a`.
    let cases = [
        (2048usize, 32usize, 32usize),
        (2048, 256, 1),
        (2048, 100, 28),
        (2048, 127, 2),
        (256, 8, 24),
    ];

    // Held for the whole test: `resident_f32_t_pair` caches on both host
    // pointers, and a per-case buffer that is dropped lets the allocator
    // recycle an address into a later case, which then reads the earlier
    // case's weights. This test failed once inside a full suite run and passed
    // alone; that is the shape of a defect which depends on what allocated
    // before it.
    let held: Vec<(Vec<u8>, Vec<u8>)> = cases
        .iter()
        .map(|&(n_in, n_a, n_b)| {
            let af = noise(n_in * n_a, 0x9a11 + n_a as u64);
            let bf = noise(n_in * n_b, 0x5b22 + n_b as u64);
            (
                af.iter().flat_map(|v| v.to_le_bytes()).collect(),
                bf.iter().flat_map(|v| v.to_le_bytes()).collect(),
            )
        })
        .collect();

    for (&(n_in, n_a, n_b), (abytes, bbytes)) in cases.iter().zip(&held) {
        let wa = Weights { data: abytes, ty: GgmlType::F32, n_in, n_out: n_a, pooled: false };
        let wb = Weights { data: bbytes, ty: GgmlType::F32, n_in, n_out: n_b, pooled: false };

        for n_tok in [1usize, 5] {
            let x = noise(n_in * n_tok, 0xc0de + n_tok as u64);
            let mut want_a = vec![0.0f32; n_a * n_tok];
            let mut want_b = vec![0.0f32; n_b * n_tok];
            cpu.matmul(&wa, &x, &mut want_a);
            cpu.matmul(&wb, &x, &mut want_b);

            let mut got_a = vec![0.0f32; n_a * n_tok];
            let mut got_b = vec![0.0f32; n_b * n_tok];
            gpu.begin_pass(n_tok);
            gpu.host_wrote(&x);
            gpu.matmul_pair(&wa, &wb, &x, &mut got_a, &mut got_b);
            gpu.host_needs(&mut got_a);
            gpu.host_needs(&mut got_b);
            gpu.end_pass();
            assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

            for (label, got, want) in
                [("a", &got_a, &want_a), ("b", &got_b, &want_b)]
            {
                for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "pair {{{n_in},{n_a}}}+{{{n_in},{n_b}}} n_tok {n_tok}: out_{label} \
                         element {i} is {g:e} against {w:e}. Merging two matmuls cannot \
                         change any accumulation, so suspect the interleave in \
                         resident_f32_t_pair (b's rows follow a's within every super-block, \
                         not after all of a) or the out_a/out_b split at row n_a."
                    );
                }
            }
        }
    }
}

/// **Which layer does the warp attention path break?**
///
/// A bisector, not an assertion. `the_warp_attention_agrees_with_the_oracle`
/// says the kernel is right at op level across 72 comparisons; the whole-model
/// test says the 0.6B answers a different token with it on. Both are true, so
/// the question is *where*.
///
/// **It does not compare intermediates, and that is deliberate.** The first
/// version of this test captured every tensor through `Ctx::trace` and reported
/// that only the final logits differed — which is impossible if an accumulation
/// order changed. `Ctx::trace` hands out *host* slices, and on this backend the
/// host copy is stale by design, so it was comparing buffers neither run wrote.
/// `CLAUDE.md`: never read a device result before `end_pass`.
///
/// So this uses the one value the model does bring home. `Cuda::attn_warp_only`
/// runs the warp phase for a single `attend` call per pass — one call is one
/// attending layer — and the logits are compared against an all-thread run.
/// A layer that breaks the answer names itself.
#[test]
#[ignore = "needs an sm_120 device and the 0.6B; a diagnostic, run with --nocapture"]
fn which_layer_does_the_warp_attention_break() {
    common::model_or_skip!(path);
    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let tokens = tk.encode("The capital of France is", true, true);

    let run = |only: Option<usize>, force: Option<bool>| -> Vec<f32> {
        let gpu = Cuda::new(0).expect("cuda device");
        gpu.use_graphs(false);
        gpu.attn_warp(force);
        gpu.attn_warp_only(only);
        // The f32 `attn_flash` path is what this is about; the tensor-core
        // kernel is the prefill default.
        gpu.set_attn_vmma(false);
        gpu.set_attn_mma(false);
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, &gpu, tokens.len() + 4, false);
        let l = e.prefill(&tokens).expect("prefill");
        assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");
        l
    };

    let base = run(None, Some(false));
    let all = run(None, Some(true));
    let (w, n) = compare(&base, &all);
    let mag = base.iter().fold(0.0f32, |m: f32, &v| m.max(v.abs()));
    println!(
        "  all layers warp   {n:>6} of {} differ  worst {w:e}  rel {:e}  argmax {} vs {}",
        base.len(),
        w / mag,
        argmax(&base),
        argmax(&all)
    );

    println!("
  per-layer, warp on for one attend call only:");
    let mut culprits = Vec::new();
    for il in 0..28usize {
        let got = run(Some(il), None);
        let (worst, differ) = compare(&base, &got);
        let rel = worst / mag;
        let am = argmax(&got);
        if differ > 0 {
            println!(
                "    layer {il:>2}  {differ:>6} differ  worst {worst:e}  rel {rel:e}  argmax {am}"
            );
        }
        if rel > 9.0e-2 || am != argmax(&base) {
            culprits.push((il, rel, am));
        }
    }
    println!("
  layers whose own warp call moves the answer past the floor: {culprits:?}");
}

/// **The 2x2 that isolates the warp attention failure.**
///
/// `which_layer_does_the_warp_attention_break` showed the warp path does *not*
/// break a prefill with graphs off — argmax identical, 1.7e-2 of magnitude
/// against a 9e-2 floor, and no single layer moves it. But
/// `the_model_agrees_with_the_oracle_to_the_quantization_floor` fails by a
/// whole argmax, and it differs in two ways: it decodes twelve steps after the
/// prefill, and it runs with CUDA graphs on.
///
/// Two candidates, so vary both independently rather than together. The
/// previous session's bisection changed the harness and the code at the same
/// time and produced a comparison that "proved" two byte-identical branches
/// behaved differently.
#[test]
#[ignore = "needs an sm_120 device and the 0.6B; a diagnostic, run with --nocapture"]
fn what_the_warp_attention_needs_to_fail() {
    common::model_or_skip!(path);
    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let tokens = tk.encode("The capital of France is", true, true);

    let run = |warp: bool, graphs: bool, steps: usize| -> (Vec<f32>, bool) {
        let gpu = Cuda::new(0).expect("cuda device");
        gpu.use_graphs(graphs);
        gpu.attn_warp(Some(warp));
        // The f32 `attn_flash` path is what this is about; the tensor-core
        // kernel is the prefill default.
        gpu.set_attn_vmma(false);
        gpu.set_attn_mma(false);
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, &gpu, tokens.len() + steps + 4, false);
        let mut l = e.prefill(&tokens).expect("prefill");
        for _ in 0..steps {
            l = e.decode(Qwen3::argmax(&l)).expect("decode");
        }
        let active = gpu.graph_active();
        assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");
        (l, active)
    };

    println!("  {:<26} {:>12} {:>12} {:>8} {:>8}", "case", "rel", "differing", "argmax", "graph");
    for (label, graphs, steps) in [
        ("prefill only, no graphs", false, 0usize),
        ("prefill only, graphs", true, 0),
        ("+12 decode, no graphs", false, 12),
        ("+12 decode, graphs", true, 12),
    ] {
        let (base, _) = run(false, graphs, steps);
        let (warp, active) = run(true, graphs, steps);
        let (worst, differing) = compare(&base, &warp);
        let mag = base.iter().fold(0.0f32, |m: f32, &v| m.max(v.abs()));
        println!(
            "  {label:<26} {:>12e} {differing:>12} {:>4}/{:<4} {active:>8}",
            worst / mag,
            argmax(&base),
            argmax(&warp),
        );
    }
    println!("
  the ceiling the model test uses is 9.0e-2 of magnitude");
}

/// **Is the warp attention wrong, or is the test that blocks it?**
///
/// `what_the_warp_attention_needs_to_fail` shows the divergence needs decode
/// steps and is unaffected by CUDA graphs. But
/// `the_model_agrees_with_the_oracle_to_the_quantization_floor` drives decode
/// with `e.decode(Qwen3::argmax(&l))` — **each run picks its next token from
/// its own logits.** Prefill already differs by 1.7e-2 of magnitude, inside the
/// 9e-2 floor, and that is enough for one greedy choice to flip; after it the
/// two runs are processing different sequences and their logits are not
/// comparable at all.
///
/// `CLAUDE.md` records that reasoning as the reason Stage 5's acceptance
/// criterion was replaced: "an argmax would flip somewhere in 250 greedy
/// decisions whether or not the cache is correct — a failure could not
/// distinguish drift from a bug."
///
/// So drive both runs with the *same* fixed tokens. If they then agree within
/// the floor, the kernel is fine and the harness was the defect.
#[test]
#[ignore = "needs an sm_120 device and the 0.6B; run with --nocapture"]
fn the_warp_attention_agrees_when_both_runs_decode_the_same_tokens() {
    common::model_or_skip!(path);
    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let tokens = tk.encode("The capital of France is", true, true);
    // A fixed continuation, so neither run's own sampling can steer it.
    let fixed = tk.encode(" Paris, and the capital of Japan is Tokyo, and", false, true);

    let run = |warp: bool| -> Vec<f32> {
        let gpu = Cuda::new(0).expect("cuda device");
        gpu.attn_warp(Some(warp));
        // The f32 `attn_flash` path is what this is about; the tensor-core
        // kernel is the prefill default.
        gpu.set_attn_vmma(false);
        gpu.set_attn_mma(false);
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, &gpu, tokens.len() + fixed.len() + 4, false);
        let mut l = e.prefill(&tokens).expect("prefill");
        for &t in &fixed {
            l = e.decode(t).expect("decode");
        }
        assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");
        l
    };

    let base = run(false);
    let warp = run(true);
    let (worst, differing) = compare(&base, &warp);
    let mag = base.iter().fold(0.0f32, |m: f32, &v| m.max(v.abs()));
    let rel = worst / mag;
    let (a0, a1) = (
        base.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0,
        warp.iter().enumerate().fold((0usize, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0,
    );
    println!(
        "  {} decode steps, fixed tokens: {differing} of {} differ, worst {worst:e}, rel {rel:e}",
        fixed.len(),
        base.len()
    );
    println!("  argmax  thread {a0}  warp {a1}");

    assert_eq!(a0, a1, "the two paths disagree on the next token even with the sequence fixed");
    assert!(
        rel <= 9.0e-2,
        "warp attention differs by {rel:e} of magnitude over {} decode steps with the          sequence fixed, past the 9e-2 the quantization floor explains. This is the          version of the comparison that cannot be confounded by a flipped greedy          choice, so a failure here is the kernel.",
        fixed.len()
    );
}

/// **The token-tiled IQ4_XS matmul is bit-identical to the unbatched one.**
///
/// `matmul_iq4_xs_q8_k_batch` loads a superblock once for `IQ4_TOK` tokens
/// where `matmul_iq4_xs_q8_k` reloads it per token. That changes which outputs
/// share a weight load and nothing else: each token still walks `ibl` ascending
/// and folds the same eight lanes in the same order, so the sequence of f32
/// additions reaching its accumulator is unchanged. Equal bits, not a
/// tolerance — the same argument the Q8_0 pair already rests on.
///
/// Both sides run on the GPU here, against `Naive` as the third opinion. That
/// matters: comparing only batched-vs-oracle would pass a kernel that is wrong
/// in the same way the unbatched one is, and comparing only the two GPU paths
/// would pass two kernels that are wrong together.
///
/// The shapes are the 35B's dense IQ4_XS in-block widths, plus a tail case
/// where the last token tile is partial (`n_tok = 13` against `IQ4_TOK = 8`) —
/// which is the one thing the batched kernel does that the other cannot get
/// wrong, since it has no tile to clamp.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_batched_iq4_matmul_is_bit_identical() {
    use inferred_thoughts::gguf::GgmlType;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    // **The subject is `matmul_iq4_xs_q8_k_batch`, so say so.** The int8
    // tensor-core kernel is the default for a batch now, and without this the
    // test would pass while exercising a kernel it says nothing about — the
    // same way `batched_moe_prefill_equals_token_by_token` passed on the CPU
    // backend while claiming to cover the CUDA routed FFN.
    gpu.iq4_mma(false);
    let cpu = Naive;

    // IQ4_XS: 136 bytes per 256-weight superblock.
    let build = |n_in: usize, n_out: usize, seed: u64| -> Vec<u8> {
        let sb = n_in / 256;
        let mut w = vec![0u8; n_out * sb * 136];
        let mut x = seed | 1;
        for b in w.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x & 0xff) as u8;
        }
        // A plausible f16 scale per superblock, so the arithmetic is not
        // dominated by denormals.
        for r in 0..n_out {
            for k in 0..sb {
                let at = (r * sb + k) * 136;
                w[at] = 0x00;
                w[at + 1] = 0x38;
            }
        }
        w
    };

    // **Every weight buffer stays alive for the whole test.** `Cuda::resident`
    // caches device copies keyed on the *host address*, and its doc states the
    // precondition: callers are the mmap or a model-owned `Vec`, "so an address
    // is never recycled underneath us". A test that allocates and drops a
    // buffer per shape breaks exactly that — the allocator hands a later shape
    // the same address and the cache returns the earlier shape's weights.
    // Observed: three shapes passed and the fourth read -2.57e4 against
    // 1.65e4, which is not a rounding difference.
    let cases: Vec<(usize, usize)> =
        vec![(2048, 512), (2048, 2048), (512, 2048), (2048, 1024)];
    let held: Vec<Vec<u8>> = cases
        .iter()
        .map(|&(n_in, n_out)| build(n_in, n_out, 0x4711 + n_out as u64))
        .collect();

    for (&(n_in, n_out), bytes) in cases.iter().zip(&held) {
        let w = Weights { data: bytes, ty: GgmlType::Iq4Xs, n_in, n_out, pooled: false };

        for n_tok in [1usize, 2, 8, 13, 32] {
            let x = noise(n_in * n_tok, 0x2f1a + n_tok as u64);

            let mut want = vec![0.0f32; n_out * n_tok];
            cpu.matmul(&w, &x, &mut want);

            let mut got = vec![0.0f32; n_out * n_tok];
            gpu.begin_pass(n_tok);
            gpu.host_wrote(&x);
            gpu.matmul(&w, &x, &mut got);
            gpu.host_needs(&mut got);
            gpu.end_pass();
            assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

            // The unbatched kernel on the same data, one token at a time, which
            // is the path decode takes and therefore the reference the batched
            // form must not move away from.
            let mut per_token = vec![0.0f32; n_out * n_tok];
            for t in 0..n_tok {
                let xs = x[t * n_in..(t + 1) * n_in].to_vec();
                let mut one = vec![0.0f32; n_out];
                gpu.begin_pass(1);
                gpu.host_wrote(&xs);
                gpu.matmul(&w, &xs, &mut one);
                gpu.host_needs(&mut one);
                gpu.end_pass();
                per_token[t * n_out..(t + 1) * n_out].copy_from_slice(&one);
            }
            assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

            for (label, other) in [("unbatched GPU", &per_token), ("naive", &want)] {
                let differing = got
                    .iter()
                    .zip(other.iter())
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                if differing > 0 {
                    let (i, a, b) = got
                        .iter()
                        .zip(other.iter())
                        .enumerate()
                        .find(|(_, (a, b))| a.to_bits() != b.to_bits())
                        .map(|(i, (a, b))| (i, *a, *b))
                        .unwrap_or((0, 0.0, 0.0));
                    println!(
                        "  {{{n_in},{n_out}}} n_tok {n_tok} vs {label}: {differing} differ,                          first at {i}: {a:e} against {b:e} (ulps {})",
                        (a.to_bits() as i64 - b.to_bits() as i64).abs()
                    );
                }
                assert_eq!(
                    differing, 0,
                    "{{{n_in},{n_out}}} n_tok {n_tok}: {differing} of {} outputs differ from \
                     {label}. Tiling tokens cannot change an accumulation, so suspect the \
                     tail clamp (nt = n_tok - t0), the per-token activation offsets \
                     (t0 + u), or the grouping of d * xs * (ls - 32).",
                    got.len()
                );
            }
        }
    }
}

/// **The batched `ssm_conv` reproduces the token-by-token one, bit for bit.**
///
/// `the_gdn_ops_agree_with_the_oracle` covers `ssm_conv` at one token, which is
/// the shape decode uses and the shape the batched kernel never takes. The
/// batched form was written because the per-token loop issued **120,090
/// launches** on a 4,000-token 35B prefill — 7.8% of device time — and the
/// dependency it appeared to have was not real: the convolution is causal over
/// a fixed window, and only the state *shift* forced the ordering.
///
/// So the claim under test is that splitting output from state update changes
/// nothing. Equal bits against the same GPU kernels run one token at a time,
/// not a tolerance: the taps are summed oldest-first with the current sample
/// last in both, so no accumulation moves.
///
/// The state is checked as carefully as the output. A conv window advanced
/// twice, or written before every output has read it, produces plausible
/// numbers and would show up only as slow drift in generated text — which is
/// the failure `qwen35`'s own batched-prefill test exists to catch on the CPU
/// side, and which nothing was catching here.
///
/// `n_tok = 5` with `keep = 3` is the case where the new window spans both the
/// incoming state and the batch; `n_tok = 2` is the case where it is mostly
/// state.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_batched_ssm_conv_matches_token_by_token() {
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);

    for kernel in [4usize, 2] {
        let keep = kernel - 1;
        let nc = 1024usize;
        let w = noise(nc * kernel, 0x5c0 + kernel as u64);

        for n_tok in [1usize, 2, 5, 8, 17] {
            let x = noise(nc * n_tok, 0x9e1 + n_tok as u64);
            let seed = noise(nc * keep, 0x33a + kernel as u64);

            // **`forget_state` before each run, and it is not a formality.**
            // `state_resident` caches device slabs keyed on the host address,
            // so a `Vec` dropped at the end of one sub-case hands the next one
            // a recycled address and the *previous* sub-case's device state.
            // Without this, tokens 0..keep-1 — exactly those whose window
            // reaches into the state — read someone else's history, and tokens
            // past the window still match, which is what the failure looked
            // like: 3 of 5 tokens wrong, the first 3.
            //
            // Third instance of this hazard in these tests today; `resident`
            // documents the precondition it rests on, and a test is the one
            // place short-lived buffers get made.
            let mut s_batch = seed.clone();
            let mut o_batch = vec![0.0f32; nc * n_tok];
            gpu.forget_state();
            gpu.begin_pass(n_tok);
            gpu.host_wrote(&x);
            gpu.ssm_conv(&mut s_batch, &x, &w, kernel, &mut o_batch);
            gpu.host_needs(&mut o_batch);
            gpu.end_pass();

            // The same kernels, one token at a time — the path decode takes.
            // A distinct buffer, so it gets its own device state: `state_resident`
            // keys on the host address, and these must not alias.
            let mut s_step = seed.clone();
            let mut o_step = vec![0.0f32; nc * n_tok];
            // Once, before the loop: inside it the device state must carry
            // forward from token to token, which is the whole point.
            gpu.forget_state();
            for t in 0..n_tok {
                let xt = x[t * nc..(t + 1) * nc].to_vec();
                let mut ot = vec![0.0f32; nc];
                gpu.begin_pass(1);
                gpu.host_wrote(&xt);
                gpu.ssm_conv(&mut s_step, &xt, &w, kernel, &mut ot);
                gpu.host_needs(&mut ot);
                gpu.end_pass();
                o_step[t * nc..(t + 1) * nc].copy_from_slice(&ot);
            }
            assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

            let differing = o_batch
                .iter()
                .zip(&o_step)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            if differing > 0 {
                for t in 0..n_tok {
                    let bad = (0..nc)
                        .filter(|&c| {
                            o_batch[t * nc + c].to_bits() != o_step[t * nc + c].to_bits()
                        })
                        .count();
                    println!(
                        "    kernel {kernel} n_tok {n_tok} token {t}: {bad}/{nc} differ                           batch {:e} step {:e}",
                        o_batch[t * nc], o_step[t * nc]
                    );
                }
            }
            assert_eq!(
                differing, 0,
                "kernel {kernel}, n_tok {n_tok}: {differing} of {} outputs differ from the \
                 per-token path. The window is a fixed causal span, so batching cannot \
                 change a sum — suspect the index mapping (p < 0 reads past[keep + p]) or \
                 the tap order.",
                o_batch.len()
            );

            // And the window the next pass inherits.
            gpu.read_state_into(&mut s_batch).expect("read batched state");
            gpu.read_state_into(&mut s_step).expect("read stepwise state");
            let sdiff = s_batch
                .iter()
                .zip(&s_step)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                sdiff, 0,
                "kernel {kernel}, n_tok {n_tok}: the state left behind differs in {sdiff} of \
                 {} values. `ssm_conv_state` must write the last `keep` samples of the batch, \
                 reading the old window for a batch shorter than the window.",
                s_batch.len()
            );
        }
    }
}

/// **Grouping the routed FFN by expert changes no bit.**
///
/// The claim the grouped kernels make in their own comments is that they change
/// which block computes an output and which weight loads are shared, never how
/// one output accumulates — every dot is still the arithmetic of
/// `dot_iq4_xs_warp` over the same bytes in the same order. That is a claim
/// about equal bits, so it is tested as one rather than under a tolerance.
///
/// # Why not the existing MoE differential
///
/// `the_35b_moe_agrees_with_the_oracle_on_a_batch` compares CUDA against
/// `Spin`, so its bound is the quantization floor rather than zero, and its
/// twenty-token prompt gives 160 (token, pick) pairs over 256 experts — nearly
/// every expert collects exactly one token, so the tile-packing loop that this
/// change is entirely about essentially never runs. It would pass with the
/// grouped path broken for any tile of more than one.
///
/// This prompt is long enough to fill whole `MOE_CHUNK`s: at 128 tokens a chunk
/// is 1,024 pairs and the mean expert collects four, which is the shape the
/// kernel was written for. The tail chunk is deliberately partial.
///
/// # One engine, two passes, and that is not incidental
///
/// The obvious form — two `Engine`s, one per path — is the bug this repo
/// learned three times in one day. `Cuda` keys `resident`, `resident_f32_t_pair`
/// and `state_resident` on **host addresses**, so dropping the first engine
/// lets the allocator hand its address to the second and the cache serves the
/// first pass's device data. `Engine::reset` is the supported way through: it
/// clears the KV cache and the recurrent state and calls `Ops::forget_state`,
/// and reusing one engine keeps every host buffer at the address its mirror was
/// built for. It also loads the 35B once rather than twice, which matters when
/// one copy is 13.4 GiB of a 16 GiB card.
#[test]
#[ignore = "needs an sm_120 device and the real 35B"]
fn the_grouped_routed_ffn_is_bit_identical() {
    use inferred_thoughts::Model;

    let Some(path) = common::find_model_named("Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf") else {
        println!("SKIPPED: no 35B found");
        return;
    };
    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");

    // Long enough for two full 128-token chunks and a partial third, so both
    // the packed path and the tail are exercised.
    let mut text = String::new();
    for i in 0..40 {
        text.push_str(
            "The capital of France is Paris, and the capital of Japan is Tokyo. \
             Routing depends on the token, so a varied prompt spreads the picks. ",
        );
        text.push_str(&i.to_string());
        text.push(' ');
    }
    let tokens = tk.encode(&text, true, true);
    assert!(
        tokens.len() > 300,
        "prompt is {} tokens; it must exceed one MOE_CHUNK of 128 for tiles to pack",
        tokens.len()
    );

    let gpu = Cuda::new(0).expect("cuda device");
    let m = Model::load(&f).expect("load the 35B");
    let n_ctx = tokens.len() + 4;
    let mut e = Engine::new(m, &gpu, n_ctx, false);

    gpu.moe_ungrouped(true);
    let per_pair = e.prefill(&tokens).expect("per-pair prefill");
    assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

    // Clears the KV cache and the recurrent state, and tells the device to
    // forget its copy of the latter. Without the `forget_state` inside it the
    // second pass would continue from the first pass's GatedDeltaNet state.
    e.reset();

    gpu.moe_ungrouped(false);
    gpu.iq4_mma(false);
    let grouped = e.prefill(&tokens).expect("grouped prefill");
    assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

    // And the same tiles through the int8 tensor cores. The MMA kernels are a
    // third implementation of the same arithmetic, so they must land on the
    // same bits as both of the others: the sub-block sum is integer and the
    // f32 fold outside it is unchanged.
    e.reset();
    gpu.iq4_mma(std::env::var("MMA_ARM").is_ok());
    let mma = e.prefill(&tokens).expect("mma prefill");
    gpu.iq4_mma(false);
    assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

    // **Did tiles actually pack?** Without this the test would pass cleanly
    // with every tile holding one token -- the case grouping exists to avoid,
    // and the one where the reuse loop never runs. The last chunk of the last
    // layer is a partial one, so compare against its own pair count rather than
    // a full MOE_CHUNK.
    let tail = tokens.len() % 128;
    let pairs_in_last_chunk = if tail == 0 { 128 } else { tail } * 8;
    let tiles = gpu
        .last_moe_tiles()
        .expect("a grouped launch should have recorded a tile count")
        as usize;
    println!(
        "  last chunk      {pairs_in_last_chunk} pairs into {tiles} tiles          ({:.2} pairs per weight load)",
        pairs_in_last_chunk as f64 / tiles as f64
    );
    assert!(
        tiles < pairs_in_last_chunk * 3 / 4,
        "{tiles} tiles for {pairs_in_last_chunk} pairs: tiles are not packing, so this          test is not exercising the reuse it exists to check. Expect roughly          min(n_expert, pairs) tiles."
    );

    assert_eq!(per_pair.len(), grouped.len(), "same vocabulary");
    let differing = per_pair
        .iter()
        .zip(&grouped)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let worst = per_pair
        .iter()
        .zip(&grouped)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    println!(
        "  grouped vs per-pair   {differing} of {} logits differ, worst {worst:e}",
        per_pair.len()
    );
    let mma_differing = per_pair
        .iter()
        .zip(&mma)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!(
        "  mma vs per-pair       {mma_differing} of {} logits differ",
        per_pair.len()
    );
    assert_eq!(
        mma_differing, 0,
        "{mma_differing} of {} logits differ between the tensor-core routed FFN and          the per-pair one. Suspect the m16n8k32 fragment layout, or a tile slot past          `nt` contributing something other than zero.",
        per_pair.len()
    );

    assert_eq!(
        differing, 0,
        "{differing} of {} logits differ between the grouped routed FFN and the \
         per-pair one, worst {worst:e}. These must be equal bits: grouping changes \
         which block computes an output and which weight loads it shares, never the \
         order anything accumulates in. Suspect the token index in \
         `matmul_iq4_xs_q8_k_moe_glu_grouped` (it divides a *pair* by `n_used`, while \
         the down matmul indexes `x` by the pair itself), the tail tile where \
         `nt < MOE_TOK`, or `moe_group`'s prefix sums.",
        per_pair.len()
    );
}

/// **The probe: IQ4_XS through `mma.m16n8k32.s8`, against the oracle, bit for bit.**
///
/// This settles a question the project has now got wrong three times — that
/// going faster must cost bit-exactness. It was assumed for the fast matmul,
/// for `__dp4a`, and for batching, and was false every time, because the
/// arithmetic had an exact decomposition hiding in it.
///
/// The decomposition here: the reference sums 32 int8 products per sub-block,
/// which is *integer* and therefore cannot round, so any split of it is exact.
/// One such sub-block is precisely one `k = 32` MMA tile. What must stay put is
/// the f32 chain outside it — `dh * s` accumulated over sub-blocks ascending —
/// and in an MMA tile a lane owns its output, so that chain lives in a register
/// rather than in eight shuffles to lane 0.
///
/// If this passes, the tensor cores are reachable *without* leaving the exact
/// set, and the fold and the scalar unpack that two traffic experiments failed
/// to move (8x for 1.54x, 5.17x for 1.12x) are addressable. If it fails, the
/// direction cost an hour.
///
/// Shapes are the 35B's real IQ4_XS widths. `n_tok` includes 13 and 21, which
/// are not multiples of the 8-token tile, because the tail is where an MMA
/// fragment layout goes wrong quietly: a column past `n_tok` must contribute
/// zero rather than garbage, and it must not be written back.
///
/// Every weight buffer is held for the whole test — `Cuda::resident` keys its
/// device copies on the host address, so a per-shape buffer that is dropped
/// hands the next shape a recycled address and the previous shape's weights.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_mma_iq4_matmul_is_bit_identical() {
    use inferred_thoughts::gguf::GgmlType;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    let cpu = Naive;

    // IQ4_XS: 136 bytes per 256-weight superblock, as `build` in
    // `the_batched_iq4_matmul_is_bit_identical`.
    let build = |n_in: usize, n_out: usize, seed: u64| -> Vec<u8> {
        let sb = n_in / 256;
        let mut w = vec![0u8; n_out * sb * 136];
        let mut x = seed | 1;
        for b in w.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x & 0xff) as u8;
        }
        for r in 0..n_out {
            for k in 0..sb {
                let at = (r * sb + k) * 136;
                w[at] = 0x00;
                w[at + 1] = 0x38;
            }
        }
        w
    };

    let cases: Vec<(usize, usize)> =
        vec![(2048, 512), (2048, 2048), (512, 2048), (2048, 1024)];
    let held: Vec<Vec<u8>> = cases
        .iter()
        .map(|&(n_in, n_out)| build(n_in, n_out, 0x9e37 + n_out as u64))
        .collect();

    let mut checked = 0usize;
    for (&(n_in, n_out), bytes) in cases.iter().zip(&held) {
        let w = Weights { data: bytes, ty: GgmlType::Iq4Xs, n_in, n_out, pooled: false };

        for n_tok in [2usize, 8, 13, 21, 32] {
            let x = noise(n_in * n_tok, 0x51c7 + n_tok as u64);

            let mut want = vec![0.0f32; n_out * n_tok];
            cpu.matmul(&w, &x, &mut want);

            gpu.iq4_mma(true);
            let mut got = vec![0.0f32; n_out * n_tok];
            gpu.begin_pass(n_tok);
            gpu.host_wrote(&x);
            gpu.matmul(&w, &x, &mut got);
            gpu.host_needs(&mut got);
            gpu.end_pass();
            gpu.iq4_mma(false);
            assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

            let differing = want
                .iter()
                .zip(&got)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let worst = want
                .iter()
                .zip(&got)
                .fold(0.0f32, |m: f32, (a, b)| m.max((a - b).abs()));
            println!(
                "  mma {n_in:>5}x{n_out:<5} n_tok {n_tok:<3} \
                 {differing:>7} of {:<7} differ   worst {worst:e}",
                want.len()
            );
            assert_eq!(
                differing, 0,
                "{differing} of {} outputs differ at {n_in}x{n_out}, n_tok {n_tok}, \
                 worst {worst:e}. The MMA path must equal the oracle bit for bit: the \
                 32-product sub-block sum is integer and so exact under any \
                 decomposition, and the f32 fold outside it is unchanged. Suspect the \
                 m16n8k32 fragment layout (A rows g and g+8, B column g, C at rows \
                 g/g+8 by columns 2q/2q+1), the nibble split (low nibbles are k<16, \
                 high nibbles k>=16 of the same four bytes), or the token tail.",
                want.len()
            );
            checked += want.len();
        }
    }
    println!("  {checked} outputs compared, all bit-identical");
}

/// **The staged tile against the oracle, bit for bit.**
///
/// `matmul_iq4_xs_q8_k_staged` computes the same products in the same order as
/// `matmul_iq4_xs_q8_k_mma`; only where the operands come from changes. So this
/// is not a hopeful tolerance test — it is the assertion that a pure
/// data-movement change moved no bits, which is the only thing that makes the
/// cost A/B beside it interpretable.
///
/// Storing `d` and `ls` separately in shared, rather than pre-folding
/// `d * (ls - 32)` as `load_tiles_iq4_xs` does, is what this depends on:
/// `c8131e5` measured that reassociation as outside the exact set.
///
/// The shapes exercise both tails. `n_out` 512, 1024 and 2048 are whole
/// multiples of the 64-row tile; 2048x1024 is not a multiple of 128, so a
/// grid whose last block is half empty gets covered. `n_tok` 13 and 21 are not
/// multiples of the 64-token tile, and a clamped token column must contribute
/// nothing that survives to a written-back output — the load clamps to
/// `n_tok - 1` rather than skipping, so a write-back that forgot its bound
/// would produce a *plausible duplicate* row rather than a crash.
///
/// Every weight buffer is held for the whole test: `Cuda` keys its device
/// copies on the host address, so a per-shape buffer that is dropped hands the
/// next shape a recycled address and the previous shape's weights.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_staged_iq4_matmul_is_bit_identical() {
    use inferred_thoughts::gguf::GgmlType;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    let cpu = Naive;

    // IQ4_XS: 136 bytes per 256-weight superblock, as `build` in
    // `the_mma_iq4_matmul_is_bit_identical`. The d bytes are fixed so the
    // values stay in range; the rest is noise.
    let build = |n_in: usize, n_out: usize, seed: u64| -> Vec<u8> {
        let sb = n_in / 256;
        let mut w = vec![0u8; n_out * sb * 136];
        let mut x = seed | 1;
        for b in w.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x & 0xff) as u8;
        }
        for r in 0..n_out {
            for k in 0..sb {
                let at = (r * sb + k) * 136;
                w[at] = 0x00;
                w[at + 1] = 0x38;
            }
        }
        w
    };

    let cases: Vec<(usize, usize)> =
        vec![(2048, 512), (2048, 2048), (512, 2048), (2048, 1024)];
    let held: Vec<Vec<u8>> = cases
        .iter()
        .map(|&(n_in, n_out)| build(n_in, n_out, 0x9e37 + n_out as u64))
        .collect();

    let mut checked = 0usize;
    for (&(n_in, n_out), bytes) in cases.iter().zip(&held) {
        let w = Weights { data: bytes, ty: GgmlType::Iq4Xs, n_in, n_out, pooled: false };

        for n_tok in [2usize, 8, 13, 21, 32, 64, 100] {
            let x = noise(n_in * n_tok, 0x51c7 + n_tok as u64);

            let mut want = vec![0.0f32; n_out * n_tok];
            cpu.matmul(&w, &x, &mut want);

            gpu.iq4_staged(true);
            let mut got = vec![0.0f32; n_out * n_tok];
            gpu.begin_pass(n_tok);
            gpu.host_wrote(&x);
            gpu.matmul(&w, &x, &mut got);
            gpu.host_needs(&mut got);
            gpu.end_pass();
            gpu.iq4_staged(false);
            assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

            let differing = want
                .iter()
                .zip(&got)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let worst = want
                .iter()
                .zip(&got)
                .fold(0.0f32, |m: f32, (a, b)| m.max((a - b).abs()));
            println!(
                "  staged {n_in:>5}x{n_out:<5} n_tok {n_tok:<4} \
                 {differing:>7} of {:<7} differ   worst {worst:e}",
                want.len()
            );
            assert_eq!(
                differing, 0,
                "{differing} of {} outputs differ at {n_in}x{n_out}, n_tok {n_tok}, \
                 worst {worst:e}. Staging is a data-movement change and must move no \
                 bits. Suspect the shared row stride (68 ints, so row g starts at bank \
                 4g), the staged nibble mapping (lane i takes qs + 4i, giving sub-block \
                 i>>2 and quad i&3, low nibbles k<16 and high nibbles k>=16 of the same \
                 four bytes), the clamped row or token tail, or a missing __syncthreads \
                 between the write of one superblock's stage and the read of the next.",
                want.len()
            );
            checked += want.len();
        }
    }
    println!("  {checked} outputs compared, all bit-identical");
}

/// **The deferred fold, against the oracle, inside a derived bound.**
///
/// `dbg_iq4_mma_foldonce` is the one IQ4_XS variant here that is *not*
/// bit-identical, so it needs the opposite kind of test: proof that it computes
/// the intended quantity and differs only by the rounding it was meant to
/// change. Without this, "1.11x for a fold reorder" is indistinguishable from
/// "1.11x for doing less work incorrectly".
///
/// # The bound, derived rather than fitted
///
/// Both kernels accumulate `(ls - 32) * s` — an integer product, exact — and
/// differ only in when they convert and fold. The reference does
/// `acc += (d * xs) * (ls_t - 32) * s_t` once per sub-block, so `8 * nb`
/// roundings; this does it once per superblock, so `nb`. The two sums are the
/// same real number, so the gap is bounded by the roundings neither shares:
/// about `8 * nb` units in the last place of the running total, relative.
///
/// At n_in 2048, nb is 8, so the bound is `64 * 2^-24`, about **3.8e-6**
/// relative. Ten times that is allowed here, because the comparison is against
/// the *final* f32 result whose own magnitude can be much smaller than the
/// partial sums that produced it — cancellation inflates relative error without
/// either kernel being wrong.
///
/// **Note the direction**: this kernel rounds eight times less often than the
/// reference, so where they differ it is the more accurate of the two. Same
/// shape of argument as the RMSNorm tree, and the same reason it is still a
/// decision rather than an obvious improvement — determinism is what is given
/// up, not accuracy.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_deferred_fold_stays_inside_its_derived_bound() {
    use inferred_thoughts::gguf::GgmlType;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    let cpu = Naive;

    let build = |n_in: usize, n_out: usize, seed: u64| -> Vec<u8> {
        let sb = n_in / 256;
        let mut w = vec![0u8; n_out * sb * 136];
        let mut x = seed | 1;
        for b in w.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x & 0xff) as u8;
        }
        for r in 0..n_out {
            for k in 0..sb {
                let at = (r * sb + k) * 136;
                w[at] = 0x00;
                w[at + 1] = 0x38;
            }
        }
        w
    };

    let cases: Vec<(usize, usize)> = vec![(2048, 2048), (512, 2048)];
    let held: Vec<Vec<u8>> = cases
        .iter()
        .map(|&(n_in, n_out)| build(n_in, n_out, 0x9e37 + n_out as u64))
        .collect();

    let mut worst_rel = 0.0f64;
    let mut worst_where = (0usize, 0usize, 0usize);
    for (&(n_in, n_out), bytes) in cases.iter().zip(&held) {
        let w = Weights { data: bytes, ty: GgmlType::Iq4Xs, n_in, n_out, pooled: false };
        let nb = n_in / 256;
        // 8 * nb roundings, one f32 ulp each, times ten for cancellation.
        let bound = 10.0 * (8 * nb) as f64 * f64::from(f32::EPSILON);

        for n_tok in [8usize, 32, 64] {
            let x = noise(n_in * n_tok, 0x51c7 + n_tok as u64);

            let mut want = vec![0.0f32; n_out * n_tok];
            cpu.matmul(&w, &x, &mut want);

            gpu.iq4_mma(true);
            gpu.iq4_fold_once(true);
            let mut got = vec![0.0f32; n_out * n_tok];
            gpu.begin_pass(n_tok);
            gpu.host_wrote(&x);
            gpu.matmul(&w, &x, &mut got);
            gpu.host_needs(&mut got);
            gpu.end_pass();
            gpu.iq4_fold_once(false);
            gpu.iq4_mma(false);
            assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

            // Scaled by the row's magnitude, not by each element, so an output
            // that cancels to near zero does not read as a huge relative error.
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            let mut differing = 0usize;
            for (a, b) in want.iter().zip(&got) {
                if a.to_bits() != b.to_bits() {
                    differing += 1;
                }
                let rel = f64::from((a - b).abs()) / f64::from(scale);
                if rel > worst_rel {
                    worst_rel = rel;
                    worst_where = (n_in, n_out, n_tok);
                }
            }
            println!(
                "  foldonce {n_in:>5}x{n_out:<5} n_tok {n_tok:<3} \
                 {differing:>7} of {:<7} differ   worst rel {worst_rel:e}   bound {bound:e}",
                want.len()
            );
            assert!(
                worst_rel <= bound,
                "deferred fold is outside its derived bound at {n_in}x{n_out}, n_tok {n_tok}: \
                 {worst_rel:e} against {bound:e}. The integer part is exact and order-free, so a \
                 miss here is not a rounding difference — suspect int32 overflow in `acci` \
                 (bounded by 8 * 31 * 32 * 127 * 127, well inside), the `ls - 32` bias having \
                 moved, or `xs` being read for the wrong token."
            );
        }
    }
    println!(
        "  worst relative difference {worst_rel:e} at {worst_where:?}, \
         and it is the *more* accurate side"
    );
}

/// **Q5_K on the tensor cores, against the oracle, inside a derived bound.**
///
/// This is the first matmul in this project that is *not* bit-identical, so it
/// needs the opposite kind of test from every other one here: proof that it
/// computes the intended quantity and differs only by the reordering it was
/// meant to introduce. "12x faster" and "12x faster because it is reading the
/// wrong bytes" are indistinguishable to a cost bench — `dbg_iq4_mma_foldonce`
/// measured a 1.11x speedup while dropping `+ t * 32` from its activation
/// pointer, and only a bound like this one caught it.
///
/// # The bound, derived rather than fitted
///
/// The reference keeps eight int32 lanes and feeds each its own f32 chain across
/// superblocks; this collapses them to one. **The integers are identical on both
/// sides** — an int32 sum of int8 products cannot round — so the difference is
/// entirely how many times the f32 running total is rounded: `8 * nb` against
/// `nb`. The gap is therefore bounded by about `8 * nb` units in the last place
/// of that total.
///
/// At n_in 4096, nb is 16, so the bound is `128 * 2^-24` ~ **7.6e-6** relative.
/// Ten times that is allowed here, because the comparison is against the final
/// f32 whose magnitude can be far smaller than the partial sums that made it —
/// cancellation inflates relative error without either side being wrong.
///
/// **Note the direction**: this rounds eight times *less* often than the
/// reference, so where they differ it is the more accurate of the two. Same
/// shape of argument as the RMSNorm tree, and the same reason it is a decision
/// rather than an obvious improvement — what is given up is determinism.
///
/// `attn_output` is 4096x2048 in the 35B, which is the first shape; the others
/// exercise a narrower `n_in` and a partial 128-row grid. `n_tok` 13, 21 and 100
/// are not multiples of the 32-token tile, and a padded column must contribute
/// nothing that survives to a written-back output.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_q5_k_mma_matmul_stays_inside_its_derived_bound() {
    use inferred_thoughts::gguf::GgmlType;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    let cpu = Naive;

    // Q5_K is 176 bytes per 256-weight superblock: f16 d, f16 dmin, 12 scale
    // bytes, 32 of qh, 128 of qs. d and dmin are pinned to a sane half so the
    // f32 chains stay in range; the rest is noise, which is what a differential
    // needs.
    let build = |n_in: usize, n_out: usize, seed: u64| -> Vec<u8> {
        let sb = n_in / 256;
        let mut w = vec![0u8; n_out * sb * 176];
        let mut x = seed | 1;
        for b in w.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x & 0xff) as u8;
        }
        for r in 0..n_out * sb {
            let at = r * 176;
            w[at] = 0x00;
            w[at + 1] = 0x38;
            w[at + 2] = 0x00;
            w[at + 3] = 0x34;
        }
        w
    };

    let cases: Vec<(usize, usize)> = vec![(4096, 2048), (2048, 2048), (512, 1024)];
    let held: Vec<Vec<u8>> = cases
        .iter()
        .map(|&(n_in, n_out)| build(n_in, n_out, 0x9e37 + n_out as u64))
        .collect();

    let mut worst_rel = 0.0f64;
    for (&(n_in, n_out), bytes) in cases.iter().zip(&held) {
        let w = Weights { data: bytes, ty: GgmlType::Q5K, n_in, n_out, pooled: false };
        let nb = n_in / 256;
        // 8 * nb roundings, one f32 ulp each, times ten for cancellation.
        let bound = 10.0 * (8 * nb) as f64 * f64::from(f32::EPSILON);

        for n_tok in [2usize, 8, 13, 21, 32, 100] {
            let x = noise(n_in * n_tok, 0x51c7 + n_tok as u64);

            let mut want = vec![0.0f32; n_out * n_tok];
            cpu.matmul(&w, &x, &mut want);

            let mut got = vec![0.0f32; n_out * n_tok];
            gpu.begin_pass(n_tok);
            gpu.host_wrote(&x);
            gpu.matmul(&w, &x, &mut got);
            gpu.host_needs(&mut got);
            gpu.end_pass();
            assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            let mut differing = 0usize;
            let mut here = 0.0f64;
            for (a, b) in want.iter().zip(&got) {
                if a.to_bits() != b.to_bits() {
                    differing += 1;
                }
                here = here.max(f64::from((a - b).abs()) / f64::from(scale));
            }
            worst_rel = worst_rel.max(here);
            println!(
                "  q5k mma {n_in:>5}x{n_out:<5} n_tok {n_tok:<4} \
                 {differing:>7} of {:<7} differ   worst rel {here:e}   bound {bound:e}",
                want.len()
            );
            assert!(
                here <= bound,
                "Q5_K on the tensor cores is outside its derived bound at \
                 {n_in}x{n_out}, n_tok {n_tok}: {here:e} against {bound:e}. The integer \
                 part is exact and order-free, so a miss here is not a rounding \
                 difference. Suspect the five-bit unpack (element t of sub-block sb is \
                 nibble sb&1 of qs[(sb>>1)*32 + t], lifted by bit sb of qh[t]), the \
                 m16n8k32 fragment order (a0/a2 are row g's k-halves, a1/a3 row g+8's, \
                 and Q5_K's halves are sixteen bytes apart rather than two nibbles of \
                 one byte), the twelve-byte scale/min shuffle, or the mins term, which \
                 is supposed to stay exact."
            );
        }
    }
    println!("  worst relative difference {worst_rel:e}, and it is the *more* accurate side");
}

/// **`INFERRED_Q5K_SCALAR` really buys bit equality back.**
///
/// The MMA path above is a deliberate departure from the oracle, and the whole
/// argument for taking it is that it is reversible. A flag that is documented to
/// restore determinism but does not is worse than no flag, so this asserts the
/// scalar arm is still bit-identical rather than trusting that nothing drifted
/// into it. Same test the RMSNorm tree carries for `--rms-serial`.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn q5k_scalar_restores_bit_equality() {
    use inferred_thoughts::gguf::GgmlType;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    gpu.q5k_mma(false);
    let cpu = Naive;

    let build = |n_in: usize, n_out: usize, seed: u64| -> Vec<u8> {
        let sb = n_in / 256;
        let mut w = vec![0u8; n_out * sb * 176];
        let mut x = seed | 1;
        for b in w.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x & 0xff) as u8;
        }
        for r in 0..n_out * sb {
            let at = r * 176;
            w[at] = 0x00;
            w[at + 1] = 0x38;
            w[at + 2] = 0x00;
            w[at + 3] = 0x34;
        }
        w
    };

    let bytes = build(4096, 2048, 0x9e37 + 2048);
    let w = Weights { data: &bytes, ty: GgmlType::Q5K, n_in: 4096, n_out: 2048, pooled: false };
    let mut checked = 0usize;
    for n_tok in [2usize, 13, 32] {
        let x = noise(4096 * n_tok, 0x51c7 + n_tok as u64);
        let mut want = vec![0.0f32; 2048 * n_tok];
        cpu.matmul(&w, &x, &mut want);

        let mut got = vec![0.0f32; 2048 * n_tok];
        gpu.begin_pass(n_tok);
        gpu.host_wrote(&x);
        gpu.matmul(&w, &x, &mut got);
        gpu.host_needs(&mut got);
        gpu.end_pass();
        assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

        let differing = want
            .iter()
            .zip(&got)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!("  q5k scalar 4096x2048 n_tok {n_tok:<3} {differing} of {} differ", want.len());
        assert_eq!(
            differing, 0,
            "{differing} of {} outputs differ with the MMA path off. The flag exists so \
             the tensor-core kernel's departure from the oracle is reversible, and a flag \
             that does not restore determinism is worse than no flag at all.",
            want.len()
        );
        checked += want.len();
    }
    gpu.q5k_mma(true);
    println!("  {checked} outputs compared with the scalar arm, all bit-identical");
}

/// **What each attention variant costs, at prefill shapes, in seconds.**
///
/// `what_attention_costs_as_context_grows` drives one query row, which is the
/// decode shape — it cannot see a change that is entirely about how many rows
/// share a launch. Without this, the only way to measure row grouping was a
/// whole-model prefill: ~90 s per data point, against a 2-4% run-to-run spread,
/// which is how an afternoon went into numbers that could not be interpreted.
///
/// Both arms run in one process against the same buffers, so there is no
/// rebuild, no model load, no expert placement and no drift between them.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn what_the_attention_variants_cost() {
    const HEAD_DIM: usize = 256;
    const N_HEAD: usize = 16;
    const N_HEAD_KV: usize = 2;
    const KV_DIM: usize = N_HEAD_KV * HEAD_DIM;
    const LAYERS: usize = 10;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    // The f32 `attn_flash` path is what this is about; the tensor-core
    // kernel is the prefill default.
    gpu.set_attn_vmma(false);
    gpu.set_attn_mma(false);

    let max_pos = 32768usize;
    let k: Vec<u16> = (0..max_pos * KV_DIM)
        .map(|i| ((i * 2654435761) >> 13) as u16 & 0x3bff)
        .collect();
    let v: Vec<u16> = (0..max_pos * KV_DIM)
        .map(|i| ((i * 40503) >> 11) as u16 & 0x3bff)
        .collect();

    // One query buffer per batch shape, all held for the whole test: `Cuda`
    // keys its mirrors on host addresses, so a buffer dropped between cases
    // hands the next one a recycled address and the previous case's data.
    // **n_q 512 is the shape the model actually runs.** `DEFAULT_MAX_BATCH` is
    // 512, so a prefill attends at 512 query rows and never at 128 -- and the
    // arms rank differently there, which cost a wrong conclusion on 09-09 when
    // the MMA path was dismissed on the n_q 128 and 256 rows. The 512 sweep
    // across depth is the one that decides anything; the rest is context.
    let shapes: Vec<(usize, usize)> = vec![(64, 2048), (128, 16384), (256, 32768),
             (512, 512), (512, 1024), (512, 2048), (512, 4096),
             (512, 8192), (512, 16384), (512, 32768), (17, 17)];
    let held: Vec<Vec<f32>> = shapes
        .iter()
        .map(|&(n_q, _)| noise(n_q * N_HEAD * HEAD_DIM, 7 + n_q as u64))
        .collect();

    println!("\nattention, prefill shapes, {N_HEAD}q/{N_HEAD_KV}kv x {HEAD_DIM}, x{LAYERS} layers");
    println!(
        "  {:>5} {:>7}  {:>10} {:>10}  {:>8}  {:>12}",
        "n_q", "n_pos", "split", "fused", "speedup", "ms/tok x10"
    );

    for (&(n_q, n_pos), q) in shapes.iter().zip(&held) {
        let a = Attn {
            q,
            k: &k,
            v: &v,
            kv_dim: KV_DIM,
            n_pos,
            head_dim: HEAD_DIM,
            n_head: N_HEAD,
            n_head_kv: N_HEAD_KV,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        };
        let mut out = vec![0.0f32; n_q * N_HEAD * HEAD_DIM];

        // Interleaved, so any drift lands on both arms equally.
        let mut best = [f64::MAX; 2];
        for _ in 0..3 {
            for (slot, fused) in [false, true].iter().enumerate() {
                gpu.set_attn_fused(*fused);
                for _ in 0..2 {
                    gpu.begin_pass(n_q);
                    gpu.attend(&a, &mut out);
                    gpu.host_needs(&mut out);
                    gpu.end_pass();
                }
                let t = std::time::Instant::now();
                const REPS: u32 = 5;
                for _ in 0..REPS {
                    gpu.begin_pass(n_q);
                    gpu.attend(&a, &mut out);
                    gpu.host_needs(&mut out);
                    gpu.end_pass();
                }
                let ms = t.elapsed().as_secs_f64() * 1e3 / REPS as f64;
                if ms < best[slot] {
                    best[slot] = ms;
                }
            }
        }
        gpu.set_attn_fused(true);
        assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

        // What ten attending layers cost per prompt token at this shape.
        let per_tok = best[1] * LAYERS as f64 / n_q as f64;
        // Third arm: the score matrix on the tensor cores. Timed the same way,
        // and checked against the split path -- it is a precision change, so
        // the interesting number is how far it moves, not whether it is equal.
        let mut reference = vec![0.0f32; n_q * N_HEAD * HEAD_DIM];
        gpu.set_attn_mma(false);
        gpu.set_attn_fused(false);
        gpu.begin_pass(n_q);
        gpu.attend(&a, &mut reference);
        gpu.host_needs(&mut reference);
        gpu.end_pass();

        gpu.set_attn_mma(true);
        let mut mma_out = vec![0.0f32; n_q * N_HEAD * HEAD_DIM];
        let mut mma_ms = f64::MAX;
        for _ in 0..3 {
            for _ in 0..2 {
                gpu.begin_pass(n_q);
                gpu.attend(&a, &mut mma_out);
                gpu.host_needs(&mut mma_out);
                gpu.end_pass();
            }
            let t = std::time::Instant::now();
            const R: u32 = 5;
            for _ in 0..R {
                gpu.begin_pass(n_q);
                gpu.attend(&a, &mut mma_out);
                gpu.host_needs(&mut mma_out);
                gpu.end_pass();
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / R as f64;
            if ms < mma_ms {
                mma_ms = ms;
            }
        }
        gpu.set_attn_mma(false);
        assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");
        let mag = reference.iter().fold(0.0f32, |m: f32, &v| m.max(v.abs()));
        let worst = reference
            .iter()
            .zip(&mma_out)
            .fold(0.0f32, |m: f32, (x, y)| m.max((x - y).abs()));
        let rel = worst / mag.max(1e-30);

        // Causal, so row r attends over `n_pos - n_q + 1 + r` positions. Two
        // MACs per (query, key, dim) — the score and the weighted sum — over
        // every head, and two flops per MAC.
        let n_pos_first = (n_pos + 1).saturating_sub(n_q) as f64;
        let windows = n_q as f64 * n_pos_first + (n_q * (n_q - 1) / 2) as f64;
        let flops = 4.0 * N_HEAD as f64 * windows * HEAD_DIM as f64;
        let tf = |ms: f64| flops / (ms * 1e-3) / 1e12;
        println!(
            "  {n_q:>5} {n_pos:>7}  {:>9.3}ms {:>9.3}ms  {:>7.2}x  {per_tok:>11.3}                mma {mma_ms:>8.3}ms {:>6.2}x  rel {rel:.2e}",
            best[0],
            best[1],
            best[0] / best[1],
            best[0] / mma_ms
        );
        // **Against the fp16 tensor ceiling, which is what the MMA arm could
        // reach; the split and fused arms run on the FP32 cores at 24 TFLOP/s.**
        // Printed on its own line so the arm comparison above stays readable.
        println!(
            "        {:>7.2} TFLOP-s split, {:>6.2} fused, {:>6.2} mma   ({:.1}% of the {MMA_F16_PEAK_TFLOPS:.0} fp16 ceiling)",
            tf(best[0]),
            tf(best[1]),
            tf(mma_ms),
            100.0 * tf(mma_ms) / MMA_F16_PEAK_TFLOPS,
        );
    }
}

/// **What the dense IQ4_XS matmul costs, at prefill shapes, in seconds.**
///
/// `the_mma_iq4_matmul_is_bit_identical` proves the kernel right and says
/// nothing about what it costs, and the only other instrument was a whole-model
/// prefill: ~90 s per point against a 2-4% run-to-run spread. That is how an
/// afternoon went into numbers too noisy to interpret, and it is the same gap
/// `what_the_attention_variants_cost` was built to close on the other kernel.
///
/// The number to watch is **GB/s of weight bytes**, not ms. This kernel's tile
/// decides how many times a weight row is re-read and re-unpacked across a
/// batch -- a block covers `128 rows x 8*MMA_NTILE tokens`, so at `MMA_NTILE` 4
/// every row crosses from global once per 32 tokens. Raising the token span
/// divides that traffic by the same factor, and this reports whether the card
/// notices.
///
/// # What it has already settled, all negative
///
/// Three changes were tried against it on 09-09 and none paid, which together
/// say the kernel is bound by none of the things that are cheap to change:
///
/// | change | traffic effect | result at n_tok 512 |
/// |---|---|---|
/// | `MMA_NTILE` 4 -> 16, a 128x128 tile | A read 4x less | -8% to +10%, a wash |
/// | `MMA_NROW` 2, one B fragment per two row blocks | B read 2x less | -6% to +13%, worse |
/// | hoisting the n-invariant half of the f32 fold | none | -1 to -2.5% |
///
/// **Run-to-run spread here is 1-4% at the large shapes**, so read the first two
/// rows as "no effect" rather than as small effects, and note the hoist is
/// barely outside the noise as well as outside the exact set -- reassociating
/// `(d * xs) * ls` into `(d * ls) * xs` changes the rounding.
///
/// The arithmetic that motivated the second row is worth keeping: at
/// 2048x8192, n_tok 512, the kernel issues 2.1M MMAs and reads 8.9 MB of
/// weights against **537 MB of activation fragments** -- the same 1 MB of
/// activations once per 16-row tile, 512 times over. Halving that changed
/// nothing, which leaves access *shape* rather than volume: both A and B arrive
/// as 16 bytes per lane scattered over eight rows. That is what llama.cpp's
/// shared-memory staging fixes, and it is not fixed by reducing volume.
///
/// Shapes are the 35B's real dense IQ4_XS widths against `n_embd` 2048, and the
/// token counts bracket `DEFAULT_MAX_BATCH`. Every weight buffer is held for the
/// whole test: `Cuda` keys its device copies on the host address, so a per-shape
/// buffer that is dropped hands the next shape a recycled address and the
/// previous shape's weights.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn what_the_dense_iq4_matmul_costs() {
    use inferred_thoughts::gguf::GgmlType;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);

    // IQ4_XS: 136 bytes per 256-weight superblock, as `build` in
    // `the_mma_iq4_matmul_is_bit_identical`. The d/scale bytes are fixed so the
    // values stay in range; the rest is noise, which is all a cost bench needs.
    let build = |n_in: usize, n_out: usize, seed: u64| -> Vec<u8> {
        let sb = n_in / 256;
        let mut w = vec![0u8; n_out * sb * 136];
        let mut x = seed | 1;
        for b in w.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x & 0xff) as u8;
        }
        for r in 0..n_out {
            for k in 0..sb {
                let at = (r * sb + k) * 136;
                w[at] = 0x00;
                w[at + 1] = 0x38;
            }
        }
        w
    };

    let shapes: Vec<(usize, usize)> =
        vec![(2048, 2048), (2048, 4096), (2048, 8192), (8192, 2048)];
    let held: Vec<Vec<u8>> = shapes
        .iter()
        .map(|&(n_in, n_out)| build(n_in, n_out, 0x9e37 + (n_out * n_in) as u64))
        .collect();
    // One activation buffer per token count, also held for the whole test.
    let toks: Vec<usize> = vec![32, 128, 512];
    let acts: Vec<Vec<f32>> = toks
        .iter()
        .map(|&t| noise(8192 * t, 0x51c7 + t as u64))
        .collect();

    println!("\ndense IQ4_XS matmul, prefill shapes");
    println!(
        "  ceiling {MMA_S8_PEAK_TOPS:.0} TOPS int8 / {MMA_F16_PEAK_TFLOPS:.0} TFLOP-s fp16 / 448 GB-s"
    );
    println!(
        "  {:>6} {:>6} {:>6}  {:>8}  {:>8}  {:>8}  {:>8}   {:>6} {:>6}  {:>7}",
        "n_in", "n_out", "n_tok", "with bus", "bare", "staged", "fold1x", "stg", "fold", "bare pk"
    );

    // **Both arms inside one shape loop, alternating**, because every result
    // this project retracted came from comparing runs an hour apart and every
    // one that survived came from interleaved arms. A drifting machine cancels
    // out of a difference taken this way and does not cancel out of two runs.
    //
    // The `mma ms` column is the canary: 2048x8192 at n_tok 512 reads ~2.0 ms
    // on a healthy machine and read 6.6 during the 09-09 fault.
    // `bus`: keep the `host_wrote` / `host_needs` pair, which is what the
    // caller does. `!bus`: drop both, so the timing is the launch and the
    // kernel with an explicit sync instead.
    //
    // **The two columns exist because seven kernel changes in a row measured
    // ~1.0x here**, and at these shapes `x` and `out` are each up to 16.8 MB a
    // rep. A bench whose fixed cost is larger than the thing it varies reports
    // 1.00x whatever the kernel does, and says nothing about why.
    let time_one = |w: &Weights, x: &[f32], out: &mut [f32], n_tok: usize, bus: bool| -> f64 {
        let run = |out: &mut [f32]| {
            gpu.begin_pass(n_tok);
            if bus {
                gpu.host_wrote(x);
            }
            gpu.matmul(w, x, out);
            if bus {
                gpu.host_needs(out);
            }
            gpu.end_pass();
            if !bus {
                let _ = gpu.sync();
            }
        };
        // Upload `x` once when the timed loop will not.
        if !bus {
            gpu.begin_pass(n_tok);
            gpu.host_wrote(x);
            gpu.end_pass();
        }
        let mut best = f64::MAX;
        for _ in 0..3 {
            for _ in 0..2 {
                run(out);
            }
            let t = std::time::Instant::now();
            const REPS: u32 = 10;
            for _ in 0..REPS {
                run(out);
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / REPS as f64;
            if ms < best {
                best = ms;
            }
        }
        best
    };

    for (&(n_in, n_out), bytes) in shapes.iter().zip(&held) {
        let w = Weights { data: bytes, ty: GgmlType::Iq4Xs, n_in, n_out, pooled: false };
        for (&n_tok, act) in toks.iter().zip(&acts) {
            let x = &act[..n_in * n_tok];
            let mut out = vec![0.0f32; n_out * n_tok];

            gpu.iq4_staged(false);
            gpu.iq4_mma(true);
            let mma = time_one(&w, x, &mut out, n_tok, true);
            let bare = time_one(&w, x, &mut out, n_tok, false);

            gpu.iq4_staged(true);
            let staged = time_one(&w, x, &mut out, n_tok, false);
            gpu.iq4_staged(false);

            // The third arm removes nothing from memory and everything but one
            // fold from between the MMAs. If the two above are washes and this
            // is not, the bound was never data movement.
            gpu.iq4_fold_once(true);
            let folded = time_one(&w, x, &mut out, n_tok, false);
            gpu.iq4_fold_once(false);
            gpu.iq4_mma(false);
            assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

            // **The number that says how much of the card is idle**, and the
            // one this bench existed without. Two ops per MAC, against a
            // ceiling that was measured rather than taken from a spec sheet.
            let ops = 2.0 * (n_in * n_out * n_tok) as f64;
            let pk = |ms: f64| 100.0 * (ops / (ms * 1e-3) / 1e12) / MMA_S8_PEAK_TOPS;
            println!(
                "  {n_in:>6} {n_out:>6} {n_tok:>6}  {mma:>8.4}  {bare:>8.4}  {staged:>8.4}  \
                 {folded:>8.4}   {:>5.2}x  {:>5.2}x  {:>6.1}%",
                bare / staged,
                bare / folded,
                pk(bare)
            );
        }
    }
}

/// **The token-tiled Q6_K matmul, against the oracle, bit for bit.**
///
/// `matmul_q6_k_q8_k_tok` holds an unpacked weight across `Q6K_TOK` tokens
/// where `matmul_q6_k_q8_k` re-unpacked it for each one. That is a claim about
/// *equal bits*, not a tolerance: each output still keeps the oracle's eight
/// interleaved f32 accumulators, in the same lane, folded in the same ascending
/// order. Only how many outputs one weight load serves changes.
///
/// Which is exactly why it is worth testing rather than asserting. The eight
/// accumulators are the reason Q6_K cannot go to the tensor cores bit-exactly,
/// and a tiling that quietly disturbed them would look like a small numeric
/// drift rather than a failure.
///
/// `n_tok` includes 13 and 21, which are not multiples of the 8-token tile,
/// because the tail is where this class of change goes wrong quietly: a padding
/// slot must contribute nothing and must not be written back. The kernel points
/// padding slots at token 0 rather than branching, so a write-back that forgot
/// its bound would produce a *plausible* duplicate row rather than a crash.
///
/// Every weight buffer is held for the whole test: `Cuda` keys its device
/// copies on the host address, so a per-shape buffer that is dropped hands the
/// next shape a recycled address and the previous shape's weights.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_tiled_q6_k_matmul_is_bit_identical() {
    use inferred_thoughts::gguf::GgmlType;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    let cpu = Naive;

    // Q6_K: 210 bytes per 256-weight superblock -- ql[128], qh[64],
    // scales[16] as int8, then d as f16. Transcribed from `block_q6_K` in
    // ggml-common.h, which is what `src/quant/kquant.rs` reads.
    let build = |n_in: usize, n_out: usize, seed: u64| -> Vec<u8> {
        let sb = n_in / 256;
        let mut w = vec![0u8; n_out * sb * 210];
        let mut x = seed | 1;
        for b in w.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x & 0xff) as u8;
        }
        for r in 0..n_out {
            for k in 0..sb {
                let at = (r * sb + k) * 210;
                // Scales are int8 and multiply an int16 product; keeping them
                // small keeps the int32 accumulation well clear of overflow, on
                // both sides of the comparison.
                for s in 0..16 {
                    w[at + 192 + s] = ((w[at + 192 + s] & 0x0f) as i8 - 8) as u8;
                }
                // d = 0.5 in f16, so the f32 fold stays in a sane range.
                w[at + 208] = 0x00;
                w[at + 209] = 0x38;
            }
        }
        w
    };

    let cases: Vec<(usize, usize)> = vec![(2048, 2048), (2048, 4096), (512, 1024)];
    let held: Vec<Vec<u8>> = cases
        .iter()
        .map(|&(n_in, n_out)| build(n_in, n_out, 0x6b17 + n_out as u64))
        .collect();

    let mut checked = 0usize;
    for (&(n_in, n_out), bytes) in cases.iter().zip(&held) {
        let w = Weights { data: bytes, ty: GgmlType::Q6K, n_in, n_out, pooled: false };

        for n_tok in [2usize, 8, 13, 21, 32] {
            let x = noise(n_in * n_tok, 0x33a1 + n_tok as u64);

            let mut want = vec![0.0f32; n_out * n_tok];
            cpu.matmul(&w, &x, &mut want);

            let mut got = vec![0.0f32; n_out * n_tok];
            gpu.begin_pass(n_tok);
            gpu.host_wrote(&x);
            gpu.matmul(&w, &x, &mut got);
            gpu.host_needs(&mut got);
            gpu.end_pass();
            assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

            let differing = want
                .iter()
                .zip(&got)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let worst = want
                .iter()
                .zip(&got)
                .fold(0.0f32, |m: f32, (a, b)| m.max((a - b).abs()));
            println!(
                "  q6k tok {n_in:>5}x{n_out:<5} n_tok {n_tok:<3} \
                 {differing:>7} of {:<7} differ   worst {worst:e}",
                want.len()
            );
            assert_eq!(
                differing, 0,
                "{differing} of {} outputs differ at {n_in}x{n_out}, n_tok {n_tok}, \
                 worst {worst:e}. The tiled Q6_K path must equal the oracle bit for \
                 bit: it reuses an unpacked weight across tokens and changes no \
                 accumulation order. Suspect the padding slots (they read token 0 and \
                 must not be written back), the per-token activation scale index, or \
                 the eight-lane fold.",
                want.len()
            );
            checked += want.len();
        }
    }
    println!("  {checked} outputs compared, all bit-identical");
}

/// **The batched delta rule against the per-token one, bit for bit.**
///
/// `the_gdn_ops_agree_with_the_oracle` drives one token, which is the decode
/// shape — it cannot reach `delta_rule_batch` at all, since that path is gated
/// on `n_tokens > 1`.
///
/// # Why this compares the device against itself
///
/// Comparing the batch against `Naive` bounds nothing useful: the gate goes
/// through `expf` and `logf`, CUDA is not obliged to round them as glibc does,
/// and the recurrence **compounds that difference once per token**. At two
/// tokens the state already sits at 6.7e-6 against 4.8e-6 for one, which is the
/// library disagreeing twice, not a defect — and at 129 tokens no honest fixed
/// tolerance separates the two cases.
///
/// Running both arms on the device removes the library from the comparison
/// entirely. Same `expf`, same order, same arithmetic; the only difference is
/// whether the token loop lives on the host or inside the kernel. That is a
/// claim about **equal bits**, and it is the claim actually being made.
///
/// # What it catches
///
/// - **the per-token strides**, which the host used to apply to the device
///   pointers and the kernel now derives itself. A wrong stride reads the wrong
///   token and still produces plausible numbers.
/// - **the state carry**, which used to be a launch boundary and is now a
///   `__syncthreads()`. Getting it wrong corrupts token `t+1`, not token `t`.
/// - **the staging barrier.** `qs`, `ks`, `g` and `beta` are rewritten every
///   iteration, so without the trailing sync a thread racing ahead overwrites
///   them while a slower one still reads. That is a race, so it fails
///   intermittently — hence a token count large enough to give it many chances.
///
/// The state is compared as well as the output: a kernel that returns the right
/// activations and leaves the wrong state behind passes any check of the output
/// alone, and then looks like drift several turns later.
///
/// Every per-token buffer is allocated up front and held. `Cuda` keys its
/// mirrors and its recurrent states on the **host address**, so a buffer
/// dropped between iterations hands the next one a recycled address and the
/// previous token's device data — the bug this repo has now found four times.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_batched_delta_rule_matches_the_per_token_one() {
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);

    // The 9B's real shapes.
    let (hk, hv, nk, nv) = (128usize, 128, 16, 32);
    let (kdim, vdim) = (hk * nk, hv * nv);

    for n_tok in [2usize, 7, 64] {
        // **`forget_state` first, as the ssm_conv batch test does.** Both state
        // slabs are fresh allocations of the same size every iteration, and
        // the backend keeps recurrent state on the device keyed by host
        // address, authoritative after its first upload. A slab that lands on
        // a recycled address would silently continue the previous iteration's
        // state instead of starting from `s0` -- which is what happened once
        // unrelated tests shifted the heap, as 28,672 outputs of garbage.
        gpu.forget_state();
        let q = noise(kdim * n_tok, 41 + n_tok as u64);
        let k = noise(kdim * n_tok, 42 + n_tok as u64);
        let v = noise(vdim * n_tok, 43 + n_tok as u64);
        let alpha = noise(nv * n_tok, 44 + n_tok as u64);
        let beta = noise(nv * n_tok, 45 + n_tok as u64);
        // ssm_a is -exp(A_log) upstream, so it is negative and the gate lands
        // inside (0, 1). A positive value would make the state explode and the
        // test would pass on garbage.
        let ssm_a: Vec<f32> = noise(nv, 46).iter().map(|x| -x.abs()).collect();
        let dt = noise(nv, 47);
        let s0 = noise(nv * hk * hv, 48);

        // --- arm A: one launch per token, the path this replaced. Owned
        // per-token buffers at distinct, stable addresses.
        let qs: Vec<Vec<f32>> = (0..n_tok).map(|t| q[t * kdim..(t + 1) * kdim].to_vec()).collect();
        let ks: Vec<Vec<f32>> = (0..n_tok).map(|t| k[t * kdim..(t + 1) * kdim].to_vec()).collect();
        let vs: Vec<Vec<f32>> = (0..n_tok).map(|t| v[t * vdim..(t + 1) * vdim].to_vec()).collect();
        let als: Vec<Vec<f32>> = (0..n_tok).map(|t| alpha[t * nv..(t + 1) * nv].to_vec()).collect();
        let bes: Vec<Vec<f32>> = (0..n_tok).map(|t| beta[t * nv..(t + 1) * nv].to_vec()).collect();
        let mut outs: Vec<Vec<f32>> = (0..n_tok).map(|_| vec![0.0f32; vdim]).collect();
        let mut s_seq = s0.clone();

        for t in 0..n_tok {
            let d1 = Delta {
                q: &qs[t], k: &ks[t], v: &vs[t],
                alpha: &als[t], beta: &bes[t], ssm_a: &ssm_a, dt_bias: &dt,
                head_k_dim: hk, head_v_dim: hv, n_k_heads: nk, n_v_heads: nv,
            };
            assert_eq!(d1.n_tokens(), 1, "arm A must take the single-token path");
            gpu.begin_pass(1);
            gpu.delta_rule(&d1, &mut s_seq, &mut outs[t]);
            gpu.host_needs(&mut outs[t]);
        }
        assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

        // --- arm B: one launch for the batch. A distinct state buffer, so it
        // gets its own device state rather than continuing arm A's.
        let d = Delta {
            q: &q, k: &k, v: &v,
            alpha: &alpha, beta: &beta, ssm_a: &ssm_a, dt_bias: &dt,
            head_k_dim: hk, head_v_dim: hv, n_k_heads: nk, n_v_heads: nv,
        };
        assert_eq!(d.n_tokens(), n_tok, "the seam derives the batch from v.len()");
        let mut s_bat = s0.clone();
        let mut o_bat = vec![0.0f32; vdim * n_tok];
        gpu.begin_pass(n_tok);
        gpu.delta_rule(&d, &mut s_bat, &mut o_bat);
        gpu.host_needs(&mut o_bat);
        assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

        let seq: Vec<f32> = outs.concat();
        let differing = seq
            .iter()
            .zip(&o_bat)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let worst = seq
            .iter()
            .zip(&o_bat)
            .fold(0.0f32, |m: f32, (a, b)| m.max((a - b).abs()));
        println!(
            "  delta batch vs per-token  n_tok {n_tok:<4} {differing} of {} differ, worst {worst:e}",
            seq.len()
        );
        assert_eq!(
            differing, 0,
            "{differing} of {} outputs differ between the batched delta rule and the \
             per-token one, worst {worst:e}. These must be equal bits: the batch runs \
             the same arithmetic on the same thread in the same order, and only moves \
             the token loop into the kernel. Suspect the per-token strides the kernel \
             now derives (kper, vper, and alpha/beta indexed by t * n_v_heads), or the \
             trailing __syncthreads() that the launch boundary used to provide.",
            seq.len()
        );

        gpu.read_state_into(&mut s_seq).expect("read arm A state back");
        gpu.read_state_into(&mut s_bat).expect("read arm B state back");
        let sdiff = s_seq
            .iter()
            .zip(&s_bat)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            sdiff, 0,
            "{sdiff} of {} recurrent state values differ. The output matched, so the \
             batch is reading the right tokens; what it leaves behind is wrong, which \
             would surface as drift several turns later rather than as a failure here.",
            s_seq.len()
        );
    }
}

/// **What the routed MoE FFN costs, at prefill shapes.**
///
/// The gap this closes: the MoE path is the project's whole subject and had
/// **no isolated bench at all**. Its share of prefill was known only from
/// `--profile-kernels`, whose per-launch sync distorts exactly the kernels with
/// many launches — and that profile has already misled this project twice.
///
/// # What makes MoE prefill hard, and why the shape matters
///
/// At batch 512 with top-8-of-256, each expert sees about **16 tokens**. So the
/// work is 256 separate GEMMs of roughly `16 x n_in x n_out` — tall, thin, and
/// with far too little reuse per weight byte to fill a tensor core. The
/// interesting quantity is therefore not ms but **the fraction of the card's
/// arithmetic reached**, which is why this reports TOPS against a measured
/// ceiling rather than a rate against nothing.
///
/// `moe_group` sorts (token, expert) pairs by expert so a tile can share one
/// weight load; the tile is `MOE_MMA_TOK` wide. Whether the tiles actually fill
/// is a function of `n_tok`, and that is the thing to watch across the rows.
///
/// The expert slab is capped at 2 GiB here. Left to itself it sizes from free
/// VRAM and would allocate ~11 GiB before doing any work, which a bench has no
/// use for.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn what_the_moe_ffn_costs() {
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    // 2 GiB against a 272 MiB pool, so nothing spills and every read is a VRAM
    // read. The optimistic arm; see `what_the_moe_ffn_costs_from_the_host_tier`
    // for the other end of the bracket.
    gpu.set_expert_budget(2 << 30);
    moe_ffn_cost_table(&gpu);
}

/// **The same table with the pool spilling to the host tier.**
///
/// `what_the_moe_ffn_costs` gives every expert a 2 GiB budget against a 272 MiB
/// pool, so every read is a VRAM read and the host tier is priced at **zero**.
/// That is the optimistic case and it is not the one the engine runs: a real
/// 5,679-token prefill resolves **5.5% of expert reads across PCIe**, inside
/// the kernel, at 28.2 MiB/token.
///
/// Here the slab is capped below the pool so placement has to spill, and the
/// `host` column says what fraction actually did. Read the two tests as a
/// bracket: the other is 0% host, this is well above the real 5.5%, and the
/// engine sits between them.
///
/// `bench_expert_residency` prices a single read from each tier — 190.8 GB/s
/// from VRAM against 16.7 in-kernel from the host, **11.4x** — and warns that
/// the cost is *not* linear in the host fraction. This measures the whole
/// kernel rather than one read, which is the quantity a placement policy should
/// be tuned against. Never tune residency by hit rate; see the residency cliff.
///
/// # It used to stop at n_tok 512, and that was a bug being worked around
///
/// n_tok 1024 with a host-resident pool reproduced the
/// `CUDA_ERROR_ILLEGAL_ADDRESS` that blocked this branch. The cause was in the
/// backend, not here: `moe_glu_impl` ignored `route.ids()` and `gather_ptrs`
/// read `slot::ROUTE_IDS`, which only `moe_topk` ever wrote — so this bench,
/// which builds a `Route::Host` by hand, was routing to whatever the slot held.
/// In range below 1024, hence the wrong experts and no failure; a fresh
/// uninitialised block once `pooled` had to grow it, hence the fault.
///
/// `stage_route_ids` fixes that, and the shape is back. **Every number this
/// bench produced before that fix was taken with degenerate routing**, which
/// packs tiles and fills the cache far better than the real thing.
///
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn what_the_moe_ffn_costs_from_the_host_tier() {
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    // Both budgets must be set before the first pooled tensor: the cache reads
    // them once, at construction. 96 MiB of slab against a 272 MiB pool leaves
    // roughly two thirds of it on the host tier.
    gpu.set_expert_host_budget(1 << 30);
    gpu.set_expert_budget(96 << 20);
    moe_ffn_cost_table(&gpu);
}

/// The shared body: the 35B routed FFN at prefill shapes, whatever tier its
/// experts landed on.
fn moe_ffn_cost_table(gpu: &Cuda) {
    use inferred_thoughts::gguf::GgmlType;
    use inferred_thoughts::ops::Route;

    // The 35B's routed FFN: n_embd 2048 in, expert FFN width 512 out, 256
    // experts per layer, 8 used per token.
    const N_IN: usize = 2048;
    const N_OUT: usize = 512;
    const N_EXPERT: usize = 256;
    const N_USED: usize = 8;

    // IQ4_XS: 136 bytes per 256-weight superblock, as elsewhere in this file.
    let build = |seed: u64| -> Vec<u8> {
        let sb = N_IN / 256;
        let mut w = vec![0u8; N_EXPERT * N_OUT * sb * 136];
        let mut x = seed | 1;
        for b in w.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x & 0xff) as u8;
        }
        for r in 0..N_EXPERT * N_OUT {
            for k in 0..sb {
                let at = (r * sb + k) * 136;
                w[at] = 0x00;
                w[at + 1] = 0x38;
            }
        }
        w
    };

    // Held for the whole test: `Cuda` keys residency on the host address.
    let gdata = build(0x9e37);
    let udata = build(0x51c7);
    let gate = Experts {
        data: &gdata, ty: GgmlType::Iq4Xs, n_in: N_IN, n_out: N_OUT, n_expert: N_EXPERT, scale: &[],
    };
    let up = Experts {
        data: &udata, ty: GgmlType::Iq4Xs, n_in: N_IN, n_out: N_OUT, n_expert: N_EXPERT, scale: &[],
    };

    println!("\nrouted MoE FFN (gate+up+silu), {N_EXPERT} experts, top-{N_USED}, {N_IN}x{N_OUT}");
    println!("  ceiling {MMA_S8_PEAK_TOPS:.0} TOPS int8");
    println!(
        "  {:>6} {:>7} {:>8}  {:>9}  {:>9}  {:>8}  {:>7}  {:>6}",
        "n_tok", "pairs", "tok/exp", "with bus", "bare", "TOPS", "of peak", "host"
    );

    let toks: Vec<usize> = vec![32, 128, 512, 1024];
    let acts: Vec<Vec<f32>> = toks.iter().map(|&t| noise(N_IN * t, 7 + t as u64)).collect();

    for (&n_tok, x) in toks.iter().zip(&acts) {
        // Deterministic pseudo-random routing, descending order per token as
        // the real router emits. Uniform, which is the *optimistic* case for
        // tile packing: real routing is skewed and packs worse.
        let mut ids = Vec::with_capacity(n_tok * N_USED);
        let mut weights = Vec::with_capacity(n_tok * N_USED);
        let mut r = 0x2545f491u64;
        for _ in 0..n_tok {
            let mut pick = Vec::new();
            while pick.len() < N_USED {
                r ^= r << 13;
                r ^= r >> 7;
                r ^= r << 17;
                let e = (r % N_EXPERT as u64) as usize;
                if !pick.contains(&e) {
                    pick.push(e);
                }
            }
            for (i, e) in pick.iter().enumerate() {
                ids.push(*e);
                weights.push(1.0 / (i + 1) as f32);
            }
        }
        let route = Route::Host { ids, weights, n_used: N_USED };

        let n_pair = n_tok * N_USED;
        let mut out = vec![0.0f32; n_pair * N_OUT];
        let mut scratch = vec![0.0f32; n_pair * N_OUT];

        // Two arms. `bus` keeps the `host_wrote` / `host_needs` pair; `!bus`
        // drops both and syncs instead.
        //
        // **The pair is a harness artifact, not this engine's PCIe cost.** The
        // model never ships activations across the bus — a 5,679-token prefill
        // does 592 uploads and 37 downloads in total — while at these shapes
        // `x` and `out` are megabytes a rep. On the dense IQ4_XS bench that pair
        // was **57% of the measurement**, which is why seven kernel experiments
        // there all read ~1.00x.
        //
        // What *is* architectural is on the other side of the table: expert
        // reads that resolve to the host tier, which the `host` column reports
        // and `what_the_moe_ffn_costs_from_the_host_tier` exercises.
        let mut timed = |bus: bool| -> f64 {
            let run = |out: &mut [f32], scratch: &mut [f32]| {
                gpu.begin_pass(n_tok);
                if bus {
                    gpu.host_wrote(x);
                }
                gpu.moe_glu(&gate, &up, &route, x, out, scratch);
                if bus {
                    gpu.host_needs(out);
                }
                gpu.end_pass();
                if !bus {
                    let _ = gpu.sync();
                }
            };
            if !bus {
                gpu.begin_pass(n_tok);
                gpu.host_wrote(x);
                gpu.end_pass();
            }
            let mut best = f64::MAX;
            for _ in 0..3 {
                for _ in 0..2 {
                    run(&mut out, &mut scratch);
                }
                let t = std::time::Instant::now();
                const REPS: u32 = 5;
                for _ in 0..REPS {
                    run(&mut out, &mut scratch);
                }
                let ms = t.elapsed().as_secs_f64() * 1e3 / REPS as f64;
                if ms < best {
                    best = ms;
                }
            }
            best
        };
        let with_bus = timed(true);
        let bare = timed(false);
        if let Some(e) = gpu.take_error() {
            let st = gpu.expert_stats();
            panic!(
                "a CUDA op reported a driver error at n_tok {n_tok}, n_pair {n_pair}: {e}\n\
                 expert cache: {st:?}"
            );
        }

        // gate and up are each n_pair x n_in x n_out MACs, two flops per MAC.
        let tops = 4.0 * (n_pair * N_IN * N_OUT) as f64 / (bare * 1e-3) / 1e12;
        let per_exp = n_pair as f64 / N_EXPERT as f64;
        // **Which tier the reads came from**, so a row cannot be read as the
        // kernel's cost when it is really the kernel's cost at 100% residency.
        let host_pct = gpu.expert_stats().map_or(0.0, |s| 100.0 * s.host_read_rate());
        println!(
            "  {n_tok:>6} {n_pair:>7} {per_exp:>8.1}  {with_bus:>9.4}  {bare:>9.4}  \
             {tops:>8.2}  {:>6.1}%  {host_pct:>5.1}%",
            100.0 * tops / MMA_S8_PEAK_TOPS
        );
    }
}

/// **What the scalar k-quant matmuls cost, at prefill shapes.**
///
/// Q6_K and Q5_K were ~22% of an 11k prefill and had no bench, so the only
/// thing known about them was a share from a profiler that distorts shares.
/// They are also the last matmuls still running one warp per output row with no
/// tensor cores at all — `attn_q` and `attn_output` on every attention block,
/// plus the LM head.
///
/// Read this against `what_the_dense_iq4_matmul_costs`. That kernel is on
/// `mma.m16n8k32.s8` and reaches ~4.3% of the int8 ceiling. Whatever these
/// reach is what a tile-loader rewrite would be starting from, and the ratio
/// between them is the size of the prize.
///
/// Shapes are the 35B's: `attn_q` is 2048 -> 4096 in Q6_K, `attn_output` is
/// 4096 -> 2048 in Q5_K, and `output.weight` is the 2048 -> 151936 LM head,
/// included at n_tok 1 because prefill lifts only the last row.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn what_the_k_quant_matmuls_cost() {
    use inferred_thoughts::gguf::GgmlType;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);

    // Q6_K is 210 bytes per 256-weight superblock, Q5_K is 176. Both keep d
    // (and dmin) at a sane f16 so the f32 chains stay in range; the rest is
    // noise, which is all a cost bench needs.
    let build = |ty: GgmlType, n_in: usize, n_out: usize, seed: u64| -> Vec<u8> {
        let (bytes, is_q6) = match ty {
            GgmlType::Q6K => (210usize, true),
            _ => (176usize, false),
        };
        let sb = n_in / 256;
        let mut w = vec![0u8; n_out * sb * bytes];
        let mut x = seed | 1;
        for b in w.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x & 0xff) as u8;
        }
        for r in 0..n_out {
            for k in 0..sb {
                let at = (r * sb + k) * bytes;
                if is_q6 {
                    for s in 0..16 {
                        w[at + 192 + s] = ((w[at + 192 + s] & 0x0f) as i8 - 8) as u8;
                    }
                    w[at + 208] = 0x00;
                    w[at + 209] = 0x38;
                } else {
                    w[at] = 0x00;
                    w[at + 1] = 0x38;
                    w[at + 2] = 0x00;
                    w[at + 3] = 0x38;
                }
            }
        }
        w
    };

    let cases: Vec<(GgmlType, usize, usize, &str)> = vec![
        (GgmlType::Q6K, 2048, 4096, "attn_q"),
        (GgmlType::Q5K, 4096, 2048, "attn_output"),
        (GgmlType::Q6K, 2048, 151936, "lm head"),
    ];
    let held: Vec<Vec<u8>> = cases
        .iter()
        .map(|&(ty, n_in, n_out, _)| build(ty, n_in, n_out, 0x6b17 + n_out as u64))
        .collect();
    let toks: Vec<usize> = vec![1, 128, 512];
    let acts: Vec<Vec<f32>> = toks.iter().map(|&t| noise(4096 * t, 31 + t as u64)).collect();

    println!("\nscalar k-quant matmuls, prefill shapes");
    println!("  ceiling {MMA_S8_PEAK_TOPS:.0} TOPS int8 (what the MMA path is measured against)");
    println!(
        "  {:>12} {:>6} {:>7} {:>6}  {:>10}  {:>9}  {:>7}",
        "what", "n_in", "n_out", "n_tok", "ms", "TOPS", "of peak"
    );

    for (&(ty, n_in, n_out, what), bytes) in cases.iter().zip(&held) {
        let w = Weights { data: bytes, ty, n_in, n_out, pooled: false };
        for (&n_tok, act) in toks.iter().zip(&acts) {
            // The LM head only ever runs on the lifted last row in prefill.
            if what == "lm head" && n_tok != 1 {
                continue;
            }
            let x = &act[..n_in * n_tok];
            let mut out = vec![0.0f32; n_out * n_tok];

            let mut best = f64::MAX;
            for _ in 0..3 {
                for _ in 0..2 {
                    gpu.begin_pass(n_tok);
                    gpu.host_wrote(x);
                    gpu.matmul(&w, x, &mut out);
                    gpu.host_needs(&mut out);
                    gpu.end_pass();
                }
                let t = std::time::Instant::now();
                const REPS: u32 = 10;
                for _ in 0..REPS {
                    gpu.begin_pass(n_tok);
                    gpu.host_wrote(x);
                    gpu.matmul(&w, x, &mut out);
                    gpu.host_needs(&mut out);
                    gpu.end_pass();
                }
                let ms = t.elapsed().as_secs_f64() * 1e3 / REPS as f64;
                if ms < best {
                    best = ms;
                }
            }
            assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

            let tops = 2.0 * (n_in * n_out * n_tok) as f64 / (best * 1e-3) / 1e12;
            println!(
                "  {what:>12} {n_in:>6} {n_out:>7} {n_tok:>6}  {best:>10.4}  {tops:>9.2}  {:>6.2}%",
                100.0 * tops / MMA_S8_PEAK_TOPS
            );
        }
    }
}

/// **The token-tiled Q5_K matmul, against the oracle, bit for bit.**
///
/// The sibling of `the_tiled_q6_k_matmul_is_bit_identical`, and the harder of
/// the two: Q5_K carries **two** f32 chains, not one. The eight interleaved
/// `sums[l]` accumulate the scaled products, while a separate `dmin` chain
/// accumulates `-dmin * sumi` inside the super-block loop, *before* the eight
/// lanes are folded in — and both updates are fused where Q6_K's are not.
/// Tiling has to keep all of that per token and in order.
///
/// So the failure this guards against is specific: getting the `dmin` chain
/// right for token 0 and wrong for the rest, which a single-token test cannot
/// see and which shows up as a small bias rather than as garbage.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_tiled_q5_k_matmul_is_bit_identical() {
    use inferred_thoughts::gguf::GgmlType;

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    // **Select the tiled arm explicitly.** `matmul_q5_k_q8_k_mma` is on by
    // default and takes this shape, and it is deliberately *not* bit-exact — so
    // without this the test would fail, and worse, silently stop covering the
    // kernel its name is about. The MMA path has its own derived-bound test.
    gpu.q5k_mma(false);
    let cpu = Naive;

    // Q5_K: 176 bytes per 256-weight superblock — d and dmin as f16, then 12
    // packed six-bit scale/min bytes, qh[32] as the fifth bit-plane, qs[128] as
    // nibbles. Transcribed from `block_q5_K` in ggml-common.h.
    let build = |n_in: usize, n_out: usize, seed: u64| -> Vec<u8> {
        let sb = n_in / 256;
        let mut w = vec![0u8; n_out * sb * 176];
        let mut x = seed | 1;
        for b in w.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = (x & 0xff) as u8;
        }
        for r in 0..n_out {
            for k in 0..sb {
                let at = (r * sb + k) * 176;
                // d and dmin = 0.5 in f16, so the two f32 chains stay in range.
                w[at] = 0x00;
                w[at + 1] = 0x38;
                w[at + 2] = 0x00;
                w[at + 3] = 0x38;
            }
        }
        w
    };

    let cases: Vec<(usize, usize)> = vec![(2048, 2048), (2048, 4096), (512, 1024)];
    let held: Vec<Vec<u8>> = cases
        .iter()
        .map(|&(n_in, n_out)| build(n_in, n_out, 0x5c19 + n_out as u64))
        .collect();

    let mut checked = 0usize;
    for (&(n_in, n_out), bytes) in cases.iter().zip(&held) {
        let w = Weights { data: bytes, ty: GgmlType::Q5K, n_in, n_out, pooled: false };

        for n_tok in [2usize, 8, 13, 21, 32] {
            let x = noise(n_in * n_tok, 0x7d0b + n_tok as u64);

            let mut want = vec![0.0f32; n_out * n_tok];
            cpu.matmul(&w, &x, &mut want);

            let mut got = vec![0.0f32; n_out * n_tok];
            gpu.begin_pass(n_tok);
            gpu.host_wrote(&x);
            gpu.matmul(&w, &x, &mut got);
            gpu.host_needs(&mut got);
            gpu.end_pass();
            assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

            let differing = want
                .iter()
                .zip(&got)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let worst = want
                .iter()
                .zip(&got)
                .fold(0.0f32, |m: f32, (a, b)| m.max((a - b).abs()));
            println!(
                "  q5k tok {n_in:>5}x{n_out:<5} n_tok {n_tok:<3} \
                 {differing:>7} of {:<7} differ   worst {worst:e}",
                want.len()
            );
            assert_eq!(
                differing, 0,
                "{differing} of {} outputs differ at {n_in}x{n_out}, n_tok {n_tok}, \
                 worst {worst:e}. The tiled Q5_K path must equal the oracle bit for \
                 bit. Suspect the per-token `dmin` chain (it must accumulate inside \
                 the super-block loop, fused, ahead of the eight-lane fold), the \
                 per-token `bsums` offset, or the padding slots.",
                want.len()
            );
            checked += want.len();
        }
    }
    println!("  {checked} outputs compared, all bit-identical");
}

/// **What the server'''s slice-and-checkpoint prefill costs: nothing.**
///
/// `serve` prefills in slices of `CHECKPOINT_EVERY` = 2048 and takes a
/// checkpoint after each — 60 device reads of the recurrent state, per slice —
/// where `generate` makes one `prefill` call and takes none. That looked like
/// an obvious suspect when a server turn measured 144.8 tok/s against
/// `generate`'''s 331.6 on byte-identical input.
///
/// **It was not the cause, and neither was anything in this repository.** The
/// machine had degraded: the same commit that measured 2.03 ms on the dense
/// IQ4_XS bench measured 6.6, llama.cpp fell 1042 -> 397 tok/s on the same
/// file, and a Windows restart restored both. A whole investigation ran against
/// a moving baseline because the comparison spanned it.
///
/// The result is kept because it is the answer to a question that will be asked
/// again — slicing 1.02x, checkpoints 0.94x, together 0.96x — and because a
/// cross-time comparison of two binaries is exactly the shape of measurement
/// this project keeps having to disown. Both arms here run in one process,
/// minutes apart, on one engine.
///
/// So this drives **one engine** three ways over the same tokens:
///
/// - one `prefill` of everything, which is what `generate` does
/// - 2048-token slices, which isolates slicing from checkpointing
/// - slices plus a checkpoint after each, which is what `serve` does
///
/// One engine rather than three, and `reset` between arms, for the reason
/// `the_grouped_routed_ffn_is_bit_identical` records: `Cuda` keys its mirrors on
/// host addresses, so a dropped engine hands the next one recycled addresses.
#[test]
#[ignore = "needs an sm_120 device and the real 35B"]
fn what_the_server_prefill_pattern_costs() {
    use inferred_thoughts::Model;

    let Some(path) = common::find_model_named("Qwen_Qwen3.6-35B-A3B-IQ4_XS.gguf") else {
        println!("SKIPPED: no 35B found");
        return;
    };
    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");

    // Real content, as `prompt_real.txt` is: word salad routes to a handful of
    // experts and would flatter every arm equally but unrealistically.
    let doc = std::fs::read_to_string("measurements/prompt_real.txt")
        .expect("measurements/prompt_real.txt");
    let tokens = tk.encode(&doc, true, true);
    let n = tokens.len();
    assert!(n > 8192, "{n} tokens; this needs several 2048-token slices");

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.set_model_path(&f.path);
    gpu.set_map_base(f.map_base());
    let m = Model::load(&f).expect("load the 35B");
    let mut e = Engine::new(m, &gpu, 32768, false);
    e.set_max_batch(512);
    // **Both binaries call this and the first version of this test did not.**
    // The expert slab sizes itself from free VRAM at the first expert, which is
    // inside the first forward pass; the KV slabs are allocated later, at the
    // first attention layer. Without the reservation the slab takes VRAM the KV
    // cache then needs, and WDDM demand-pages the difference - the 4.76x cliff
    // in BENCHMARKS.md, which no counter in this engine can see.
    gpu.reserve_for_kv((e.kv_capacity_bytes() + e.recurrent_capacity_bytes()) as usize);

    const SLICE: usize = 2048;

    // Warm-up, and it pays the ~24 s of expert placement so no arm carries it.
    let _ = e.prefill(&tokens[..SLICE]).expect("warm-up");
    e.reset();

    let mut time = |label: &str, slice: Option<usize>, ckpt: bool| {
        e.reset();
        let t = std::time::Instant::now();
        match slice {
            None => {
                e.prefill(&tokens).expect("prefill");
            }
            Some(s) => {
                let mut done = 0;
                while done < n {
                    let take = s.min(n - done);
                    e.prefill(&tokens[done..done + take]).expect("prefill slice");
                    done += take;
                    if ckpt {
                        // Exactly what `Session::advance` does between slices.
                        let _ = e.checkpoint();
                    }
                }
            }
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!("  {label:<28} {ms:>9.1} ms   {:>7.2} tok/s", n as f64 / (ms * 1e-3));
        ms
    };

    println!("\nserver prefill pattern, {n} tokens of real content, ctx 32768");
    let whole = time("one prefill (generate)", None, false);
    let sliced = time("2048 slices, no checkpoint", Some(SLICE), false);
    let served = time("2048 slices + checkpoint", Some(SLICE), true);
    assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

    println!(
        "\n  slicing alone      {:.2}x\n  checkpoints add    {:.2}x\n  together           {:.2}x",
        sliced / whole,
        served / sliced,
        served / whole,
    );
}

/// The operands `check_mma_nvfp4` derives on the device, mirrored exactly. See
/// `fp4_check_a`, `fp4_check_b` and `fp4_check_scales` in
/// `kernels/diagnostics.cuh`.
fn fp4_check_a(lane: u32, i: u32) -> u32 {
    1 + (lane * 7 + i * 3) % 15
}

fn fp4_check_b(lane: u32, i: u32) -> u32 {
    1 + (lane * 5 + i * 11) % 15
}

fn fp4_check_scales(lane: u32, mul: u32, add: u32) -> u32 {
    (0..4).fold(0, |v, c| v | ((0x30 + 8 * ((lane * mul + c + add) % 4)) << (8 * c)))
}

/// **The NVFP4 block-scaled `mma` computes what the PTX ISA says it does.**
///
/// Before a ceiling for this instruction means anything, and before a kernel is
/// built on it, the operand layout has to be right. PTX ISA 9.4 defines it: the
/// arithmetic `D = (A * scale_A) * (B * scale_B) + C` and the selectors in
/// 9.7.16.3, the `m16n8k64` fragments in 9.7.16.5.11, E2M1 and UE4M3 in 5.2.3.
/// One detail -- which thread of the scale-A pair supplies row `g` and which
/// row `g + 8` -- is only in a figure, so it is taken from llama.cpp's
/// `vec_dot_fp4_fp4_mma`, which reads row `g` from lane `4g` and `g + 8` from
/// `4g + 1`.
///
/// `check_mma_nvfp4` runs one instruction over a warp on operands each lane
/// derives from its index. This rebuilds A (16x64), B (64x8) and both scale
/// matrices from the fragment tables and evaluates every accumulator. E2M1
/// values against power-of-two UE4M3 scales sum exactly in f32 at these
/// magnitudes, so the comparison is equality. On a mismatch it reports whether
/// either of two other readings -- the scale pair swapped, or scales carried
/// with llama.cpp's CPU-side factor of one half -- would have matched instead.
#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_nvfp4_block_scaled_mma_follows_the_isa() {
    // The kernel exists only in an `sm_120a` build; see `build.rs`.
    if !cfg!(nvfp4_block_scale) {
        println!("SKIPPED: built for sm_120; the NVFP4 mma needs INFERRED_SM_ARCH=sm_120a");
        return;
    }
    // E2M1, ISA 5.2.3 and `cuda_fp4.hpp`: sign bit 3, magnitudes 0..6.
    let e2m1 = |code: u32| -> f64 {
        let m = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(code & 7) as usize];
        if code & 8 != 0 { -m } else { m }
    };
    // UE4M3, ISA 5.2.3: 4 exponent bits, 3 mantissa bits. Only normal codes are
    // used here, so the bias of 7 is the one reading the test depends on.
    let ue4m3 = |code: u32| -> f64 {
        let (e, m) = ((code >> 3) & 0xf, code & 7);
        assert!(e != 0 && code != 0x7f, "check scales must be normal and not NaN");
        (1.0 + f64::from(m) / 8.0) * 2f64.powi(e as i32 - 7)
    };

    // A and B from the fragment tables, 9.7.16.5.11.
    let mut a = [[0.0f64; 64]; 16];
    let mut b = [[0.0f64; 8]; 64];
    for lane in 0..32u32 {
        let (g, t) = ((lane >> 2) as usize, (lane % 4) as usize);
        for i in 0..32u32 {
            let iu = i as usize;
            let row = if iu < 8 || (16..24).contains(&iu) { g } else { g + 8 };
            let col = t * 8 + (iu & 7) + if iu >= 16 { 32 } else { 0 };
            a[row][col] = e2m1(fp4_check_a(lane, i));
        }
        for i in 0..16u32 {
            let iu = i as usize;
            let row = t * 8 + (iu & 7) + if iu >= 8 { 32 } else { 0 };
            b[row][g] = e2m1(fp4_check_b(lane, i));
        }
    }

    // Expected accumulators, lane order, under one reading of the scale pair.
    let expected = |swap_pair: bool, scale_factor: f64| -> Vec<f64> {
        let byte = |v: u32, c: usize| (v >> (8 * c)) & 0xff;
        let mut out = vec![0.0f64; 128];
        for lane in 0..32u32 {
            let (g, t) = ((lane >> 2) as usize, (lane % 4) as usize);
            let lane_of = |g: usize, k: u32| (4 * g) as u32 + k;
            for i in 0..4usize {
                let upper = i >= 2;
                let row = if upper { g + 8 } else { g };
                let col = 2 * t + (i & 1);
                // scale_A for this row: thread-id-a 0, lanes %4 in {0, 1}.
                let pair = if upper != swap_pair { 1 } else { 0 };
                let sa = fp4_check_scales(lane_of(g, pair), 1, 0);
                // scale_B for this column: thread-id-b 0, lane 4 * col.
                let sb = fp4_check_scales(lane_of(col, 0), 3, 1);
                let mut acc = 0.0f64;
                for k in 0..64usize {
                    let fa = ue4m3(byte(sa, k / 16)) * scale_factor;
                    let fb = ue4m3(byte(sb, k / 16)) * scale_factor;
                    acc += (a[row][k] * fa) * (b[k][col] * fb);
                }
                out[lane as usize * 4 + i] = acc;
            }
        }
        out
    };

    let gpu = Cuda::new(0).expect("cuda device");
    let got = gpu.check_mma_nvfp4().expect("check_mma_nvfp4");
    assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

    let matches = |want: &[f64]| got.iter().zip(want).filter(|(g, w)| f64::from(**g) == **w).count();
    let isa = expected(false, 1.0);
    let n_isa = matches(&isa);
    let n_swapped = matches(&expected(true, 1.0));
    let n_halved = matches(&expected(false, 0.5));
    println!(
        "  accumulators matching: ISA reading {n_isa}/128, scale pair swapped {n_swapped}/128, \
         scales halved {n_halved}/128"
    );
    for (idx, (g, w)) in got.iter().zip(&isa).enumerate().filter(|(_, (g, w))| f64::from(**g) != **w).take(8) {
        println!("  lane {:>2} d{}  got {g:>12}  want {w:>12}", idx / 4, idx % 4);
    }
    assert_eq!(
        n_isa, 128,
        "the NVFP4 mma does not compute what the ISA reading says; see the counts above \
         for which alternative reading matched"
    );
}

/// **The tensor-core ceilings -- int8, fp16 and NVFP4 -- in one sitting.**
///
/// `MMA_S8_PEAK_TOPS` and `MMA_F16_PEAK_TFLOPS` came from a harness that was
/// never committed; this re-takes both by the recorded method beside the NVFP4
/// block-scaled form, so the new row is comparable by construction. Method as
/// in BENCHMARKS-v1 09-09 (ceilings): `mma` back to back on constant register
/// operands, no memory in the loop, over `sm_count * 48` warps (1,728 on 36
/// SMs). Reported as trillions of element products per second, one instruction
/// being 16 x 8 x k of them.
///
/// Each arm first times 1,000 iterations and sizes its real run to ~20 ms a
/// launch, so an arm far slower than expected cannot hold the device long
/// enough to trip the display driver's watchdog.
///
/// **And no rate is believed until the work is proven.** Every thread stores
/// its accumulators; each arm runs one iteration first, and every accumulator
/// of the timed run must be exactly `iters` times that value -- exact in s32 and
/// f32 at these magnitudes. The first version of this bench had no such check
/// and reported int8 at 7,353x its ceiling from a loop that cannot have run.
#[test]
#[ignore = "needs an sm_120 device; a measurement, not an assertion"]
fn what_the_fp4_tensor_cores_can_do() {
    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    let warps = gpu.sm_count() as u32 * 48;

    // (label, kernel, k, float accumulators, recorded ceiling)
    let arms: [(&str, &'static str, u64, bool, Option<f64>); 4] = [
        ("int8  m16n8k32.s8", "bench_mma_ceiling_s8", 32, false, Some(MMA_S8_PEAK_TOPS)),
        ("fp16  m16n8k16", "bench_mma_ceiling_f16", 16, true, Some(MMA_F16_PEAK_TFLOPS)),
        ("fp8   m16n8k32.e4m3", "bench_mma_ceiling_e4m3", 32, true, None),
        ("nvfp4 m16n8k64 block-scaled", "bench_mma_ceiling_nvfp4", 64, true, None),
    ];

    println!("\ntensor-core ceilings, {warps} warps requested on {} SMs", gpu.sm_count());
    println!(
        "  {:<30} {:>9} {:>11} {:>14} {:>10}",
        "arm", "iters", "us/launch", "1e12 prod/s", "vs v1"
    );
    let mut rates = Vec::new();
    for (label, kernel, k, float, reference) in arms {
        if kernel.ends_with("nvfp4") && !cfg!(nvfp4_block_scale) {
            println!("  {label:<30} SKIPPED: built for sm_120; needs INFERRED_SM_ARCH=sm_120a");
            continue;
        }
        let value = |bits: u32| {
            if float { f64::from(f32::from_bits(bits)) } else { f64::from(bits as i32) }
        };
        let (_, _, one) = gpu.bench_mma_ceiling(kernel, 1, warps, 1).expect("one iteration");
        let (probe_us, _, _) = gpu.bench_mma_ceiling(kernel, 1_000, warps, 2).expect("probe");
        let iters = ((1_000.0 * 20_000.0 / probe_us.max(1.0)) as i32).clamp(1_000, 4_000_000);
        let (us, launched, finals) = gpu.bench_mma_ceiling(kernel, iters, warps, 4).expect("bench");

        let wrong = finals
            .iter()
            .zip(&one)
            .filter(|(f, o)| value(**f) != value(**o) * f64::from(iters))
            .count();
        assert!(
            value(one[0]) != 0.0 && wrong == 0,
            "{label}: {wrong} of {} accumulators are not {iters} x one iteration ({}), so the \
             timed loop did not do the work and its time means nothing",
            finals.len(),
            value(one[0]),
        );

        let products = f64::from(launched) * f64::from(iters) * (16 * 8 * k) as f64;
        let rate = products / (us * 1e-6) / 1e12;
        let vs = reference.map_or("--".to_string(), |r| format!("{:.2}x", rate / r));
        println!("  {label:<30} {iters:>9} {us:>11.1} {rate:>14.1} {vs:>10}");
        rates.push((label, rate, k));
    }
    assert!(gpu.take_error().is_none(), "a CUDA op reported a driver error");

    // Ratios by label, not by index: the NVFP4 arm is skipped on an `sm_120`
    // build, so a position in `rates` is not a fixed arm.
    let rate_of = |name: &str| rates.iter().find(|(l, _, _)| l.starts_with(name));
    if let (Some((_, s8, k8)), Some((_, other, ko))) = (rate_of("int8"), rate_of("nvfp4")) {
        println!(
            "\n  NVFP4 against int8: {:.2}x products per second, {:.2}x instructions per second",
            other / s8,
            other / s8 * (*k8 as f64) / (*ko as f64),
        );
    }
    if let (Some((_, s8, _)), Some((_, fp8, _))) = (rate_of("int8"), rate_of("fp8")) {
        println!(
            "  FP8 e4m3 against int8, the same m16n8k32 shape: {:.2}x. Both are 4,096 products \n  \
             an instruction, so this is the instruction rate too.",
            fp8 / s8,
        );
    }
}

/// The NVFP4 routed-expert kernels on the GPU, against the oracle, on layer 0's
/// real weights and per-expert scales from the converted NVFP4 GGUF.
///
/// Two claims, each bit-exact:
///
/// - `matmul_experts` (gate, up and down, grouped by expert) equals `Naive`'s
///   default — each pick a `dot_nvfp4_q8_0` row times its expert's scale. The
///   dot is integer inside a serial f32 chain on both sides, so there is no
///   rounding to excuse.
/// - The fused `moe_glu` equals the GPU's own gate and up rows through the GPU's
///   `silu_mul`. Compared on one device because `expf` is the one op whose bits
///   differ between CPU and GPU (`only_the_expf_ops_diverge`); everything before
///   it is covered by the first claim.
///
/// The route makes tokens share experts, so a tile holds several pairs and the
/// grouped loop is exercised rather than degenerate.
#[test]
#[ignore = "needs an sm_120 device and the NVFP4 GGUF"]
fn the_nvfp4_expert_kernels_match_the_oracle() {
    use inferred_thoughts::gguf::{GgmlType, GgufFile};
    use inferred_thoughts::ops::{Experts, Ops, Route};

    let Some(path) = common::find_model_named("Qwen3.6-35B-A3B-NVFP4-Q8_0.gguf") else {
        println!("SKIPPED: no Qwen3.6-35B-A3B-NVFP4-Q8_0.gguf found; set INFERRED_MODEL_DIR");
        return;
    };
    let f = GgufFile::open(&path).expect("open the NVFP4 GGUF");
    let exps = |name: &str, n_in: usize, n_out: usize| {
        let info = f.tensor(&format!("blk.0.{name}.weight")).expect("expert tensor");
        let scale = f
            .tensor(&format!("blk.0.{name}.scale"))
            .map(|s| f.tensor_bytes(s))
            .unwrap_or(&[]);
        Experts { data: f.tensor_bytes(info), ty: info.ty, n_in, n_out, n_expert: 256, scale }
    };
    let gate = exps("ffn_gate_exps", 2048, 512);
    let up = exps("ffn_up_exps", 2048, 512);
    let down = exps("ffn_down_exps", 512, 2048);
    assert_eq!(gate.ty, GgmlType::Nvfp4, "layer 0's experts should be NVFP4");
    assert!(!gate.scale.is_empty(), "the converted file should carry per-expert scales");

    let gpu = inferred_thoughts::Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    // The exact path, which FP4 x FP4 replaces by default: this test is also
    // the proof that `nvfp4_fp4(false)` restores bit-equality with the oracle.
    gpu.nvfp4_fp4(false);

    // Tokens alternate between two overlapping expert sets, so the six tokens'
    // 48 pairs fall into tiles of up to six.
    let (n_tok, n_used) = (6usize, 8usize);
    let pairs = n_tok * n_used;
    let ids: Vec<usize> = (0..pairs).map(|p| (13 * (p % n_used) + 13 * ((p / n_used) % 2)) % 256).collect();
    let route = Route::Host { ids, weights: vec![0.125; pairs], n_used };

    // Held for the whole test: the backend keys its mirrors on host addresses.
    let wave = |n: usize, seed: f32| -> Vec<f32> {
        (0..n).map(|i| ((i as f32 * 0.7137 + seed).sin() * (1.0 + (i % 7) as f32))).collect()
    };
    let x = wave(n_tok * 2048, 0.3);
    let x_pairs: Vec<f32> = (0..pairs).flat_map(|p| x[(p / n_used) * 2048..(p / n_used + 1) * 2048].to_vec()).collect();
    let h = wave(pairs * 512, 1.9);

    let mut kept: Vec<Vec<f32>> = Vec::new();
    let mut against_oracle = |what: &str, w: &Experts<'_>, xin: &[f32]| -> usize {
        let mut cpu = vec![0.0f32; pairs * w.n_out];
        inferred_thoughts::Naive.matmul_experts(w, &route, xin, &mut cpu);
        let mut dev = vec![0.0f32; pairs * w.n_out];
        gpu.begin_pass(n_tok);
        gpu.matmul_experts(w, &route, xin, &mut dev);
        gpu.host_needs(&mut dev);
        if let Some(e) = gpu.take_error() {
            panic!("{what}: driver error: {e}");
        }
        let differing = cpu.iter().zip(&dev).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        if differing != 0 {
            let at = cpu.iter().zip(&dev).position(|(a, b)| a.to_bits() != b.to_bits()).unwrap_or(0);
            panic!(
                "{what}: {differing} of {} outputs differ from the oracle; first at {at} (pair {}, row {}): \
                 naive {:e} vs cuda {:e}",
                cpu.len(), at / w.n_out, at % w.n_out, cpu[at], dev[at]
            );
        }
        println!("  {what:<5} {} outputs bit-identical to naive", dev.len());
        kept.push(dev);
        kept.len() - 1
    };
    let gi = against_oracle("gate", &gate, &x_pairs);
    let ui = against_oracle("up", &up, &x_pairs);
    against_oracle("down", &down, &h);

    let tiles = gpu.last_moe_tiles().expect("a grouped launch ran");
    assert!((tiles as usize) < pairs, "{tiles} tiles for {pairs} pairs: no tile held more than one pair");

    let mut g = kept[gi].clone();
    let u = kept[ui].clone();
    gpu.begin_pass(n_tok);
    gpu.silu_mul(&mut g, &u);
    gpu.host_needs(&mut g);

    let mut fused = vec![0.0f32; pairs * 512];
    let mut scratch = vec![0.0f32; pairs * 512];
    gpu.begin_pass(n_tok);
    gpu.moe_glu(&gate, &up, &route, &x, &mut fused, &mut scratch);
    gpu.host_needs(&mut fused);
    if let Some(e) = gpu.take_error() {
        panic!("moe_glu: driver error: {e}");
    }
    let differing = g.iter().zip(&fused).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(
        differing, 0,
        "the fused NVFP4 gate+up+SiLU differs from gate, up and silu_mul run separately on the \
         same device in {differing} of {} outputs",
        fused.len()
    );
    println!("  glu   {} outputs bit-identical to gate, up and silu_mul; {tiles} tiles for {pairs} pairs", fused.len());
}

/// The NVFP4 routed-expert kernels as **FP4 x FP4 on the tensor cores**,
/// against the CPU reference on layer 0's real experts and scales.
///
/// - `matmul_experts` (gate, up, down): every pick's row within the chain bound
///   of `quant::vec_dot_nvfp4_fp4` times its expert's scale. The bound is the
///   one `the_fp4_tensor_core_matmul_is_within_the_chain_bound` derives,
///   `n_sub * EPSILON * sum|terms|`, times the scale, plus an ulp on each side
///   for the multiply by it.
/// - The fused `moe_glu` bit-identical to the device's own gate and up rows
///   through its `silu_mul`, for the `expf` reason the exact-path test gives.
///
/// Twenty tokens over two alternating expert sets put 10 or 20 pairs on an
/// expert, so a tile's second 8-pair sub-tile runs, and runs partial.
#[test]
#[ignore = "needs an sm_120 device and the NVFP4 GGUF"]
fn the_fp4_expert_kernels_are_within_the_chain_bound() {
    use inferred_thoughts::gguf::{GgmlType, GgufFile};
    use inferred_thoughts::ops::{Experts, Ops, Route};

    if !cfg!(nvfp4_block_scale) {
        println!("SKIPPED: built for sm_120; the FP4 kernels need INFERRED_SM_ARCH=sm_120a");
        return;
    }
    let Some(path) = common::find_model_named("Qwen3.6-35B-A3B-NVFP4-Q8_0.gguf") else {
        println!("SKIPPED: no Qwen3.6-35B-A3B-NVFP4-Q8_0.gguf found; set INFERRED_MODEL_DIR");
        return;
    };
    let f = GgufFile::open(&path).expect("open the NVFP4 GGUF");
    let exps = |name: &str, n_in: usize, n_out: usize| {
        let info = f.tensor(&format!("blk.0.{name}.weight")).expect("expert tensor");
        let scale = f
            .tensor(&format!("blk.0.{name}.scale"))
            .map(|s| f.tensor_bytes(s))
            .unwrap_or(&[]);
        Experts { data: f.tensor_bytes(info), ty: info.ty, n_in, n_out, n_expert: 256, scale }
    };
    let gate = exps("ffn_gate_exps", 2048, 512);
    let up = exps("ffn_up_exps", 2048, 512);
    let down = exps("ffn_down_exps", 512, 2048);

    let gpu = Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    gpu.nvfp4_fp4(true);

    let (n_tok, n_used) = (20usize, 8usize);
    let pairs = n_tok * n_used;
    let ids: Vec<usize> = (0..pairs).map(|p| (13 * (p % n_used) + 13 * ((p / n_used) % 2)) % 256).collect();
    let route = Route::Host { ids: ids.clone(), weights: vec![0.125; pairs], n_used };

    // Held for the whole test: the backend keys its mirrors on host addresses.
    let wave = |n: usize, seed: f32| -> Vec<f32> {
        (0..n).map(|i| ((i as f32 * 0.7137 + seed).sin() * (1.0 + (i % 7) as f32))).collect()
    };
    let x = wave(n_tok * 2048, 0.3);
    let x_pairs: Vec<f32> = (0..pairs).flat_map(|p| x[(p / n_used) * 2048..(p / n_used + 1) * 2048].to_vec()).collect();
    let h = wave(pairs * 512, 1.9);

    let e2m1 = |c: u8| -> f64 {
        let m = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(c & 7) as usize];
        if c & 8 != 0 { -m } else { m }
    };
    let ue4m3 = |c: u8| -> f64 {
        let (e, m) = (((c >> 3) & 0xf) as i32, f64::from(c & 7));
        if c == 0 || c == 0x7f { 0.0 } else if e == 0 { m * 2f64.powi(-9) } else { (1.0 + m / 8.0) * 2f64.powi(e - 7) }
    };

    let mut kept: Vec<Vec<f32>> = Vec::new();
    let mut against_reference = |what: &str, w: &Experts<'_>, xin: &[f32]| -> usize {
        let mut dev = vec![0.0f32; pairs * w.n_out];
        gpu.begin_pass(n_tok);
        gpu.matmul_experts(w, &route, xin, &mut dev);
        gpu.host_needs(&mut dev);
        if let Some(e) = gpu.take_error() {
            panic!("{what}: driver error: {e}");
        }
        let row_bytes = GgmlType::Nvfp4.n_bytes(w.n_in as u64) as usize;
        let (stride, n_sub) = (w.stride(), w.n_in / 16);
        let (mut worst, mut exact) = (0.0f64, 0usize);
        for p in 0..pairs {
            let e = ids[p];
            let xr = &xin[p * w.n_in..(p + 1) * w.n_in];
            let (scales, codes) = inferred_thoughts::quant::fp4_activation(xr);
            let s = w.scale_of(e);
            for r in 0..w.n_out {
                let rb = &w.data[e * stride + r * row_bytes..e * stride + (r + 1) * row_bytes];
                let cpu = (inferred_thoughts::quant::vec_dot_nvfp4_fp4(rb, xr) * s) as f64;
                let deq = inferred_thoughts::quant::dequantize(rb, GgmlType::Nvfp4, w.n_in).expect("dequantize");
                let mut sum_abs = 0.0f64;
                for sb in 0..n_sub {
                    let xs = ue4m3(scales[sb]);
                    let term: f64 = (sb * 16..(sb + 1) * 16).map(|k| deq[k] as f64 * e2m1(codes[k]) * xs).sum();
                    sum_abs += term.abs();
                }
                let eps = f32::EPSILON as f64;
                let bound = (s as f64).abs() * n_sub as f64 * eps * sum_abs + 2.0 * eps * cpu.abs();
                let got = dev[p * w.n_out + r] as f64;
                let diff = (got - cpu).abs();
                assert!(
                    diff <= bound,
                    "{what}: pair {p} (expert {e}) row {r}: tensor core {got:e} vs reference {cpu:e}, off by \
                     {diff:e} against a bound of {bound:e}. Not addition order: suspect a sub-tile's pair, the \
                     register layout or the scale."
                );
                exact += (diff == 0.0) as usize;
                if bound > 0.0 {
                    worst = worst.max(diff / bound);
                }
            }
        }
        println!(
            "  {what:<5} {} outputs within the chain bound; {exact} bit-identical, worst at {worst:.3} of the bound",
            dev.len()
        );
        kept.push(dev);
        kept.len() - 1
    };
    let gi = against_reference("gate", &gate, &x_pairs);
    let ui = against_reference("up", &up, &x_pairs);
    against_reference("down", &down, &h);

    let tiles = gpu.last_moe_tiles().expect("a grouped launch ran");
    assert!((tiles as usize) * 8 < pairs, "{tiles} tiles for {pairs} pairs: the sub-tiles were never shared");

    let mut g = kept[gi].clone();
    let u = kept[ui].clone();
    gpu.begin_pass(n_tok);
    gpu.silu_mul(&mut g, &u);
    gpu.host_needs(&mut g);

    let mut fused = vec![0.0f32; pairs * 512];
    let mut scratch = vec![0.0f32; pairs * 512];
    gpu.begin_pass(n_tok);
    gpu.moe_glu(&gate, &up, &route, &x, &mut fused, &mut scratch);
    gpu.host_needs(&mut fused);
    if let Some(e) = gpu.take_error() {
        panic!("moe_glu: driver error: {e}");
    }
    let differing = g.iter().zip(&fused).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(
        differing, 0,
        "the fused FP4 gate+up+SiLU differs from gate, up and silu_mul run separately on the same device in \
         {differing} of {} outputs",
        fused.len()
    );
    println!("  glu   {} outputs bit-identical to its parts; {tiles} tiles for {pairs} pairs", fused.len());
}

/// What resolving one expert miss costs, computed on the CPU against computed on
/// the GPU from the pinned host tier. SSD-TIER.md fork O2.
///
/// **Both options share the SSD read** (stage 0, `TIERS.md`), so this times only
/// what differs after it: one real expert -- gate, up, SiLU, down -- of layer 0 of
/// the 35B NVFP4, for 1 token (decode) and 16 (a prefill-shaped batch).
///
/// - **CPU option**: `Naive` (the one-thread oracle) and `Spin` (the 8-thread CPU
///   engine). Their NVFP4 path is NVFP4 x Q8_0.
/// - **GPU option**: the same expert with its weights in the VRAM slab, then in
///   the pinned device-mapped host tier, FP4 x FP4 as by default. The difference
///   is what a host-tier expert adds. Each context is built alone, and both arms
///   **assert they never reached the eviction fallback**, which computes with the
///   wrong experts (BENCHMARKS-v2 14-09-2026, stage 1 corrected).
///
/// Also asserts the two GPU arms produce **bit-identical** output: the same
/// kernel over the same bytes must not care which tier holds them, which is the
/// premise tier 3 rests on.
///
/// Not measured here: numerics between the options. The CPU path matches the GPU
/// only under `INFERRED_NVFP4_Q8=1`, as the NVFP4 kernel tests already show. The
/// hidden-state transfers the CPU option adds are timed by
/// `scripts/small_copy_cost.cu`.
#[test]
#[ignore = "needs an sm_120 device and the NVFP4 GGUF"]
fn what_a_miss_costs_on_cpu_and_gpu() {
    use inferred_thoughts::ops::Route;
    use inferred_thoughts::Spin;

    let Some(path) = common::find_model_named("Qwen3.6-35B-A3B-NVFP4-Q8_0.gguf") else {
        println!("SKIPPED: no Qwen3.6-35B-A3B-NVFP4-Q8_0.gguf found; set INFERRED_MODEL_DIR");
        return;
    };
    let f = GgufFile::open(&path).expect("open the NVFP4 GGUF");
    let exps = |name: &str, n_in: usize, n_out: usize| {
        let info = f.tensor(&format!("blk.0.{name}.weight")).expect("expert tensor");
        let scale = f
            .tensor(&format!("blk.0.{name}.scale"))
            .map(|s| f.tensor_bytes(s))
            .unwrap_or(&[]);
        Experts { data: f.tensor_bytes(info), ty: info.ty, n_in, n_out, n_expert: 256, scale }
    };
    let gate = exps("ffn_gate_exps", 2048, 512);
    let up = exps("ffn_up_exps", 2048, 512);
    let down = exps("ffn_down_exps", 512, 2048);

    // Expert 200: past the handful of gate experts a tiny slab can hold, so in the
    // host arm its gate, up and down all land in the pinned tier.
    const EXPERT: usize = 200;
    const WARMUP: usize = 5;
    const SAMPLES: usize = 21;
    let batches = [1usize, 16];

    // Every buffer held for the whole test: the backend keys mirrors on host
    // addresses.
    let wave = |n: usize, seed: f32| -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.7137 + seed).sin() * (1.0 + (i % 7) as f32)).collect()
    };
    let max_tok = batches.iter().copied().max().unwrap_or(1);
    let x_all = wave(max_tok * 2048, 0.3);
    let routes: Vec<Route> = batches
        .iter()
        .map(|&n| Route::Host { ids: vec![EXPERT; n], weights: vec![1.0; n], n_used: 1 })
        .collect();

    fn median_us(mut run: impl FnMut()) -> f64 {
        for _ in 0..WARMUP {
            run();
        }
        let mut xs: Vec<f64> = (0..SAMPLES)
            .map(|_| {
                let t = std::time::Instant::now();
                run();
                t.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        xs.sort_by(|a, b| a.total_cmp(b));
        xs[SAMPLES / 2]
    }

    println!("\nwhat one expert miss costs after the read, 35B layer 0, expert {EXPERT}");
    println!("  {:<38} {:>12} {:>12}", "", "1 token", "16 tokens");

    // -- CPU option ---------------------------------------------------------
    let spin = Spin::new(8);
    let mut cpu_bufs: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> =
        batches.iter().map(|&n| (vec![0.0; n * 512], vec![0.0; n * 512], vec![0.0; n * 2048])).collect();
    for label in ["CPU  Naive, 1 thread (oracle)", "CPU  Spin, 8 threads"] {
        let mut row = Vec::new();
        for (bi, &n) in batches.iter().enumerate() {
            let x = &x_all[..n * 2048];
            let route = &routes[bi];
            let (g, u, o) = &mut cpu_bufs[bi];
            let us = if label.contains("Naive") {
                median_us(|| {
                    Naive.matmul_experts(&gate, route, x, g);
                    Naive.matmul_experts(&up, route, x, u);
                    Naive.silu_mul(g, u);
                    Naive.matmul_experts(&down, route, g, o);
                })
            } else {
                median_us(|| {
                    spin.matmul_experts(&gate, route, x, g);
                    spin.matmul_experts(&up, route, x, u);
                    spin.silu_mul(g, u);
                    spin.matmul_experts(&down, route, g, o);
                })
            };
            row.push(us);
        }
        println!("  {label:<38} {:>9.1} us {:>9.1} us", row[0], row[1]);
    }

    // -- GPU option ---------------------------------------------------------
    // Each arm builds its own context and drops it before the next: concurrent
    // contexts fault intermittently. Buffers live outside the loop, for the
    // whole test.
    let mut gpu_bufs: Vec<Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>> = (0..2)
        .map(|_| {
            batches.iter().map(|&n| (vec![0.0; n * 512], vec![0.0; n * 512], vec![0.0; n * 2048])).collect()
        })
        .collect();
    let mut gpu_rows: Vec<(String, Vec<f64>)> = Vec::new();
    for (arm, host_tier) in [false, true].into_iter().enumerate() {
        let gpu = Cuda::new(0).expect("cuda device");
        gpu.use_graphs(false);
        if host_tier {
            // A slab of a few slots and a pinned tier far larger than layer 0's
            // 768 slices (452 MiB): every expert placed, none evicted.
            gpu.set_expert_budget(8 << 20);
            gpu.set_expert_host_budget(1 << 30);
        }
        let mut row = Vec::new();
        for (bi, &n) in batches.iter().enumerate() {
            let x = &x_all[..n * 2048];
            let route = &routes[bi];
            let (g, scratch, o) = &mut gpu_bufs[arm][bi];
            let us = median_us(|| {
                gpu.begin_pass(n);
                gpu.moe_glu(&gate, &up, route, x, g, scratch);
                gpu.matmul_experts(&down, route, g, o);
                gpu.end_pass();
                gpu.sync().expect("sync");
            });
            gpu.begin_pass(n);
            gpu.moe_glu(&gate, &up, route, x, g, scratch);
            gpu.matmul_experts(&down, route, g, o);
            gpu.host_needs(o);
            if let Some(e) = gpu.take_error() {
                panic!("driver error: {e}");
            }
            row.push(us);
        }
        let st = gpu.expert_stats().expect("the expert cache was built");
        let name = if host_tier { "host-tier" } else { "VRAM" };
        assert!(
            !st.oversubscribed && st.evictions == 0,
            "the {name} arm was oversubscribed ({} evictions, {} fetched): it would time fetches, \
             not the tier",
            st.evictions,
            st.fetched
        );
        if host_tier {
            assert!(st.host_slots > 0, "the host-tier arm placed nothing in the pinned tier");
        } else {
            assert_eq!(st.host_slots, 0, "the VRAM arm spilled {} tensors to the host tier", st.host_slots);
        }
        gpu_rows.push((format!("GPU  weights in {name} ({} slab slots)", st.slots), row));
    }
    for (label, row) in &gpu_rows {
        println!("  {label:<38} {:>9.1} us {:>9.1} us", row[0], row[1]);
    }
    println!(
        "  {:<38} {:>9.1} us {:>9.1} us",
        "GPU  the host tier adds",
        gpu_rows[1].1[0] - gpu_rows[0].1[0],
        gpu_rows[1].1[1] - gpu_rows[0].1[1]
    );

    for (bi, &n) in batches.iter().enumerate() {
        let (vram, host) = (&gpu_bufs[0][bi].2, &gpu_bufs[1][bi].2);
        let differing = vram.iter().zip(host).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        assert_eq!(
            differing, 0,
            "{n} tokens: {differing} of {} outputs differ between the VRAM and host-tier arms -- \
             the tier changed the arithmetic",
            vram.len()
        );
    }
    println!("  VRAM and host-tier outputs bit-identical at every batch size");
    println!(
        "\n  A 125B expert is 2560 x 640, 1.56x the 35B's per matmul, so CPU compute there is\n  \
         ~1.56x these rows by the arithmetic -- not measured."
    );
}

/// **The SSD tier's acceptance test** (SSD-TIER.md D14): with the expert caches
/// capped below the pool, the 35B generates exactly the tokens it generates when
/// every expert fits.
///
/// Placement may change speed, never output. The same kernel over the same bytes
/// computes the same result wherever the bytes live —
/// `what_a_miss_costs_on_cpu_and_gpu` shows it for the VRAM and host tiers — and
/// `Engine::generate` is greedy, so a different token means a different
/// computation, not drift.
///
/// **It fails on the code before tier 3.** Once both tiers fill, that code evicts
/// VRAM slots without rewriting the evicted experts' table entries, so it computes
/// with the wrong experts (BENCHMARKS-v2 14-09-2026, stage 1 corrected). The capped
/// arm asserts it really is oversubscribed, so this cannot pass vacuously, and the
/// default arm asserts it is not.
///
/// **Graphs are off in both arms**, so the tier is the only variable: the MVP runs
/// ungraphed while oversubscribed (D13). An arm with graphs on joins when graphs
/// return to the oversubscribed path.
#[test]
#[ignore = "needs an sm_120 device and the NVFP4 35B"]
fn the_35b_generates_identically_with_the_expert_pool_oversubscribed() {
    use inferred_thoughts::Model;

    let Some(path) = common::find_model_named("Qwen3.6-35B-A3B-NVFP4-Q8_0.gguf") else {
        println!("SKIPPED: no Qwen3.6-35B-A3B-NVFP4-Q8_0.gguf found; set INFERRED_MODEL_DIR");
        return;
    };
    let f = GgufFile::open(&path).expect("open the NVFP4 GGUF");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");

    // Real text rather than a repeated sentence: repetitive input routes to a
    // narrow set of experts and flatters a cache (CLAUDE.md, on `p20k.txt`).
    // Past 512 tokens so the prompt takes **two** prefill passes and two MoE
    // chunks (`DEFAULT_MAX_BATCH`, `MOE_CHUNK`): the lease must clear between
    // them, and one layer resolves a union of up to all its experts at once.
    const PROMPT_TOKENS: usize = 700;
    let text = include_str!("../measurements/prompt_6k.txt");
    let mut tokens = tk.encode(text, true, true);
    assert!(tokens.len() > PROMPT_TOKENS, "prompt_6k.txt tokenized shorter than {PROMPT_TOKENS}");
    tokens.truncate(PROMPT_TOKENS);

    // The pool, read from the file rather than written down: every routed-expert
    // tensor, times the experts each holds.
    let n_expert = u64::from(f.metadata.get_arch_u32("expert_count").expect("expert_count"));
    let expert_tensors = f.tensors.iter().filter(|t| t.name.ends_with("_exps.weight")).count() as u64;
    let pool = expert_tensors * n_expert;

    const MAX_NEW: usize = 32;
    let n_ctx = tokens.len() + MAX_NEW + 8;
    // 4 + 4 GiB: near the 125B's projected ~47% addressable. 2 + 2 GiB: about 19%,
    // where eviction pressure is far higher and a lease bug has the most chances
    // to show.
    let arms: [(&str, Option<(f64, f64)>); 3] = [
        ("default budget", None),
        ("capped 4 + 4 GiB", Some((4.0, 4.0))),
        ("capped 2 + 2 GiB", Some((2.0, 2.0))),
    ];

    println!("\nexpert pool: {expert_tensors} tensors x {n_expert} experts = {pool} slices");
    println!("prompt: {} tokens of prompt_6k.txt, {MAX_NEW} generated", tokens.len());
    let mut runs: Vec<Vec<u32>> = Vec::new();
    for (label, caps) in arms {
        let gpu = Cuda::new(0).expect("cuda device");
        gpu.use_graphs(false);
        if let Some((vram, host)) = caps {
            gpu.set_expert_budget((vram * 1073741824.0) as usize);
            gpu.set_expert_host_budget((host * 1073741824.0) as usize);
        }
        // Both, or the cache has no file to read from and takes the serial
        // mapped-memory path: until 17-09 this test set only the path, so the
        // parallel `O_DIRECT` reads and grouped fetches never ran in it.
        gpu.set_model_path(&f.path);
        gpu.set_map_base(f.map_base());
        let m = Model::load(&f).expect("load the 35B");
        let mut e = Engine::new(m, &gpu, n_ctx, false);
        let (produced, _) = e.generate(&tokens, MAX_NEW, None, |_| {}).expect("generate");
        if let Some(err) = gpu.take_error() {
            panic!("{label}: driver error: {err}");
        }

        let st = gpu.expert_stats().expect("the expert cache was built");
        let addressable = st.slots + st.host_slots;
        let text = tk.decode(&produced, false).unwrap_or_default();
        println!(
            "  {label:<18} slab {} + host {} = {addressable} of {pool} addressable   {} evictions\n  \
             {:<18} {:?}",
            st.slots, st.host_slots, st.evictions, "", text
        );
        // The instruments that lied on the first CLI run (SSD-TIER.md, "The first
        // run through the CLI"), held to what the run did.
        let dev = gpu.stats();
        println!(
            "  {:<18} placed {} VRAM + {} host + {} cold of {}   {} fetched   {} cache uploads, {} up counted",
            "",
            st.distinct.saturating_sub(st.host_slots + st.cold_at_load),
            st.host_slots,
            st.cold_at_load,
            st.distinct,
            st.fetched,
            st.up_calls,
            dev.h2d_calls
        );
        assert!(
            dev.h2d_calls >= st.up_calls,
            "{label}: the crossing counter saw {} uploads but the expert cache issued {}",
            dev.h2d_calls,
            st.up_calls
        );
        // The parallel read path is the one that ran: it alone records whether
        // its reads are `O_DIRECT`, so an unset flag means the serial fallback.
        if caps.is_some()
            && std::env::var("INFERRED_FETCH_THREADS").map_or(true, |v| v != "1")
            && std::env::var("INFERRED_FETCH_DIRECT").map_or(true, |v| v != "0")
        {
            assert!(
                st.fetch_direct,
                "{label}: the fetches did not take the parallel O_DIRECT path, so this run tested the serial fallback"
            );
        }
        if caps.is_some() {
            assert!(
                addressable < pool,
                "{label}: {addressable} of {pool} slices addressable, so nothing is oversubscribed \
                 and the comparison would pass vacuously"
            );
            assert_eq!(st.distinct, pool, "{label}: not every expert was seen when the tables were built");
            assert_eq!(
                st.distinct - st.host_slots - st.cold_at_load,
                st.slots,
                "{label}: VRAM placements do not match the slab's slots; cold experts miscounted"
            );
            // A fetch into a full slab is the bytes, plus a table entry and a
            // residency flag for both the fetched expert and its victim. Those
            // four writes queue during a resolve and go up as one list per flush
            // (`tier3-speed`), so the copies are one a fetch plus one a flush,
            // and the queued writes are still four a fetch.
            assert!(
                st.patches >= 4 * st.fetched,
                "{label}: {} table and flag writes queued for {} fetches, under four a fetch",
                st.patches,
                st.fetched
            );
            assert!(
                st.up_calls >= st.fetched + st.patch_flushes,
                "{label}: {} uploads counted for {} fetches and {} patch flushes",
                st.up_calls,
                st.fetched,
                st.patch_flushes
            );
        } else {
            assert_eq!(st.cold_at_load, 0, "{label}: the reference arm left experts cold");
            assert!(
                addressable >= pool && st.evictions == 0,
                "{label}: the reference arm is itself oversubscribed ({addressable} of {pool}, {} \
                 evictions), so it is not a reference",
                st.evictions
            );
        }
        runs.push(produced);
    }

    let reference = &runs[0];
    for (i, capped) in runs.iter().enumerate().skip(1) {
        let label = arms[i].0;
        if let Some(at) = reference.iter().zip(capped).position(|(a, b)| a != b) {
            panic!(
                "{label} diverges from the reference at generated token {at} of {MAX_NEW} \
                 ({} against {}): the expert tier changed the computation",
                capped[at], reference[at]
            );
        }
        assert_eq!(reference.len(), capped.len(), "{label} produced a different number of tokens");
        println!("  {label}: identical, all {} generated tokens match", capped.len());
    }
}
