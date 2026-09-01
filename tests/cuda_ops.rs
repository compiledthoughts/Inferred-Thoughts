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

use inferred_thoughts::ops::{Attn, Ops, Weights};
use inferred_thoughts::quant::half::f32_to_f16;
use inferred_thoughts::{Cuda, Engine, GgufFile, Naive, Qwen3, Tokenizer};

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
        Naive.rope_neox(&mut a, 37, head_dim, n_heads, 1.0e6);
        gpu.begin_pass(1);
        gpu.rope_neox(&mut b, 37, head_dim, n_heads, 1.0e6);
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
        Naive.softmax(&mut a);
        gpu.begin_pass(1);
        gpu.softmax(&mut b);
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

    // Twelve decode steps, which is past the point where the GPU stops
    // launching kernels one by one and starts replaying the step as a CUDA
    // graph. That transition is invisible to this comparison and should stay
    // that way, so this is where it gets checked.
    let steps = 12;

    let cpu_logits = {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, Naive, tokens.len() + steps + 4, false);
        let mut l = e.prefill(&tokens).expect("prefill");
        for _ in 0..steps {
            l = e.decode(Qwen3::argmax(&l)).expect("decode");
        }
        l
    };
    let gpu_logits = {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, &gpu, tokens.len() + steps + 4, false);
        let mut l = e.prefill(&tokens).expect("prefill");
        for _ in 0..steps {
            l = e.decode(Qwen3::argmax(&l)).expect("decode");
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
        fn matmul(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) {
            self.0.matmul(w, x, out);
            self.0.host_needs(out);
        }
        fn rope_neox(&self, x: &mut [f32], p: usize, hd: usize, nh: usize, theta: f32) {
            self.0.rope_neox(x, p, hd, nh, theta);
            self.0.host_needs(x);
        }
        fn add_assign(&self, a: &mut [f32], b: &[f32]) {
            self.0.add_assign(a, b);
            self.0.host_needs(a);
        }

        fn softmax(&self, x: &mut [f32]) {
            Naive.softmax(x);
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
