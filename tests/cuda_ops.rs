//! Per-op differential test: the CUDA backend against the `naive` oracle.
//!
//! This runs *before* the whole-model comparison for a reason. A wrong token
//! deep in a generation tells you the GPU disagrees; it does not tell you which
//! of eight kernels is responsible. Each op here is driven with synthetic data
//! and compared on its own, so a failure names the kernel.
//!
//! Two classes of result are expected, and the test encodes the difference:
//!
//! * **Bit-identical** — `matmul`, `rms_norm`, `rms_norm_heads`, `rope_neox`,
//!   `add_assign`. Integer and f32 arithmetic in the oracle's order, with
//!   `--fmad=false` preventing contraction. Anything less is a bug.
//! * **Close** — `softmax`, `silu_mul`, `attend`. These call `expf`, and CUDA's
//!   is not obliged to match glibc's to the last bit. The tolerance is a few
//!   ulp, not a fudge factor: if one of these is off by more than that, the
//!   accumulation order is wrong, not the library.
//!
//! ```text
//! cargo test --release --features cuda --test cuda_ops -- --ignored --nocapture
//! ```

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

#[test]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn every_op_agrees_with_the_oracle() {
    let gpu = Cuda::new(0).expect("cuda device");
    println!("device {} sm_{}{}", gpu.name(), gpu.capability().0, gpu.capability().1);

    let n = 1024usize;
    let eps = 1e-6f32;

    // --- rms_norm ------------------------------------------------------
    {
        let x = noise(n, 1);
        let w = noise(n, 2);
        let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
        Naive.rms_norm(&x, &w, eps, &mut a);
        gpu.rms_norm(&x, &w, eps, &mut b);
        exact("rms_norm", &a, &b);
    }

    // --- rms_norm_heads ------------------------------------------------
    {
        let head_dim = 128;
        let mut a = noise(n, 3);
        let mut b = a.clone();
        let w = noise(head_dim, 4);
        Naive.rms_norm_heads(&mut a, &w, head_dim, eps);
        gpu.rms_norm_heads(&mut b, &w, head_dim, eps);
        exact("rms_norm_heads", &a, &b);
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
        gpu.matmul(&w, &x, &mut b);
        exact("matmul_q8_0", &a, &b);
    }

    // --- rope_neox -----------------------------------------------------
    {
        let (head_dim, n_heads) = (128usize, 8usize);
        let mut a = noise(head_dim * n_heads, 7);
        let mut b = a.clone();
        Naive.rope_neox(&mut a, 37, head_dim, n_heads, 1.0e6);
        gpu.rope_neox(&mut b, 37, head_dim, n_heads, 1.0e6);
        exact("rope_neox", &a, &b);
    }

    // --- add_assign ----------------------------------------------------
    {
        let mut a = noise(n, 8);
        let mut b = a.clone();
        let other = noise(n, 9);
        Naive.add_assign(&mut a, &other);
        gpu.add_assign(&mut b, &other);
        exact("add_assign", &a, &b);
    }

    // --- softmax -------------------------------------------------------
    {
        let mut a = noise(384, 10);
        let mut b = a.clone();
        Naive.softmax(&mut a);
        gpu.softmax(&mut b);
        close("softmax", &a, &b, 1e-7);
    }

    // --- silu_mul ------------------------------------------------------
    {
        let mut a = noise(n, 11);
        let mut b = a.clone();
        let up = noise(n, 12);
        Naive.silu_mul(&mut a, &up);
        gpu.silu_mul(&mut b, &up);
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
        gpu.attend(&attn, &mut b);
        close("attend", &a, &b, 1e-6);
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

    let cpu_logits = {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, Naive, tokens.len() + 4, false);
        e.prefill(&tokens).expect("prefill")
    };
    let gpu_logits = {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, &gpu, tokens.len() + 4, false);
        e.prefill(&tokens).expect("prefill")
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

    /// The five exact kernels on the GPU, the three exp-dependent ones on the
    /// CPU. Not a backend anyone should run — a bisection instrument.
    struct ExactOnly<'a>(&'a Cuda);
    impl Ops for ExactOnly<'_> {
        fn rms_norm(&self, x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
            self.0.rms_norm(x, w, eps, out)
        }
        fn rms_norm_heads(&self, x: &mut [f32], w: &[f32], head_dim: usize, eps: f32) {
            self.0.rms_norm_heads(x, w, head_dim, eps)
        }
        fn matmul(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) {
            self.0.matmul(w, x, out)
        }
        fn rope_neox(&self, x: &mut [f32], p: usize, hd: usize, nh: usize, theta: f32) {
            self.0.rope_neox(x, p, hd, nh, theta)
        }
        fn add_assign(&self, a: &mut [f32], b: &[f32]) {
            self.0.add_assign(a, b)
        }
        fn softmax(&self, x: &mut [f32]) {
            Naive.softmax(x)
        }
        fn silu_mul(&self, gate: &mut [f32], up: &[f32]) {
            Naive.silu_mul(gate, up)
        }
        fn attend(&self, a: &Attn<'_>, out: &mut [f32]) {
            Naive.attend(a, out)
        }
    }

    let f = GgufFile::open(&path).expect("open model");
    let tk = Tokenizer::from_metadata(&f.metadata).expect("tokenizer");
    let tokens = tk.encode("The capital of France is", true, true);

    let cpu = {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, Naive, tokens.len() + 4, false);
        e.prefill(&tokens).expect("prefill")
    };
    let mixed = {
        let m = Qwen3::load(&f).expect("load model");
        let mut e = Engine::new(m, ExactOnly(&gpu), tokens.len() + 4, false);
        e.prefill(&tokens).expect("prefill")
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

    let t = Instant::now();
    for _ in 0..reps {
        gpu.saxpy(1.0, &x, &y, n).expect("saxpy");
    }
    let launch = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

    let t = Instant::now();
    for _ in 0..reps {
        x.write(&host).expect("h2d");
        gpu.saxpy(1.0, &x, &y, n).expect("saxpy");
        y.read(&mut back).expect("d2h");
    }
    let round_trip = t.elapsed().as_secs_f64() * 1e6 / reps as f64;

    println!("  launch + sync        {launch:>7.1} us");
    println!("  + 4 KiB h2d/d2h      {round_trip:>7.1} us");
    println!("  a decode step runs ~450 ops, so the floor is {:.1} ms/token",
             round_trip * 450.0 / 1000.0);
}
