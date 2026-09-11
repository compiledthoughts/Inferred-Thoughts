//! The k-quant dot products, against ggml's own kernels.
//!
//! The 35B needs `vec_dot` for IQ4_XS, Q5_K and Q6_K, and ggml pairs all three
//! with **Q8_K** activations rather than f32. These fixtures come from
//! `scripts/dump_kquant_dots.py`, which loads `libggml-cpu.so` through `ctypes`
//! and calls `quantize_row_q8_K_ref` and `ggml_vec_dot_*_q8_K` directly.
//!
//! **That is a stronger oracle than Stage 2 had.** `check_q8_matmul.py` models
//! the reference in numpy and compares against our Rust, which catches a
//! transcription slip only if the two slip differently. Here the fixture *is*
//! the reference's answer, so a shared misreading of the C is impossible.
//!
//! The dumper records both the dispatching entry point and the `_generic`
//! scalar path, and on this machine **they disagree**: 7 to 9 of 64 rows match
//! bit-for-bit, the rest differ by up to 2.9e-6 absolute on values of order 1.
//!
//! That is f32 accumulation-order noise, not a semantic difference — the
//! integer products are exact, but the reference sums eight f32 lanes where
//! AVX-512 sums a different width in a different order. **We match `_generic`,
//! the portable definition.** The consequence worth stating: llama.cpp on this
//! machine runs the SIMD path, so being bit-exact here means being bit-exact
//! against ggml's *definition*, and llama.cpp's own output will differ from
//! both of us by about that much. Unlike Q8_0 — where the 32-element block sum
//! is integer and every implementation must agree — there is no bit-exactness
//! to be had against llama.cpp's actual arithmetic for the k-quants.
//!
//! **NVFP4 rides in the same fixture format with a Q8_0 activation**, its
//! `vec_dot_type`, in the slot the k-quants use for Q8_K. On x86 ggml has no
//! SIMD NVFP4 kernel, so its dispatch and generic entries are one function.

use std::fs;
use std::path::{Path, PathBuf};

use inferred_thoughts::gguf::GgmlType;

const QK_K: usize = 256;
/// `{ float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; }`
const Q8K_BYTES: usize = 4 + QK_K + 2 * (QK_K / 16);

struct Fixture {
    name: String,
    ty: GgmlType,
    /// Elements per row.
    n: usize,
    rows: usize,
    weights: Vec<u8>,
    activation: Vec<f32>,
    /// ggml's `block_q8_K` bytes for `activation`.
    q8k: Vec<u8>,
    /// One dot product per weight row, from the portable kernel.
    generic: Vec<f32>,
    /// The same, from whichever kernel ggml dispatches to here.
    dispatch: Vec<f32>,
    source: PathBuf,
}

fn load(path: &Path) -> Fixture {
    let b = fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let mut p = 0usize;
    let mut take = |n: usize| {
        let s = &b[p..p + n];
        p += n;
        s.to_vec()
    };

    assert_eq!(take(4), b"ITDP", "bad magic in {}", path.display());
    assert_eq!(
        u32::from_le_bytes(take(4).try_into().unwrap()),
        1,
        "unsupported fixture version in {}",
        path.display()
    );
    let ty_code = u32::from_le_bytes(take(4).try_into().unwrap());
    let n = u64::from_le_bytes(take(8).try_into().unwrap()) as usize;
    let rows = u64::from_le_bytes(take(8).try_into().unwrap()) as usize;
    let name_len = u32::from_le_bytes(take(4).try_into().unwrap()) as usize;
    let name = String::from_utf8(take(name_len)).unwrap();
    let w_len = u64::from_le_bytes(take(8).try_into().unwrap()) as usize;
    let weights = take(w_len);

    let mut activation = Vec::with_capacity(n);
    for _ in 0..n {
        activation.push(f32::from_le_bytes(take(4).try_into().unwrap()));
    }
    let q8k_len = u64::from_le_bytes(take(8).try_into().unwrap()) as usize;
    let q8k = take(q8k_len);

    let mut f32s = |k: usize| {
        (0..k)
            .map(|_| f32::from_le_bytes(take(4).try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let generic = f32s(rows);
    let dispatch = f32s(rows);

    Fixture {
        name,
        ty: GgmlType::from_u32(ty_code)
            .unwrap_or_else(|| panic!("{}: unknown ggml type {ty_code}", path.display())),
        n,
        rows,
        weights,
        activation,
        q8k,
        generic,
        dispatch,
        source: path.to_path_buf(),
    }
}

fn fixtures() -> Vec<Fixture> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut out: Vec<_> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "itdp"))
        .map(|p| load(&p))
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The fixtures whose activation is Q8_K: every format but NVFP4.
fn kquant_fixtures() -> Vec<Fixture> {
    fixtures().into_iter().filter(|f| f.ty != GgmlType::Nvfp4).collect()
}

/// The NVFP4 fixtures, whose activation is Q8_0.
fn nvfp4_fixtures() -> Vec<Fixture> {
    fixtures().into_iter().filter(|f| f.ty == GgmlType::Nvfp4).collect()
}

/// ggml's SIMD and portable kernels differ, and only by rounding.
///
/// **Recorded rather than asserted away.** An earlier version of this test
/// claimed they were bit-identical; they are not, and the "evidence" was a
/// dumper bug — `ggml_cpu_init()` had not been called, so ggml's f16 -> f32
/// table was zero, every super-block scale read back as 0.0, and both paths
/// returned exactly 0.0 for every row. Two kernels agreeing on zero is not
/// agreement.
///
/// The bound below is the claim that matters: the divergence is f32
/// summation-order noise on an otherwise exact integer dot, so it stays at the
/// 1e-6 level relative to the magnitudes involved. If it ever grew, one of the
/// two paths would have changed meaning rather than merely changed order.
#[test]
fn the_simd_and_portable_kernels_differ_only_by_rounding() {
    let fx = fixtures();
    assert!(!fx.is_empty(), "no .itdp fixtures; run scripts/dump_kquant_dots.py");
    for f in &fx {
        let scale = f
            .generic
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()))
            .max(f32::MIN_POSITIVE);
        let worst = f
            .generic
            .iter()
            .zip(&f.dispatch)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        let same = f
            .generic
            .iter()
            .zip(&f.dispatch)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        println!(
            "  {:<52} {same:>3}/{} bit-identical, worst {:.2e} ({:.1e} relative)",
            f.name,
            f.rows,
            worst,
            worst / scale
        );
        assert!(
            worst / scale < 1e-4,
            "{}: ggml's two kernels differ by {:.3e} relative, far past summation-order noise. \n             One of them has changed meaning.",
            f.name,
            worst / scale
        );
        assert!(
            f.generic.iter().any(|v| *v != 0.0),
            "{}: every reference dot is 0.0 — did the dumper call ggml_cpu_init()?",
            f.name
        );
    }
}

/// Our Q8_K activation quantization must reproduce ggml's, byte for byte.
///
/// Checked before any dot product, because all three formats share this input:
/// a wrong `d`, a wrong rounding rule or a missing `bsums` would make all three
/// dots fail at once and look like three kernel bugs instead of one.
#[test]
fn q8_k_quantization_matches_the_reference_byte_for_byte() {
    for f in kquant_fixtures() {
        let ours = inferred_thoughts::quant::q8_k_blocks(&f.activation);
        assert_eq!(
            ours.len(),
            f.q8k.len(),
            "{}: produced {} bytes of Q8_K, reference has {}",
            f.name,
            ours.len(),
            f.q8k.len()
        );

        let nb = f.n / QK_K;
        for b in 0..nb {
            let (a, e) = (
                &ours[b * Q8K_BYTES..(b + 1) * Q8K_BYTES],
                &f.q8k[b * Q8K_BYTES..(b + 1) * Q8K_BYTES],
            );
            if a == e {
                continue;
            }
            // Name the field rather than dumping 292 bytes: the three have
            // different causes and the first byte that differs says which.
            let field = |i: usize| match i {
                0..=3 => "d (the super-block scale)",
                4..=259 => "qs (a quant -- check nearest_int's tie rule)",
                _ => "bsums (the per-16 sums Q5_K's mins term needs)",
            };
            let at = a.iter().zip(e).position(|(x, y)| x != y).unwrap_or(0);
            panic!(
                "{}: Q8_K block {b} differs from the reference at byte {at}, in {}\n  \
                 ours {:02x?}\n  ggml {:02x?}",
                f.name,
                field(at),
                &a[at..(at + 8).min(a.len())],
                &e[at..(at + 8).min(e.len())],
            );
        }
    }
}

/// Every k-quant dot product, bit-exact against ggml.
///
/// Bit-exact and not a tolerance, for the same reason the Q8_0 matmul is: the
/// products accumulate as integers and cannot round, so the only float
/// operations are the per-sub-block scalings, which both sides perform
/// identically. Any difference at all is a transcription bug. Per `CLAUDE.md`,
/// the fix is never to loosen this.
#[test]
fn every_k_quant_dot_reproduces_ggml() {
    let fx = kquant_fixtures();
    assert!(!fx.is_empty(), "no .itdp fixtures; run scripts/dump_kquant_dots.py");

    let mut checked = 0usize;
    for f in &fx {
        let row_bytes = f.weights.len() / f.rows;
        for r in 0..f.rows {
            let w = &f.weights[r * row_bytes..(r + 1) * row_bytes];
            let ours = inferred_thoughts::quant::vec_dot_q8_k(f.ty, w, &f.activation);
            let want = f.generic[r];
            assert_eq!(
                ours.to_bits(),
                want.to_bits(),
                "{} row {r} ({:?}, n={}): ours {ours:e} vs ggml {want:e}\n  \
                 from {}",
                f.name,
                f.ty,
                f.n,
                f.source.display()
            );
            checked += 1;
        }
    }
    println!("  {checked} dot products, bit-exact against ggml");
}

/// Our Q8_0 activation quantization must reproduce `quantize_row_q8_0_ref`,
/// byte for byte — checked before the NVFP4 dot for the reason the Q8_K test
/// is checked before the k-quant dots.
#[test]
fn nvfp4_s_q8_0_activation_matches_the_reference_byte_for_byte() {
    let fx = nvfp4_fixtures();
    assert!(!fx.is_empty(), "no NVFP4 .itdp fixture; run scripts/dump_kquant_dots.py on the NVFP4 GGUF");
    for f in fx {
        let ours = inferred_thoughts::quant::q8_0_blocks(&f.activation);
        assert_eq!(ours.len(), f.q8k.len(), "{}: Q8_0 byte count", f.name);
        const Q8_0_BYTES: usize = 34;
        for b in 0..ours.len() / Q8_0_BYTES {
            let (a, e) = (
                &ours[b * Q8_0_BYTES..(b + 1) * Q8_0_BYTES],
                &f.q8k[b * Q8_0_BYTES..(b + 1) * Q8_0_BYTES],
            );
            assert_eq!(
                a, e,
                "{}: Q8_0 block {b} differs ({})",
                f.name,
                if a[..2] != e[..2] { "the f16 scale" } else { "a quant: check roundf's tie rule" }
            );
        }
    }
}

/// Every NVFP4 dot product, bit-exact against ggml's
/// `ggml_vec_dot_nvfp4_q8_0`: integer sums per sub-block inside a serial f32
/// chain, so any difference is a transcription bug.
#[test]
fn every_nvfp4_dot_reproduces_ggml() {
    let fx = nvfp4_fixtures();
    assert!(!fx.is_empty(), "no NVFP4 .itdp fixture; run scripts/dump_kquant_dots.py on the NVFP4 GGUF");
    let mut checked = 0usize;
    for f in &fx {
        let row_bytes = f.weights.len() / f.rows;
        for r in 0..f.rows {
            let w = &f.weights[r * row_bytes..(r + 1) * row_bytes];
            let ours = inferred_thoughts::quant::vec_dot_nvfp4_q8_0(w, &f.activation);
            let want = f.generic[r];
            assert_eq!(
                ours.to_bits(),
                want.to_bits(),
                "{} row {r}: ours {ours:e} vs ggml {want:e}",
                f.name
            );
            checked += 1;
        }
    }
    println!("  {checked} NVFP4 dot products, bit-exact against ggml");
}

/// A structural check that cannot be fooled by a regenerated fixture.
///
/// **Stage 6's lesson, and it matters more here.** The fixtures now encode a
/// compiler's FMA decisions, so regenerating them after a llama.cpp rebuild
/// could quietly bless whatever we produce. This test does not consult them: it
/// dequantizes each weight row with `quant::dequantize` — proven bit-exact
/// against `gguf.quants` in Stage 6, by an entirely separate path — and takes a
/// plain f32 dot against the *unquantized* activation.
///
/// The two cannot agree exactly: the k-quant dot sees the activation through
/// Q8_K, which is ~8 bits over a 256-element super-block. What it does catch is
/// a wrong bit-unpacking, a swapped scale, or a lost `min` — errors that move
/// the answer by far more than quantization does.
#[test]
fn the_dots_agree_with_the_proven_dequantizer_to_the_quantization_floor() {
    for f in fixtures() {
        let row_bytes = f.weights.len() / f.rows;
        let rows = f.rows.min(16);
        let mut diffs = Vec::with_capacity(rows);
        let mut scale = 0.0f32;
        for r in 0..rows {
            let w = &f.weights[r * row_bytes..(r + 1) * row_bytes];
            let deq = inferred_thoughts::quant::dequantize(w, f.ty, f.n)
                .expect("Stage 6 dequantization");
            let exact: f32 = deq.iter().zip(&f.activation).map(|(a, b)| a * b).sum();
            // Each format through its own `vec_dot_type`: Q8_0 for NVFP4.
            let ours = if f.ty == GgmlType::Nvfp4 {
                inferred_thoughts::quant::vec_dot_nvfp4_q8_0(w, &f.activation)
            } else {
                inferred_thoughts::quant::vec_dot_q8_k(f.ty, w, &f.activation)
            };
            diffs.push((ours - exact).abs());
            scale = scale.max(exact.abs());
        }
        // Normalized by the fixture's magnitude, not each row's. A row whose
        // dot lands near zero is a cancellation, not a kernel failure, and
        // dividing by it would report a huge ratio for a tiny error.
        let worst = diffs.iter().fold(0.0f32, |m, d| m.max(*d)) / scale.max(1e-6);
        println!("  {:<52} worst {worst:.3e} relative to an f32 dot", f.name);
        // Measured at 0.8-1.7% across the three formats, which is what Q8_K's
        // ~8 bits per 256-element super-block predicts -- and the fixture's
        // activation deliberately spans ~800:1 within a block, so the smallest
        // elements quantize to zero and drop out.
        //
        // What this separates is layout from rounding: a swapped nibble, a
        // missed high bit or a lost `min` moves the answer by order 100%, not
        // by 2%. That is the error class this test exists for; the bit-exact
        // test above covers everything finer.
        assert!(
            worst < 0.05,
            "{}: the k-quant dot is {worst:.3e} from a dequantized f32 dot. \
             That is past what Q8_K's ~8 bits explain -- suspect the bit \
             unpacking or a scale, not rounding.",
            f.name
        );
    }
}

// ------------------------------------------------------------------- on CUDA
//
// The same fixtures, against the GPU kernels. These are in this file rather
// than `tests/cuda_ops.rs` because the fixture loader and the real 35B weight
// rows are here, and a k-quant kernel checked against synthetic weights is a
// much weaker test: the bit-packings are what break, and real weights exercise
// every nibble, high bit and scale the format has.

/// The device Q8_K quantizer, against the host one.
///
/// **Checked before the dots, and separately from them, for one reason.** All
/// three formats share this input, so a wrong `d` or a wrong rounding rule
/// fails three kernels at once and looks like three bugs. Worse, `bsums` is
/// read only by Q5_K, so a defect confined to it presents as *one* broken
/// matmul beside two healthy ones — which reads like a Q5_K transcription slip
/// and sends you to the wrong file. The handoff for this work named that exact
/// failure shape; this test is the answer to it.
#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_device_q8_k_quantizer_matches_the_host_one() {
    let gpu = inferred_thoughts::Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);

    let fx = kquant_fixtures();
    assert!(!fx.is_empty(), "no .itdp fixtures; run scripts/dump_kquant_dots.py");

    for f in &fx {
        let host = inferred_thoughts::quant::Q8KRow::from_f32(&f.activation);
        let (scales, quants, bsums) = gpu
            .quantize_q8_k_readback(&f.activation)
            .expect("quantize_q8_k");

        // Bit-for-bit on the scale, not approximately: `d` is 1/iscale and
        // iscale is -127 divided by the signed extreme, so a tie broken the
        // other way in the argmax moves the whole block's quants.
        for (b, (a, e)) in scales.iter().zip(host.scales()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{}: super-block {b} scale differs, device {a:e} vs host {e:e}. \
                 The argmax that picks the extreme must break ties toward the \
                 LOWER index, as the reference's linear scan does.",
                f.name
            );
        }
        assert_eq!(quants, host.quants(), "{}: Q8_K quants differ", f.name);
        assert_eq!(
            bsums,
            host.bsums(),
            "{}: Q8_K bsums differ. Only Q5_K reads these, so this would have \
             surfaced as a lone Q5_K failure.",
            f.name
        );
        println!(
            "  {:<52} {} super-blocks identical (scales, quants, bsums)",
            f.name,
            scales.len()
        );
    }
}

/// Every k-quant matmul on the GPU, **bit-identical to the CPU oracle**.
///
/// The handoff for this work expected a derived tolerance here, on the grounds
/// that the CPU dots are exact only because they mirror one build's per-function
/// FMA contraction. That turned out to be recoverable rather than lost: nvcc's
/// contraction is off globally (`--fmad=false` in build.rs) and re-enabled by
/// hand with `__fmaf_rn` in exactly the two places the reference fused, so the
/// GPU can mirror the same asymmetry deliberately.
///
/// What makes the *parallel* kernel exact is the argument `matmul_q8_0_warp`
/// found, which generalizes: every k-quant dot is an integer inner sum inside
/// an f32 outer chain. The integer part cannot round, so the warp may split it
/// however it likes; the f32 chain is walked serially and ascending, exactly as
/// the oracle walks it. See `kernels/kernels.cu`.
///
/// Per `CLAUDE.md`, the fix if this ever fails is never to loosen it.
#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_k_quant_matmuls_are_bit_identical_on_the_gpu() {
    use inferred_thoughts::ops::{Ops, Weights};

    let gpu = inferred_thoughts::Cuda::new(0).expect("cuda device");
    // One op at a time, read straight back: a graph defers the whole pass to
    // `end_pass`, so it cannot serve this shape.
    gpu.use_graphs(false);
    println!(
        "device {} sm_{}{}",
        gpu.name(),
        gpu.capability().0,
        gpu.capability().1
    );

    let fx = fixtures();
    assert!(!fx.is_empty(), "no .itdp fixtures; run scripts/dump_kquant_dots.py");

    for f in &fx {
        // The fixture's rows are contiguous and each is a whole number of
        // super-blocks, which is exactly a `Weights` of `rows` outputs.
        let w = Weights {
            data: &f.weights,
            ty: f.ty,
            n_in: f.n,
            n_out: f.rows,
            pooled: false,
        };

        let mut cpu = vec![0.0f32; f.rows];
        inferred_thoughts::Naive.matmul(&w, &f.activation, &mut cpu);

        let mut dev = vec![0.0f32; f.rows];
        gpu.begin_pass(1);
        gpu.matmul(&w, &f.activation, &mut dev);
        gpu.host_needs(&mut dev);
        if let Some(e) = gpu.take_error() {
            panic!("{}: driver error: {e}", f.name);
        }

        // The CPU side is already pinned to ggml's portable kernel by
        // `every_k_quant_dot_reproduces_ggml`, so agreeing with it bit-for-bit
        // chains the GPU to ggml's definition too.
        let differing = cpu
            .iter()
            .zip(&dev)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        if differing != 0 {
            let at = cpu
                .iter()
                .zip(&dev)
                .position(|(a, b)| a.to_bits() != b.to_bits())
                .unwrap_or(0);
            panic!(
                "{} ({:?}, n_in {}, {} rows): {differing} rows differ from the oracle.\n  \
                 first at row {at}: naive {:e} vs cuda {:e}\n  \
                 The integer part of this dot is order-free and the f32 chain is \
                 meant to be serial and ascending -- a difference is a defect in \
                 one of those two claims, not rounding. Check the FMA choices \
                 first: Q5_K fuses, Q6_K and IQ4_XS must not.",
                f.name, f.ty, f.n, f.rows, cpu[at], dev[at],
            );
        }
        println!(
            "  {:<52} {} rows bit-identical to naive",
            f.name, f.rows
        );
    }
}

/// The batch axis changes which outputs share a launch, never how one
/// accumulates.
///
/// Q8_0 gets this property from two separate kernels that must be kept in
/// agreement; the k-quants get it from one kernel with the batch on
/// `blockIdx.y`, so decode is `gridDim.y == 1` of the same code. This asserts
/// that rather than assuming it — and it is the test that would catch a
/// per-token pointer offset computed from the wrong stride, which is the one
/// thing the single-token path cannot see.
#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn a_batched_k_quant_matmul_agrees_with_its_own_single_token_path() {
    use inferred_thoughts::ops::{Ops, Weights};

    let gpu = inferred_thoughts::Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);

    for f in fixtures() {
        let w = Weights {
            data: &f.weights,
            ty: f.ty,
            n_in: f.n,
            n_out: f.rows,
            pooled: false,
        };

        // Four tokens: the fixture's activation, then three cheap variations
        // of it, so a stride bug cannot hide behind identical rows.
        let n_tok = 4usize;
        let mut batch = Vec::with_capacity(n_tok * f.n);
        for t in 0..n_tok {
            let k = 1.0 + t as f32 * 0.37;
            batch.extend(f.activation.iter().map(|v| v * k));
        }

        let mut want = vec![0.0f32; n_tok * f.rows];
        for t in 0..n_tok {
            let x = &batch[t * f.n..(t + 1) * f.n];
            let mut row = vec![0.0f32; f.rows];
            gpu.begin_pass(1);
            gpu.matmul(&w, x, &mut row);
            gpu.host_needs(&mut row);
            want[t * f.rows..(t + 1) * f.rows].copy_from_slice(&row);
        }

        let mut got = vec![0.0f32; n_tok * f.rows];
        gpu.begin_pass(n_tok);
        gpu.matmul(&w, &batch, &mut got);
        gpu.host_needs(&mut got);
        if let Some(e) = gpu.take_error() {
            panic!("{}: driver error: {e}", f.name);
        }

        let differing = want
            .iter()
            .zip(&got)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differing,
            0,
            "{} ({:?}): {differing} of {} batched outputs differ from the same \
             kernel run one token at a time. Batching is meant to be exact by \
             construction, so suspect a per-token stride.",
            f.name,
            f.ty,
            want.len()
        );
        println!(
            "  {:<52} {n_tok} tokens x {} rows match the single-token path",
            f.name, f.rows
        );
    }
}

// ------------------------------------------------------- FP4 x FP4 on CUDA
//
// The tensor-core NVFP4 path against the CPU reference in `ops::naive`, which
// transcribes llama.cpp's CUDA arithmetic. The quantizer is discrete and must
// match to the byte; the product may differ only by the order the core adds
// exact sub-block terms in.

/// E2M1 and UE4M3 as real numbers, unhalved: ISA 5.2.3.
fn e2m1_value(code: u8) -> f64 {
    let m = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(code & 7) as usize];
    if code & 8 != 0 { -m } else { m }
}

fn ue4m3_value(code: u8) -> f64 {
    let (e, m) = (((code >> 3) & 0xf) as i32, f64::from(code & 7));
    if code == 0 || code == 0x7f {
        0.0
    } else if e == 0 {
        m * 2f64.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

/// The device FP4 activation quantizer against `quant::fp4_activation`, byte
/// for byte, at four scales of the fixture's activation: the seed's CUDA
/// rounding, the +-2 search's fused error and the E2M1 ties all have to agree,
/// and 4000x pushes blocks past 448 * 6 into the saturated seed.
#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_device_fp4_activation_quantizer_matches_the_reference() {
    if !cfg!(nvfp4_block_scale) {
        println!("SKIPPED: built for sm_120; the FP4 kernels need INFERRED_SM_ARCH=sm_120a");
        return;
    }
    let gpu = inferred_thoughts::Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    let fx = nvfp4_fixtures();
    assert!(!fx.is_empty(), "no NVFP4 .itdp fixture; run scripts/dump_kquant_dots.py on the NVFP4 GGUF");

    // Held for the whole test: the backend keys its mirrors on host addresses.
    let mut held: Vec<Vec<f32>> = Vec::new();
    for f in &fx {
        for k in [1.0f32, 0.003, 90.0, 4000.0] {
            held.push(f.activation.iter().map(|v| v * k).collect());
            let x = held.last().unwrap();
            let (want_scales, want_codes) = inferred_thoughts::quant::fp4_activation(x);
            let n_sub = x.len() / 16;
            let mut want_packed = vec![0u8; n_sub * 8];
            for s in 0..n_sub {
                for j in 0..8 {
                    want_packed[s * 8 + j] = want_codes[s * 16 + j] | (want_codes[s * 16 + j + 8] << 4);
                }
            }
            let (scales, packed) = gpu.quantize_nvfp4_act_readback(x).expect("quantize_nvfp4_act");
            if let Some(e) = gpu.take_error() {
                panic!("{}: driver error: {e}", f.name);
            }
            if let Some(s) = scales.iter().zip(&want_scales).position(|(a, b)| a != b) {
                panic!(
                    "{} x{k}: sub-block {s} scale code {:#04x} on the device, {:#04x} in the reference \
                     (amax {:e}). Check the seed's rounding first, then the fused error.",
                    f.name,
                    scales[s],
                    want_scales[s],
                    x[s * 16..(s + 1) * 16].iter().fold(0.0f32, |m, v| m.max(v.abs()))
                );
            }
            if let Some(b) = packed.iter().zip(&want_packed).position(|(a, w)| a != w) {
                panic!("{} x{k}: packed code byte {b} (sub-block {}) differs: {:#04x} vs {:#04x}",
                    f.name, b / 8, packed[b], want_packed[b]);
            }
            println!("  {:<52} x{k:<6} {n_sub} sub-blocks identical", f.name);
        }
    }
}

/// FP4 x FP4 on the tensor cores against `quant::vec_dot_nvfp4_fp4`, output by
/// output.
///
/// **The bound is derived, not chosen.** Every sub-block term is exact in f32
/// (a 12-bit integer times an 8-bit scale mantissa), so the reference and the
/// core differ only in the order they add `n_sub` exact terms. Each addition
/// rounds by at most half an ulp of its result, and no partial sum exceeds
/// `sum|terms|`, so each side is within `n_sub * EPSILON / 2 * sum|terms|` of the
/// exact sum and the two within `n_sub * EPSILON * sum|terms|` of each other.
///
/// 50 of the fixture's 64 rows and 11 tokens, so both the last row tile and the
/// last token tile are partial.
#[test]
#[cfg(feature = "cuda")]
#[ignore = "needs an sm_120 device; run with --release --features cuda -- --ignored"]
fn the_fp4_tensor_core_matmul_is_within_the_chain_bound() {
    use inferred_thoughts::ops::{Ops, Weights};

    if !cfg!(nvfp4_block_scale) {
        println!("SKIPPED: built for sm_120; the FP4 kernels need INFERRED_SM_ARCH=sm_120a");
        return;
    }
    let gpu = inferred_thoughts::Cuda::new(0).expect("cuda device");
    gpu.use_graphs(false);
    gpu.nvfp4_fp4(true);
    let fx = nvfp4_fixtures();
    assert!(!fx.is_empty(), "no NVFP4 .itdp fixture; run scripts/dump_kquant_dots.py on the NVFP4 GGUF");

    for f in &fx {
        let (n_out, n_tok, n_sub) = (50usize, 11usize, f.n / 16);
        let row_bytes = f.weights.len() / f.rows;
        let wbytes = &f.weights[..n_out * row_bytes];
        let w = Weights { data: wbytes, ty: f.ty, n_in: f.n, n_out, pooled: false };

        let mut batch = Vec::with_capacity(n_tok * f.n);
        for t in 0..n_tok {
            let k = (1.0 + t as f32 * 0.37) * if t % 3 == 2 { -1.0 } else { 1.0 };
            batch.extend(f.activation.iter().map(|v| v * k));
        }
        let mut dev = vec![0.0f32; n_tok * n_out];
        gpu.begin_pass(n_tok);
        gpu.matmul(&w, &batch, &mut dev);
        gpu.host_needs(&mut dev);
        if let Some(e) = gpu.take_error() {
            panic!("{}: driver error: {e}", f.name);
        }

        let wdeq = inferred_thoughts::quant::dequantize(wbytes, GgmlType::Nvfp4, n_out * f.n)
            .expect("dequantize the weight rows");
        let (mut worst, mut exact_hits) = (0.0f64, 0usize);
        for t in 0..n_tok {
            let x = &batch[t * f.n..(t + 1) * f.n];
            let (scales, codes) = inferred_thoughts::quant::fp4_activation(x);
            for r in 0..n_out {
                let cpu = inferred_thoughts::quant::vec_dot_nvfp4_fp4(&wbytes[r * row_bytes..(r + 1) * row_bytes], x) as f64;
                let mut sum_abs = 0.0f64;
                for s in 0..n_sub {
                    let xs = ue4m3_value(scales[s]);
                    let term: f64 = (s * 16..(s + 1) * 16)
                        .map(|e| wdeq[r * f.n + e] as f64 * e2m1_value(codes[e]) * xs)
                        .sum();
                    sum_abs += term.abs();
                }
                let bound = n_sub as f64 * f32::EPSILON as f64 * sum_abs;
                let got = dev[t * n_out + r] as f64;
                let diff = (got - cpu).abs();
                assert!(
                    diff <= bound,
                    "{} token {t} row {r}: tensor core {got:e} vs reference {cpu:e}, off by {diff:e} \
                     against a chain bound of {bound:e}. That is not addition order: suspect the \
                     register layout, a scale selector or the activation packing.",
                    f.name
                );
                if diff == 0.0 {
                    exact_hits += 1;
                }
                if bound > 0.0 {
                    worst = worst.max(diff / bound);
                }
            }
        }
        println!(
            "  {:<52} {n_tok} tokens x {n_out} rows within the chain bound; {exact_hits} bit-identical, \
             worst at {worst:.3} of the bound",
            f.name
        );
    }
}
