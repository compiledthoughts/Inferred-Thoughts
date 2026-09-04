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
    for f in fixtures() {
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
    let fx = fixtures();
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
            let ours = inferred_thoughts::quant::vec_dot_q8_k(f.ty, w, &f.activation);
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
