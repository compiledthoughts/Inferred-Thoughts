//! Stage 2 acceptance: our dequantization must reproduce llama.cpp's, exactly.
//!
//! Fixtures under `tests/fixtures/` are produced by `scripts/dump_fixtures.py`
//! and carry both the raw quantized bytes and the reference f32 output from
//! `gguf.quants.dequantize`. They are self-contained, so these tests do not
//! need a model file.
//!
//! The comparison is bit-exact, not within a tolerance. Q8_0 dequantization is
//! an int8 widened to f32 times an f16 scale widened to f32; both sides perform
//! the same IEEE f32 multiply, so any difference at all is a real bug and not
//! rounding. Per `CLAUDE.md`, the fix for a failure here is never to loosen
//! this.

use std::fs;
use std::path::{Path, PathBuf};

use inferred_thoughts::gguf::GgmlType;
use inferred_thoughts::quant::dequantize;

struct Fixture {
    name: String,
    ty: GgmlType,
    raw: Vec<u8>,
    reference: Vec<f32>,
    source: PathBuf,
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load(path: &Path) -> Fixture {
    let b = fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));

    let mut p = 0usize;
    let mut take = |n: usize| {
        let s = &b[p..p + n];
        p += n;
        s
    };

    assert_eq!(take(4), b"ITFX", "bad magic in {}", path.display());
    let version = u32::from_le_bytes(take(4).try_into().unwrap());
    assert_eq!(version, 1, "unsupported fixture version in {}", path.display());

    let ty_code = u32::from_le_bytes(take(4).try_into().unwrap());
    let n = u64::from_le_bytes(take(8).try_into().unwrap()) as usize;
    let raw_len = u64::from_le_bytes(take(8).try_into().unwrap()) as usize;
    let name_len = u32::from_le_bytes(take(4).try_into().unwrap()) as usize;
    let name = String::from_utf8(take(name_len).to_vec()).unwrap();
    let raw = take(raw_len).to_vec();

    let mut reference = Vec::with_capacity(n);
    for _ in 0..n {
        reference.push(f32::from_le_bytes(take(4).try_into().unwrap()));
    }

    let ty = GgmlType::from_u32(ty_code)
        .unwrap_or_else(|| panic!("fixture {} has unknown ggml type {ty_code}", path.display()));

    Fixture {
        name,
        ty,
        raw,
        reference,
        source: path.to_path_buf(),
    }
}

fn all_fixtures() -> Vec<Fixture> {
    let dir = fixture_dir();
    let mut out: Vec<Fixture> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "itfx"))
        .map(|p| load(&p))
        .collect();
    out.sort_by(|a, b| a.source.cmp(&b.source));
    assert!(
        !out.is_empty(),
        "no fixtures in {} -- run scripts/dump_fixtures.py",
        dir.display()
    );
    out
}

/// On a mismatch, report the actual numbers rather than just a count: the
/// first differing index, both values there, and the worst difference anywhere.
fn compare(f: &Fixture) {
    let ours = dequantize(&f.raw, f.ty, f.reference.len())
        .unwrap_or_else(|e| panic!("dequantize failed for {}: {e}", f.name));

    assert_eq!(ours.len(), f.reference.len(), "length mismatch for {}", f.name);

    let mut first_diff = None;
    let mut worst = (0usize, 0.0f64);
    for (i, (&a, &b)) in ours.iter().zip(&f.reference).enumerate() {
        if a.to_bits() != b.to_bits() {
            if first_diff.is_none() {
                first_diff = Some(i);
            }
            let d = (a as f64 - b as f64).abs();
            if d > worst.1 {
                worst = (i, d);
            }
        }
    }

    if let Some(i) = first_diff {
        let lo = i.saturating_sub(2);
        let hi = (i + 3).min(ours.len());
        panic!(
            "{} ({}): dequantization differs from llama.cpp\n\
             first difference at index {i}: ours {:?} (0x{:08x}), reference {:?} (0x{:08x})\n\
             worst difference {:e} at index {}\n\
             ours[{lo}..{hi}]      = {:?}\n\
             reference[{lo}..{hi}] = {:?}",
            f.name,
            f.ty.name(),
            ours[i],
            ours[i].to_bits(),
            f.reference[i],
            f.reference[i].to_bits(),
            worst.1,
            worst.0,
            &ours[lo..hi],
            &f.reference[lo..hi],
        );
    }
}

#[test]
fn matches_llama_cpp_bit_exactly() {
    let fixtures = all_fixtures();
    for f in &fixtures {
        compare(f);
        println!("ok  {:<8} {} elements  [{}]", f.ty.name(), f.reference.len(), f.name);
    }
}

/// Every type v0 claims to support should actually have a fixture behind it,
/// so the suite fails loudly if one is dropped rather than silently thinning.
#[test]
fn covers_every_supported_type() {
    let have: Vec<&'static str> = all_fixtures().iter().map(|f| f.ty.name()).collect();
    for want in ["F32", "F16", "Q8_0"] {
        assert!(
            have.contains(&want),
            "no fixture covers {want}; have {have:?}"
        );
    }
}

/// The buffer-reusing path used by the forward pass must agree with the
/// allocating one, since only the latter is covered by the fixture comparison.
#[test]
fn dequantize_into_agrees_with_dequantize() {
    for f in &all_fixtures() {
        let mut buf = vec![0.0f32; f.reference.len()];
        inferred_thoughts::quant::dequantize_into(&f.raw, f.ty, &mut buf).unwrap();
        let owned = dequantize(&f.raw, f.ty, f.reference.len()).unwrap();
        assert!(
            buf.iter().zip(&owned).all(|(a, b)| a.to_bits() == b.to_bits()),
            "dequantize_into disagrees with dequantize for {}",
            f.name
        );
    }
}
