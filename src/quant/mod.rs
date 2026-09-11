//! Dequantization from GGUF block formats to f32.
//!
//! F32, F16, BF16, Q8_0, the three formats the 35B needs: Q5_K, Q6_K and
//! IQ4_XS, and NVFP4 for the 35B's NVFP4 checkpoint. Nothing is added
//! speculatively — a format arrives when a model we actually run declares it.
//!
//! Layouts are read from the reference, not guessed:
//!   - `dequantize_row_q8_0` in `ggml/src/ggml-quants.c`
//!   - the `block_q8_0` struct and its `static_assert` in `ggml-common.h`
//!
//! A Q8_0 block is 34 bytes: an f16 scale `d`, then 32 int8 quants. The value
//! of element `j` is `qs[j] * d`, with `d` widened to f32 before the multiply —
//! matching `y[i*qk + j] = x[i].qs[j]*d` where `d` is already
//! `GGML_FP16_TO_FP32(x[i].d)`.

pub mod half;
pub mod kquant;

pub use kquant::{Q8KRow, q8_k_blocks, vec_dot_q8_k};

use crate::error::{Error, Result};
use crate::gguf::GgmlType;
use half::{bf16_to_f32, f16_to_f32};

/// Dequantize `n` elements from `data` into a new `Vec<f32>`.
///
/// This is the fixture-testing entry point. It is deliberately *not* what the
/// forward pass uses: materializing an f32 copy of a weight tensor would need
/// ~4x the model's file size in RAM. See `CLAUDE.md`.
pub fn dequantize(data: &[u8], ty: GgmlType, n: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; n];
    dequantize_into(data, ty, &mut out)?;
    Ok(out)
}

/// Dequantize into a caller-provided buffer, allocating nothing.
///
/// `out.len()` determines how many elements are read, and must be a whole
/// number of blocks.
pub fn dequantize_into(data: &[u8], ty: GgmlType, out: &mut [f32]) -> Result<()> {
    let n = out.len();
    let block_size = ty.block_size();

    if (n as u64) % block_size != 0 {
        return Err(Error::QuantNotBlockAligned {
            ty: ty.name(),
            n,
            block_size,
        });
    }

    let expected = ty.n_bytes(n as u64);
    if data.len() as u64 != expected {
        return Err(Error::QuantSizeMismatch {
            ty: ty.name(),
            n,
            expected,
            got: data.len(),
        });
    }

    match ty {
        GgmlType::F32 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        GgmlType::F16 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        GgmlType::Bf16 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(2)) {
                *o = bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        GgmlType::Q8_0 => dequantize_q8_0(data, out),
        GgmlType::Q5K => dequantize_q5_k(data, out),
        GgmlType::Q6K => dequantize_q6_k(data, out),
        GgmlType::Iq4Xs => dequantize_iq4_xs(data, out),
        GgmlType::Nvfp4 => dequantize_nvfp4(data, out),
        other => {
            return Err(Error::UnsupportedQuantType { ty: other.name() });
        }
    }

    Ok(())
}

/// Elements per Q8_0 block (`QK8_0`).
const QK8_0: usize = 32;
/// Bytes per Q8_0 block: `sizeof(ggml_half) + QK8_0`.
const Q8_0_BLOCK_BYTES: usize = 2 + QK8_0;

fn dequantize_q8_0(data: &[u8], out: &mut [f32]) {
    for (block, dst) in data
        .chunks_exact(Q8_0_BLOCK_BYTES)
        .zip(out.chunks_exact_mut(QK8_0))
    {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (o, &q) in dst.iter_mut().zip(&block[2..]) {
            *o = (q as i8) as f32 * d;
        }
    }
}

// ------------------------------------------------------- k-quants and i-quants
//
// The 35B needs three more formats. All three are super-block layouts: 256
// elements share one f16 scale, and sub-blocks carry their own smaller scales
// packed into the leftover bits. Everything below is transcribed from
// `ggml/src/ggml-quants.c` and `ggml/src/ggml-common.h`, function by function —
// `CLAUDE.md` forbids inventing format constants, and these layouts have no
// redundancy to catch a guess.

/// Elements per super-block (`QK_K` in ggml-common.h).
const QK_K: usize = 256;
/// Bytes of packed 6-bit scales-and-mins in Q4_K and Q5_K (`K_SCALE_SIZE`).
const K_SCALE_SIZE: usize = 12;

/// `kvalues_iq4nl` from ggml-common.h: the 16-entry non-linear codebook that
/// IQ4_NL and IQ4_XS both index.
///
/// **This is what makes IQ4_XS an i-quant rather than a k-quant.** A 4-bit
/// k-quant would store a uniform grid; here the four bits are an index into a
/// fixed, unevenly spaced table chosen to fit the distribution of weights.
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// `get_scale_min_k4` from ggml-quants.c.
///
/// Q4_K and Q5_K pack eight 6-bit scales and eight 6-bit mins into 12 bytes.
/// The first four of each sit in the low 6 bits of bytes 0..8; the last four
/// are split, taking their low nibble from bytes 8..12 and their high two bits
/// from the top of an earlier byte.
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// `dequantize_row_q5_K`.
///
/// Eight sub-blocks of 32, each with its own 6-bit scale and 6-bit min, so a
/// value is `d * sc * q - dmin * m`. The fifth bit of each quant lives in a
/// separate `qh` plane, selected by a mask that shifts left by two per 64
/// elements.
fn dequantize_q5_k(data: &[u8], out: &mut [f32]) {
    const BYTES: usize = 176;
    for (b, chunk) in data.chunks_exact(BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([chunk[2], chunk[3]]));
        let scales = &chunk[4..4 + K_SCALE_SIZE];
        let qh = &chunk[16..16 + QK_K / 8];
        let qs = &chunk[48..48 + QK_K / 2];
        let y = &mut out[b * QK_K..(b + 1) * QK_K];

        let (mut u1, mut u2) = (1u8, 2u8);
        for half in 0..QK_K / 64 {
            let ql = &qs[half * 32..half * 32 + 32];
            let (sc, m) = get_scale_min_k4(half * 2, scales);
            let (d1, m1) = (d * sc as f32, dmin * m as f32);
            let (sc, m) = get_scale_min_k4(half * 2 + 1, scales);
            let (d2, m2) = (d * sc as f32, dmin * m as f32);

            let base = half * 64;
            for l in 0..32 {
                let hi = if qh[l] & u1 != 0 { 16u32 } else { 0 };
                y[base + l] = d1 * ((ql[l] & 0xF) as u32 + hi) as f32 - m1;
            }
            for l in 0..32 {
                let hi = if qh[l] & u2 != 0 { 16u32 } else { 0 };
                y[base + 32 + l] = d2 * ((ql[l] >> 4) as u32 + hi) as f32 - m2;
            }
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/// `dequantize_row_q6_K`.
///
/// Sixteen sub-blocks of 16, each with a signed 8-bit scale, so a value is
/// `d * sc * q` with no min term. Six bits per quant: four in `ql`, two in
/// `qh`, biased by -32 to centre the range.
fn dequantize_q6_k(data: &[u8], out: &mut [f32]) {
    const BYTES: usize = 210;
    for (b, chunk) in data.chunks_exact(BYTES).enumerate() {
        let ql_all = &chunk[0..QK_K / 2];
        let qh_all = &chunk[QK_K / 2..QK_K / 2 + QK_K / 4];
        let sc_all = &chunk[QK_K / 2 + QK_K / 4..QK_K / 2 + QK_K / 4 + QK_K / 16];
        let d = f16_to_f32(u16::from_le_bytes([chunk[BYTES - 2], chunk[BYTES - 1]]));
        let y = &mut out[b * QK_K..(b + 1) * QK_K];

        // Two passes of 128 elements; each consumes 64 ql, 32 qh and 8 scales.
        for n in 0..QK_K / 128 {
            let ql = &ql_all[n * 64..];
            let qh = &qh_all[n * 32..];
            let sc = &sc_all[n * 8..];
            let base = n * 128;
            for l in 0..32 {
                let is = l / 16;
                // The int8 cast happens before the bias in the reference, and
                // the value always fits, so it is a no-op there and here.
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                // Grouped as `(d * sc) * q`, matching the reference's
                // left-to-right float evaluation.
                y[base + l] = d * sc[is] as i8 as f32 * q1 as f32;
                y[base + l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                y[base + l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                y[base + l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
            }
        }
    }
}

/// `dequantize_row_iq4_xs`.
///
/// Eight sub-blocks of 32. Each has a 6-bit scale split across a nibble in
/// `scales_l` and two bits in the 16-bit `scales_h`, biased by -32. The quants
/// are 4-bit indices into [`KVALUES_IQ4NL`], with the low nibble covering
/// elements 0..16 of the sub-block and the high nibble 16..32.
fn dequantize_iq4_xs(data: &[u8], out: &mut [f32]) {
    const BYTES: usize = 136;
    for (b, chunk) in data.chunks_exact(BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        let scales_h = u16::from_le_bytes([chunk[2], chunk[3]]);
        let scales_l = &chunk[4..4 + QK_K / 64];
        let qs = &chunk[8..8 + QK_K / 2];
        let y = &mut out[b * QK_K..(b + 1) * QK_K];

        for ib in 0..QK_K / 32 {
            let low = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32;
            let high = (((scales_h >> (2 * ib)) & 3) as i32) << 4;
            let dl = d * ((low | high) - 32) as f32;

            let q = &qs[ib * 16..ib * 16 + 16];
            let base = ib * 32;
            for j in 0..16 {
                y[base + j] = dl * KVALUES_IQ4NL[(q[j] & 0xF) as usize] as f32;
                y[base + j + 16] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
            }
        }
    }
}

// ------------------------------------------------------------------ NVFP4
//
// The NVFP4 checkpoint of the 35B, converted by llama.cpp's
// `convert_hf_to_gguf.py`, which copies ModelOpt's nibbles and E4M3 scale bits
// unchanged and regroups four 16-element blocks into one 64-element block.

/// `kvalues_mxfp4` from ggml-common.h: the E2M1 values **doubled**, sign in
/// bit 3. NVFP4 and MXFP4 share it.
pub(crate) const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// `ggml_ue4m3_to_fp32` from ggml-impl.h: an unsigned E4M3 scale, bias 7,
/// **halved** to pair with the doubled [`KVALUES_MXFP4`], so a weight is the
/// ModelOpt value E2M1 x E4M3 exactly. 0 and 0x7F (the NaN code) read as 0.
/// Bit 7 is not examined, as in the reference.
pub(crate) fn ue4m3_to_f32(x: u8) -> f32 {
    if x == 0 || x == 0x7f {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xf) as i32;
    let man = (x & 7) as f32;
    // `ldexpf` in the reference; a power of two is exact either way.
    let raw = if exp == 0 {
        man * 2f32.powi(-9)
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    };
    raw * 0.5
}

/// `dequantize_row_nvfp4`.
///
/// A 36-byte block is four UE4M3 scales `d`, then 32 bytes of nibbles. Byte
/// `j` of sub-block `s` is `qs[8s + j]`: its low nibble is element `j` of the
/// sub-block and its high nibble element `j + 8`. The per-tensor second scale
/// is a separate tensor and is **not** applied here, as in the reference.
fn dequantize_nvfp4(data: &[u8], out: &mut [f32]) {
    const BYTES: usize = 36;
    for (block, y) in data.chunks_exact(BYTES).zip(out.chunks_exact_mut(64)) {
        for s in 0..4 {
            let d = ue4m3_to_f32(block[s]);
            let qs = &block[4 + s * 8..4 + s * 8 + 8];
            let yb = &mut y[s * 16..s * 16 + 16];
            for j in 0..8 {
                yb[j] = KVALUES_MXFP4[(qs[j] & 0xf) as usize] as f32 * d;
                yb[j + 8] = KVALUES_MXFP4[(qs[j] >> 4) as usize] as f32 * d;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_passthrough() {
        let vals = [1.0f32, -2.5, 0.0, 1e-30];
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(dequantize(&bytes, GgmlType::F32, 4).unwrap(), vals);
    }

    #[test]
    fn q8_0_applies_scale_per_block() {
        // One block: scale 0.5, quants 0,1,2,...  Values are q * 0.5.
        let mut block = Vec::new();
        block.extend_from_slice(&0x3800u16.to_le_bytes()); // f16 0.5
        for j in 0..QK8_0 {
            block.push(j as u8);
        }
        let got = dequantize(&block, GgmlType::Q8_0, QK8_0).unwrap();
        for (j, v) in got.iter().enumerate() {
            assert_eq!(*v, j as f32 * 0.5, "element {j}");
        }
    }

    #[test]
    fn q8_0_quants_are_signed() {
        // 0xff must read as -1, not 255.
        let mut block = Vec::new();
        block.extend_from_slice(&0x3c00u16.to_le_bytes()); // f16 1.0
        block.push(0xff);
        block.extend(std::iter::repeat_n(0u8, QK8_0 - 1));
        let got = dequantize(&block, GgmlType::Q8_0, QK8_0).unwrap();
        assert_eq!(got[0], -1.0);
    }

    #[test]
    fn q8_0_scales_are_per_block_not_shared() {
        // Two blocks with different scales; element 0 of each must differ.
        let mut data = Vec::new();
        for scale in [0x3c00u16, 0x4000] {
            // 1.0, then 2.0
            data.extend_from_slice(&scale.to_le_bytes());
            data.push(1);
            data.extend(std::iter::repeat_n(0u8, QK8_0 - 1));
        }
        let got = dequantize(&data, GgmlType::Q8_0, QK8_0 * 2).unwrap();
        assert_eq!(got[0], 1.0);
        assert_eq!(got[QK8_0], 2.0);
    }

    #[test]
    fn rejects_partial_block() {
        let data = vec![0u8; Q8_0_BLOCK_BYTES];
        assert!(matches!(
            dequantize(&data, GgmlType::Q8_0, 16),
            Err(Error::QuantNotBlockAligned { .. })
        ));
    }

    #[test]
    fn rejects_wrong_input_size() {
        let data = vec![0u8; Q8_0_BLOCK_BYTES - 1];
        assert!(matches!(
            dequantize(&data, GgmlType::Q8_0, QK8_0),
            Err(Error::QuantSizeMismatch { .. })
        ));
    }

    #[test]
    fn unimplemented_types_say_so() {
        let data = vec![0u8; 144];
        match dequantize(&data, GgmlType::Q4K, 256) {
            Err(Error::UnsupportedQuantType { ty }) => assert_eq!(ty, "Q4_K"),
            other => panic!("expected UnsupportedQuantType, got {other:?}"),
        }
    }
    use half::f32_to_f16;

    // ---------------------------------------------- k-quants and i-quants
    //
    // The fixtures in `tests/dequantize.rs` are the primary evidence: 65,536
    // real weights per format, compared bit for bit against `gguf.quants`.
    // These are structural checks that hold independently of them, so a
    // regenerated fixture cannot quietly bless a layout error.

    /// IQ4_XS is the only format here whose quants are *indices*, not numbers.
    /// This pins both the codebook and the split 6-bit scale: with the scale
    /// arranged to make `dl` exactly 1.0, the output must be the codebook
    /// itself.
    #[test]
    fn iq4_xs_indexes_the_nonlinear_codebook() {
        let mut blk = vec![0u8; 136];
        blk[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes()); // d = 1
        // dl = d * (ls - 32), and ls is (low nibble of scales_l) | (2 bits of
        // scales_h << 4). Want ls = 33 => low = 1, high = 32.
        blk[2] = 2; // scales_h bits 0..2 = 2, so high = 2 << 4 = 32
        blk[4] = 1; // scales_l[0] low nibble = 1
        // Sub-block 0: low nibble of byte j is j, high nibble is 0.
        for j in 0..16 {
            blk[8 + j] = j as u8;
        }

        let got = dequantize(&blk, GgmlType::Iq4Xs, 256).unwrap();
        for j in 0..16 {
            assert_eq!(got[j], KVALUES_IQ4NL[j] as f32, "low nibble {j}");
            assert_eq!(got[j + 16], KVALUES_IQ4NL[0] as f32, "high nibble {j}");
        }
        // The codebook is deliberately uneven -- that is the point of an
        // i-quant -- so a uniform grid would fail here.
        assert_ne!(
            KVALUES_IQ4NL[1] - KVALUES_IQ4NL[0],
            KVALUES_IQ4NL[8] - KVALUES_IQ4NL[7]
        );
    }

    /// Q6_K biases its 6-bit quants by -32, so an all-zero quant plane is not
    /// zero output. Getting the bias wrong is invisible in a magnitude check
    /// and obvious here.
    #[test]
    fn q6_k_biases_quants_by_negative_32() {
        let mut blk = vec![0u8; 210];
        blk[192] = 1; // scales[0] = 1
        blk[208..210].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        let got = dequantize(&blk, GgmlType::Q6K, 256).unwrap();
        assert_eq!(got[0], -32.0);

        // Now set the low nibble of element 0 to 15 and its high bits to 3:
        // q = 15 | (3 << 4) = 63, minus the bias is 31.
        blk[0] = 0x0F;
        blk[128] = 0x03;
        let got = dequantize(&blk, GgmlType::Q6K, 256).unwrap();
        assert_eq!(got[0], 31.0);
    }

    /// Q5_K is the only one of the three with a min term, so a zero quant does
    /// not give zero output either -- it gives `-dmin * m`.
    #[test]
    fn q5_k_subtracts_the_block_minimum() {
        let mut blk = vec![0u8; 176];
        blk[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes()); // d
        blk[2..4].copy_from_slice(&f32_to_f16(2.0).to_le_bytes()); // dmin
        // scales[0] is the 6-bit scale for sub-block 0, scales[4] its min.
        blk[4] = 3;
        blk[8] = 5;
        let got = dequantize(&blk, GgmlType::Q5K, 256).unwrap();
        // quant 0, so value = d * sc * 0 - dmin * m = -2 * 5.
        assert_eq!(got[0], -10.0);

        // The fifth bit lives in the qh plane: setting it adds 16 quant steps.
        blk[16] = 1; // qh[0] bit 0
        let got = dequantize(&blk, GgmlType::Q5K, 256).unwrap();
        assert_eq!(got[0], 1.0 * 3.0 * 16.0 - 10.0);
    }

    /// Super-blocks must not bleed into each other. Two blocks with different
    /// scales, checked at the seam -- the classic stride bug.
    #[test]
    fn super_blocks_are_independent() {
        for (ty, bytes, d_off) in [
            (GgmlType::Q5K, 176usize, 0usize),
            (GgmlType::Q6K, 210, 208),
            (GgmlType::Iq4Xs, 136, 0),
        ] {
            let mut data = vec![0u8; bytes * 2];
            data[d_off..d_off + 2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
            data[bytes + d_off..bytes + d_off + 2]
                .copy_from_slice(&f32_to_f16(4.0).to_le_bytes());
            let got = dequantize(&data, ty, 512).unwrap();
            // Whatever the layout decodes to, the second block's values must
            // be exactly 4x the first's -- same quants, scale 4x.
            for i in 0..256 {
                assert_eq!(
                    got[256 + i],
                    got[i] * 4.0,
                    "{}: element {i} across the block boundary",
                    ty.name()
                );
            }
        }
    }

    /// UE4M3 scale 0x38 is 2^0 = 1.0 before the halving, so every weight is
    /// the E2M1 value itself: this pins the doubled table and the half together.
    /// Sub-block 1 carries 0x40 (2.0) to catch a scale applied to the wrong
    /// sixteen, and byte `j`'s two nibbles land on elements `j` and `j + 8`.
    #[test]
    fn nvfp4_is_e2m1_times_e4m3_with_split_nibbles() {
        const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
        let mut blk = vec![0u8; 36];
        blk[0] = 0x38;
        blk[1] = 0x40;
        for j in 0..8 {
            blk[4 + j] = (j as u8) | ((15 - j as u8) << 4);
            blk[12 + j] = j as u8;
        }
        let got = dequantize(&blk, GgmlType::Nvfp4, 64).unwrap();
        for j in 0..8 {
            assert_eq!(got[j], E2M1[j], "low nibble {j}");
            assert_eq!(got[j + 8], E2M1[15 - j], "high nibble {j}");
            assert_eq!(got[16 + j], 2.0 * E2M1[j], "sub-block 1, element {j}");
        }
        // Sub-blocks 2 and 3 have scale code 0, which reads as zero.
        assert!(got[32..].iter().all(|&v| v == 0.0));
    }

    /// The scale decode against the formula, exhaustively: subnormals are
    /// `m * 2^-9`, the NaN code reads 0, and bit 7 is not examined.
    #[test]
    fn ue4m3_decodes_every_code() {
        for x in 0u8..=255 {
            // The reference tests `x == 0x7F` on the whole byte, so 0xff is not
            // the NaN code there: it decodes as 0x7f's bits would, to 240.
            if x == 0xff {
                assert_eq!(ue4m3_to_f32(x), 240.0);
                continue;
            }
            let c = x & 0x7f;
            let (e, m) = ((c >> 3) as i32, (c & 7) as f64);
            let want = if c == 0 || c == 0x7f {
                0.0
            } else if e == 0 {
                m * 2f64.powi(-9) * 0.5
            } else {
                (1.0 + m / 8.0) * 2f64.powi(e - 7) * 0.5
            };
            assert_eq!(ue4m3_to_f32(x) as f64, want, "code {x:#04x}");
        }
        // The largest finite code, 448, halved.
        assert_eq!(ue4m3_to_f32(0x7e), 224.0);
    }

}
