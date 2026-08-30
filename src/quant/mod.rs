//! Dequantization from GGUF block formats to f32.
//!
//! v0 scope: F32, F16, BF16 and Q8_0. k-quants and i-quants come later.
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
}
