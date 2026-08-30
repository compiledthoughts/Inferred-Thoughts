//! ggml tensor type codes and their block geometry.
//!
//! Every constant here is transcribed from the reference implementation, not
//! derived or remembered:
//!   - discriminants: `enum ggml_type` in `ggml/include/ggml.h`
//!   - block sizes:   the `type_traits` table in `ggml/src/ggml.c`
//!   - byte sizes:    the `static_assert(sizeof(block_*) == ...)` line beside
//!                    each block struct in `ggml/src/ggml-common.h`
//!
//! Where a size is a formula in the reference, the formula is reproduced in the
//! comment so it can be re-checked against the source without arithmetic.
//! Relevant constants: QK_K = 256, K_SCALE_SIZE = 12, QK4_NL = 32,
//! QK_MXFP4 = 32, QK_NVFP4 = 64, QK_NVFP4_SUB = 16, QK1_0 = 128,
//! IQ3S_N_SCALE = QK_K/64 = 4, sizeof(ggml_half) = 2.

use crate::error::{Error, Result};

/// ggml's maximum tensor rank (`GGML_MAX_DIMS`).
pub const MAX_DIMS: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    I8,
    I16,
    I32,
    I64,
    F64,
    Iq1M,
    Bf16,
    Tq1_0,
    Tq2_0,
    Mxfp4,
    Nvfp4,
    Q1_0,
}

impl GgmlType {
    /// Discriminants from `enum ggml_type`. Codes 4, 5, 31, 32, 33, 36, 37 and
    /// 38 were removed from ggml and are reported as such rather than guessed.
    pub fn from_u32(ty: u32) -> Option<Self> {
        Some(match ty {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::Iq1M,
            30 => Self::Bf16,
            34 => Self::Tq1_0,
            35 => Self::Tq2_0,
            39 => Self::Mxfp4,
            40 => Self::Nvfp4,
            41 => Self::Q1_0,
            _ => return None,
        })
    }

    /// The name ggml prints for this type, so our output can be diffed against
    /// `gguf_dump.py` directly.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Iq2Xs => "IQ2_XS",
            Self::Iq3Xxs => "IQ3_XXS",
            Self::Iq1S => "IQ1_S",
            Self::Iq4Nl => "IQ4_NL",
            Self::Iq3S => "IQ3_S",
            Self::Iq2S => "IQ2_S",
            Self::Iq4Xs => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::Iq1M => "IQ1_M",
            Self::Bf16 => "BF16",
            Self::Tq1_0 => "TQ1_0",
            Self::Tq2_0 => "TQ2_0",
            Self::Mxfp4 => "MXFP4",
            Self::Nvfp4 => "NVFP4",
            Self::Q1_0 => "Q1_0",
        }
    }

    /// Elements per block. 1 for unquantized types.
    pub fn block_size(self) -> u64 {
        match self {
            Self::F32
            | Self::F16
            | Self::Bf16
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::F64 => 1,

            Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1
            | Self::Iq4Nl
            | Self::Mxfp4 => 32,

            Self::Nvfp4 => 64,
            Self::Q1_0 => 128,

            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::Iq2Xxs
            | Self::Iq2Xs
            | Self::Iq3Xxs
            | Self::Iq1S
            | Self::Iq3S
            | Self::Iq2S
            | Self::Iq4Xs
            | Self::Iq1M
            | Self::Tq1_0
            | Self::Tq2_0 => 256,
        }
    }

    /// Bytes per block.
    pub fn type_size(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Bf16 => 2,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::F64 => 8,

            Self::Q4_0 => 18,  // half + QK4_0/2       = 2 + 16
            Self::Q4_1 => 20,  // 2*half + QK4_1/2     = 4 + 16
            Self::Q5_0 => 22,  // half + u32 + QK5_0/2 = 2 + 4 + 16
            Self::Q5_1 => 24,  // 2*half + u32 + QK5_1/2 = 4 + 4 + 16
            Self::Q8_0 => 34,  // half + QK8_0         = 2 + 32
            Self::Q8_1 => 36,  // 2*half + QK8_1       = 4 + 32
            Self::Q1_0 => 18,  // half + QK1_0/8       = 2 + 16
            Self::Mxfp4 => 17, // u8 + QK_MXFP4/2      = 1 + 16
            Self::Nvfp4 => 36, // u8*(64/16) + 64/2    = 4 + 32

            Self::Q2K => 84,  // 2*half + QK_K/16 + QK_K/4          = 4 + 16 + 64
            Self::Q3K => 110, // half + QK_K/4 + QK_K/8 + 12        = 2 + 64 + 32 + 12
            Self::Q4K => 144, // 2*half + K_SCALE_SIZE + QK_K/2     = 4 + 12 + 128
            Self::Q5K => 176, // 2*half + K_SCALE + QK_K/2 + QK_K/8 = 4 + 12 + 128 + 32
            Self::Q6K => 210, // half + QK_K/16 + 3*QK_K/4          = 2 + 16 + 192
            Self::Q8K => 292, // f32 + QK_K + QK_K/16*i16           = 4 + 256 + 32

            Self::Iq2Xxs => 66,  // half + QK_K/8*u16              = 2 + 64
            Self::Iq2Xs => 74,   // half + QK_K/8*u16 + QK_K/32    = 2 + 64 + 8
            Self::Iq2S => 82,    // half + QK_K/4 + QK_K/16        = 2 + 64 + 16
            Self::Iq3Xxs => 98,  // half + 3*(QK_K/8)              = 2 + 96
            Self::Iq3S => 110,   // half + 13*(QK_K/32) + QK_K/64  = 2 + 104 + 4
            Self::Iq1S => 50,    // half + QK_K/8 + QK_K/16        = 2 + 32 + 16
            Self::Iq1M => 56,    // QK_K/8 + QK_K/16 + QK_K/32     = 32 + 16 + 8
            Self::Iq4Nl => 18,   // half + QK4_NL/2                = 2 + 16
            Self::Iq4Xs => 136,  // half + u16 + QK_K/64 + QK_K/2  = 2 + 2 + 4 + 128
            Self::Tq1_0 => 54,   // half + QK_K/64 + (QK_K - 4*QK_K/64)/5 = 2 + 4 + 48
            Self::Tq2_0 => 66,   // half + QK_K/4                  = 2 + 64
        }
    }

    pub fn is_quantized(self) -> bool {
        self.block_size() != 1
    }

    /// Bytes occupied by `n_elements` values of this type.
    ///
    /// `n_elements` must be a whole number of blocks; ggml guarantees this by
    /// requiring the row length `ne[0]` to be block-aligned, which
    /// [`super::TensorInfo`] checks at parse time.
    pub fn n_bytes(self, n_elements: u64) -> u64 {
        n_elements / self.block_size() * self.type_size()
    }
}

/// Type codes that once existed but were removed from ggml. Named so a file
/// containing one produces a useful message instead of "unknown type 31".
pub(crate) fn removed_type_name(ty: u32) -> Option<&'static str> {
    Some(match ty {
        4 => "Q4_2 (removed)",
        5 => "Q4_3 (removed)",
        31 => "Q4_0_4_4 (removed, use Q4_0 with runtime repacking)",
        32 => "Q4_0_4_8 (removed, use Q4_0 with runtime repacking)",
        33 => "Q4_0_8_8 (removed, use Q4_0 with runtime repacking)",
        36 => "IQ4_NL_4_4 (removed, use IQ4_NL with runtime repacking)",
        37 => "IQ4_NL_4_8 (removed, use IQ4_NL with runtime repacking)",
        38 => "IQ4_NL_8_8 (removed, use IQ4_NL with runtime repacking)",
        _ => return None,
    })
}

pub(crate) fn tensor_type_from_u32(name: &str, ty: u32) -> Result<GgmlType> {
    if let Some(t) = GgmlType::from_u32(ty) {
        return Ok(t);
    }
    if let Some(removed) = removed_type_name(ty) {
        return Err(Error::RemovedTensorType {
            name: name.to_string(),
            ty: removed,
        });
    }
    Err(Error::UnknownTensorType {
        name: name.to_string(),
        ty,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Spot-checks against the static_asserts in ggml-common.h. These are the
    // types this project actually reads; a wrong size here silently shifts
    // every subsequent tensor offset.
    #[test]
    fn block_geometry_matches_reference() {
        assert_eq!((GgmlType::Q8_0.block_size(), GgmlType::Q8_0.type_size()), (32, 34));
        assert_eq!((GgmlType::Q4K.block_size(), GgmlType::Q4K.type_size()), (256, 144));
        assert_eq!((GgmlType::Q6K.block_size(), GgmlType::Q6K.type_size()), (256, 210));
        assert_eq!((GgmlType::Iq4Xs.block_size(), GgmlType::Iq4Xs.type_size()), (256, 136));
        assert_eq!((GgmlType::F32.block_size(), GgmlType::F32.type_size()), (1, 4));
        assert_eq!((GgmlType::F16.block_size(), GgmlType::F16.type_size()), (1, 2));
    }

    #[test]
    fn q8_0_is_8_5_bits_per_weight() {
        // 34 bytes per 32 weights.
        assert_eq!(GgmlType::Q8_0.n_bytes(32), 34);
        assert_eq!(GgmlType::Q8_0.n_bytes(4096), 4096 / 32 * 34);
    }

    #[test]
    fn removed_types_are_not_parsed_as_valid() {
        for ty in [4u32, 5, 31, 32, 33, 36, 37, 38] {
            assert!(GgmlType::from_u32(ty).is_none());
            assert!(removed_type_name(ty).is_some());
        }
    }
}
