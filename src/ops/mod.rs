//! Tensor primitives, behind a backend seam.
//!
//! Three implementations are planned: `naive` (scalar f32 Rust, the oracle we
//! own), `ggml` (CPU SIMD via FFI, for the dense path), and `cuda` (ours, for
//! the MoE expert path). Only `naive` exists today. Model code is written
//! against the [`Ops`] trait so adding a backend never touches it.
//!
//! Shapes follow ggml's convention: a weight is `{ne0, ne1}` where **ne0 is the
//! contraction dimension**, so `{n_in, n_out}` maps `n_in -> n_out`, and rows of
//! length `n_in` are contiguous.

pub mod naive;

use crate::gguf::GgmlType;

/// A weight matrix, left in whatever quantized form the file stores it.
///
/// Deliberately a borrow of the mmap rather than owned f32: materializing the
/// 9B's weights as f32 would need ~36 GB against 32 GB of RAM. See `CLAUDE.md`.
#[derive(Debug, Clone, Copy)]
pub struct Weights<'a> {
    pub data: &'a [u8],
    pub ty: GgmlType,
    /// Contraction dimension (`ne0`): the length of one row.
    pub n_in: usize,
    /// Number of rows (`ne1`): the output width.
    pub n_out: usize,
}

impl<'a> Weights<'a> {
    /// Bytes of row `j`. Rows are contiguous and each is a whole number of
    /// quantization blocks, which the GGUF parser enforces at load.
    pub fn row(&self, j: usize) -> &'a [u8] {
        let stride = self.ty.n_bytes(self.n_in as u64) as usize;
        &self.data[j * stride..(j + 1) * stride]
    }
}

/// Every primitive the qwen3 forward pass needs.
///
/// Methods write into caller-provided buffers so a backend never allocates on
/// the forward path.
pub trait Ops {
    /// `out = x / sqrt(mean(x^2) + eps) * weight`
    ///
    /// No `+1` on the weight — that is Gemma's variant, not Qwen's.
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]);

    /// RMSNorm applied independently to each `head_dim`-sized slice of `x`,
    /// in place. This is Qwen3's QK-norm: `weight` has length `head_dim` and is
    /// shared across heads.
    fn rms_norm_heads(&self, x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32);

    /// `out[j] = dot(w.row(j), x)` for every row, dequantizing as it goes.
    fn matmul(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]);

    /// NEOX-style rotary embedding, in place, over `n_heads` heads of
    /// `head_dim` each.
    ///
    /// NEOX pairs dimension `i` with `i + head_dim/2`, **not** with `i + 1`.
    /// `LLM_ARCH_QWEN3` sits under llama.cpp's "the pairs of head values are
    /// offset by n_rot/2" group.
    fn rope_neox(&self, x: &mut [f32], pos: usize, head_dim: usize, n_heads: usize, theta_base: f32);

    /// Numerically stable softmax, in place.
    fn softmax(&self, x: &mut [f32]);

    /// `gate = silu(gate) * up`, in place — the SwiGLU nonlinearity.
    fn silu_mul(&self, gate: &mut [f32], up: &[f32]);

    /// `a += b`, in place.
    fn add_assign(&self, a: &mut [f32], b: &[f32]);
}
