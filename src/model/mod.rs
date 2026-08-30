//! Model architectures, written against the [`crate::ops::Ops`] seam.

pub mod qwen3;
pub mod qwen35;

pub use qwen3::Qwen3;

use crate::error::{Error, Result};
use crate::gguf::{GgufFile, TensorInfo};
use crate::ops::Weights;
use crate::quant::dequantize;

/// Look up a tensor by name, failing with the name rather than an index panic.
pub(crate) fn tensor<'a>(f: &'a GgufFile, name: &str) -> Result<&'a TensorInfo> {
    f.tensor(name).ok_or_else(|| Error::MissingTensor {
        name: name.to_string(),
    })
}

/// A 2-D weight matrix as a borrow of the mapped file, shape-checked.
///
/// `expected` is `{n_in, n_out}` in ggml order, where `n_in` is the contraction
/// dimension. Checking here means a wrong architecture assumption fails at load
/// with both shapes named, instead of silently producing wrong numbers.
pub(crate) fn matrix<'a>(
    f: &'a GgufFile,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Weights<'a>> {
    let info = tensor(f, name)?;
    if info.dims != vec![n_in as u64, n_out as u64] {
        return Err(Error::TensorShapeMismatch {
            name: name.to_string(),
            expected: vec![n_in as u64, n_out as u64],
            got: info.dims.clone(),
        });
    }
    Ok(Weights {
        data: f.tensor_bytes(info),
        ty: info.ty,
        n_in,
        n_out,
    })
}

/// A 1-D vector (norm weights), dequantized once at load.
///
/// These are tiny — a few hundred KB across the whole model — so unlike the
/// weight matrices there is no reason to keep them packed.
pub(crate) fn vector(f: &GgufFile, name: &str, len: usize) -> Result<Vec<f32>> {
    let info = tensor(f, name)?;
    if info.dims != vec![len as u64] {
        return Err(Error::TensorShapeMismatch {
            name: name.to_string(),
            expected: vec![len as u64],
            got: info.dims.clone(),
        });
    }
    dequantize(f.tensor_bytes(info), info.ty, len)
}
