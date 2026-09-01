//! Model architectures, written against the [`crate::ops::Ops`] seam.

pub mod qwen3;
pub mod qwen35;

pub use qwen3::Qwen3;
pub use qwen35::Qwen35;

use crate::cache::{KvCache, RecurrentState};
use crate::error::{Error, Result};
use crate::gguf::{GgufFile, TensorInfo};
use crate::ops::{Ops, Weights};
use crate::profile::Ctx;
use crate::quant::dequantize;

/// The architectures this engine runs, chosen at load time from the file.
///
/// **An enum rather than a trait, deliberately.** `CLAUDE.md` scopes the
/// project to a handful of architectures on purpose, and the two here do not
/// have the same shape: `qwen3` runs a whole batch in one pass, `qwen35` is a
/// sequential scan that runs one token at a time, and only one of them needs
/// recurrent state. A trait would have to either take the union of those
/// signatures — which is this enum with extra indirection — or grow associated
/// cache types that every caller then has to name.
///
/// The other reason is what comes next. Layer placement makes the engine ask
/// "where does layer `il` run", which is a question about the *schedule*, not
/// about the architecture. Keeping the architectures as data rather than
/// behind a `dyn` boundary leaves the engine free to interleave them.
pub enum Model<'a> {
    Qwen3(Qwen3<'a>),
    Qwen35(Qwen35<'a>),
}

impl<'a> Model<'a> {
    /// Read `general.architecture` and load the matching stack.
    ///
    /// Fails with the architecture name rather than guessing, per the rule
    /// against inventing format constants.
    pub fn load(f: &'a GgufFile) -> Result<Self> {
        match f.metadata.architecture()? {
            "qwen3" => Ok(Model::Qwen3(Qwen3::load(f)?)),
            "qwen35" => Ok(Model::Qwen35(Qwen35::load(f)?)),
            other => Err(Error::UnsupportedArchitecture {
                arch: other.to_string(),
                supported: "qwen3, qwen35",
            }),
        }
    }

    pub fn arch(&self) -> &'static str {
        match self {
            Model::Qwen3(_) => "qwen3",
            Model::Qwen35(_) => "qwen35",
        }
    }

    /// Blocks actually executed. For `qwen35` this excludes MTP blocks, which
    /// are loaded by llama.cpp but never run in a normal pass.
    pub fn n_layer(&self) -> usize {
        match self {
            Model::Qwen3(m) => m.cfg.n_layer,
            Model::Qwen35(m) => m.cfg.n_main_layer(),
        }
    }

    pub fn n_vocab(&self) -> usize {
        match self {
            Model::Qwen3(m) => m.cfg.n_vocab,
            Model::Qwen35(m) => m.cfg.n_vocab,
        }
    }

    pub fn kv_dim(&self) -> usize {
        match self {
            Model::Qwen3(m) => m.cfg.kv_dim(),
            Model::Qwen35(m) => m.cfg.kv_dim(),
        }
    }

    /// KV slabs the cache must hold.
    ///
    /// Not the same as [`Model::n_layer`] for `qwen35`, where only the
    /// attention layers attend — 8 of 32 on the 9B, 10 of 40 on the 35B. That
    /// gap is the reason a 262k context is affordable at all.
    pub fn n_kv_layer(&self) -> usize {
        match self {
            Model::Qwen3(m) => m.cfg.n_layer,
            Model::Qwen35(m) => m.n_kv_layer(),
        }
    }

    /// `(n_layer, conv_len, ssm_len)` for architectures with recurrent state.
    pub fn recurrent_dims(&self) -> Option<(usize, usize, usize)> {
        match self {
            Model::Qwen3(_) => None,
            Model::Qwen35(m) => Some((
                m.cfg.n_main_layer(),
                m.cfg.conv_state_len(),
                m.cfg.ssm_state_len(),
            )),
        }
    }

    pub fn weight_bytes_per_pass(&self) -> u64 {
        match self {
            Model::Qwen3(m) => m.weight_bytes_per_pass(),
            Model::Qwen35(m) => m.weight_bytes_per_pass(),
        }
    }

    /// Run `tokens` from absolute position `start_pos` and return logits for
    /// the last one.
    ///
    /// The two architectures differ in a way the caller does not have to care
    /// about: `qwen3` takes the whole batch, and `qwen35` is fed one token at a
    /// time because the delta rule is a sequential scan — token `t`'s state
    /// update is token `t+1`'s input, so there is no batched form of it short
    /// of llama.cpp's separate chunked algorithm. Prefill on `qwen35` is
    /// therefore correct and slow, which is the right order to get them in.
    pub fn forward<O: Ops>(
        &self,
        ops: &O,
        tokens: &[u32],
        start_pos: usize,
        kv: &mut KvCache,
        rs: Option<&mut RecurrentState>,
        ctx: &mut Ctx<'_>,
    ) -> Result<Vec<f32>> {
        match self {
            Model::Qwen3(m) => m.forward(ops, tokens, start_pos, kv, ctx),
            Model::Qwen35(m) => {
                let rs = rs.ok_or_else(|| Error::InconsistentArchitecture {
                    what: "recurrent state",
                    detail: "qwen35 needs recurrent state and none was supplied".to_string(),
                })?;
                let mut logits = Vec::new();
                for (i, &token) in tokens.iter().enumerate() {
                    logits = m.forward(ops, token, start_pos + i, kv, rs, ctx)?;
                }
                Ok(logits)
            }
        }
    }
}

/// So a caller holding a concrete stack can hand it to the engine without
/// naming the enum. The per-op tests build a `Qwen3` directly because they
/// call its batch `forward`, and should not have to care that the engine
/// stores architectures as a sum type.
impl<'a> From<Qwen3<'a>> for Model<'a> {
    fn from(m: Qwen3<'a>) -> Self {
        Model::Qwen3(m)
    }
}

impl<'a> From<Qwen35<'a>> for Model<'a> {
    fn from(m: Qwen35<'a>) -> Self {
        Model::Qwen35(m)
    }
}

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
