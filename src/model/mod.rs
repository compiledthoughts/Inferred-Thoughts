//! Model architectures, written against the [`crate::ops::Ops`] seam.

pub mod qwen3;
pub mod qwen35;
pub mod qwen4exp;

pub use qwen3::Qwen3;
pub use qwen35::Qwen35;
pub use qwen4exp::Qwen4Exp;

use crate::cache::{KvCache, RecurrentState};
use crate::error::{Error, Result};
use crate::gguf::{GgufFile, TensorInfo};
use crate::ops::{Experts, Ops, Weights};
use crate::profile::Ctx;
use crate::quant::dequantize;

/// The architectures this engine runs, chosen at load time from the file.
///
/// **An enum rather than a trait, deliberately.** `CLAUDE.md` scopes the
/// project to a handful of architectures on purpose, and only one of the two
/// here needs recurrent state. A trait would have to either take the union of
/// those signatures — which is this enum with extra indirection — or grow
/// associated cache types that every caller then has to name.
///
/// The shapes have since converged: both now take a batch and a start position,
/// which is what batched prefill bought. The remaining asymmetry is the state.
///
/// The other reason is what comes next. Layer placement makes the engine ask
/// "where does layer `il` run", which is a question about the *schedule*, not
/// about the architecture. Keeping the architectures as data rather than
/// behind a `dyn` boundary leaves the engine free to interleave them.
pub enum Model<'a> {
    Qwen3(Qwen3<'a>),
    Qwen35(Qwen35<'a>),
    /// Qwen3.8-Flash-Next (`src/model/qwen4exp.md`). The forward pass runs on the
    /// CPU backends; its new ops have no CUDA kernels until step 4.
    Qwen4Exp(Qwen4Exp<'a>),
}

impl<'a> Model<'a> {
    /// Read `general.architecture` and load the matching stack.
    ///
    /// Fails with the architecture name rather than guessing, per the rule
    /// against inventing format constants.
    pub fn load(f: &'a GgufFile) -> Result<Self> {
        match f.metadata.architecture()? {
            "qwen3" => Ok(Model::Qwen3(Qwen3::load(f)?)),
            // One stack serves both: `qwen35moe` is `qwen35` with the dense FFN
            // replaced by a router, 256 experts and a shared expert. Everything
            // else -- GatedDeltaNet, the attention blocks, the norms -- is
            // identical, which is what `CLAUDE.md` meant by keeping the FFN
            // behind a seam so the MoE variant is a delta rather than a rewrite.
            "qwen35" | "qwen35moe" => Ok(Model::Qwen35(Qwen35::load(f)?)),
            // Its own stack, not a `qwen35` variant: hyper-connections, PLE and the
            // QSA indexer change the block itself (SSD-TIER.md D17).
            "qwen4exp" => Ok(Model::Qwen4Exp(Qwen4Exp::load(f)?)),
            other => Err(Error::UnsupportedArchitecture {
                arch: other.to_string(),
                supported: "qwen3, qwen35, qwen35moe, qwen4exp (load only)",
            }),
        }
    }

    pub fn arch(&self) -> &'static str {
        match self {
            Model::Qwen3(_) => "qwen3",
            Model::Qwen35(m) if m.cfg.is_moe() => "qwen35moe",
            Model::Qwen35(_) => "qwen35",
            Model::Qwen4Exp(_) => "qwen4exp",
        }
    }

    /// The settings read from the file, for `inferred inspect --model`.
    pub fn describe(&self) -> String {
        match self {
            Model::Qwen3(m) => format!("{:#?}", m.cfg),
            Model::Qwen35(m) => format!("{:#?}", m.cfg),
            Model::Qwen4Exp(m) => format!("{:#?}", m.cfg),
        }
    }

    /// Blocks actually executed. For `qwen35` this excludes MTP blocks, which
    /// are loaded by llama.cpp but never run in a normal pass.
    pub fn n_layer(&self) -> usize {
        match self {
            Model::Qwen3(m) => m.cfg.n_layer,
            Model::Qwen35(m) => m.cfg.n_main_layer(),
            Model::Qwen4Exp(m) => m.cfg.n_layer,
        }
    }

    pub fn n_vocab(&self) -> usize {
        match self {
            Model::Qwen3(m) => m.cfg.n_vocab,
            Model::Qwen35(m) => m.cfg.n_vocab,
            Model::Qwen4Exp(m) => m.cfg.n_vocab,
        }
    }

    pub fn kv_dim(&self) -> usize {
        match self {
            Model::Qwen3(m) => m.cfg.kv_dim(),
            Model::Qwen35(m) => m.cfg.kv_dim(),
            Model::Qwen4Exp(m) => m.cfg.kv_dim(),
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
            Model::Qwen4Exp(m) => m.n_kv_layer(),
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
            // GDN state only; the PLE conv history joins it with the forward pass.
            Model::Qwen4Exp(m) => Some((m.cfg.n_layer, m.cfg.conv_state_len(), m.cfg.ssm_state_len())),
        }
    }

    pub fn weight_bytes_per_pass(&self) -> u64 {
        match self {
            Model::Qwen3(m) => m.weight_bytes_per_pass(),
            Model::Qwen35(m) => m.weight_bytes_per_pass(),
            Model::Qwen4Exp(m) => m.weight_bytes_per_pass(),
        }
    }

    /// Bytes of weight a device backend holds whole — every matmul weight except
    /// the routed experts, which the expert cache tiers. qwen3 has no experts, so
    /// that is all of a pass.
    pub fn dense_weight_bytes(&self) -> u64 {
        match self {
            Model::Qwen3(m) => m.weight_bytes_per_pass(),
            Model::Qwen35(m) => m.dense_weight_bytes(),
            Model::Qwen4Exp(m) => m.dense_weight_bytes(),
        }
    }

    /// Experts across every routed expert tensor, which a device backend's expert
    /// cache has to be able to count.
    pub fn expert_pool(&self) -> usize {
        match self {
            Model::Qwen3(_) => 0,
            Model::Qwen35(m) => m.expert_pool(),
            Model::Qwen4Exp(m) => m.expert_pool(),
        }
    }

    /// Run `tokens` from absolute position `start_pos` and return logits for
    /// the last one.
    ///
    /// Both architectures now take the whole batch. `qwen35`'s two sequential
    /// scans — the delta rule and the causal convolution — iterate over it
    /// behind the seam rather than forcing the whole layer to run a token at a
    /// time, so its matmuls still read each weight once per batch.
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
                m.forward(ops, tokens, start_pos, kv, rs, ctx)
            }
            Model::Qwen4Exp(m) => {
                let rs = rs.ok_or_else(|| Error::InconsistentArchitecture {
                    what: "recurrent state",
                    detail: "qwen4exp needs recurrent state and none was supplied".to_string(),
                })?;
                m.forward(ops, tokens, start_pos, kv, rs, ctx)
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

impl<'a> From<Qwen4Exp<'a>> for Model<'a> {
    fn from(m: Qwen4Exp<'a>) -> Self {
        Model::Qwen4Exp(m)
    }
}

/// Does a tensor's shape equal `want`, ignoring trailing dimensions of 1?
///
/// llama.cpp's `check_tensor_dims` (llama-model-loader.cpp) accepts exactly
/// that: every listed dimension must match and every further one must be 1. A
/// file is free to store `ffn_gate_inp_shexp` as `{2048}` or `{2048, 1}`, and
/// `convert_hf_to_gguf.py` has written both.
pub(crate) fn shape_is(got: &[u64], want: &[u64]) -> bool {
    let trim = |d: &[u64]| {
        let mut n = d.len();
        while n > 0 && d[n - 1] == 1 {
            n -= 1;
        }
        n
    };
    let (g, w) = (trim(got), trim(want));
    g == w && got[..g] == want[..w]
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
    if !shape_is(&info.dims, &[n_in as u64, n_out as u64]) {
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
        pooled: false,
    })
}

/// A 3-D stack of expert matrices, shape-checked as `{n_in, n_out, n_expert}`.
///
/// Separate from [`matrix`] because a wrong guess about which of the three
/// dimensions is the contraction axis produces plausible numbers rather than an
/// error — `ffn_gate_exps` is `{2048, 512, 256}` and `ffn_down_exps` is
/// `{512, 2048, 256}`, so the two are transposes of each other and a swap would
/// be invisible until the output was wrong.
pub(crate) fn experts<'a>(
    f: &'a GgufFile,
    name: &str,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
) -> Result<Experts<'a>> {
    let info = tensor(f, name)?;
    let want = vec![n_in as u64, n_out as u64, n_expert as u64];
    if !shape_is(&info.dims, &want) {
        return Err(Error::TensorShapeMismatch {
            name: name.to_string(),
            expected: want,
            got: info.dims.clone(),
        });
    }
    // NVFP4's per-expert second scale, written beside the weight by
    // `convert_hf_to_gguf.py` (`_flush_nvfp4_experts`) and absent otherwise.
    let scale_name = name.replace(".weight", ".scale");
    let scale = match f.tensor(&scale_name) {
        None => &[][..],
        Some(s) => {
            vector(f, &scale_name, n_expert)?;
            if s.ty != crate::gguf::GgmlType::F32 {
                return Err(Error::UnsupportedQuantType { ty: s.ty.name() });
            }
            f.tensor_bytes(s)
        }
    };
    Ok(Experts {
        data: f.tensor_bytes(info),
        ty: info.ty,
        n_in,
        n_out,
        n_expert,
        scale,
    })
}

/// A weight's optional per-tensor second scale (`<weight>.scale`, one element,
/// NVFP4 only): `None` when the file carries none.
pub(crate) fn optional_scalar(f: &GgufFile, name: &str) -> Result<Option<f32>> {
    if f.tensor(name).is_none() {
        return Ok(None);
    }
    Ok(Some(vector(f, name, 1)?[0]))
}

/// A 1-D vector (norm weights), dequantized once at load.
///
/// These are tiny — a few hundred KB across the whole model — so unlike the
/// weight matrices there is no reason to keep them packed.
/// A 1-D tensor as a one-row weight matrix.
///
/// `ffn_gate_inp_shexp` is stored as `{n_embd}`, not `{n_embd, 1}`, so
/// [`matrix`] rejects it. Borrowing it as a `Weights` rather than dequantizing
/// to a `Vec<f32>` is what lets its dot product run through the seam — and on a
/// device backend that is the difference between a host sync per layer per
/// token and none.
pub(crate) fn row_matrix<'a>(f: &'a GgufFile, name: &str, n_in: usize) -> Result<Weights<'a>> {
    let info = tensor(f, name)?;
    if !shape_is(&info.dims, &[n_in as u64]) {
        return Err(Error::TensorShapeMismatch {
            name: name.to_string(),
            expected: vec![n_in as u64],
            got: info.dims.clone(),
        });
    }
    Ok(Weights {
        data: f.tensor_bytes(info),
        ty: info.ty,
        n_in,
        n_out: 1,
        pooled: false,
    })
}

/// A small 2-D tensor, `{a, b}` exactly, dequantized: conv kernels, which the
/// ops index as plain floats. `qwen35.rs` keeps its own `conv_weights`; this is
/// the same check for `qwen4exp`, added beside it rather than moved, so the
/// 35B's file is untouched (SSD-TIER.md D17).
pub(crate) fn dense_2d(f: &GgufFile, name: &str, a: usize, b: usize) -> Result<Vec<f32>> {
    let info = tensor(f, name)?;
    if info.dims != vec![a as u64, b as u64] {
        return Err(Error::TensorShapeMismatch {
            name: name.to_string(),
            expected: vec![a as u64, b as u64],
            got: info.dims.clone(),
        });
    }
    dequantize(f.tensor_bytes(info), info.ty, a * b)
}

pub(crate) fn vector(f: &GgufFile, name: &str, len: usize) -> Result<Vec<f32>> {
    let info = tensor(f, name)?;
    if !shape_is(&info.dims, &[len as u64]) {
        return Err(Error::TensorShapeMismatch {
            name: name.to_string(),
            expected: vec![len as u64],
            got: info.dims.clone(),
        });
    }
    dequantize(f.tensor_bytes(info), info.ty, len)
}
