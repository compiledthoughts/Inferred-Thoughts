//! Qwen3.8-Flash-Next, architecture `qwen4exp`: settings and weights, **load only**.
//!
//! Step 1 of `src/model/qwen4exp.md` — the architecture reference, and where the
//! forward pass below will be specified block by block. Every setting is read from the
//! file and every tensor is mapped and shape-checked, so a wrong assumption fails
//! at load with names attached. There is no forward pass yet: `Model::forward`
//! refuses this architecture with an error rather than computing anything.
//!
//! Transcribed from llama.cpp `src/models/qwen4exp.cpp` at `3057bb66c` — settings
//! from `load_arch_hparams` (lines 26–148), tensors and shapes from
//! `load_arch_tensors` (150–258). What differs from `qwen35moe`:
//!
//! - **Hyper-connections replace the norms.** The residual stream is `n_stream`
//!   copies wide (4); each layer has one module before its mixer and one before its
//!   MoE, and a final one before the head. There is **no `output_norm`**: the final
//!   mixer carries it (`qwen4exp.cpp:159`).
//! - **Every layer is MoE**, top-k of `expert_count` plus a sigmoid-gated shared
//!   expert.
//! - **QSA layers carry indexer weights**; a GDN layer does not.
//! - **One PLE layer**: a hashed n-gram table read by row, and a key/value/conv
//!   mixer on that layer, which must be a GatedDeltaNet layer
//!   (`qwen4exp.cpp:137-142`).
//!
//! Its own code path, per SSD-TIER.md D17: nothing here is shared with `qwen35`.

use std::cell::RefCell;
use std::collections::HashSet;

use crate::cache::{KvCache, RecurrentState};
use crate::error::{Error, Result};
use crate::gguf::{Array, GgufFile, Metadata};
use crate::ops::{Attn, Delta, Experts, Ops, QsaPool, QsaSelect, Weights};
use crate::profile::Ctx;
use crate::quant::dequantize_into;

/// Experts guessed per layer for lookahead prefetch.
///
/// **Ten, measured, not the recall study's 16.** The offline study ranked
/// predictors by recall, where more guesses are always better: top-16 recalled
/// 81.4% of the 125B's picks against top-10's 67.7%. On the machine the opposite
/// end binds — a wrong guess is a read the host pays for — and the 125B's decode
/// peaked at 10: medians 7.00 tok/s at k=10, 6.76 at 8, 6.73 at 12, 6.48 at 16,
/// against 6.74 with prefetch off (SSD-TIER.md D20). `INFERRED_PREFETCH_K`
/// overrides it.
const PREFETCH_K: usize = 10;

/// Whether the hyper-connection mixer averages its streams in one op.
/// `INFERRED_HC_FUSED=0` restores the eight-launch sequence.
///
/// **An A/B lever, not an exact-path switch.** `Ops::mean_streams` is
/// bit-identical to the sequence it replaces — same order, same final multiply
/// — so both arms must produce the same tokens, and a test asserts it.
fn hc_fused() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INFERRED_HC_FUSED").map(|v| v != "0").unwrap_or(true))
}

/// [`PREFETCH_K`], or `INFERRED_PREFETCH_K` when set; read once.
fn prefetch_k() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("INFERRED_PREFETCH_K").ok().and_then(|v| v.parse().ok()).unwrap_or(PREFETCH_K)
    })
}

/// llama.cpp `src/llama-hparams.h:13`, `LLAMA_MAX_PLE_NGRAM`.
pub const MAX_PLE_NGRAM: usize = 8;
/// llama.cpp `src/llama-hparams.h:14`, `LLAMA_MAX_PLE_HEADS`.
pub const MAX_PLE_HEADS: usize = 64;

/// Everything the forward pass will need, read from metadata.
#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    /// `attention.key_length`, 256; equal to `value_length`.
    pub head_dim: usize,
    pub n_vocab: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    /// `rope.dimension_count`: how many of `head_dim` rotate.
    pub n_rot: usize,
    /// `rope.dimension_sections`, the mRoPE split.
    pub rope_sections: [i32; 4],

    /// `ssm.conv_kernel`.
    pub ssm_d_conv: usize,
    /// `ssm.inner_size`; must equal `ssm_d_state * ssm_dt_rank`.
    pub ssm_d_inner: usize,
    /// `ssm.state_size`: both the key and the value head dimension
    /// (`qwen4exp.cpp:197-198`).
    pub ssm_d_state: usize,
    /// `ssm.time_step_rank`: the value head count.
    pub ssm_dt_rank: usize,
    /// `ssm.group_count`: the key head count.
    pub ssm_n_group: usize,

    /// Per layer: GatedDeltaNet (`true`) or QSA attention (`false`), from
    /// `attention.recurrent_layers`, else `full_attention_interval`
    /// (`qwen4exp.cpp:128-135`).
    pub recurrent: Vec<bool>,
    /// `attention.compress_ratios`, per layer: the QSA block size, 0 where a layer
    /// has no indexer.
    pub compress_ratios: Vec<u32>,

    pub moe: Moe,
    pub hc: HyperConnections,
    pub indexer: IndexerConfig,
    /// `None` when the file carries no `ple.layers`; every PLE field is then
    /// absent, as llama.cpp leaves them zero (`qwen4exp.cpp:64-70`).
    pub ple: Option<Ple>,
}

#[derive(Debug, Clone, Copy)]
pub struct Moe {
    /// `expert_count`.
    pub n_expert: usize,
    /// `expert_used_count`: picks per token.
    pub n_expert_used: usize,
    /// `expert_feed_forward_length`: one routed expert's width.
    pub expert_ff: usize,
    /// `expert_shared_feed_forward_length`.
    pub shared_ff: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct HyperConnections {
    /// `hyper_connection.count`: parallel residual streams. Must exceed 1
    /// (`qwen4exp.cpp:47-52`).
    pub n_stream: usize,
    /// `hyper_connection.low_rank`: the gate's bottleneck width.
    pub low_rank: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct IndexerConfig {
    /// `attention.indexer.head_count`.
    pub n_head: usize,
    /// `attention.indexer.key_length`.
    pub head_dim: usize,
    /// `attention.indexer.top_k`: positions QSA keeps; dense attention is exact
    /// below it (llama.cpp PR #27742).
    pub top_k: usize,
}

#[derive(Debug, Clone)]
pub struct Ple {
    /// `ple.layers`, of which exactly one is supported (`qwen4exp.cpp:73-77`).
    pub layer: usize,
    /// `ple.ngram_size`: the longest n-gram hashed (3 = bigrams and trigrams).
    pub ngram_size: usize,
    /// `ple.heads_per_ngram`.
    pub heads_per_ngram: usize,
    /// `ple.conv_kernel`.
    pub conv_kernel: usize,
    /// `ple.eos_token_id`: the n-gram history resets on it.
    pub eos_token_id: u32,
    /// `ple.image_token_id`, optional (`qwen4exp.cpp:89-90`).
    pub image_token_id: Option<u32>,
    /// `embedding_length_per_layer_input`: one table row's width.
    pub head_dim: usize,
    /// `ple.layer_multipliers`, the first `ngram_size` of them. u64 in the file and
    /// exact: they reach ~2.4e13, beyond any float.
    pub layer_multipliers: Vec<u64>,
    /// `ple.head_offsets` and `ple.head_vocab_sizes`, the first `n_heads()` of
    /// each, narrowed to the int32 row index the gather uses
    /// (`qwen4exp.cpp:110-124`).
    pub head_offsets: Vec<u32>,
    pub head_vocab_sizes: Vec<u32>,
}

impl Ple {
    /// Hash heads: `(ngram_size - 1) * heads_per_ngram` (`qwen4exp.cpp:95`).
    pub fn n_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// Rows the head ranges index; the table must have at least these
    /// (`qwen4exp.cpp:171-183`).
    pub fn min_rows(&self) -> u64 {
        self.head_offsets
            .iter()
            .zip(&self.head_vocab_sizes)
            .map(|(&o, &v)| u64::from(o) + u64::from(v))
            .max()
            .unwrap_or(0)
    }
}

impl Config {
    /// The widened residual stream: `n_stream * n_embd`.
    pub fn hc_dim(&self) -> usize {
        self.hc.n_stream * self.n_embd
    }

    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    /// `attn_q` emits query and gate interleaved per head (`qwen4exp.cpp:216-217`).
    pub fn q_gate_dim(&self) -> usize {
        self.head_dim * 2 * self.n_head
    }

    pub fn head_k_dim(&self) -> usize {
        self.ssm_d_state
    }

    pub fn head_v_dim(&self) -> usize {
        self.ssm_d_state
    }

    pub fn n_k_heads(&self) -> usize {
        self.ssm_n_group
    }

    pub fn n_v_heads(&self) -> usize {
        self.ssm_dt_rank
    }

    pub fn key_dim(&self) -> usize {
        self.head_k_dim() * self.n_k_heads()
    }

    pub fn value_dim(&self) -> usize {
        self.head_v_dim() * self.n_v_heads()
    }

    /// Channels the GDN conv runs over: q, k and v (`qwen4exp.cpp:203`).
    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }

    /// Floats of GDN conv state per layer.
    pub fn conv_state_len(&self) -> usize {
        (self.ssm_d_conv - 1) * self.conv_dim()
    }

    /// Floats of GDN recurrent state per layer.
    pub fn ssm_state_len(&self) -> usize {
        self.head_k_dim() * self.head_v_dim() * self.n_v_heads()
    }

    pub fn is_recurrent(&self, il: usize) -> bool {
        self.recurrent.get(il).copied().unwrap_or(false)
    }

    /// Layers with a KV cache: the QSA layers.
    pub fn n_kv_layer(&self) -> usize {
        self.recurrent.iter().filter(|&&r| !r).count()
    }

    pub fn from_gguf(f: &GgufFile) -> Result<Self> {
        let md = &f.metadata;
        let arch = md.architecture()?;
        if arch != "qwen4exp" {
            return Err(Error::UnsupportedArchitecture {
                arch: arch.to_string(),
                supported: "qwen4exp",
            });
        }
        let u = |suffix: &str| -> Result<usize> { Ok(md.get_arch_u32(suffix)? as usize) };
        let nonzero = |what: &'static str, v: usize| -> Result<usize> {
            if v == 0 {
                return Err(Error::InconsistentArchitecture {
                    what,
                    detail: "must be greater than zero".to_string(),
                });
            }
            Ok(v)
        };

        // MTP blocks sit past `n_layer()` in llama.cpp; our conversion drops them
        // (the converter's qwen4exp class sets `no_mtp`). Refuse rather than load
        // blocks this file would describe but this loader does not map.
        let nextn = match md.get(&md.arch_key("nextn_predict_layers")?) {
            Some(_) => u("nextn_predict_layers")?,
            None => 0,
        };
        if nextn != 0 {
            return Err(Error::InconsistentArchitecture {
                what: "nextn_predict_layers",
                detail: format!("{nextn} MTP blocks; qwen4exp MTP is not supported"),
            });
        }
        let n_layer = nonzero("block_count", u("block_count")?)?;

        let n_head = u("attention.head_count")?;
        let n_head_kv = u("attention.head_count_kv")?;
        let head_dim = u("attention.key_length")?;
        let v_len = u("attention.value_length")?;
        if n_head_kv == 0 || n_head % n_head_kv != 0 {
            return Err(Error::InconsistentArchitecture {
                what: "GQA grouping",
                detail: format!("{n_head} query heads do not divide into {n_head_kv} kv heads"),
            });
        }
        if v_len != head_dim {
            return Err(Error::InconsistentArchitecture {
                what: "head dimension",
                detail: format!("key_length {head_dim} != value_length {v_len}"),
            });
        }

        let sections = match arch_array(md, "rope.dimension_sections")? {
            Some(Array::I32(v)) if v.len() == 4 => [v[0], v[1], v[2], v[3]],
            Some(other) => {
                return Err(Error::InconsistentArchitecture {
                    what: "rope.dimension_sections",
                    detail: format!("expected 4 i32, found {} {}", other.len(), other.elem_type_name()),
                });
            }
            None => return Err(Error::MissingKey(md.arch_key("rope.dimension_sections")?)),
        };

        let n_vocab = super::tensor(f, "token_embd.weight")?
            .dims
            .get(1)
            .copied()
            .ok_or_else(|| Error::InconsistentArchitecture {
                what: "token_embd.weight",
                detail: "expected 2 dimensions".to_string(),
            })? as usize;

        // `qwen4exp.cpp:33-42`: every ssm key required and nonzero.
        let ssm_d_conv = nonzero("ssm.conv_kernel", u("ssm.conv_kernel")?)?;
        let ssm_d_inner = nonzero("ssm.inner_size", u("ssm.inner_size")?)?;
        let ssm_d_state = nonzero("ssm.state_size", u("ssm.state_size")?)?;
        let ssm_dt_rank = nonzero("ssm.time_step_rank", u("ssm.time_step_rank")?)?;
        let ssm_n_group = nonzero("ssm.group_count", u("ssm.group_count")?)?;

        // `qwen4exp.cpp:128-135`.
        let recurrent = match arch_array(md, "attention.recurrent_layers")? {
            Some(Array::Bool(v)) => v.clone(),
            Some(other) => {
                return Err(Error::InconsistentArchitecture {
                    what: "attention.recurrent_layers",
                    detail: format!("expected bool, found {}", other.elem_type_name()),
                });
            }
            None => {
                let interval = match md.get(&md.arch_key("full_attention_interval")?) {
                    Some(_) => u("full_attention_interval")?,
                    None => 4,
                };
                let interval = nonzero("full_attention_interval", interval)?;
                (0..n_layer).map(|i| (i + 1) % interval != 0).collect()
            }
        };
        if recurrent.len() != n_layer {
            return Err(Error::InconsistentArchitecture {
                what: "attention.recurrent_layers",
                detail: format!("{} entries for {n_layer} layers", recurrent.len()),
            });
        }

        let compress_ratios = match arch_array(md, "attention.compress_ratios")? {
            Some(a) => indices(a, "attention.compress_ratios")?,
            None => vec![0; n_layer],
        };
        if compress_ratios.len() != n_layer {
            return Err(Error::InconsistentArchitecture {
                what: "attention.compress_ratios",
                detail: format!("{} entries for {n_layer} layers", compress_ratios.len()),
            });
        }

        let n_expert = nonzero("expert_count", u("expert_count")?)?;
        let moe = Moe {
            n_expert,
            n_expert_used: u("expert_used_count")?,
            expert_ff: nonzero("expert_feed_forward_length", u("expert_feed_forward_length")?)?,
            shared_ff: nonzero(
                "expert_shared_feed_forward_length",
                u("expert_shared_feed_forward_length")?,
            )?,
        };
        if moe.n_expert_used == 0 || moe.n_expert_used > n_expert {
            return Err(Error::InconsistentArchitecture {
                what: "expert_used_count",
                detail: format!("{} of {n_expert} experts", moe.n_expert_used),
            });
        }

        // `qwen4exp.cpp:45-53`.
        let hc = HyperConnections {
            n_stream: u("hyper_connection.count")?,
            low_rank: nonzero("hyper_connection.low_rank", u("hyper_connection.low_rank")?)?,
        };
        if hc.n_stream <= 1 {
            return Err(Error::InconsistentArchitecture {
                what: "hyper_connection.count",
                detail: format!("must be greater than one, got {}", hc.n_stream),
            });
        }

        // `qwen4exp.cpp:56-61`.
        let indexer = IndexerConfig {
            n_head: nonzero("attention.indexer.head_count", u("attention.indexer.head_count")?)?,
            head_dim: nonzero("attention.indexer.key_length", u("attention.indexer.key_length")?)?,
            top_k: nonzero("attention.indexer.top_k", u("attention.indexer.top_k")?)?,
        };

        let ple = read_ple(md, n_layer)?;
        let n_embd = nonzero("embedding_length", u("embedding_length")?)?;
        if let Some(p) = &ple {
            if !recurrent[p.layer] {
                return Err(Error::InconsistentArchitecture {
                    what: "ple.layers",
                    detail: format!("PLE layer {} is not a linear attention layer", p.layer),
                });
            }
            // `build_inp_ple` reshapes the gathered rows to `[head_dim * n_heads, T]`
            // and `ple_key` reads that as `n_embd` (`qwen4exp.cpp:1186, 241`).
            if p.head_dim * p.n_heads() != n_embd {
                return Err(Error::InconsistentArchitecture {
                    what: "embedding_length_per_layer_input",
                    detail: format!(
                        "{} x {} hash heads != embedding_length {n_embd}",
                        p.head_dim,
                        p.n_heads()
                    ),
                });
            }
        }

        let cfg = Self {
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            n_vocab,
            rope_theta: md.get_arch_f32("rope.freq_base")?,
            rms_eps: md.get_arch_f32("attention.layer_norm_rms_epsilon")?,
            n_rot: u("rope.dimension_count")?,
            rope_sections: sections,
            ssm_d_conv,
            ssm_d_inner,
            ssm_d_state,
            ssm_dt_rank,
            ssm_n_group,
            recurrent,
            compress_ratios,
            moe,
            hc,
            indexer,
            ple,
        };

        // Relationships the llama.cpp graph assumes without checking; a file that
        // broke one would read as a numerics bug later, so name it here.
        if cfg.ssm_d_inner != cfg.value_dim() {
            return Err(Error::InconsistentArchitecture {
                what: "ssm.inner_size",
                detail: format!(
                    "{} != state_size {} x time_step_rank {}",
                    cfg.ssm_d_inner, cfg.ssm_d_state, cfg.ssm_dt_rank
                ),
            });
        }
        if cfg.n_v_heads() % cfg.n_k_heads() != 0 {
            return Err(Error::InconsistentArchitecture {
                what: "delta rule grouping",
                detail: format!(
                    "{} value heads do not divide into {} key heads",
                    cfg.n_v_heads(),
                    cfg.n_k_heads()
                ),
            });
        }
        if cfg.n_rot > cfg.head_dim {
            return Err(Error::InconsistentArchitecture {
                what: "rope.dimension_count",
                detail: format!("{} exceeds head_dim {}", cfg.n_rot, cfg.head_dim),
            });
        }
        Ok(cfg)
    }
}

/// The PLE settings, `qwen4exp.cpp:64-125`.
fn read_ple(md: &Metadata, n_layer: usize) -> Result<Option<Ple>> {
    let Some(layers) = arch_array(md, "ple.layers")? else { return Ok(None) };
    let layers = indices(layers, "ple.layers")?;
    if layers.is_empty() {
        return Ok(None);
    }
    if layers.len() != 1 {
        return Err(Error::InconsistentArchitecture {
            what: "ple.layers",
            detail: format!("{} layers listed; only one PLE layer is supported", layers.len()),
        });
    }
    let layer = layers[0] as usize;
    if layer >= n_layer {
        return Err(Error::InconsistentArchitecture {
            what: "ple.layers",
            detail: format!("PLE layer {layer} is out of range for {n_layer} layers"),
        });
    }
    let u = |suffix: &str| -> Result<usize> { Ok(md.get_arch_u32(suffix)? as usize) };
    let image_token_id = match md.get(&md.arch_key("ple.image_token_id")?) {
        Some(_) => Some(md.get_arch_u32("ple.image_token_id")?),
        None => None,
    };
    let ngram_size = u("ple.ngram_size")?;
    let heads_per_ngram = u("ple.heads_per_ngram")?;
    let conv_kernel = u("ple.conv_kernel")?;
    let head_dim = u("embedding_length_per_layer_input")?;
    if conv_kernel == 0 || head_dim == 0 {
        return Err(Error::InconsistentArchitecture {
            what: "ple.conv_kernel / embedding_length_per_layer_input",
            detail: format!("{conv_kernel} / {head_dim}; both must be greater than zero"),
        });
    }
    if !(2..=MAX_PLE_NGRAM).contains(&ngram_size) {
        return Err(Error::InconsistentArchitecture {
            what: "ple.ngram_size",
            detail: format!("{ngram_size} is outside 2..={MAX_PLE_NGRAM}"),
        });
    }
    let n_heads = (ngram_size - 1) * heads_per_ngram;
    if n_heads == 0 || n_heads > MAX_PLE_HEADS {
        return Err(Error::InconsistentArchitecture {
            what: "ple.heads_per_ngram",
            detail: format!("{n_heads} hash heads is outside 1..={MAX_PLE_HEADS}"),
        });
    }

    // `qwen4exp.cpp:104-106`: at least this many entries, the rest ignored.
    let at_least = |suffix: &'static str, n: usize| -> Result<Vec<u64>> {
        let a = arch_array(md, suffix)?.ok_or_else(|| Error::MissingKey(suffix.to_string()))?;
        let v = unsigned(a, suffix)?;
        if v.len() < n {
            return Err(Error::InconsistentArchitecture {
                what: suffix,
                detail: format!("{} entries, at least {n} required", v.len()),
            });
        }
        Ok(v[..n].to_vec())
    };
    let layer_multipliers = at_least("ple.layer_multipliers", ngram_size)?;
    let offsets = at_least("ple.head_offsets", n_heads)?;
    let vocab = at_least("ple.head_vocab_sizes", n_heads)?;

    // `qwen4exp.cpp:115-124`: every range nonempty and inside the int32 row index.
    let limit = i32::MAX as u64;
    let mut head_offsets = Vec::with_capacity(n_heads);
    let mut head_vocab_sizes = Vec::with_capacity(n_heads);
    for (h, (&o, &v)) in offsets.iter().zip(&vocab).enumerate() {
        if v == 0 || o > limit || v > limit || o + v > limit {
            return Err(Error::InconsistentArchitecture {
                what: "ple.head_offsets / ple.head_vocab_sizes",
                detail: format!("head {h} range {o}+{v} does not fit the int32 row index"),
            });
        }
        head_offsets.push(o as u32);
        head_vocab_sizes.push(v as u32);
    }

    Ok(Some(Ple {
        layer,
        ngram_size,
        heads_per_ngram,
        conv_kernel,
        eos_token_id: md.get_arch_u32("ple.eos_token_id")?,
        image_token_id,
        head_dim,
        layer_multipliers,
        head_offsets,
        head_vocab_sizes,
    }))
}

/// An array under the file's architecture prefix, or `None` when the key is absent.
fn arch_array<'m>(md: &'m Metadata, suffix: &str) -> Result<Option<&'m Array>> {
    let key = md.arch_key(suffix)?;
    match md.get(&key) {
        None => Ok(None),
        Some(_) => md.get_array(&key).map(Some),
    }
}

/// An unsigned integer array, widened to u64 as `Metadata::get_u64` widens scalars.
fn unsigned(a: &Array, what: &'static str) -> Result<Vec<u64>> {
    match a {
        Array::U64(v) => Ok(v.clone()),
        Array::U32(v) => Ok(v.iter().map(|&x| u64::from(x)).collect()),
        other => Err(Error::InconsistentArchitecture {
            what,
            detail: format!("expected an unsigned integer array, found {}", other.elem_type_name()),
        }),
    }
}

/// A small non-negative integer array. The converter writes `ple.layers` and
/// `attention.compress_ratios` as i32 where llama.cpp reads them as uint32.
fn indices(a: &Array, what: &'static str) -> Result<Vec<u32>> {
    match a {
        Array::U32(v) => Ok(v.clone()),
        Array::I32(v) => v
            .iter()
            .map(|&x| {
                u32::try_from(x).map_err(|_| Error::InconsistentArchitecture {
                    what,
                    detail: format!("negative entry {x}"),
                })
            })
            .collect(),
        other => Err(Error::InconsistentArchitecture {
            what,
            detail: format!("expected an integer array, found {}", other.elem_type_name()),
        }),
    }
}

// ------------------------------------------------------------------- weights

/// One hyper-connection module: the low-rank gate that reads the `n_stream`
/// residual copies and, except on the head, the inject weights that write back.
///
/// Fields are read by the forward pass, stage 2 step 3.
#[allow(dead_code)]
pub(crate) struct Hc<'a> {
    /// `{hc_dim}`.
    norm: Vec<f32>,
    /// `{hc_dim, low_rank}`.
    down: Weights<'a>,
    /// `{low_rank, hc_dim}`.
    up: Weights<'a>,
    /// `{hc_dim, n_stream}`; `None` on the head's final mixer.
    inject: Option<Weights<'a>>,
}

#[allow(dead_code)]
pub(crate) struct Indexer<'a> {
    /// `{n_embd, indexer.n_head * indexer.head_dim}`.
    q_proj: Weights<'a>,
    /// `{n_embd, indexer.head_dim}`.
    k_proj: Weights<'a>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

#[allow(dead_code)]
pub(crate) enum Mixer<'a> {
    /// QSA attention: `wq` holds query and gate interleaved per head.
    Attn {
        wq: Weights<'a>,
        wk: Weights<'a>,
        wv: Weights<'a>,
        wo: Weights<'a>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        indexer: Indexer<'a>,
    },
    /// GatedDeltaNet, `qwen4exp.cpp:229-237`.
    Delta {
        wqkv: Weights<'a>,
        wgate: Weights<'a>,
        conv1d: Vec<f32>,
        dt_bias: Vec<f32>,
        ssm_a: Vec<f32>,
        ssm_beta: Weights<'a>,
        ssm_alpha: Weights<'a>,
        ssm_norm: Vec<f32>,
        ssm_out: Weights<'a>,
    },
}

/// The PLE layer's mixer, `qwen4exp.cpp:240-247`.
#[allow(dead_code)]
pub(crate) struct PleMixer<'a> {
    /// `{n_embd, hc_dim}`.
    key: Weights<'a>,
    /// `{n_embd, n_embd}`.
    value: Weights<'a>,
    norm_key: Vec<f32>,
    norm_query: Vec<f32>,
    norm_conv: Vec<f32>,
    /// `{conv_kernel, hc_dim}`, dequantized.
    conv1d: Vec<f32>,
}

#[allow(dead_code)]
pub(crate) struct Ffn<'a> {
    /// The router, `{n_embd, n_expert}`.
    gate_inp: Weights<'a>,
    gate: Experts<'a>,
    up: Experts<'a>,
    down: Experts<'a>,
    shared_gate: Weights<'a>,
    shared_up: Weights<'a>,
    shared_down: Weights<'a>,
    /// `ffn_gate_inp_shexp`, `{n_embd}`, the shared expert's sigmoid gate.
    shared_gate_inp: Weights<'a>,
    /// NVFP4 second scales, when the file carries them.
    shared_gate_s: Option<f32>,
    shared_up_s: Option<f32>,
    shared_down_s: Option<f32>,
}

#[allow(dead_code)]
pub(crate) struct Layer<'a> {
    hc_attn: Hc<'a>,
    hc_ffn: Hc<'a>,
    mixer: Mixer<'a>,
    ple: Option<PleMixer<'a>>,
    ffn: Ffn<'a>,
}

/// Tensors a checkpoint carries that this engine reads on purpose never:
/// NVFP4 activation scales. The FP4 x FP4 path quantizes activations as
/// llama.cpp's CUDA backend does, and the exact path uses NVFP4 x Q8_0; neither
/// reads them (`ops::naive`, the `Fp4Row` notes).
pub const UNUSED_SUFFIXES: &[&str] = &[".input_scale"];

#[allow(dead_code)]
pub struct Qwen4Exp<'a> {
    pub cfg: Config,
    tok_embd: Weights<'a>,
    output: Weights<'a>,
    output_s: Option<f32>,
    /// The final mixer before the head; it carries the output norm.
    head_hc: Hc<'a>,
    /// `per_layer_token_embd`, `{ple.head_dim, rows}`, read by row.
    ple_table: Option<Weights<'a>>,
    layers: Vec<Layer<'a>>,
    /// Absolute layer -> KV slab; only the QSA layers have one.
    kv_slot: Vec<usize>,
    /// Every tensor name this loader mapped, for [`Qwen4Exp::unmapped`].
    mapped: HashSet<String>,
    /// Activation buffers, owned so their addresses never change (CLAUDE.md).
    scratch: RefCell<Scratch>,
    /// The PLE layer's per-sequence state: its n-gram window and conv history.
    ple_state: RefCell<PleState>,
}

/// What PLE carries between passes of one sequence (`qwen4exp.md`, "PLE").
///
/// **Held by the model, not by `RecurrentState`**, and tied to the position it
/// expects next: a pass at position 0 starts a new sequence, and a pass anywhere
/// else must continue exactly where the last one ended, or it is refused. A rewind
/// — `serve` returning to a checkpoint — goes through [`PleSnapshot`], which the
/// engine's checkpoints carry beside the GatedDeltaNet state.
#[derive(Default)]
struct PleState {
    next_pos: usize,
    /// The sequence's last `ngram_size - 1` tokens, oldest first; shorter near its start.
    prev: Vec<u32>,
    /// `[hc_dim][(conv_kernel - 1) * ngram_size]`, oldest first; zero at a sequence start.
    conv: Vec<f32>,
}

/// A copy of PLE's per-sequence state at one position, for a checkpoint.
///
/// **Without it `serve` could answer one turn and no more**: a new request whose
/// history differs from what was cached returns to a checkpoint, and PLE refused
/// every pass that did not continue exactly where the last ended. Other
/// implementations report the same gap (`qwen4exp.md`, "Other implementations").
#[derive(Clone, Debug)]
pub struct PleSnapshot {
    next_pos: usize,
    prev: Vec<u32>,
    conv: Vec<f32>,
}

impl PleSnapshot {
    /// Bytes held, for a caller bounding how many checkpoints it keeps.
    pub fn bytes(&self) -> usize {
        self.prev.len() * std::mem::size_of::<u32>() + self.conv.len() * std::mem::size_of::<f32>()
    }
}

/// Records every tensor it maps, so load can prove nothing in the file went
/// unaccounted for.
struct Loader<'a> {
    f: &'a GgufFile,
    mapped: HashSet<String>,
}

impl<'a> Loader<'a> {
    fn matrix(&mut self, name: &str, n_in: usize, n_out: usize) -> Result<Weights<'a>> {
        self.mapped.insert(name.to_string());
        super::matrix(self.f, name, n_in, n_out)
    }

    fn row_matrix(&mut self, name: &str, n_in: usize) -> Result<Weights<'a>> {
        self.mapped.insert(name.to_string());
        super::row_matrix(self.f, name, n_in)
    }

    fn vector(&mut self, name: &str, len: usize) -> Result<Vec<f32>> {
        self.mapped.insert(name.to_string());
        super::vector(self.f, name, len)
    }

    fn dense_2d(&mut self, name: &str, a: usize, b: usize) -> Result<Vec<f32>> {
        self.mapped.insert(name.to_string());
        super::dense_2d(self.f, name, a, b)
    }

    fn experts(&mut self, name: &str, n_in: usize, n_out: usize, n_expert: usize) -> Result<Experts<'a>> {
        self.mapped.insert(name.to_string());
        let scale = name.replace(".weight", ".scale");
        if self.f.tensor(&scale).is_some() {
            self.mapped.insert(scale);
        }
        super::experts(self.f, name, n_in, n_out, n_expert)
    }

    fn optional_scalar(&mut self, name: &str) -> Result<Option<f32>> {
        if self.f.tensor(name).is_some() {
            self.mapped.insert(name.to_string());
        }
        super::optional_scalar(self.f, name)
    }

    fn hc(&mut self, prefix: &str, cfg: &Config, with_inject: bool) -> Result<Hc<'a>> {
        let (hc_dim, lr) = (cfg.hc_dim(), cfg.hc.low_rank);
        Ok(Hc {
            norm: self.vector(&format!("{prefix}_norm.weight"), hc_dim)?,
            down: self.matrix(&format!("{prefix}_down.weight"), hc_dim, lr)?,
            up: self.matrix(&format!("{prefix}_up.weight"), lr, hc_dim)?,
            inject: if with_inject {
                Some(self.matrix(&format!("{prefix}_inject.weight"), hc_dim, cfg.hc.n_stream)?)
            } else {
                None
            },
        })
    }
}

impl<'a> Qwen4Exp<'a> {
    pub fn load(f: &'a GgufFile) -> Result<Self> {
        let cfg = Config::from_gguf(f)?;
        let mut ld = Loader { f, mapped: HashSet::new() };
        let n_embd = cfg.n_embd;
        let m = cfg.moe;

        let tok_embd = ld.matrix("token_embd.weight", n_embd, cfg.n_vocab)?;
        // Tied to the embedding when the file has no head (`qwen4exp.cpp:164-167`).
        let (output, output_s) = match f.tensor("output.weight") {
            Some(_) => (
                ld.matrix("output.weight", n_embd, cfg.n_vocab)?,
                ld.optional_scalar("output.scale")?,
            ),
            None => (tok_embd, None),
        };
        // `LLM_TENSOR_HC_HEAD_*` is written as `output_hc_*` (`qwen4exp.cpp:160-162`).
        let head_hc = ld.hc("output_hc", &cfg, false)?;

        let ple_table = match &cfg.ple {
            None => None,
            Some(p) => {
                let name = "per_layer_token_embd.weight";
                let rows = super::tensor(f, name)?.dims.get(1).copied().unwrap_or(0);
                if rows < p.min_rows() {
                    return Err(Error::InconsistentArchitecture {
                        what: "per_layer_token_embd.weight",
                        detail: format!("{rows} rows, too few for the PLE head ranges ({})", p.min_rows()),
                    });
                }
                Some(ld.matrix(name, p.head_dim, rows as usize)?)
            }
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let p = |name: &str| format!("blk.{il}.{name}");
            let hc_attn = ld.hc(&p("hc_attn"), &cfg, true)?;
            let hc_ffn = ld.hc(&p("hc_ffn"), &cfg, true)?;
            let mixer = if cfg.is_recurrent(il) {
                Mixer::Delta {
                    wqkv: ld.matrix(&p("attn_qkv.weight"), n_embd, cfg.conv_dim())?,
                    wgate: ld.matrix(&p("attn_gate.weight"), n_embd, cfg.value_dim())?,
                    conv1d: ld.dense_2d(&p("ssm_conv1d.weight"), cfg.ssm_d_conv, cfg.conv_dim())?,
                    dt_bias: ld.vector(&p("ssm_dt.bias"), cfg.n_v_heads())?,
                    ssm_a: ld.vector(&p("ssm_a"), cfg.n_v_heads())?,
                    ssm_beta: ld.matrix(&p("ssm_beta.weight"), n_embd, cfg.n_v_heads())?,
                    ssm_alpha: ld.matrix(&p("ssm_alpha.weight"), n_embd, cfg.n_v_heads())?,
                    ssm_norm: ld.vector(&p("ssm_norm.weight"), cfg.head_v_dim())?,
                    ssm_out: ld.matrix(&p("ssm_out.weight"), cfg.value_dim(), n_embd)?,
                }
            } else {
                let ix = cfg.indexer;
                Mixer::Attn {
                    wq: ld.matrix(&p("attn_q.weight"), n_embd, cfg.q_gate_dim())?,
                    wk: ld.matrix(&p("attn_k.weight"), n_embd, cfg.kv_dim())?,
                    wv: ld.matrix(&p("attn_v.weight"), n_embd, cfg.kv_dim())?,
                    wo: ld.matrix(&p("attn_output.weight"), cfg.head_dim * cfg.n_head, n_embd)?,
                    q_norm: ld.vector(&p("attn_q_norm.weight"), cfg.head_dim)?,
                    k_norm: ld.vector(&p("attn_k_norm.weight"), cfg.head_dim)?,
                    indexer: Indexer {
                        q_proj: ld.matrix(&p("indexer.q_proj.weight"), n_embd, ix.n_head * ix.head_dim)?,
                        k_proj: ld.matrix(&p("indexer.k_proj.weight"), n_embd, ix.head_dim)?,
                        q_norm: ld.vector(&p("indexer.q_norm.weight"), ix.head_dim)?,
                        k_norm: ld.vector(&p("indexer.k_norm.weight"), ix.head_dim)?,
                    },
                }
            };
            let ple = match &cfg.ple {
                Some(pc) if pc.layer == il => Some(PleMixer {
                    key: ld.matrix(&p("ple_key.weight"), n_embd, cfg.hc_dim())?,
                    value: ld.matrix(&p("ple_value.weight"), n_embd, n_embd)?,
                    norm_key: ld.vector(&p("ple_norm_key.weight"), cfg.hc_dim())?,
                    norm_query: ld.vector(&p("ple_norm_query.weight"), cfg.hc_dim())?,
                    norm_conv: ld.vector(&p("ple_norm_conv.weight"), cfg.hc_dim())?,
                    conv1d: ld.dense_2d(&p("ple_conv1d.weight"), pc.conv_kernel, cfg.hc_dim())?,
                }),
                _ => None,
            };
            let ffn = Ffn {
                gate_inp: ld.matrix(&p("ffn_gate_inp.weight"), n_embd, m.n_expert)?,
                gate: ld.experts(&p("ffn_gate_exps.weight"), n_embd, m.expert_ff, m.n_expert)?,
                up: ld.experts(&p("ffn_up_exps.weight"), n_embd, m.expert_ff, m.n_expert)?,
                down: ld.experts(&p("ffn_down_exps.weight"), m.expert_ff, n_embd, m.n_expert)?,
                shared_gate: ld.matrix(&p("ffn_gate_shexp.weight"), n_embd, m.shared_ff)?,
                shared_up: ld.matrix(&p("ffn_up_shexp.weight"), n_embd, m.shared_ff)?,
                shared_down: ld.matrix(&p("ffn_down_shexp.weight"), m.shared_ff, n_embd)?,
                shared_gate_inp: ld.row_matrix(&p("ffn_gate_inp_shexp.weight"), n_embd)?,
                shared_gate_s: ld.optional_scalar(&p("ffn_gate_shexp.scale"))?,
                shared_up_s: ld.optional_scalar(&p("ffn_up_shexp.scale"))?,
                shared_down_s: ld.optional_scalar(&p("ffn_down_shexp.scale"))?,
            };
            layers.push(Layer { hc_attn, hc_ffn, mixer, ple, ffn });
        }

        let mut kv_slot = Vec::with_capacity(cfg.n_layer);
        let mut next = 0;
        for il in 0..cfg.n_layer {
            kv_slot.push(next);
            if !cfg.is_recurrent(il) {
                next += 1;
            }
        }

        Ok(Self {
            cfg,
            tok_embd,
            output,
            output_s,
            head_hc,
            ple_table,
            layers,
            kv_slot,
            mapped: ld.mapped,
            scratch: RefCell::new(Scratch::default()),
            ple_state: RefCell::new(PleState::default()),
        })
    }

    pub fn n_kv_layer(&self) -> usize {
        self.cfg.n_kv_layer()
    }

    /// Size the per-token scratch for passes of up to `n` tokens, the engine's
    /// batch, so no later pass moves a buffer (`Scratch::fit`). Only ever grows.
    pub fn reserve_batch(&self, n: usize) {
        let mut s = self.scratch.borrow_mut();
        s.reserve = s.reserve.max(n);
    }

    /// The most bytes of activations a pass can hold with the scratch reserved
    /// for `reserve_batch` tokens and a context of `n_ctx`, for a device backend
    /// that keeps a copy of each: the per-token scratch at a full batch, the
    /// layer-major residuals for a whole group (`engine::LAYER_MAJOR_MAX_TOKENS`)
    /// and QSA's per-query buffers at full depth. An upper bound: a buffer the
    /// backend never mirrors is counted anyway.
    pub fn activation_peak_bytes(&self, n_ctx: usize) -> usize {
        let c = &self.cfg;
        let mut s = self.scratch.borrow_mut();
        let cap = s.reserve.max(1);
        s.fit(c, cap);
        let fitted = s.fitted_bytes();
        let f = std::mem::size_of::<f32>();
        let group = crate::engine::LAYER_MAJOR_MAX_TOKENS.min(n_ctx);
        let residuals = group * c.hc.n_stream * c.n_embd * f;
        let ratio = c.compress_ratios.iter().copied().max().unwrap_or(0) as usize;
        let qsa = if ratio > 0 {
            cap * (n_ctx / ratio + 1) * f + cap * (c.indexer.top_k as usize + ratio) * std::mem::size_of::<u32>()
        } else {
            0
        };
        fitted + residuals + qsa
    }

    /// PLE's state as it stands, for a checkpoint; `None` for a model without PLE.
    ///
    /// **The conv history is read back from the device first.** A device backend
    /// owns it once `dilated_conv` has written it and never sends it home on the
    /// forward path, so without the read a checkpoint would save a stale host copy
    /// and restore "successfully" into a history the sequence never had.
    pub fn ple_snapshot<O: Ops>(&self, ops: &O) -> Option<PleSnapshot> {
        self.cfg.ple.as_ref()?;
        let mut st = self.ple_state.borrow_mut();
        if !st.conv.is_empty() {
            ops.read_state(&mut st.conv);
        }
        Some(PleSnapshot { next_pos: st.next_pos, prev: st.prev.clone(), conv: st.conv.clone() })
    }

    /// Put PLE's state back as a checkpoint held it. The caller must tell the
    /// backend the host copy is authoritative again (`Ops::forget_state`).
    pub fn ple_restore(&self, s: &PleSnapshot) -> Result<()> {
        let mut st = self.ple_state.borrow_mut();
        if st.conv.len() == s.conv.len() {
            // In place, so the conv history keeps the address its device copy is keyed on.
            st.conv.copy_from_slice(&s.conv);
        } else if st.conv.is_empty() {
            st.conv = s.conv.clone();
        } else {
            return Err(Error::InconsistentArchitecture {
                what: "PLE checkpoint",
                detail: format!("holds {} conv floats, this model's history is {}", s.conv.len(), st.conv.len()),
            });
        }
        st.prev = s.prev.clone();
        st.next_pos = s.next_pos;
        Ok(())
    }

    /// Tensors this loader mapped.
    pub fn n_mapped(&self) -> usize {
        self.mapped.len()
    }

    /// Tensors in the file this loader did not map, split into those unread on
    /// purpose ([`UNUSED_SUFFIXES`]) and the rest. A nonempty second list means the
    /// file has something this transcription does not know about.
    pub fn unmapped(&self, f: &GgufFile) -> (Vec<String>, Vec<String>) {
        let mut on_purpose = Vec::new();
        let mut unknown = Vec::new();
        for t in &f.tensors {
            if self.mapped.contains(&t.name) {
                continue;
            }
            if UNUSED_SUFFIXES.iter().any(|s| t.name.ends_with(s)) {
                on_purpose.push(t.name.clone());
            } else {
                unknown.push(t.name.clone());
            }
        }
        (on_purpose, unknown)
    }

    /// Bytes of quantized weight one decode token reads, derived from shapes: the
    /// dense weights, `n_expert_used / n_expert` of each expert tensor, and one
    /// PLE row per hash head.
    pub fn weight_bytes_per_pass(&self) -> u64 {
        self.weight_bytes(true)
    }

    /// Bytes of the weights a device holds whole: everything a pass reads except
    /// the routed experts (the expert cache's tiers) and the PLE rows (read from
    /// the file on the host). What the expert slab has to leave room for.
    pub fn dense_weight_bytes(&self) -> u64 {
        self.weight_bytes(false)
    }

    /// Experts across every routed tensor: `n_expert` for each layer's gate, up
    /// and down.
    pub fn expert_pool(&self) -> usize {
        self.layers.iter().map(|l| l.ffn.gate.n_expert + l.ffn.up.n_expert + l.ffn.down.n_expert).sum()
    }

    /// The matmul weights; with `routed`, also the used share of each expert
    /// tensor and one PLE row per hash head.
    fn weight_bytes(&self, routed: bool) -> u64 {
        let w = |m: &Weights<'_>| m.ty.n_bytes(m.n_in as u64) * m.n_out as u64;
        let n_used = if routed { self.cfg.moe.n_expert_used as u64 } else { 0 };
        let e = |x: &Experts<'_>| x.data.len() as u64 * n_used / self.cfg.moe.n_expert as u64;
        let hc = |h: &Hc<'_>| w(&h.down) + w(&h.up) + h.inject.as_ref().map_or(0, w);
        let mut total = w(&self.output) + hc(&self.head_hc);
        if let (true, Some(t), Some(p)) = (routed, &self.ple_table, &self.cfg.ple) {
            total += t.ty.n_bytes(t.n_in as u64) * p.n_heads() as u64;
        }
        for l in &self.layers {
            total += hc(&l.hc_attn) + hc(&l.hc_ffn);
            total += match &l.mixer {
                Mixer::Attn { wq, wk, wv, wo, indexer, .. } => {
                    w(wq) + w(wk) + w(wv) + w(wo) + w(&indexer.q_proj) + w(&indexer.k_proj)
                }
                Mixer::Delta { wqkv, wgate, ssm_beta, ssm_alpha, ssm_out, .. } => {
                    w(wqkv) + w(wgate) + w(ssm_beta) + w(ssm_alpha) + w(ssm_out)
                }
            };
            if let Some(pm) = &l.ple {
                total += w(&pm.key) + w(&pm.value);
            }
            let f = &l.ffn;
            total += w(&f.gate_inp) + e(&f.gate) + e(&f.up) + e(&f.down);
            total += w(&f.shared_gate) + w(&f.shared_up) + w(&f.shared_down) + w(&f.shared_gate_inp);
        }
        total
    }
}

// --------------------------------------------------------------- forward pass

/// Every intermediate a pass of `n` tokens needs, token-major, resized in place so
/// addresses never change (CLAUDE.md; `qwen35::Scratch` records why).
#[derive(Default)]
struct Scratch {
    /// Embeddings, `[n][n_embd]`.
    x: Vec<f32>,
    /// The wide residual, `[n][n_stream][n_embd]` — ggml's `[n_embd, hc, T]`.
    res: Vec<f32>,
    /// `[n][n_stream]` of 1.0, for `hc_init`'s broadcast.
    ones_streams: Vec<f32>,
    /// `[n_embd]` of 1.0: the unit weight a per-stream RMSNorm runs with.
    ones: Vec<f32>,
    // hyper-connection read and write
    xn: Vec<f32>,
    lo: Vec<f32>,
    hgate: Vec<f32>,
    tmp: Vec<f32>,
    mixed: Vec<f32>,
    inject: Vec<f32>,
    wide: Vec<f32>,
    /// A block's output, `[n][n_embd]`.
    block: Vec<f32>,
    // attention
    qg: Vec<f32>,
    q: Vec<f32>,
    g: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    // gated delta
    qkv: Vec<f32>,
    z: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    conv: Vec<f32>,
    q_part: Vec<f32>,
    k_part: Vec<f32>,
    v_part: Vec<f32>,
    core: Vec<f32>,
    // MoE
    router: Vec<f32>,
    logit: Vec<f32>,
    g_all: Vec<f32>,
    u_all: Vec<f32>,
    o_all: Vec<f32>,
    e_gate: Vec<f32>,
    e_up: Vec<f32>,
    e_out: Vec<f32>,
    // PLE
    emb: Vec<f32>,
    pkey: Vec<f32>,
    pvalue: Vec<f32>,
    pquery: Vec<f32>,
    psdot: Vec<f32>,
    pgated: Vec<f32>,
    pconv_in: Vec<f32>,
    pconv_out: Vec<f32>,
    // the head, one token
    hres: Vec<f32>,
    hxn: Vec<f32>,
    hlo: Vec<f32>,
    hgate1: Vec<f32>,
    htmp: Vec<f32>,
    // QSA: indexer keys and queries for this pass, block scores, and the cells each
    // query attends to, `QsaSelect::stride` per row.
    ik: Vec<f32>,
    iq: Vec<f32>,
    bscores: Vec<f32>,
    cells: Vec<u32>,
    hmixed: Vec<f32>,
    logits: Vec<f32>,
    /// Layer-major prefill's residuals, one per sub-chunk, each swapped into
    /// `res` while its sub-chunk runs. Separate buffers rather than one wide
    /// one, so every buffer the seam sees is whole and keeps its own device
    /// mirror across the layers of the pass (CLAUDE.md: never a sub-slice).
    parts: Vec<Vec<f32>>,
    /// Tokens a pass may hold, so per-token buffers are allocated once at that
    /// size (`Scratch::fit`). The engine sets it to its batch; zero until then.
    reserve: usize,
}

impl Scratch {
    /// Bytes of the buffers `fit` sizes, at their current lengths. QSA's buffers
    /// and the layer-major residuals are sized elsewhere and not counted here.
    fn fitted_bytes(&self) -> usize {
        let v = [
            &self.x, &self.res, &self.ones_streams, &self.ones, &self.xn, &self.lo, &self.hgate,
            &self.tmp, &self.mixed, &self.inject, &self.wide, &self.block, &self.qg, &self.q,
            &self.g, &self.k, &self.v, &self.attn, &self.qkv, &self.z, &self.alpha, &self.beta,
            &self.conv, &self.q_part, &self.k_part, &self.v_part, &self.core, &self.router,
            &self.logit, &self.g_all, &self.u_all, &self.o_all, &self.e_gate, &self.e_up,
            &self.e_out, &self.emb, &self.pkey, &self.pvalue, &self.pquery, &self.psdot,
            &self.pgated, &self.pconv_in, &self.pconv_out, &self.hres, &self.hxn, &self.hlo,
            &self.hgate1, &self.htmp, &self.ik, &self.iq, &self.hmixed, &self.logits,
        ];
        v.iter().map(|b| b.len()).sum::<usize>() * std::mem::size_of::<f32>()
    }

    /// Every buffer sized for a pass of `n` tokens.
    ///
    /// **Per-token buffers are allocated once, for `reserve` tokens** (06-10). A
    /// device backend keeps a copy of each buffer keyed on its host address and
    /// never frees it, so a buffer that grows past its capacity moves and
    /// orphans its old device copy. `serve` runs passes of many sizes in one
    /// process (its warm-up, then each turn's sub-chunks) and orphaned ~600 MiB
    /// that way on 06-10, which filled the card and slowed every kernel 2-4x.
    /// Reserved up front, a buffer only resizes inside its allocation, so its
    /// address never changes and a device copy that has to grow is replaced
    /// under the same key, freeing the old one.
    fn fit(&mut self, c: &Config, n: usize) {
        let cap = n.max(self.reserve);
        let (nd, hc, lr, m) = (c.n_embd, c.hc.n_stream, c.hc.low_rank, c.moe);
        // `per` values per token, room for `cap` tokens.
        let z = |b: &mut Vec<f32>, per: usize| {
            if b.capacity() < cap * per {
                b.reserve_exact(cap * per - b.len());
            }
            b.resize(n * per, 0.0);
        };
        // Sizes that do not scale with the pass.
        let f = |b: &mut Vec<f32>, k: usize| b.resize(k, 0.0);
        z(&mut self.x, nd);
        z(&mut self.res, hc * nd);
        self.ones_streams.clear();
        self.ones_streams.reserve_exact(cap * hc);
        self.ones_streams.resize(n * hc, 1.0);
        self.ones.clear();
        self.ones.resize(nd, 1.0);
        z(&mut self.xn, hc * nd);
        z(&mut self.lo, lr);
        z(&mut self.hgate, hc * nd);
        z(&mut self.tmp, nd);
        z(&mut self.mixed, nd);
        z(&mut self.inject, hc);
        z(&mut self.wide, hc * nd);
        z(&mut self.block, nd);
        z(&mut self.qg, c.q_gate_dim());
        z(&mut self.q, c.head_dim * c.n_head);
        z(&mut self.g, c.head_dim * c.n_head);
        z(&mut self.k, c.kv_dim());
        z(&mut self.v, c.kv_dim());
        z(&mut self.attn, c.head_dim * c.n_head);
        z(&mut self.qkv, c.conv_dim());
        z(&mut self.z, c.value_dim());
        z(&mut self.alpha, c.n_v_heads());
        z(&mut self.beta, c.n_v_heads());
        z(&mut self.conv, c.conv_dim());
        z(&mut self.q_part, c.key_dim());
        z(&mut self.k_part, c.key_dim());
        z(&mut self.v_part, c.value_dim());
        z(&mut self.core, c.value_dim());
        z(&mut self.router, m.n_expert);
        z(&mut self.logit, 1);
        z(&mut self.g_all, m.n_expert_used * m.expert_ff);
        z(&mut self.u_all, m.n_expert_used * m.expert_ff);
        z(&mut self.o_all, m.n_expert_used * nd);
        z(&mut self.e_gate, m.shared_ff);
        z(&mut self.e_up, m.shared_ff);
        z(&mut self.e_out, nd);
        z(&mut self.emb, nd);
        z(&mut self.pkey, hc * nd);
        z(&mut self.pvalue, nd);
        z(&mut self.pquery, hc * nd);
        z(&mut self.psdot, hc);
        z(&mut self.pgated, hc * nd);
        z(&mut self.pconv_in, hc * nd);
        z(&mut self.pconv_out, hc * nd);
        f(&mut self.hres, hc * nd);
        f(&mut self.hxn, hc * nd);
        f(&mut self.hlo, lr);
        f(&mut self.hgate1, hc * nd);
        f(&mut self.htmp, nd);
        z(&mut self.ik, c.indexer.head_dim);
        z(&mut self.iq, c.indexer.n_head * c.indexer.head_dim);
        f(&mut self.hmixed, nd);
        f(&mut self.logits, c.n_vocab);
    }
}

/// A hyper-connection read (`build_hc_mix`, `qwen4exp.cpp:266-312`): RMSNorm each
/// stream, times the `hc_dim` weight; a low-rank sigmoid gate; the gated streams'
/// mean into `mixed`; and, when asked, the inject logits from the normed input.
///
/// `res` is `[n][n_stream][n_embd]`; `xn`, `gate` the same size; `lo` `[n][low_rank]`;
/// `tmp` and `mixed` `[n][n_embd]`; `inject` `[n][n_stream]`.
#[allow(clippy::too_many_arguments)]
fn hc_read<O: Ops>(
    ops: &O,
    c: &Config,
    hc: &Hc<'_>,
    res: &[f32],
    ones: &[f32],
    xn: &mut [f32],
    lo: &mut [f32],
    gate: &mut [f32],
    tmp: &mut [f32],
    mixed: &mut [f32],
    inject: Option<&mut [f32]>,
    trace: Option<(&mut Ctx<'_>, usize)>,
) -> Result<()> {
    let (nd, n_stream) = (c.n_embd, c.hc.n_stream);
    let inv = 1.0 / n_stream as f32;
    ops.gather_chunks(res, res.len(), res.len(), 0, xn);
    ops.rms_norm_heads(xn, ones, nd, c.rms_eps);
    ops.mul_rows(xn, &hc.norm);
    ops.matmul(&hc.down, xn, lo);
    ops.scale(lo, inv);
    ops.silu(lo);
    ops.matmul(&hc.up, lo, gate);
    // The inject logits read the normed input, before it is gated in place.
    if let Some(inj) = inject {
        let w = hc.inject.as_ref().ok_or_else(|| Error::InconsistentArchitecture {
            what: "hyper-connection",
            detail: "an inject was asked of a module without inject weights".to_string(),
        })?;
        ops.matmul(w, xn, inj);
    }
    let mut trace = trace;
    if let Some((ctx, il)) = trace.as_mut() {
        ctx.trace("hc_norm", *il, xn);
    }
    ops.sigmoid_mul(xn, gate);
    // The mean in the reference's order: stream 0, + 1, + 2, ..., then x 1/n.
    //
    // **One op, not eight** (20-09). The sequence below it ran four
    // `gather_chunks`, three `add_assign` and one `scale` to average four
    // strided slices — twice a layer, 768 launches a token on the 125B, where
    // those two kernels are 9.2% and 5.8% of GPU time and this mixer is ~62%
    // and ~74% of them. `Ops::mean_streams` is bit-identical to it, so the
    // control below is an A/B lever rather than an exact-path escape.
    if hc_fused() {
        ops.mean_streams(xn, n_stream, nd, inv, mixed);
    } else {
        ops.gather_chunks(xn, nd, n_stream * nd, 0, mixed);
        for s in 1..n_stream {
            ops.gather_chunks(xn, nd, n_stream * nd, s * nd, tmp);
            ops.add_assign(mixed, tmp);
        }
        ops.scale(mixed, inv);
    }
    Ok(())
}

impl<'a> Qwen4Exp<'a> {
    /// Run `tokens` from `start_pos` and return the last one's logits.
    ///
    /// The graph of `qwen4exp.cpp:336-441`, written out in `src/model/qwen4exp.md`
    /// ("The forward pass"). Dense attention only: a pass that would put more than
    /// `top_k + compress_ratio - 1` cells in a QSA layer's cache is refused, since
    /// only up to there is dense attention exactly QSA.
    pub fn forward<O: Ops>(
        &self,
        ops: &O,
        tokens: &[u32],
        start_pos: usize,
        kv: &mut KvCache,
        rs: &mut RecurrentState,
        ctx: &mut Ctx<'_>,
    ) -> Result<Vec<f32>> {
        let n = tokens.len();
        self.check_pass(n, start_pos, kv, rs)?;
        let mut st = self.ple_state.borrow_mut();
        self.ple_begin(ops, start_pos, &mut st)?;

        ops.begin_pass(n);
        let s = &mut *self.scratch.borrow_mut();
        s.fit(&self.cfg, n);
        self.embed(ops, tokens, s, ctx)?;
        for il in 0..self.cfg.n_layer {
            self.layer(ops, il, tokens, start_pos, kv, rs, &mut st, s, ctx)?;
        }
        kv.commit(start_pos + n);
        st.next_pos = start_pos + n;
        self.head(ops, n, s, ctx)
    }

    /// [`Qwen4Exp::forward`] over a prompt longer than one pass should hold,
    /// **layer-major**: every sub-chunk of at most `max_sub` tokens runs through
    /// layer L before any runs through L+1, all inside one pass.
    ///
    /// **Why** (`TODO.md`, layer-major prefill): run chunk by chunk, a 512-token
    /// chunk of the 125B touches nearly every expert of every layer (512 tokens x
    /// top-10 over 512 experts), so each chunk fetched most of the ~48 GiB cold
    /// pool again — 470,938 experts, 404 GiB, for a 5,688-token prompt. Layer by
    /// layer, a layer's experts are fetched once and every sub-chunk then hits
    /// them, since one layer's 1,536 expert tensors fit the slab many times over.
    ///
    /// **The same arithmetic as chunk-major prefill over the same split**, in a
    /// different order. Each sub-chunk runs [`Qwen4Exp::layer`] exactly as a pass
    /// of its own would, at its own `start_pos`: attention reads positions from
    /// the arguments, not from the committed length, the recurrent state and
    /// QSA's pooled-key watermark are per layer and visited in token order, and
    /// PLE's window advances inside [`Qwen4Exp::ple`] as each sub-chunk passes
    /// through it. `the_0_2b_prefills_layer_major_bit_identically` holds it to the
    /// bit.
    ///
    /// **One pass, because `begin_pass` marks every device mirror stale**: a
    /// sub-chunk's residual must stay on the device from one layer to the next,
    /// so each has its own buffer in `Scratch::parts`, swapped into `res` while
    /// it runs.
    ///
    /// **The split is even** — sizes differ by at most one token — because
    /// `Scratch::fit` regrows a buffer by writing it, and a short last sub-chunk
    /// would regrow every scratch buffer at every layer: ~380 MiB of host writes
    /// per layer for a 5,688-token prompt at 512, against one token's worth here.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_layer_major<O: Ops>(
        &self,
        ops: &O,
        tokens: &[u32],
        start_pos: usize,
        max_sub: usize,
        kv: &mut KvCache,
        rs: &mut RecurrentState,
        ctx: &mut Ctx<'_>,
    ) -> Result<Vec<f32>> {
        let n = tokens.len();
        self.check_pass(n, start_pos, kv, rs)?;
        let mut st = self.ple_state.borrow_mut();
        self.ple_begin(ops, start_pos, &mut st)?;

        let subs = even_split(n, max_sub);
        let w = self.cfg.hc.n_stream * self.cfg.n_embd;
        let s = &mut *self.scratch.borrow_mut();
        // Every allocation before `begin_pass`, so no buffer is freed inside the
        // pass and no address a mirror was written through this pass is reused.
        // The first sub-chunk is the largest, so no later `fit` grows past it.
        if s.parts.len() < subs.len() {
            s.parts.resize_with(subs.len(), Vec::new);
        }
        // Each residual is allocated for a whole sub-chunk once, for the reason
        // `Scratch::fit` gives: a part that moved would orphan its device copy.
        let room = max_sub.max(1).max(s.reserve) * w;
        for (part, &(_, len)) in s.parts.iter_mut().zip(&subs) {
            if part.capacity() < room {
                part.reserve_exact(room - part.len());
            }
            part.resize(len * w, 0.0);
        }
        s.fit(&self.cfg, subs[0].1);

        ops.begin_pass(n);
        for (k, &(off, len)) in subs.iter().enumerate() {
            s.fit(&self.cfg, len);
            std::mem::swap(&mut s.res, &mut s.parts[k]);
            let r = self.embed(ops, &tokens[off..off + len], s, ctx);
            std::mem::swap(&mut s.res, &mut s.parts[k]);
            r?;
        }
        for il in 0..self.cfg.n_layer {
            for (k, &(off, len)) in subs.iter().enumerate() {
                s.fit(&self.cfg, len);
                std::mem::swap(&mut s.res, &mut s.parts[k]);
                let r = self.layer(ops, il, &tokens[off..off + len], start_pos + off, kv, rs, &mut st, s, ctx);
                std::mem::swap(&mut s.res, &mut s.parts[k]);
                r?;
            }
        }
        kv.commit(start_pos + n);
        st.next_pos = start_pos + n;

        let last = subs.len() - 1;
        let len = subs[last].1;
        s.fit(&self.cfg, len);
        std::mem::swap(&mut s.res, &mut s.parts[last]);
        let out = self.head(ops, len, s, ctx);
        std::mem::swap(&mut s.res, &mut s.parts[last]);
        out
    }

    /// What every pass checks before it touches anything.
    fn check_pass(&self, n: usize, start_pos: usize, kv: &KvCache, rs: &RecurrentState) -> Result<()> {
        let c = &self.cfg;
        if n == 0 {
            return Err(Error::InconsistentArchitecture {
                what: "forward",
                detail: "no tokens supplied".to_string(),
            });
        }
        rs.check(c.n_layer, c.conv_state_len(), c.ssm_state_len())?;
        if kv.kv_dim() != c.kv_dim() {
            return Err(Error::InconsistentArchitecture {
                what: "kv cache",
                detail: format!("cache holds {} lanes per position, model needs {}", kv.kv_dim(), c.kv_dim()),
            });
        }
        if start_pos + n > kv.n_ctx() {
            return Err(Error::ContextOverflow { pos: start_pos + n - 1, n_ctx: kv.n_ctx() });
        }
        // Past QSA's budget the indexer reads keys cached at every earlier position,
        // so the cache must have been built with the lane (`Engine::new` does).
        if c.compress_ratios.iter().any(|&r| r > 0) && kv.idx_dim() != c.indexer.head_dim {
            return Err(Error::InconsistentArchitecture {
                what: "kv cache",
                detail: format!(
                    "the cache holds {} indexer values per position, QSA needs {}",
                    kv.idx_dim(),
                    c.indexer.head_dim
                ),
            });
        }
        Ok(())
    }

    /// PLE's per-sequence state at the start of a pass: cleared at position 0,
    /// and otherwise required to continue exactly where the last pass ended.
    fn ple_begin<O: Ops>(&self, ops: &O, start_pos: usize, st: &mut PleState) -> Result<()> {
        let c = &self.cfg;
        if let Some(p) = &c.ple {
            if start_pos == 0 {
                st.prev.clear();
                st.conv.clear();
                st.conv.resize(c.hc_dim() * (p.conv_kernel - 1) * p.ngram_size, 0.0);
                // A device backend owns the conv history once a kernel has written
                // it, so zeroing the host copy is invisible to it without this —
                // the same reason `Engine::reset` calls it for the GDN state.
                ops.forget_state();
            } else if start_pos != st.next_pos {
                return Err(Error::NotImplemented {
                    what: "PLE history across a rewind",
                    detail: format!(
                        "pass at position {start_pos}, but the n-gram window and conv history end at {}",
                        st.next_pos
                    ),
                });
            }
        }
        Ok(())
    }

    /// The embeddings of `tokens` into `s.x`, broadcast into the wide residual
    /// `s.res`. `s` must already fit `tokens.len()`.
    fn embed<O: Ops>(&self, ops: &O, tokens: &[u32], s: &mut Scratch, ctx: &mut Ctx<'_>) -> Result<()> {
        let c = &self.cfg;
        let (nd, n_stream) = (c.n_embd, c.hc.n_stream);
        for (t, &token) in tokens.iter().enumerate() {
            if token as usize >= c.n_vocab {
                return Err(Error::TokenOutOfRange { id: token, vocab_size: c.n_vocab });
            }
            dequantize_into(self.tok_embd.row(token as usize), self.tok_embd.ty, &mut s.x[t * nd..(t + 1) * nd])?;
        }
        ops.host_wrote(&s.x);
        ops.host_wrote(&s.ones_streams);
        ops.host_wrote(&s.ones);
        ctx.trace("inp_embd", 0, &s.x);

        ops.mul_streams(&mut s.res, &s.x, &s.ones_streams, n_stream);
        ctx.trace("hc_init", 0, &s.res);
        Ok(())
    }

    /// Layer `il` over `tokens`, which start at `start_pos`, on the residual in
    /// `s.res`. The whole of a layer, so a pass and a layer-major pass run the
    /// same code for it.
    #[allow(clippy::too_many_arguments)]
    fn layer<O: Ops>(
        &self,
        ops: &O,
        il: usize,
        tokens: &[u32],
        start_pos: usize,
        kv: &mut KvCache,
        rs: &mut RecurrentState,
        st: &mut PleState,
        s: &mut Scratch,
        ctx: &mut Ctx<'_>,
    ) -> Result<()> {
        let c = &self.cfg;
        let n = tokens.len();
        let layer = &self.layers[il];
        if layer.ple.is_some() {
            self.ple(ops, il, tokens, st, s, ctx)?;
        }

        hc_read(
            ops, c, &layer.hc_attn, &s.res, &s.ones, &mut s.xn, &mut s.lo, &mut s.hgate,
            &mut s.tmp, &mut s.mixed, Some(&mut s.inject), None,
        )?;
        ctx.trace("hc_attn_mixed", il, &s.mixed);
        match &layer.mixer {
            Mixer::Delta { .. } => self.gated_delta(ops, layer, il, rs, s, ctx)?,
            Mixer::Attn { .. } => self.attention(ops, layer, il, start_pos, n, kv, s, ctx)?,
        }
        self.hc_write(ops, s);
        // The reference's second combine is renamed `l_last` (cb overwrites the
        // name), so its printed `hc_combine` is this one, after the mixer.
        ctx.trace("hc_combine", il, &s.res);

        hc_read(
            ops, c, &layer.hc_ffn, &s.res, &s.ones, &mut s.xn, &mut s.lo, &mut s.hgate,
            &mut s.tmp, &mut s.mixed, Some(&mut s.inject), Some((&mut *ctx, il)),
        )?;
        ctx.trace("hc_mixed", il, &s.mixed);
        ctx.trace("hc_inject", il, &s.inject);
        // `INFERRED_ROUTER_DUMP`: the router's input, for the prefetch study.
        // Reads it home, so graphs go off for the run; a measurement of what
        // ran, not of speed.
        if let Some(d) = crate::dump::router() {
            ops.host_needs(&mut s.mixed);
            let head = [il as u32, n as u32, c.n_embd as u32];
            d.record(&[
                b"RDMP",
                crate::dump::bytes_of(&head[..1]),
                &(start_pos as u64).to_le_bytes(),
                crate::dump::bytes_of(&head[1..]),
                crate::dump::bytes_of(&s.mixed),
            ]);
        }
        let next_router = self.layers.get(il + 1).map(|l| &l.ffn.gate_inp);
        self.moe(ops, layer, next_router, n, s);
        ctx.trace("ffn_moe_out", il, &s.block);
        self.hc_write(ops, s);
        ctx.trace("l_last", il, &s.res);
        Ok(())
    }

    /// The output head on the last of the `n` rows in `s.res`, and the end of the
    /// pass.
    fn head<O: Ops>(&self, ops: &O, n: usize, s: &mut Scratch, ctx: &mut Ctx<'_>) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let (nd, n_stream) = (c.n_embd, c.hc.n_stream);
        // The final mixer carries the output norm; only the last row is needed.
        let w = n_stream * nd;
        ops.gather_chunks(&s.res, w, w, (n - 1) * w, &mut s.hres);
        hc_read(
            ops, c, &self.head_hc, &s.hres, &s.ones, &mut s.hxn, &mut s.hlo, &mut s.hgate1,
            &mut s.htmp, &mut s.hmixed, None, None,
        )?;
        ctx.trace("result_norm", 0, &s.hmixed);
        ops.matmul(&self.output, &s.hmixed, &mut s.logits);
        if let Some(v) = self.output_s {
            ops.scale(&mut s.logits, v);
        }
        ops.end_pass();
        ops.host_needs(&mut s.logits);
        ctx.trace("result_output", 0, &s.logits);
        Ok(s.logits.clone())
    }

    /// A hyper-connection write (`build_hc_combine`, `qwen4exp.cpp:314-334`):
    /// `res += block · 2·sigmoid(inject / n_stream)`, per stream.
    fn hc_write<O: Ops>(&self, ops: &O, s: &mut Scratch) {
        let n_stream = self.cfg.hc.n_stream;
        ops.scale(&mut s.inject, 1.0 / n_stream as f32);
        ops.sigmoid(&mut s.inject);
        ops.scale(&mut s.inject, 2.0);
        ops.mul_streams(&mut s.wide, &s.block, &s.inject, n_stream);
        ops.add_assign(&mut s.res, &s.wide);
    }

    /// GatedDeltaNet (`build_layer_attn_linear`, `qwen4exp.cpp:847-972`): as
    /// `qwen35`'s, with the output gate `sigmoid(z)` rather than SiLU.
    fn gated_delta<O: Ops>(
        &self,
        ops: &O,
        layer: &Layer<'_>,
        il: usize,
        rs: &mut RecurrentState,
        s: &mut Scratch,
        ctx: &mut Ctx<'_>,
    ) -> Result<()> {
        let c = &self.cfg;
        let Mixer::Delta { wqkv, wgate, conv1d, dt_bias, ssm_a, ssm_beta, ssm_alpha, ssm_norm, ssm_out } =
            &layer.mixer
        else {
            return Err(Error::InconsistentArchitecture {
                what: "layer kind",
                detail: format!("layer {il} is attention but was routed to the delta rule"),
            });
        };
        let (kdim, vdim, cdim) = (c.key_dim(), c.value_dim(), c.conv_dim());
        ops.matmul(wqkv, &s.mixed, &mut s.qkv);
        ctx.trace("linear_attn_qkv_mixed", il, &s.qkv);
        ops.matmul(wgate, &s.mixed, &mut s.z);
        ctx.trace("z", il, &s.z);
        ops.matmul_pair(ssm_alpha, ssm_beta, &s.mixed, &mut s.alpha, &mut s.beta);
        ops.ssm_conv(rs.conv_mut(il), &s.qkv, conv1d, c.ssm_d_conv, &mut s.conv);
        ctx.trace("conv_output_silu", il, &s.conv);
        ops.gather_chunks(&s.conv, kdim, cdim, 0, &mut s.q_part);
        ops.gather_chunks(&s.conv, kdim, cdim, kdim, &mut s.k_part);
        ops.gather_chunks(&s.conv, vdim, cdim, 2 * kdim, &mut s.v_part);
        ops.l2_norm_heads(&mut s.q_part, c.head_k_dim(), c.rms_eps);
        ops.l2_norm_heads(&mut s.k_part, c.head_k_dim(), c.rms_eps);
        ctx.trace("q_conv_predelta", il, &s.q_part);
        ctx.trace("k_conv_predelta", il, &s.k_part);
        let d = Delta {
            q: &s.q_part,
            k: &s.k_part,
            v: &s.v_part,
            alpha: &s.alpha,
            beta: &s.beta,
            ssm_a,
            dt_bias,
            head_k_dim: c.head_k_dim(),
            head_v_dim: c.head_v_dim(),
            n_k_heads: c.n_k_heads(),
            n_v_heads: c.n_v_heads(),
        };
        ops.delta_rule(&d, rs.ssm_mut(il), &mut s.core);
        // build_norm_gated: rms_norm(core, ssm_norm) · sigmoid(z) (qwen4exp.cpp:459-469).
        ops.rms_norm_heads(&mut s.core, ssm_norm, c.head_v_dim(), c.rms_eps);
        ops.sigmoid_mul(&mut s.core, &s.z);
        ctx.trace("final_output", il, &s.core);
        ops.matmul(ssm_out, &s.core, &mut s.block);
        ctx.trace("linear_attn_out", il, &s.block);
        Ok(())
    }

    /// Gated attention (`build_layer_attn`, `qwen4exp.cpp:761-845`), dense: as
    /// `qwen35`'s, which is exactly QSA within the budget `forward` enforces.
    #[allow(clippy::too_many_arguments)]
    fn attention<O: Ops>(
        &self,
        ops: &O,
        layer: &Layer<'_>,
        il: usize,
        start_pos: usize,
        n: usize,
        kv: &mut KvCache,
        s: &mut Scratch,
        ctx: &mut Ctx<'_>,
    ) -> Result<()> {
        let c = &self.cfg;
        let Mixer::Attn { wq, wk, wv, wo, q_norm, k_norm, indexer } = &layer.mixer else {
            return Err(Error::InconsistentArchitecture {
                what: "layer kind",
                detail: format!("layer {il} is recurrent but was routed to attention"),
            });
        };
        let (hd, kd) = (c.head_dim, c.kv_dim());
        let slot = self.kv_slot[il];
        let ratio = c.compress_ratios[il] as usize;
        // QSA: every pass caches this layer's raw indexer keys, below the budget too,
        // because a later query pools them (`qwen4exp.cpp:599-603`) — but only when
        // this cache is long enough for a pass ever to pass the budget. A context of
        // 2,051 or fewer is dense attention throughout and never reads them, so it
        // runs no indexer at all (and the 0.2B's F16 indexer, which CUDA has no
        // matmul for yet, stays out of GPU runs that cannot use it).
        let qsa_reachable = ratio > 0 && kv.n_ctx() > c.indexer.top_k + ratio - 1;
        if qsa_reachable {
            ops.matmul(&indexer.k_proj, &s.mixed, &mut s.ik);
            ctx.trace("indexer_k_raw", il, &s.ik);
            // These cells' pooled blocks are stale from here on, whether or not
            // this pass pools (`KvCache::lower_pooled`).
            kv.lower_pooled(slot, start_pos);
            ops.kv_write(kv.idx_layer_mut(slot), start_pos * c.indexer.head_dim, &s.ik);
        }
        // Past the budget, choose each query's cells before K and V are published:
        // the selection reads only indexer keys, so the order does not matter to it.
        // `INFERRED_QSA_DENSE=1` keeps every cell past the budget: the control arm,
        // and what the model did before the indexer existed. Slower, not exact — the
        // model was trained to see the selection.
        //
        // **Decode selects at every depth once QSA is reachable**, so its kernel
        // sequence never changes as the context crosses the budget (CLAUDE.md);
        // below it the selection keeps every cell, which is dense attention. A
        // prefill wholly below the budget stays dense.
        let sparse = qsa_reachable
            && (n == 1 || start_pos + n > c.indexer.top_k + ratio - 1)
            && std::env::var("INFERRED_QSA_DENSE").is_err();
        let sel = sparse.then(|| QsaSelect {
            n_head: c.indexer.n_head,
            dim: c.indexer.head_dim,
            n_blocks: (start_pos + n) / ratio,
            start_pos,
            ratio,
            budget: c.indexer.top_k / ratio,
        });
        if let Some(sel) = &sel {
            self.qsa_select(ops, indexer, il, slot, n, sel, kv, s, ctx);
        }
        ops.matmul(wq, &s.mixed, &mut s.qg);
        ctx.trace("Qcur_full", il, &s.qg);
        ops.gather_chunks(&s.qg, hd, 2 * hd, 0, &mut s.q);
        ops.gather_chunks(&s.qg, hd, 2 * hd, hd, &mut s.g);
        ops.matmul(wk, &s.mixed, &mut s.k);
        ops.matmul(wv, &s.mixed, &mut s.v);
        ops.rms_norm_heads(&mut s.q, q_norm, hd, c.rms_eps);
        ctx.trace("Qcur_normed", il, &s.q);
        ops.rms_norm_heads(&mut s.k, k_norm, hd, c.rms_eps);
        ctx.trace("Kcur_normed", il, &s.k);
        ops.rope_neox(&mut s.q, start_pos, hd, c.n_rot, c.n_head, c.rope_theta);
        ctx.trace("Qcur", il, &s.q);
        ops.rope_neox(&mut s.k, start_pos, hd, c.n_rot, c.n_head_kv, c.rope_theta);
        ctx.trace("Kcur", il, &s.k);
        ops.kv_write(kv.k_layer_mut(slot), start_pos * kd, &s.k);
        ops.kv_write(kv.v_layer_mut(slot), start_pos * kd, &s.v);
        let a = Attn {
            q: &s.q,
            k: kv.k_layer(slot),
            v: kv.v_layer(slot),
            kv_dim: kd,
            n_pos: start_pos + n,
            head_dim: hd,
            n_head: c.n_head,
            n_head_kv: c.n_head_kv,
            scale: 1.0 / (hd as f32).sqrt(),
        };
        if let Some(sel) = &sel {
            ops.attend_sparse(&a, &s.cells, sel, &mut s.attn);
        } else {
            ops.attend(&a, &mut s.attn);
        }
        ctx.trace("attn_pregate", il, &s.attn);
        ops.sigmoid_mul(&mut s.attn, &s.g);
        ctx.trace("attn_gated", il, &s.attn);
        ops.matmul(wo, &s.attn, &mut s.block);
        ctx.trace("attn_output", il, &s.block);
        Ok(())
    }

    /// QSA's indexer: which cells each query of this pass attends to, into
    /// `s.cells` at `sel.stride()` per row (`build_qsa_top_k`,
    /// `qwen4exp.cpp:525-674`, with the model's reference selection —
    /// `ops::naive::select_cells`).
    ///
    /// - queries: `index_q_proj`, RMSNorm per head, RoPE at their positions;
    /// - keys: every block whole at the pass's last position, pooled once each into
    ///   the cache's pooled lane ([`Ops::qsa_pool`]) from the layer's watermark,
    ///   which the cache lowers to this pass's start first;
    /// - scores and selection: [`Ops::qsa_select`], one score per (query, block).
    #[allow(clippy::too_many_arguments)]
    fn qsa_select<O: Ops>(
        &self,
        ops: &O,
        ix: &Indexer<'_>,
        il: usize,
        slot: usize,
        n: usize,
        sel: &QsaSelect,
        kv: &mut KvCache,
        s: &mut Scratch,
        ctx: &mut Ctx<'_>,
    ) {
        let c = &self.cfg;
        let (idim, ih) = (sel.dim, sel.n_head);

        ops.matmul(&ix.q_proj, &s.mixed, &mut s.iq);
        ops.rms_norm_heads(&mut s.iq, &ix.q_norm, idim, c.rms_eps);
        ops.rope_neox(&mut s.iq, sel.start_pos, idim, c.n_rot, ih, c.rope_theta);
        ctx.trace("indexer_q", il, &s.iq);

        let (raw, pooled, from) = kv.pooled_mut(slot, sel.start_pos);
        let pool = QsaPool {
            from,
            to: sel.n_blocks,
            ratio: sel.ratio,
            dim: idim,
            norm: &ix.k_norm,
            eps: c.rms_eps,
            n_rot: c.n_rot,
            theta: c.rope_theta,
        };
        ops.qsa_pool(raw, pooled, &pool);
        kv.set_pooled_through(slot, sel.n_blocks);

        // Resized, never sliced: a device backend keys buffers on their address.
        // And reserved for a full batch at the deepest the context allows, once:
        // `bscores` grows with depth, and a buffer that moved would orphan its
        // device copy (`Scratch::fit`).
        let cap = n.max(s.reserve);
        let blocks_max = (kv.n_ctx() / sel.ratio.max(1)).max(sel.n_blocks).max(1);
        if s.bscores.capacity() < cap * blocks_max {
            s.bscores.reserve_exact(cap * blocks_max - s.bscores.len());
        }
        if s.cells.capacity() < cap * sel.stride() {
            s.cells.reserve_exact(cap * sel.stride() - s.cells.len());
        }
        s.bscores.resize(n * sel.n_blocks.max(1), 0.0);
        s.cells.resize(n * sel.stride(), 0);
        ops.qsa_select(&s.iq, kv.pooled_layer(slot), sel, &mut s.bscores, &mut s.cells);
    }

    /// The routed experts plus the gated shared expert (`build_layer_ffn`,
    /// `qwen4exp.cpp:974-1022`) into `s.block`, from `s.mixed`. The same seam calls
    /// as `qwen35::moe_batch`, for the whole pass at once.
    fn moe<O: Ops>(&self, ops: &O, layer: &Layer<'_>, next_router: Option<&Weights<'_>>, n: usize, s: &mut Scratch) {
        let (m, nd, f) = (self.cfg.moe, self.cfg.n_embd, &layer.ffn);
        ops.matmul_pair(&f.gate_inp, &f.shared_gate_inp, &s.mixed, &mut s.router, &mut s.logit);
        ops.softmax(&mut s.router, m.n_expert);
        let route = ops.route(&mut s.router, m.n_expert, m.n_expert_used);
        // Decode: the next layer's likely experts, from this layer's input, so
        // their reads can start while this layer computes (SSD-TIER.md D20).
        if n == 1
            && let Some(w) = next_router
        {
            ops.prefetch_hint(w, &s.mixed, prefetch_k());
        }
        ops.moe_glu(&f.gate, &f.up, &route, &s.mixed, &mut s.g_all, &mut s.u_all);
        ops.matmul_experts(&f.down, &route, &s.g_all, &mut s.o_all);
        ops.matmul(&f.shared_gate, &s.mixed, &mut s.e_gate);
        ops.matmul(&f.shared_up, &s.mixed, &mut s.e_up);
        if let Some(v) = f.shared_up_s {
            ops.scale(&mut s.e_up, v);
        }
        if let Some(v) = f.shared_gate_s {
            ops.scale(&mut s.e_gate, v);
        }
        ops.silu_mul(&mut s.e_gate, &s.e_up);
        ops.matmul(&f.shared_down, &s.e_gate, &mut s.e_out);
        if let Some(v) = f.shared_down_s {
            ops.scale(&mut s.e_out, v);
        }
        ops.moe_finish(&mut s.block, 0, nd, &s.o_all, &route, &s.e_out, &s.logit, 0);
        let _ = n;
    }

    /// The PLE block, applied to the wide residual before layer `il`'s HC read
    /// (`build_inp_ple` and `build_ple`, `qwen4exp.cpp:1024-1283`).
    #[allow(clippy::too_many_arguments)]
    fn ple<O: Ops>(
        &self,
        ops: &O,
        il: usize,
        tokens: &[u32],
        st: &mut PleState,
        s: &mut Scratch,
        ctx: &mut Ctx<'_>,
    ) -> Result<()> {
        let c = &self.cfg;
        let (Some(p), Some(table), Some(pm)) = (&c.ple, &self.ple_table, &self.layers[il].ple) else {
            return Err(Error::InconsistentArchitecture {
                what: "PLE",
                detail: format!("layer {il} has a PLE mixer but the model has no PLE table or settings"),
            });
        };
        let (nd, n_stream, hd, nh) = (c.n_embd, c.hc.n_stream, p.head_dim, p.n_heads());
        let eos = p.eos_token_id;

        // The row indices, on the host: ggml has no int64 or xor (qwen4exp.cpp:1084-1110).
        for (t, &token) in tokens.iter().enumerate() {
            let mut window = vec![0u64; p.ngram_size];
            window[0] = u64::from(token);
            let mut cut = false;
            for back in 1..p.ngram_size {
                // `back` positions before token `t`: in this pass, else in the window
                // carried from the last one, else before the sequence start.
                let pred = if t >= back {
                    Some(tokens[t - back])
                } else {
                    let from_prev = back - t;
                    st.prev.len().checked_sub(from_prev).map(|i| st.prev[i])
                };
                let tok = if cut { None } else { pred };
                cut = cut || tok.is_none() || tok == Some(eos);
                window[back] = if cut { u64::from(eos) } else { u64::from(tok.unwrap_or(eos)) };
            }
            for ng in 2..=p.ngram_size {
                let mut mixed = window[0].wrapping_mul(p.layer_multipliers[0]);
                for j in 1..ng {
                    mixed ^= window[j].wrapping_mul(p.layer_multipliers[j]);
                }
                let base = (ng - 2) * p.heads_per_ngram;
                for g in 0..p.heads_per_ngram {
                    let h = base + g;
                    let row = mixed % u64::from(p.head_vocab_sizes[h]) + u64::from(p.head_offsets[h]);
                    let at = t * nd + h * hd;
                    dequantize_into(table.row(row as usize), table.ty, &mut s.emb[at..at + hd])?;
                }
            }
        }
        let _ = nh;
        ops.host_wrote(&s.emb);
        ctx.trace("ple_embd", 0, &s.emb);

        ops.matmul(&pm.key, &s.emb, &mut s.pkey);
        ops.matmul(&pm.value, &s.emb, &mut s.pvalue);
        ops.rms_norm_heads(&mut s.pkey, &s.ones, nd, c.rms_eps);
        ops.mul_rows(&mut s.pkey, &pm.norm_key);
        ops.gather_chunks(&s.res, s.res.len(), s.res.len(), 0, &mut s.pquery);
        ops.rms_norm_heads(&mut s.pquery, &s.ones, nd, c.rms_eps);
        ops.mul_rows(&mut s.pquery, &pm.norm_query);
        ops.row_dot(&s.pkey, &s.pquery, nd, &mut s.psdot);
        ops.scale(&mut s.psdot, 1.0 / (nd as f32).sqrt());
        ops.signed_sqrt_sigmoid(&mut s.psdot);
        ctx.trace("ple_gate", il, &s.psdot);

        ops.mul_streams(&mut s.pgated, &s.pvalue, &s.psdot, n_stream);
        ctx.trace("ple_gated_value", il, &s.pgated);
        ops.gather_chunks(&s.pgated, s.pgated.len(), s.pgated.len(), 0, &mut s.pconv_in);
        ops.rms_norm_heads(&mut s.pconv_in, &s.ones, nd, c.rms_eps);
        ops.mul_rows(&mut s.pconv_in, &pm.norm_conv);
        ops.dilated_conv(&mut st.conv, &s.pconv_in, &pm.conv1d, p.conv_kernel, p.ngram_size, &mut s.pconv_out);
        ops.silu(&mut s.pconv_out);
        ctx.trace("ple_conv_out", il, &s.pconv_out);

        ops.add_assign(&mut s.pgated, &s.pconv_out);
        ops.add_assign(&mut s.res, &s.pgated);

        // The window the next tokens to pass through here will read. Advanced
        // here rather than at the end of the pass, so a layer-major pass, whose
        // sub-chunks all reach this layer before any reaches the head, sees each
        // sub-chunk's predecessors exactly as a pass of its own would.
        let keep = p.ngram_size - 1;
        let mut seq: Vec<u32> = st.prev.iter().copied().chain(tokens.iter().copied()).collect();
        if seq.len() > keep {
            seq.drain(..seq.len() - keep);
        }
        st.prev = seq;
        Ok(())
    }
}

/// `n` tokens cut into the fewest sub-chunks of at most `max` tokens, as evenly as
/// they go: `(offset, len)`, the longer ones first, lengths differing by at most
/// one. See [`Qwen4Exp::forward_layer_major`] for why even.
fn even_split(n: usize, max: usize) -> Vec<(usize, usize)> {
    let k = n.div_ceil(max.max(1)).max(1);
    let (q, r) = (n / k, n % k);
    let mut out = Vec::with_capacity(k);
    let mut off = 0;
    for i in 0..k {
        let len = q + usize::from(i < r);
        out.push((off, len));
        off += len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_split_covers_the_prompt_in_the_fewest_near_equal_sub_chunks() {
        assert_eq!(even_split(5688, 512), (0..12).map(|i| (i * 474, 474)).collect::<Vec<_>>());
        assert_eq!(even_split(1025, 512), vec![(0, 342), (342, 342), (684, 341)]);
        assert_eq!(even_split(512, 512), vec![(0, 512)]);
        assert_eq!(even_split(7, 1), (0..7).map(|i| (i, 1)).collect::<Vec<_>>());
        for (n, max) in [(5688, 512), (5689, 512), (2061, 1024), (513, 512), (3, 8)] {
            let s = even_split(n, max);
            assert_eq!(s.len(), n.div_ceil(max), "{n}/{max}: not the fewest");
            assert_eq!(s.iter().map(|p| p.1).sum::<usize>(), n, "{n}/{max}: does not cover");
            assert!(s.windows(2).all(|w| w[0].0 + w[0].1 == w[1].0), "{n}/{max}: not contiguous");
            assert!(s.iter().all(|p| p.1 <= max), "{n}/{max}: a sub-chunk over the cap");
            assert!(s.windows(2).all(|w| w[0].1 >= w[1].1 && w[0].1 - w[1].1 <= 1), "{n}/{max}: not even");
        }
    }

    /// The 125B's settings, from `inferred inspect` on
    /// `Qwen3.8-Flash-Next-NVFP4-Q8_0.gguf`, so the derived dimensions can be
    /// checked without the file.
    fn flash_next() -> Config {
        Config {
            n_layer: 48,
            n_embd: 2560,
            n_head: 24,
            n_head_kv: 2,
            head_dim: 256,
            n_vocab: 248_320,
            rope_theta: 1.0e7,
            rms_eps: 1.0e-6,
            n_rot: 64,
            rope_sections: [11, 11, 10, 0],
            ssm_d_conv: 4,
            ssm_d_inner: 6144,
            ssm_d_state: 128,
            ssm_dt_rank: 48,
            ssm_n_group: 16,
            recurrent: (0..48).map(|i| (i + 1) % 4 != 0).collect(),
            compress_ratios: (0..48).map(|i| if (i + 1) % 4 == 0 { 4 } else { 0 }).collect(),
            moe: Moe { n_expert: 512, n_expert_used: 10, expert_ff: 640, shared_ff: 640 },
            hc: HyperConnections { n_stream: 4, low_rank: 320 },
            indexer: IndexerConfig { n_head: 4, head_dim: 128, top_k: 2048 },
            ple: None,
        }
    }

    /// The derived dimensions agree with the tensor shapes in the file: `attn_qkv`
    /// `{2560, 10240}`, `attn_gate` `{2560, 6144}`, `attn_q` `{2560, 12288}`,
    /// `hc_attn_norm` `{10240}`, and 12 QSA layers at `il % 4 == 3`.
    #[test]
    fn derived_dimensions_match_the_125b_tensors() {
        let c = flash_next();
        assert_eq!(c.conv_dim(), 10_240);
        assert_eq!(c.value_dim(), 6144);
        assert_eq!(c.q_gate_dim(), 12_288);
        assert_eq!(c.hc_dim(), 10_240);
        assert_eq!(c.kv_dim(), 512);
        assert_eq!(c.n_kv_layer(), 12);
        assert!(!c.is_recurrent(3) && c.is_recurrent(4) && !c.is_recurrent(47));
    }

    /// `min_rows` is the furthest end of any head's range, not the sum of sizes:
    /// the table is indexed by `offset + local`, so that is what must fit.
    #[test]
    fn ple_min_rows_is_the_furthest_head_end() {
        let p = Ple {
            layer: 1,
            ngram_size: 3,
            heads_per_ngram: 1,
            conv_kernel: 4,
            eos_token_id: 0,
            image_token_id: None,
            head_dim: 16,
            layer_multipliers: vec![1, 2, 3],
            head_offsets: vec![0, 2053],
            head_vocab_sizes: vec![2053, 2063],
        };
        assert_eq!(p.n_heads(), 2);
        assert_eq!(p.min_rows(), 4116);
    }
}
