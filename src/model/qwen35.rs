//! The `qwen35` architecture — a hybrid of GatedDeltaNet and full attention.
//!
//! Ported from `llama_model_qwen35` in llama.cpp's `src/models/qwen35.cpp` and
//! `src/models/delta-net-base.cpp`. This applies to **both** target models: the
//! 9B is `qwen35`, the 35B is `qwen35moe`, and the only differences are the MoE
//! FFN and the embedding width. Everything below is shared.
//!
//! # Which layers are which
//!
//! ```c
//! recurrent_layer_arr[i] = (i < n_main) && ((i + 1) % full_attn_interval != 0);
//! ```
//!
//! With `full_attention_interval = 4`, layer `i` is **recurrent when
//! `(i + 1) % 4 != 0`** — so 0, 1, 2 are GatedDeltaNet, 3 is full attention, and
//! so on. Three of every four layers are recurrent, which is why the KV cache
//! is small enough for a 262,144-token context to be plausible at all.
//!
//! # Layer skeleton, both kinds
//!
//! ```text
//! x_in = x
//! cur  = rms_norm(x, attn_norm)
//! cur  = gated_delta_net(cur) | attention(cur)
//! x    = cur + x_in                       // residual
//! r    = x
//! cur  = rms_norm(x, post_attention_norm)
//! x    = ffn(cur) + r                     // residual
//! ```
//!
//! Note the norm is named `post_attention_norm` in the file but plays the role
//! `ffn_norm` does in `qwen3`.
//!
//! # Dimensions, derived not guessed
//!
//! From `load_arch_tensors`, in terms of the `ssm.*` metadata keys:
//!
//! ```text
//! head_k_dim = head_v_dim = ssm.state_size      = 128
//! n_k_heads                = ssm.group_count    = 16
//! n_v_heads                = ssm.time_step_rank = 32
//! key_dim    = head_k_dim * n_k_heads           = 2048
//! value_dim  = head_v_dim * n_v_heads           = 4096   ( = ssm.inner_size)
//! conv_dim   = key_dim * 2 + value_dim          = 8192
//! ```
//!
//! Which is what the tensors say: `ssm_conv1d` is `{4, 8192}`, `attn_qkv` is
//! `{n_embd, 8192}`, `attn_gate` is `{n_embd, 4096}`, `ssm_alpha` and `ssm_beta`
//! are `{n_embd, 32}`, `ssm_a` and `ssm_dt.bias` are `{32}`, `ssm_norm` is
//! `{128}`.
//!
//! Note `n_v_heads` is **twice** `n_k_heads`, so the delta rule is grouped the
//! way GQA is — but **by modulo, not by division**: value head `h` reads
//! key/query head `h % n_k_heads`. Verified against both of llama.cpp's paths,
//! which have to agree:
//!
//! * the unfused path calls `ggml_repeat_4d(q_conv, head_k_dim, num_v_heads, ..)`,
//!   and `ggml_compute_forward_repeat_f32` *tiles* — `dst[i1*ne01 + k1]` reads
//!   `src[k1]`, so destination head `h` reads source head `h % ne01`;
//! * the fused kernel says it outright:
//!   `const int64_t iq1 = iv1 % neq1; const int64_t ik1 = iv1 % nek1;`
//!   (`ggml_compute_forward_gated_delta_net_one_chunk`).
//!
//! An earlier version of this note said `h / 2`. That is blocked grouping, it
//! is what GQA does elsewhere in this file, and it is wrong here — it agrees
//! with the truth only for `h = 0` and `h = 1`. It would have produced
//! plausible garbage rather than an error, which is exactly the failure mode
//! `HANDOFF.md` warned about when it called the key-vs-value axis the risky
//! part of this layer.
//!
//! # The GatedDeltaNet layer
//!
//! ```text
//! qkv   = attn_qkv @ cur                         // [conv_dim]
//! z     = attn_gate @ cur                        // [value_dim]
//! beta  = sigmoid(ssm_beta @ cur)                // [n_v_heads]
//! gate  = softplus(ssm_alpha @ cur + ssm_dt) * ssm_a   // [n_v_heads]
//!
//! conv_in  = concat(conv_state, qkv)             // causal depthwise conv1d,
//! conv_out = silu(ssm_conv(conv_in, ssm_conv1d)) // kernel 4, per channel
//!
//! q, k = l2_norm(split(conv_out))                // NOT rms_norm; see below
//! v    = split(conv_out)
//! o    = delta_rule(q, k, v, gate, beta, state)
//! out  = ssm_out @ (rms_norm(o, ssm_norm) * silu(z))
//! ```
//!
//! # The delta rule itself
//!
//! `build_delta_net_autoregressive`, one token, per value head `h`, with state
//! `S` of shape `[head_k_dim, head_v_dim]` — index 0 is the key axis, index 1
//! the value axis:
//!
//! ```text
//! q     *= 1 / sqrt(head_k_dim)
//! S     *= exp(gate[h])                          // scalar decay
//! pred[j] = sum_i S[i,j] * k[i]                  // current prediction, S^T k
//! d[j]    = beta[h] * (v[j] - pred[j])           // the error
//! S[i,j] += k[i] * d[j]                          // rank-1 update, k (x) d
//! out[j]  = sum_i S[i,j] * q[i]                  // S^T q
//! ```
//!
//! That is the whole mechanism: a per-head associative memory that *corrects*
//! its stored value for the current key rather than merely accumulating, with
//! an exponential forget gate. Cost per token is O(head_k_dim * head_v_dim),
//! independent of context length — which is why 30 of 40 layers need no KV
//! cache and why the state is 128x128 per head regardless of how long the
//! sequence gets.
//!
//! # Two details that would silently produce plausible wrong numbers
//!
//! 1. **`q` and `k` get `l2_norm`, not `rms_norm`.** `ggml_compute_forward_l2_norm_f32`
//!    scales by `1 / max(sqrt(sum(x^2)), eps)` — no division by `n`, and `eps`
//!    clamps the *norm*, where RMSNorm adds `eps` to the mean under the square
//!    root. Different function, same-looking output.
//! 2. **The attention layers' `attn_q` produces query *and* gate, interleaved
//!    per head**: `[q_0, gate_0, q_1, gate_1, ...]` with stride `head_dim * 2`.
//!    The attention output is multiplied by `sigmoid(gate)` before `wo`. That
//!    is why `attn_q` is `{n_embd, head_dim * 2 * n_head}` and there is no
//!    separate gate tensor on those layers.
//!
//! # mRoPE reduces to ordinary RoPE for text, and the sections are inert
//!
//! The attention layers call `ggml_rope_multi` with sections `{11, 11, 10, 0}`
//! over `rope.dimension_count = 64` of the 256-wide head. That looks like a
//! fourth thing to implement. It is not, for text-only input:
//!
//! * `llm_graph_input_pos::set_input` fills the four position components as
//!   `p_t = p_h = p_w = pos` and `p_e = 0` when the batch is tokens rather than
//!   an image.
//! * `indep_sects` in `ggml_mrope_cache_init` is `is_vision`, false here, so
//!   all four thetas start from their base and are scaled identically each
//!   step. With three of them equal, the section a dimension falls in cannot
//!   change its angle.
//! * `theta_e` is the only one that differs, and it is never selected. See
//!   below for why — the reason is not the obvious one.
//!
//! So what actually has to be implemented is **partial RoPE**: rotate the first
//! `n_rot = 64` of each 256-wide head, pass the remaining 192 through
//! unchanged. The sections can be read, asserted to sum to `n_rot / 2`, and
//! otherwise ignored — but *only* while input is text. An image path would
//! make them live, which is why the reasoning is recorded rather than the
//! conclusion alone.
//!
//! ## It is IMROPE, not MROPE, and the earlier derivation was of the wrong branch
//!
//! `llama-model.cpp` puts `qwen35` and `qwen35moe` with the Qwen3VL family on
//! `LLAMA_ROPE_TYPE_IMROPE` — *interleaved* mRoPE — not `LLAMA_ROPE_TYPE_MROPE`.
//! `ggml_mrope_cache_init` branches on that, and the two branches select thetas
//! completely differently:
//!
//! ```c
//! if (is_imrope) {                                     // ours
//!     if      (sector % 3 == 1 && sector < 3*sections[1]) theta = theta_h;
//!     else if (sector % 3 == 2 && sector < 3*sections[2]) theta = theta_w;
//!     else if (sector % 3 == 0 && sector < 3*sections[0]) theta = theta_t;
//!     else                                               theta = theta_e;
//! } else {                                             // plain MROPE
//!     if      (sector >= sections[0] && sector < sec_w) theta = theta_h;
//!     ...
//! }
//! ```
//!
//! An earlier version of this note derived `theta_e`'s unreachability from the
//! `else` branch — the one this architecture does not take. The conclusion
//! survives, by different arithmetic: with sections `{11, 11, 10, 0}`,
//! `sect_dims = 32` and `n_rot = 64`, `sector = (i0/2) % 32` covers 0..=31, and
//! every one of those satisfies one of the three modular tests. The tightest is
//! `sector % 3 == 2 && sector < 30`, whose largest qualifying value is 29. So
//! the `else` is never reached and `theta_e` stays unused.
//!
//! ## Pairing is NEOX — verified, no longer an open question
//!
//! `ggml-cpu/ops.cpp` falls `GGML_ROPE_TYPE_IMROPE` through to the same call
//! NEOX makes, `rotate_pairs<T>(n_dims, n_dims/2, cache, src, dst_data)`, so
//! dimension `i` pairs with `i + n_rot/2` and `Ops::rope_neox` is the right
//! kernel. This was flagged unverified precisely because getting it wrong would
//! look like a numerics bug rather than a structural one.

use std::cell::RefCell;

use crate::cache::{KvCache, RecurrentState};
use crate::error::{Error, Result};
use crate::gguf::GgufFile;
use crate::ops::{Attn, Delta, Experts, Ops, Weights};
use crate::profile::{Ctx, Part};
use crate::quant::{dequantize, dequantize_into};

use super::{experts, matrix, tensor, vector};

/// Everything the forward pass needs, read from metadata.
#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    /// `attention.key_length`, 256 — not `n_embd / n_head`.
    pub head_dim: usize,
    pub n_vocab: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,

    /// `rope.dimension_count`: how many of `head_dim` actually rotate.
    pub n_rot: usize,
    /// `rope.dimension_sections`, the mRoPE split.
    pub rope_sections: [i32; 4],

    /// `ssm.conv_kernel`, 4.
    pub ssm_d_conv: usize,
    /// `ssm.inner_size`, 4096. Equals `head_v_dim * n_v_heads`.
    pub ssm_d_inner: usize,
    /// `ssm.state_size`, 128. This is both `head_k_dim` and `head_v_dim`.
    pub ssm_d_state: usize,
    /// `ssm.time_step_rank`, 32. This is `n_v_heads`.
    pub ssm_dt_rank: usize,
    /// `ssm.group_count`, 16. This is `n_k_heads`.
    pub ssm_n_group: usize,

    /// `full_attention_interval`, 4.
    pub full_attention_interval: usize,
    /// `nextn_predict_layers` — MTP blocks appended past the main stack and not
    /// executed in a normal forward pass.
    pub nextn_predict_layers: usize,

    /// Present on `qwen35moe`, absent on `qwen35`. The only structural
    /// difference between the two architectures.
    pub moe: Option<Moe>,
}

/// The mixture-of-experts half of `qwen35moe`.
///
/// A separate struct rather than four `Option` fields on [`Config`] so the
/// invariant "either all of these or none" is held by the type instead of by
/// convention — an FFN cannot be half routed.
#[derive(Debug, Clone, Copy)]
pub struct Moe {
    /// `expert_count`, 256.
    pub n_expert: usize,
    /// `expert_used_count`, 8. How many of the 256 each token routes to.
    pub n_expert_used: usize,
    /// `expert_feed_forward_length`, 512. The FFN width of **one** expert, so
    /// the routed FFN is 8 x 512 wide per token rather than 256 x 512.
    pub expert_ff: usize,
    /// `expert_shared_feed_forward_length`, 512. The always-on expert, which is
    /// not routed and not counted in `n_expert_used`.
    pub shared_ff: usize,
}

impl Config {
    /// Whether this is the routed variant.
    pub fn is_moe(&self) -> bool {
        self.moe.is_some()
    }

    /// Query heads per key/value head, on the attention layers.
    pub fn gqa_group(&self) -> usize {
        self.n_head / self.n_head_kv
    }

    /// Width of the concatenated query-and-gate projection: `attn_q` emits two
    /// `head_dim` blocks per head.
    pub fn q_gate_dim(&self) -> usize {
        self.head_dim * 2 * self.n_head
    }

    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    // ---- GatedDeltaNet dimensions, named as `qwen35.cpp` names them ----

    pub fn head_k_dim(&self) -> usize {
        self.ssm_d_state
    }

    pub fn head_v_dim(&self) -> usize {
        self.ssm_d_inner / self.ssm_dt_rank
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

    /// Channels the depthwise conv runs over: q, k and v concatenated.
    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }

    /// Value heads per key head — the delta rule's GQA factor.
    pub fn delta_group(&self) -> usize {
        self.n_v_heads() / self.n_k_heads()
    }

    /// Floats of recurrent state per layer: one `head_k_dim x head_v_dim`
    /// matrix per value head. **Independent of context length.**
    pub fn ssm_state_len(&self) -> usize {
        self.head_k_dim() * self.head_v_dim() * self.n_v_heads()
    }

    /// Floats of convolution state per layer: the `kernel - 1` past inputs for
    /// every channel.
    pub fn conv_state_len(&self) -> usize {
        (self.ssm_d_conv - 1) * self.conv_dim()
    }

    /// Layers actually executed; MTP blocks sit past this.
    pub fn n_main_layer(&self) -> usize {
        self.n_layer - self.nextn_predict_layers
    }

    /// Is layer `il` a GatedDeltaNet layer?
    ///
    /// Transcribed from `load_arch_hparams`: recurrent when `(il + 1) %
    /// full_attention_interval != 0`, and never for the MTP blocks.
    pub fn is_recurrent(&self, il: usize) -> bool {
        il < self.n_main_layer() && (il + 1) % self.full_attention_interval != 0
    }

    pub fn from_gguf(f: &GgufFile) -> Result<Self> {
        let md = &f.metadata;
        let arch = md.architecture()?;
        if arch != "qwen35" && arch != "qwen35moe" {
            return Err(Error::UnsupportedArchitecture {
                arch: arch.to_string(),
                supported: "qwen35, qwen35moe",
            });
        }
        // The routed variant carries `expert_count` and has no
        // `feed_forward_length`; the dense one is the other way round. Keying
        // on the presence of the expert keys rather than on the architecture
        // string means a file that declares one and ships the other fails at
        // load with a missing key, instead of later with wrong shapes.
        let moe = match md.get_arch_u32("expert_count") {
            Ok(n_expert) => Some(Moe {
                n_expert: n_expert as usize,
                n_expert_used: md.get_arch_u32("expert_used_count")? as usize,
                expert_ff: md.get_arch_u32("expert_feed_forward_length")? as usize,
                shared_ff: md.get_arch_u32("expert_shared_feed_forward_length")? as usize,
            }),
            Err(_) => None,
        };

        let n_head = md.get_arch_u32("attention.head_count")? as usize;
        let n_head_kv = md.get_arch_u32("attention.head_count_kv")? as usize;
        let head_dim = md.get_arch_u32("attention.key_length")? as usize;
        let v_len = md.get_arch_u32("attention.value_length")? as usize;
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

        // mRoPE sections. The file stores them as i32; anything else means the
        // key does not mean what we think it means, so refuse rather than coerce.
        let key = md.arch_key("rope.dimension_sections")?;
        let sections = match md.get_array(&key)? {
            crate::gguf::Array::I32(v) => v.clone(),
            other => {
                return Err(Error::InconsistentArchitecture {
                    what: "rope.dimension_sections",
                    detail: format!("expected an i32 array, found {} entries of another type", other.len()),
                });
            }
        };
        if sections.len() != 4 {
            return Err(Error::InconsistentArchitecture {
                what: "rope.dimension_sections",
                detail: format!("expected 4 entries, found {}", sections.len()),
            });
        }
        let mut rope_sections = [0i32; 4];
        rope_sections.copy_from_slice(&sections);

        let n_vocab = super::tensor(f, "token_embd.weight")?
            .dims
            .get(1)
            .copied()
            .ok_or_else(|| Error::InconsistentArchitecture {
                what: "token_embd.weight",
                detail: "expected 2 dimensions".to_string(),
            })? as usize;

        let cfg = Self {
            n_layer: md.get_arch_u32("block_count")? as usize,
            n_embd: md.get_arch_u32("embedding_length")? as usize,
            // The routed variant has no dense FFN width. `n_ff` then describes
            // one expert, which is what every buffer sized from it needs.
            n_ff: match &moe {
                Some(m) => m.expert_ff,
                None => md.get_arch_u32("feed_forward_length")? as usize,
            },
            n_head,
            n_head_kv,
            head_dim,
            n_vocab,
            rope_theta: md.get_arch_f32("rope.freq_base")?,
            rms_eps: md.get_arch_f32("attention.layer_norm_rms_epsilon")?,
            n_rot: md.get_arch_u32("rope.dimension_count")? as usize,
            rope_sections,
            ssm_d_conv: md.get_arch_u32("ssm.conv_kernel")? as usize,
            ssm_d_inner: md.get_arch_u32("ssm.inner_size")? as usize,
            ssm_d_state: md.get_arch_u32("ssm.state_size")? as usize,
            ssm_dt_rank: md.get_arch_u32("ssm.time_step_rank")? as usize,
            ssm_n_group: md.get_arch_u32("ssm.group_count")? as usize,
            full_attention_interval: md
                .get_arch_u32("full_attention_interval")
                .unwrap_or(4) as usize,
            nextn_predict_layers: md.get_arch_u32("nextn_predict_layers").unwrap_or(0) as usize,
            moe,
        };

        if let Some(m) = &cfg.moe {
            if m.n_expert_used == 0 || m.n_expert_used > m.n_expert {
                return Err(Error::InconsistentArchitecture {
                    what: "expert_used_count",
                    detail: format!("{} of {} experts", m.n_expert_used, m.n_expert),
                });
            }
        }

        // Relationships `qwen35.cpp` assumes without checking. If a file
        // violates one, every number downstream is wrong in a way that looks
        // like a numerics bug, so fail here with the names instead.
        if cfg.ssm_dt_rank == 0 || cfg.ssm_d_inner % cfg.ssm_dt_rank != 0 {
            return Err(Error::InconsistentArchitecture {
                what: "ssm.inner_size / ssm.time_step_rank",
                detail: format!(
                    "{} does not divide into {} value heads",
                    cfg.ssm_d_inner, cfg.ssm_dt_rank
                ),
            });
        }
        if cfg.n_k_heads() == 0 || cfg.n_v_heads() % cfg.n_k_heads() != 0 {
            return Err(Error::InconsistentArchitecture {
                what: "delta rule grouping",
                detail: format!(
                    "{} value heads do not divide into {} key heads",
                    cfg.n_v_heads(),
                    cfg.n_k_heads()
                ),
            });
        }
        if cfg.head_v_dim() != cfg.head_k_dim() {
            return Err(Error::InconsistentArchitecture {
                what: "delta rule head dimensions",
                detail: format!(
                    "head_v_dim {} != head_k_dim {}; the recurrence assumes a square state",
                    cfg.head_v_dim(),
                    cfg.head_k_dim()
                ),
            });
        }
        if cfg.full_attention_interval == 0 {
            return Err(Error::InconsistentArchitecture {
                what: "full_attention_interval",
                detail: "must be non-zero".to_string(),
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

/// One block's weights. Two shapes share a struct because the trunk tensors —
/// the two norms and the dense FFN — are identical either way, and only the
/// mixer differs.
enum Mixer<'a> {
    /// A full-attention block. `wq` emits query *and* gate interleaved per
    /// head, which is why there is no separate gate tensor here.
    Attn {
        wq: Weights<'a>,
        wk: Weights<'a>,
        wv: Weights<'a>,
        wo: Weights<'a>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    },
    /// A GatedDeltaNet block.
    Delta {
        wqkv: Weights<'a>,
        wgate: Weights<'a>,
        conv1d: Vec<f32>,
        ssm_beta: Weights<'a>,
        ssm_alpha: Weights<'a>,
        ssm_a: Vec<f32>,
        dt_bias: Vec<f32>,
        ssm_norm: Vec<f32>,
        ssm_out: Weights<'a>,
    },
}

struct Layer<'a> {
    attn_norm: Vec<f32>,
    /// Named `post_attention_norm` in the file; plays the role `ffn_norm` does
    /// in `qwen3`.
    ffn_norm: Vec<f32>,
    mixer: Mixer<'a>,
    ffn: Ffn<'a>,
}

/// The feed-forward half of a block: dense on `qwen35`, routed on `qwen35moe`.
///
/// An enum for the same reason [`Mixer`] is one — the two are alternatives at
/// the same point in the block, and the rest of the layer does not vary with
/// which it is. This is the seam `CLAUDE.md` asked to keep the FFN behind so
/// the MoE variant would be a delta rather than a rewrite; the delta turned out
/// to be this type and the arm that reads it.
enum Ffn<'a> {
    Dense {
        gate: Weights<'a>,
        up: Weights<'a>,
        down: Weights<'a>,
    },
    /// **8 of 256 experts per token, plus one that always runs.**
    ///
    /// Storage and traffic diverge sharply here, which is the whole reason this
    /// model is interesting: the routed experts are 408 MiB per block, 94% of
    /// the block, but a token reads 8/256 of them — 12.75 MiB. So placement of
    /// these tensors governs a third of a token's bytes while dominating what
    /// has to be resident.
    Moe {
        /// The router: `{n_embd, n_expert}`, **F32**, so its matmul is exact
        /// and the expert choice can be checked against llama.cpp directly.
        gate_inp: Weights<'a>,
        gate: Experts<'a>,
        up: Experts<'a>,
        down: Experts<'a>,
        /// The always-on expert, stored at Q8_0 where the routed ones are
        /// IQ4_XS — it is read every token, so it is worth more bits.
        shared_gate: Weights<'a>,
        shared_up: Weights<'a>,
        shared_down: Weights<'a>,
        /// `ffn_gate_inp_shexp`: a length-`n_embd` vector, not a matrix. It
        /// gates the shared expert's contribution with a sigmoid.
        shared_gate_inp: Vec<f32>,
    },
}

/// Every intermediate one token needs, allocated once per pass.
///
/// **Stable addresses are the point, not the saved `malloc`.** A device backend
/// keys its mirrors on the host address of a buffer, and a recorded CUDA graph
/// holds device pointers in its nodes. Allocating inside the layer loop gives
/// each of the 32 layers fresh addresses every token, which churns the mirror
/// map and leaves a graph's nodes pointing at buffers that no longer exist.
/// `qwen3::forward` hoists its buffers above the loop for exactly this reason.
///
/// Sized for the widest layer of each kind, so an attention block and a
/// recurrent block share the same allocations.
#[derive(Default)]
struct Scratch {
    /// The residual stream, here with the rest so it too keeps one address.
    x: Vec<f32>,
    /// Router logits then probabilities, `n_expert` wide. **Not batched**: the
    /// MoE FFN is a per-token loop, because each token routes to its own eight
    /// experts and there is nothing to share across a batch until the tokens
    /// are grouped by expert, which is what llama.cpp's `mul_mat_id` does.
    router: Vec<f32>,
    /// One expert's intermediates, `expert_ff` wide, reused across all eight.
    e_gate: Vec<f32>,
    e_up: Vec<f32>,
    /// One expert's output and the running weighted sum, `n_embd` wide.
    e_out: Vec<f32>,
    moe_acc: Vec<f32>,
    /// The last row lifted out of `x`, its norm, and the logits. Not batched —
    /// only the final position produces output — but owned for the same reason.
    last: Vec<f32>,
    final_norm: Vec<f32>,
    logits: Vec<f32>,
    normed: Vec<f32>,
    mixed: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    ffn_out: Vec<f32>,
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
}

impl Scratch {
    /// Size every buffer for a batch of `n` tokens, token-major, **without
    /// moving it**.
    ///
    /// Every buffer is `n` copies of the single-token shape, which is what lets
    /// [`Qwen35::attention`] and [`Qwen35::gated_delta`] batch with almost no
    /// change: they were already written as whole-buffer calls through the
    /// seam, so a longer buffer is simply a bigger call.
    ///
    /// **Resized rather than reallocated, and owned by the model rather than
    /// built per pass.** `Vec::resize` moves nothing while capacity holds, so
    /// capacity settles at the largest batch ever seen and the addresses then
    /// never change again.
    ///
    /// That matters because a device backend keys its activation mirrors on
    /// host addresses and never frees them — `Mirror::invalidate` records why.
    /// Allocating these per pass left a whole new set of device buffers behind
    /// every time: measured at **64 mirrors holding 84 MiB** after one prompt
    /// on the 0.6B, climbing for as long as a session ran. It cost throughput
    /// too, because a mirror at a new address is not device-current and every
    /// buffer was re-uploaded — 261 -> 656 tok/s of prefill on that same run.
    fn fit(&mut self, c: &Config, n: usize) {
        let z = |b: &mut Vec<f32>, k: usize| b.resize(n * k, 0.0);
        z(&mut self.x, c.n_embd);
        z(&mut self.normed, c.n_embd);
        z(&mut self.mixed, c.n_embd);
        z(&mut self.gate, c.n_ff);
        z(&mut self.up, c.n_ff);
        z(&mut self.ffn_out, c.n_embd);
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

        // The MoE buffers are single-token: routing differs per token, so that
        // FFN runs as a loop and shares one set of scratch across the batch.
        if let Some(m) = &c.moe {
            self.router.resize(m.n_expert, 0.0);
            self.e_gate.resize(m.expert_ff.max(m.shared_ff), 0.0);
            self.e_up.resize(m.expert_ff.max(m.shared_ff), 0.0);
            self.e_out.resize(c.n_embd, 0.0);
            self.moe_acc.resize(c.n_embd, 0.0);
        }

        // One row of output per pass, whatever the batch.
        self.last.resize(c.n_embd, 0.0);
        self.final_norm.resize(c.n_embd, 0.0);
        self.logits.resize(c.n_vocab, 0.0);
    }
}

pub struct Qwen35<'a> {
    pub cfg: Config,
    tok_embd: Weights<'a>,
    output_norm: Vec<f32>,
    output: Weights<'a>,
    layers: Vec<Layer<'a>>,
    /// Absolute layer index to KV slab. Only attention layers have one.
    ///
    /// **Worth the second numbering here, unlike the recurrent state.** On the
    /// 35B at its full 262,144 context, a slab per absolute layer would be
    /// ~21.5 GiB against ~5.4 for the 10 layers that actually attend -- the
    /// difference between fitting on this card and not. `RecurrentState` makes
    /// the opposite call for the opposite reason: there the waste is tens of
    /// megabytes, and one numbering is worth more than the memory.
    kv_slot: Vec<usize>,
    n_kv_layer: usize,
    /// Activation buffers, owned so their **addresses never change**. See
    /// [`Scratch`].
    scratch: RefCell<Scratch>,
}

impl<'a> Qwen35<'a> {
    pub fn load(f: &'a GgufFile) -> Result<Self> {
        let cfg = Config::from_gguf(f)?;
        let (n_embd, n_ff) = (cfg.n_embd, cfg.n_ff);

        let tok_embd = matrix(f, "token_embd.weight", n_embd, cfg.n_vocab)?;
        // Same fallback llama.cpp uses (TENSOR_DUPLICATED): tie to the
        // embedding when the file carries no separate head. The 9B has one; the
        // 35B has one too, at Q6_K.
        let output = match f.tensor("output.weight") {
            Some(_) => matrix(f, "output.weight", n_embd, cfg.n_vocab)?,
            None => tok_embd,
        };

        // MTP blocks are loaded by llama.cpp but not executed in a normal pass,
        // and they carry tensors this engine has no use for. Skipping them here
        // rather than loading and ignoring them keeps a missing-tensor error
        // meaningful.
        let mut layers = Vec::with_capacity(cfg.n_main_layer());
        for i in 0..cfg.n_main_layer() {
            let p = |name: &str| format!("blk.{i}.{name}");
            let mixer = if cfg.is_recurrent(i) {
                Mixer::Delta {
                    wqkv: matrix(f, &p("attn_qkv.weight"), n_embd, cfg.conv_dim())?,
                    wgate: matrix(f, &p("attn_gate.weight"), n_embd, cfg.value_dim())?,
                    // Stored {kernel, conv_dim} in ggml order, which is
                    // channel-major with the taps contiguous — the layout
                    // `Ops::ssm_conv` reads.
                    conv1d: conv_weights(f, &p("ssm_conv1d.weight"), cfg.ssm_d_conv, cfg.conv_dim())?,
                    ssm_beta: matrix(f, &p("ssm_beta.weight"), n_embd, cfg.n_v_heads())?,
                    ssm_alpha: matrix(f, &p("ssm_alpha.weight"), n_embd, cfg.n_v_heads())?,
                    ssm_a: vector(f, &p("ssm_a"), cfg.n_v_heads())?,
                    dt_bias: vector(f, &p("ssm_dt.bias"), cfg.n_v_heads())?,
                    ssm_norm: vector(f, &p("ssm_norm.weight"), cfg.head_v_dim())?,
                    ssm_out: matrix(f, &p("ssm_out.weight"), cfg.value_dim(), n_embd)?,
                }
            } else {
                Mixer::Attn {
                    wq: matrix(f, &p("attn_q.weight"), n_embd, cfg.q_gate_dim())?,
                    wk: matrix(f, &p("attn_k.weight"), n_embd, cfg.kv_dim())?,
                    wv: matrix(f, &p("attn_v.weight"), n_embd, cfg.kv_dim())?,
                    wo: matrix(f, &p("attn_output.weight"), cfg.head_dim * cfg.n_head, n_embd)?,
                    q_norm: vector(f, &p("attn_q_norm.weight"), cfg.head_dim)?,
                    k_norm: vector(f, &p("attn_k_norm.weight"), cfg.head_dim)?,
                }
            };
            let ffn = match &cfg.moe {
                None => Ffn::Dense {
                    gate: matrix(f, &p("ffn_gate.weight"), n_embd, n_ff)?,
                    up: matrix(f, &p("ffn_up.weight"), n_embd, n_ff)?,
                    down: matrix(f, &p("ffn_down.weight"), n_ff, n_embd)?,
                },
                Some(m) => Ffn::Moe {
                    gate_inp: matrix(f, &p("ffn_gate_inp.weight"), n_embd, m.n_expert)?,
                    // `{n_in, n_out, n_expert}`. Note `down` is the transpose of
                    // the other two, which is why these go through `experts()`
                    // rather than being reshaped by hand.
                    gate: experts(f, &p("ffn_gate_exps.weight"), n_embd, m.expert_ff, m.n_expert)?,
                    up: experts(f, &p("ffn_up_exps.weight"), n_embd, m.expert_ff, m.n_expert)?,
                    down: experts(f, &p("ffn_down_exps.weight"), m.expert_ff, n_embd, m.n_expert)?,
                    shared_gate: matrix(f, &p("ffn_gate_shexp.weight"), n_embd, m.shared_ff)?,
                    shared_up: matrix(f, &p("ffn_up_shexp.weight"), n_embd, m.shared_ff)?,
                    shared_down: matrix(f, &p("ffn_down_shexp.weight"), m.shared_ff, n_embd)?,
                    shared_gate_inp: vector(f, &p("ffn_gate_inp_shexp.weight"), n_embd)?,
                },
            };
            layers.push(Layer {
                attn_norm: vector(f, &p("attn_norm.weight"), n_embd)?,
                ffn_norm: vector(f, &p("post_attention_norm.weight"), n_embd)?,
                mixer,
                ffn,
            });
        }

        // Slab per attention layer, assigned in order, so a recurrent layer
        // costs nothing. Recurrent entries hold the next free slot and are
        // never read -- `attention` is the only caller and it runs only on
        // attention layers.
        let mut kv_slot = Vec::with_capacity(cfg.n_main_layer());
        let mut next = 0;
        for il in 0..cfg.n_main_layer() {
            kv_slot.push(next);
            if !cfg.is_recurrent(il) {
                next += 1;
            }
        }

        Ok(Self {
            cfg,
            tok_embd,
            output_norm: vector(f, "output_norm.weight", n_embd)?,
            output,
            layers,
            kv_slot,
            n_kv_layer: next,
            scratch: RefCell::new(Scratch::default()),
        })
    }

    /// KV slabs the cache must hold: one per attention layer, not per layer.
    pub fn n_kv_layer(&self) -> usize {
        self.n_kv_layer
    }

    fn embed(&self, id: u32, out: &mut [f32]) -> Result<()> {
        if id as usize >= self.cfg.n_vocab {
            return Err(Error::TokenOutOfRange {
                id,
                vocab_size: self.cfg.n_vocab,
            });
        }
        dequantize_into(self.tok_embd.row(id as usize), self.tok_embd.ty, out)
    }

    /// Bytes of quantized weight one forward pass reads.
    ///
    /// Derived from shapes rather than counted, per rule 3 in
    /// [`crate::profile`]. Recurrent state is excluded because it is not a
    /// weight; the KV cache is accounted separately.
    pub fn weight_bytes_per_pass(&self) -> u64 {
        let w = |m: &Weights<'_>| m.ty.n_bytes(m.n_in as u64) * m.n_out as u64;
        // 8 of 256 on the routed variant; irrelevant on the dense one.
        let n_used = self.cfg.moe.map_or(0, |m| m.n_expert_used);
        let per_layer: u64 = self
            .layers
            .iter()
            .map(|l| {
                let mixer = match &l.mixer {
                    Mixer::Attn { wq, wk, wv, wo, .. } => w(wq) + w(wk) + w(wv) + w(wo),
                    Mixer::Delta {
                        wqkv,
                        wgate,
                        ssm_beta,
                        ssm_alpha,
                        ssm_out,
                        ..
                    } => w(wqkv) + w(wgate) + w(ssm_beta) + w(ssm_alpha) + w(ssm_out),
                };
                mixer + ffn_bytes(&l.ffn, n_used)
            })
            .sum();
        per_layer + w(&self.output)
    }

    /// Run `tokens` from absolute position `start_pos` against the recurrent
    /// state and the KV cache, and return logits for the last one.
    ///
    /// **One function serves both phases**, as in [`super::Qwen3::forward`]:
    /// prefill is the prompt at `start_pos = 0`, decode is one token at
    /// `start_pos = kv.len()`.
    ///
    /// The delta rule and the causal convolution are sequential scans — token
    /// `t`'s state update feeds `t+1` — so they cannot be parallelized over the
    /// batch. That does **not** make them a reason to run the whole layer one
    /// token at a time, which is what this used to do. Everything carrying real
    /// bytes here is a matmul: the fused `attn_qkv`, `attn_gate`, `ssm_out`,
    /// the attention projections and the whole FFN. Those batch, and a batch
    /// reads each weight once instead of once per token. The two scans iterate
    /// *behind the seam*, so llama.cpp's chunked algorithm can arrive later as a
    /// backend change with no edit here.
    pub fn forward<O: Ops>(
        &self,
        ops: &O,
        tokens: &[u32],
        start_pos: usize,
        kv: &mut KvCache,
        rs: &mut RecurrentState,
        ctx: &mut Ctx<'_>,
    ) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let n = tokens.len();
        if n == 0 {
            return Err(Error::InconsistentArchitecture {
                what: "forward",
                detail: "no tokens supplied".to_string(),
            });
        }
        rs.check(c.n_main_layer(), c.conv_state_len(), c.ssm_state_len())?;
        if kv.kv_dim() != c.kv_dim() {
            return Err(Error::InconsistentArchitecture {
                what: "kv cache",
                detail: format!(
                    "cache holds {} lanes per position, model needs {}",
                    kv.kv_dim(),
                    c.kv_dim()
                ),
            });
        }
        // Checked up front so neither cache is left half-written.
        if start_pos + n > kv.n_ctx() {
            return Err(Error::ContextOverflow {
                pos: start_pos + n - 1,
                n_ctx: kv.n_ctx(),
            });
        }

        let step = ctx.prof.begin_step();
        ops.begin_pass(n);

        let nd = c.n_embd;
        // Owned by the model and resized in place, so every buffer keeps the
        // address it had last pass. See `Scratch`.
        let s = &mut *self.scratch.borrow_mut();
        s.fit(c, n);
        for (t, &token) in tokens.iter().enumerate() {
            self.embed(token, &mut s.x[t * nd..(t + 1) * nd])?;
        }
        ops.host_wrote(&s.x);
        ctx.trace("inp_embd", 0, &s.x);

        for il in 0..c.n_main_layer() {
            let layer = &self.layers[il];
            let t_mix = ctx.prof.layer_begin();

            ops.rms_norm(&s.x, &layer.attn_norm, c.rms_eps, &mut s.normed);
            ctx.trace("attn_norm", il, &s.normed);

            match &layer.mixer {
                Mixer::Attn { .. } => self.attention(ops, layer, il, start_pos, n, kv, s, ctx)?,
                Mixer::Delta { .. } => self.gated_delta(ops, layer, il, rs, s, ctx)?,
            }
            ops.add_assign(&mut s.x, &s.mixed);
            ctx.trace("attn_residual", il, &s.x);
            ctx.prof.layer_end(t_mix, step, il, Part::Attn);

            let t_ffn = ctx.prof.layer_begin();
            ops.rms_norm(&s.x, &layer.ffn_norm, c.rms_eps, &mut s.normed);
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    ops.matmul(gate, &s.normed, &mut s.gate);
                    ops.matmul(up, &s.normed, &mut s.up);
                    ops.silu_mul(&mut s.gate, &s.up);
                    ops.matmul(down, &s.gate, &mut s.ffn_out);
                }
                // Deliberately not yet implemented, and refusing rather than
                // approximating. Two things are missing and they are different
                // kinds of missing:
                //
                //  - the k-quant dot products. `Ops::matmul` covers F32, F16
                //    and Q8_0; every routed expert here is IQ4_XS. ggml's
                //    `vec_dot_type` for IQ4_XS, Q5_K and Q6_K is **Q8_K**, so
                //    matching it means quantizing the activation to Q8_K and
                //    doing an integer dot, exactly as the Q8_0 path does.
                //    Dequantizing to f32 instead would run, and would diverge
                //    from `llama-eval-callback` systematically -- which is the
                //    comparison that has found every architecture bug here.
                //
                //  - the routing rule itself: whether the router is softmaxed
                //    or sigmoided, whether the top-k weights are renormalized,
                //    and how the shared expert's sigmoid gate composes. Those
                //    are constants to be read out of the llama.cpp source, not
                //    guessed, per `CLAUDE.md`.
                Ffn::Moe {
                    gate_inp,
                    gate,
                    up,
                    down,
                    shared_gate,
                    shared_up,
                    shared_down,
                    shared_gate_inp,
                } => {
                    let m = match &c.moe {
                        Some(m) => *m,
                        None => {
                            return Err(Error::InconsistentArchitecture {
                                what: "moe config",
                                detail: "a routed FFN without expert counts".to_string(),
                            });
                        }
                    };
                    for t in 0..n {
                        let at = t * nd;
                        moe_token(
                            ops,
                            &m,
                            MoeWeights {
                                gate_inp,
                                gate,
                                up,
                                down,
                                shared_gate,
                                shared_up,
                                shared_down,
                                shared_gate_inp,
                            },
                            at,
                            s,
                        );
                    }
                    ctx.trace("ffn_moe_out", il, &s.ffn_out);
                }
            }
            ops.add_assign(&mut s.x, &s.ffn_out);
            ctx.trace("post_ffn", il, &s.x);
            ctx.prof.layer_end(t_ffn, step, il, Part::Ffn);
        }

        // Publish this token's position. Without it `KvCache::len` never moves,
        // so `Engine::run` starts every decode step at 0: each token overwrites
        // slot 0, attends only to itself, and is RoPE'd at position 0.
        //
        // That failure is nearly invisible from a single pass -- `trace` makes
        // one `forward` call and gets its positions right internally -- and it
        // only shows up *across* calls. The 24 GatedDeltaNet layers keep
        // advancing correctly either way, because their state is sequential and
        // does not care about `pos`, so the model still emits plausible text
        // for a few tokens before collapsing. That is what made it look like
        // numerical drift.
        kv.commit(start_pos + n);

        // Only the final position's logits are needed, and the last row is
        // taken through the seam rather than by sub-slicing `x`: a device
        // backend keys its mirrors on host addresses, so `&x[(n-1)*nd..]` would
        // be an unseen address uploaded from a host copy the device never
        // wrote. See the same note in `qwen3::forward`.
        ops.gather_chunks(&s.x, nd, nd, (n - 1) * nd, &mut s.last);
        ops.rms_norm(&s.last, &self.output_norm, c.rms_eps, &mut s.final_norm);
        ctx.trace("result_norm", 0, &s.final_norm);

        ops.matmul(&self.output, &s.final_norm, &mut s.logits);
        ops.end_pass();
        ops.host_needs(&mut s.logits);
        ctx.trace("result_output", 0, &s.logits);
        // Copied out rather than moved: the buffer keeps its address, or the
        // next pass allocates a new one and strands a device mirror.
        Ok(s.logits.clone())
    }

    /// A full-attention block.
    ///
    /// Two things here are not in `qwen3`. `attn_q` emits query *and* gate
    /// interleaved per head with stride `head_dim * 2`, and the attention
    /// output is multiplied by `sigmoid(gate)` before the output projection.
    /// And RoPE is partial: only the first `n_rot` of each `head_dim` rotate,
    /// the rest pass through. See the module header for why mRoPE reduces to
    /// exactly that for text.
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
        let Mixer::Attn {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
        } = &layer.mixer
        else {
            return Err(Error::InconsistentArchitecture {
                what: "layer kind",
                detail: format!("layer {il} is recurrent but was routed to attention"),
            });
        };

        let (hd, kd) = (c.head_dim, c.kv_dim());

        ops.matmul(wq, &s.normed, &mut s.qg);
        ctx.trace("Qcur_full", il, &s.qg);

        // De-interleave. Per head the projection emits [q | gate], so head h's
        // query starts at h * 2 * head_dim and its gate half a head later.
        // Through the seam rather than as a host loop: `qg` is a matmul result,
        // so on a device backend reading it here would drag the activation home
        // in the middle of a layer and break graph capture.
        ops.gather_chunks(&s.qg, hd, 2 * hd, 0, &mut s.q);
        ops.gather_chunks(&s.qg, hd, 2 * hd, hd, &mut s.g);

        ops.matmul(wk, &s.normed, &mut s.k);
        ops.matmul(wv, &s.normed, &mut s.v);

        // QK-norm strictly before RoPE, as in qwen3.
        ops.rms_norm_heads(&mut s.q, q_norm, hd, c.rms_eps);
        ctx.trace("Qcur_normed", il, &s.q);
        ops.rms_norm_heads(&mut s.k, k_norm, hd, c.rms_eps);
        ctx.trace("Kcur_normed", il, &s.k);

        // From row 0's absolute position; the op advances one per row.
        ops.rope_neox(&mut s.q, start_pos, hd, c.n_rot, c.n_head, c.rope_theta);
        ctx.trace("Qcur", il, &s.q);
        ops.rope_neox(&mut s.k, start_pos, hd, c.n_rot, c.n_head_kv, c.rope_theta);
        ctx.trace("Kcur", il, &s.k);

        // The whole batch is published before attending, because row `t`
        // attends to rows this same call wrote. One contiguous run: the cache
        // is position-major and the batch occupies consecutive positions.
        let slot = self.kv_slot[il];
        ops.kv_write(kv.k_layer_mut(slot), start_pos * kd, &s.k);
        ops.kv_write(kv.v_layer_mut(slot), start_pos * kd, &s.v);

        let a = Attn {
            q: &s.q,
            k: kv.k_layer(slot),
            v: kv.v_layer(slot),
            kv_dim: kd,
            // The *last* row's window; earlier rows are masked to fewer by
            // `Attn::n_pos_of`, which is what keeps a batched prefill causal.
            n_pos: start_pos + n,
            head_dim: hd,
            n_head: c.n_head,
            n_head_kv: c.n_head_kv,
            scale: 1.0 / (hd as f32).sqrt(),
        };
        ops.attend(&a, &mut s.attn);
        ctx.trace("attn_pregate", il, &s.attn);

        // sigmoid(gate) * attention, then the output projection.
        ops.sigmoid_mul(&mut s.attn, &s.g);
        ctx.trace("attn_gated", il, &s.attn);

        ops.matmul(wo, &s.attn, &mut s.mixed);
        ctx.trace("attn_output", il, &s.mixed);
        Ok(())
    }

    /// A GatedDeltaNet block.
    ///
    /// The order is load-bearing and is the module header's, verified against
    /// `build_layer_attn_linear`: project, convolve over the stored window,
    /// l2-normalize q and k but not v, run the delta rule, then normalize the
    /// output by `ssm_norm` and gate it with `silu(z)` before projecting out.
    #[allow(clippy::too_many_arguments)]
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
        let Mixer::Delta {
            wqkv,
            wgate,
            conv1d,
            ssm_beta,
            ssm_alpha,
            ssm_a,
            dt_bias,
            ssm_norm,
            ssm_out,
        } = &layer.mixer
        else {
            return Err(Error::InconsistentArchitecture {
                what: "layer kind",
                detail: format!("layer {il} is attention but was routed to the delta rule"),
            });
        };

        let (kdim, vdim, cdim) = (c.key_dim(), c.value_dim(), c.conv_dim());

        ops.matmul(wqkv, &s.normed, &mut s.qkv);
        ctx.trace("linear_attn_qkv_mixed", il, &s.qkv);

        ops.matmul(wgate, &s.normed, &mut s.z);
        ctx.trace("z", il, &s.z);

        ops.matmul(ssm_alpha, &s.normed, &mut s.alpha);
        ops.matmul(ssm_beta, &s.normed, &mut s.beta);

        // The seam takes the state slab and advances it, so nothing about the
        // conv window is assembled here. That is what lets a device backend
        // keep this layer's history in its own memory -- and what lets a
        // CPU-resident layer and a GPU-resident one coexist without either
        // slab migrating.
        ops.ssm_conv(rs.conv_mut(il), &s.qkv, conv1d, c.ssm_d_conv, &mut s.conv);
        ctx.trace("conv_output_silu", il, &s.conv);

        // The convolved output is [q | k | v] concatenated along the channel
        // axis, in that order — the same order `attn_qkv` emits and the same
        // one the conv preserved, since it is depthwise.
        //
        // **Split through the seam, not with `split_at_mut`.** A device backend
        // keys its mirrors on the host address of a slice, so a sub-slice
        // taken here would look like a *different* buffer starting mid-way
        // through `conv` -- and one with no mirror, so it would be uploaded
        // from a host copy that the convolution never wrote. That produced
        // fluent-looking garbage on the first GPU run. Copying into owned
        // buffers costs three small kernels and keeps every slice something the
        // seam has seen.
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
        ctx.trace("dnet_out", il, &s.core);

        // build_norm_gated: rms_norm(core, ssm_norm) * silu(z). Per head over
        // head_v_dim, with ssm_norm shared across heads.
        ops.rms_norm_heads(&mut s.core, ssm_norm, c.head_v_dim(), c.rms_eps);
        ops.silu_mul(&mut s.z, &s.core);
        ctx.trace("final_output", il, &s.z);

        ops.matmul(ssm_out, &s.z, &mut s.mixed);
        ctx.trace("linear_attn_out", il, &s.mixed);
        Ok(())
    }
}

/// Bytes one block's FFN reads for a single token.
///
/// **Storage and traffic differ by 32x for the routed variant**, which is the
/// fact the whole project turns on. A block holds 408 MiB of routed experts —
/// 94% of the block — but a token touches `n_expert_used` of `n_expert` of
/// them, 8 of 256, so it reads 12.75 MiB. The shared expert and the router are
/// read in full every token.
///
/// Counting storage here instead would overstate a token's traffic by ~15 GiB
/// and make every bandwidth figure derived from it meaningless.
fn ffn_bytes(ffn: &Ffn<'_>, n_used: usize) -> u64 {
    let w = |m: &Weights<'_>| m.data.len() as u64;
    match ffn {
        Ffn::Dense { gate, up, down } => w(gate) + w(up) + w(down),
        Ffn::Moe {
            gate_inp,
            gate,
            up,
            down,
            shared_gate,
            shared_up,
            shared_down,
            ..
        } => {
            let used = |e: &Experts<'_>| (e.stride() * n_used) as u64;
            w(gate_inp)
                + used(gate)
                + used(up)
                + used(down)
                + w(shared_gate)
                + w(shared_up)
                + w(shared_down)
        }
    }
}

/// `ssm_conv1d.weight`, dequantized and shape-checked as `{kernel, channels}`.
///
/// A 1-D `vector` helper will not do: this is 2-D in the file, and it is small
/// enough (128 KB) that keeping it unpacked costs nothing while letting
/// `Ops::ssm_conv` index it as plain floats.
fn conv_weights(f: &GgufFile, name: &str, kernel: usize, channels: usize) -> Result<Vec<f32>> {
    let info = tensor(f, name)?;
    if info.dims != vec![kernel as u64, channels as u64] {
        return Err(Error::TensorShapeMismatch {
            name: name.to_string(),
            expected: vec![kernel as u64, channels as u64],
            got: info.dims.clone(),
        });
    }
    dequantize(f.tensor_bytes(info), info.ty, kernel * channels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 9B's numbers, so the derived dimensions can be checked without the
    /// file. These come from `inferred inspect` on the real model.
    fn qwen35_9b() -> Config {
        Config {
            n_layer: 32,
            n_embd: 4096,
            n_ff: 12288,
            n_head: 16,
            n_head_kv: 4,
            head_dim: 256,
            n_vocab: 248_320,
            rope_theta: 1e7,
            rms_eps: 1e-6,
            n_rot: 64,
            rope_sections: [11, 11, 10, 0],
            ssm_d_conv: 4,
            ssm_d_inner: 4096,
            ssm_d_state: 128,
            ssm_dt_rank: 32,
            ssm_n_group: 16,
            full_attention_interval: 4,
            nextn_predict_layers: 0,
            moe: None,
        }
    }

    /// Every one of these is checkable against a tensor shape in the file, and
    /// each was wrong in at least one plausible reading of the metadata.
    #[test]
    fn derived_dimensions_match_the_tensor_shapes() {
        let c = qwen35_9b();
        assert_eq!(c.head_k_dim(), 128);
        assert_eq!(c.head_v_dim(), 128, "ssm_norm is {{128}}");
        assert_eq!(c.n_k_heads(), 16);
        assert_eq!(c.n_v_heads(), 32, "ssm_a and ssm_dt.bias are {{32}}");
        assert_eq!(c.key_dim(), 2048);
        assert_eq!(c.value_dim(), 4096, "attn_gate is {{n_embd, 4096}}");
        assert_eq!(c.conv_dim(), 8192, "ssm_conv1d is {{4, 8192}}");
        assert_eq!(c.q_gate_dim(), 8192, "attn_q is {{n_embd, 8192}}");
        assert_eq!(c.delta_group(), 2);
    }

    /// Three of every four layers are recurrent, and the attention layers are
    /// the ones whose index is 3 mod 4 — which is what the file shows.
    #[test]
    fn layer_types_follow_the_full_attention_interval() {
        let c = qwen35_9b();
        let attention: Vec<usize> = (0..c.n_layer).filter(|&i| !c.is_recurrent(i)).collect();
        assert_eq!(attention, vec![3, 7, 11, 15, 19, 23, 27, 31]);
        assert_eq!((0..c.n_layer).filter(|&i| c.is_recurrent(i)).count(), 24);
    }

    /// The 35B has the same recurrent hyperparameters and a different width,
    /// so the same code has to serve both.
    #[test]
    fn the_35b_shares_every_recurrent_dimension() {
        let mut c = qwen35_9b();
        c.n_layer = 40;
        c.n_embd = 2048;
        c.n_head_kv = 2;
        assert_eq!(c.conv_dim(), 8192);
        assert_eq!(c.ssm_state_len(), 128 * 128 * 32);
        let attention: Vec<usize> = (0..40).filter(|&i| !c.is_recurrent(i)).collect();
        assert_eq!(attention, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);
    }

    /// State is constant in context length. That is the whole reason a 262,144
    /// token context is affordable, so it gets an assertion rather than a
    /// comment.
    #[test]
    fn recurrent_state_does_not_grow_with_context() {
        let c = qwen35_9b();
        assert_eq!(c.ssm_state_len(), 524_288); // 2 MiB of f32 per layer
        assert_eq!(c.conv_state_len(), 3 * 8192);
    }
}

/// The weights one routed FFN needs, grouped so `moe_token` takes four
/// arguments instead of eleven.
struct MoeWeights<'a, 'b> {
    gate_inp: &'b Weights<'a>,
    gate: &'b Experts<'a>,
    up: &'b Experts<'a>,
    down: &'b Experts<'a>,
    shared_gate: &'b Weights<'a>,
    shared_up: &'b Weights<'a>,
    shared_down: &'b Weights<'a>,
    shared_gate_inp: &'b [f32],
}

/// One token through the mixture of experts, writing into `s.ffn_out` at `at`.
///
/// Transcribed from `llm_graph_context::build_moe_ffn` and
/// `llama_model_qwen35moe::graph::build_layer_ffn`. The rule, with the two
/// arguments qwen35moe passes:
///
/// ```text
/// probs = softmax(ffn_gate_inp . x)             over all 256, before top-k
/// sel   = the 8 largest                          LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX
/// w     = probs[sel] / max(sum, 6.103515625e-5)  norm_w = true
/// out   = sum_i w_i * down_i(silu(gate_i . x) * up_i . x)
///       + sigmoid(ffn_gate_inp_shexp . x) * down_sh(silu(gate_sh . x) * up_sh . x)
/// ```
///
/// Three details that are easy to get wrong and produce plausible text anyway:
///
/// - **The softmax is over all 256 experts, before selection**, so the weights
///   are the full distribution's probabilities and not a softmax of the top 8.
///   `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT` is the variant that does the
///   latter, and qwen35moe does not use it.
/// - **`expert_weights_scale` is not applied.** It defaults to 0.0, the file
///   carries no such key, and llama.cpp skips the scale when it is 0.0 or 1.0 —
///   so a naive reading that multiplies by it would zero the whole FFN.
/// - **The clamp is `6.103515625e-5`**, f16's smallest normal, guarding the
///   division rather than the weights.
///
/// The experts are summed in top-k order, serially, as the reference does.
fn moe_token<O: Ops>(ops: &O, m: &Moe, w: MoeWeights<'_, '_>, at: usize, s: &mut Scratch) {
    let nd = s.e_out.len();
    let x = &s.normed[at..at + nd];

    // The router is F32, so this matmul is exact and the expert choice can be
    // compared against llama.cpp directly.
    ops.matmul(w.gate_inp, x, &mut s.router);
    ops.softmax(&mut s.router);
    // Selection is a host decision, so the probabilities have to come home. On
    // a device backend that is one crossing per layer per token and is the
    // first thing to remove when the MoE moves to the GPU.
    ops.host_needs(&mut s.router);

    // Top-k by probability, ties broken by the lower expert id — `argsort` in
    // the reference is descending and stable, and exact ties in a softmax over
    // 256 logits are vanishingly rare either way.
    let mut pick: Vec<(usize, f32)> = Vec::with_capacity(m.n_expert_used);
    for _ in 0..m.n_expert_used {
        let mut best = usize::MAX;
        for e in 0..m.n_expert {
            if pick.iter().any(|(p, _)| *p == e) {
                continue;
            }
            if best == usize::MAX || s.router[e] > s.router[best] {
                best = e;
            }
        }
        pick.push((best, s.router[best]));
    }

    let sum: f32 = pick.iter().map(|(_, p)| *p).sum();
    let denom = sum.max(6.103_515_625e-5);

    s.moe_acc[..nd].fill(0.0);
    ops.host_wrote(&s.moe_acc[..nd]);

    for (e, p) in &pick {
        let weight = p / denom;
        ops.matmul(&w.gate.expert(*e), x, &mut s.e_gate[..m.expert_ff]);
        ops.matmul(&w.up.expert(*e), x, &mut s.e_up[..m.expert_ff]);
        let (g, u) = (&mut s.e_gate[..m.expert_ff], &s.e_up[..m.expert_ff]);
        ops.silu_mul(g, u);
        ops.matmul(&w.down.expert(*e), &s.e_gate[..m.expert_ff], &mut s.e_out);
        let (acc, out) = (&mut s.moe_acc[..nd], &s.e_out[..nd]);
        ops.add_scaled(acc, out, weight);
    }

    // The shared expert: always run, gated by a sigmoid of a single logit.
    ops.matmul(w.shared_gate, x, &mut s.e_gate[..m.shared_ff]);
    ops.matmul(w.shared_up, x, &mut s.e_up[..m.shared_ff]);
    let (g, u) = (&mut s.e_gate[..m.shared_ff], &s.e_up[..m.shared_ff]);
    ops.silu_mul(g, u);
    ops.matmul(w.shared_down, &s.e_gate[..m.shared_ff], &mut s.e_out);

    // `ffn_gate_inp_shexp` is a vector, not a matrix: one logit per token.
    let logit: f32 = w
        .shared_gate_inp
        .iter()
        .zip(x)
        .map(|(a, b)| a * b)
        .sum();
    let sg = 1.0 / (1.0 + (-logit).exp());
    let (acc, out) = (&mut s.moe_acc[..nd], &s.e_out[..nd]);
    ops.add_scaled(acc, out, sg);

    s.ffn_out[at..at + nd].copy_from_slice(&s.moe_acc[..nd]);
    ops.host_wrote(&s.ffn_out[at..at + nd]);
}
