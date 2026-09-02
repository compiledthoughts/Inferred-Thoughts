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

use crate::cache::{KvCache, RecurrentState};
use crate::error::{Error, Result};
use crate::gguf::GgufFile;
use crate::ops::{Attn, Delta, Ops, Weights};
use crate::profile::{Ctx, Part};
use crate::quant::{dequantize, dequantize_into};

use super::{matrix, tensor, vector};

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
}

impl Config {
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
        if arch != "qwen35" {
            return Err(Error::UnsupportedArchitecture {
                arch: arch.to_string(),
                supported: "qwen35",
            });
        }

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
            n_ff: md.get_arch_u32("feed_forward_length")? as usize,
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
        };

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
    ffn_gate: Weights<'a>,
    ffn_up: Weights<'a>,
    ffn_down: Weights<'a>,
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
struct Scratch {
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
    fn new(c: &Config) -> Self {
        let z = |n: usize| vec![0.0f32; n];
        Self {
            normed: z(c.n_embd),
            mixed: z(c.n_embd),
            gate: z(c.n_ff),
            up: z(c.n_ff),
            ffn_out: z(c.n_embd),
            qg: z(c.q_gate_dim()),
            q: z(c.head_dim * c.n_head),
            g: z(c.head_dim * c.n_head),
            k: z(c.kv_dim()),
            v: z(c.kv_dim()),
            attn: z(c.head_dim * c.n_head),
            qkv: z(c.conv_dim()),
            z: z(c.value_dim()),
            alpha: z(c.n_v_heads()),
            beta: z(c.n_v_heads()),
            conv: z(c.conv_dim()),
            q_part: z(c.key_dim()),
            k_part: z(c.key_dim()),
            v_part: z(c.value_dim()),
            core: z(c.value_dim()),
        }
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
            layers.push(Layer {
                attn_norm: vector(f, &p("attn_norm.weight"), n_embd)?,
                ffn_norm: vector(f, &p("post_attention_norm.weight"), n_embd)?,
                mixer,
                ffn_gate: matrix(f, &p("ffn_gate.weight"), n_embd, n_ff)?,
                ffn_up: matrix(f, &p("ffn_up.weight"), n_embd, n_ff)?,
                ffn_down: matrix(f, &p("ffn_down.weight"), n_ff, n_embd)?,
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
                mixer + w(&l.ffn_gate) + w(&l.ffn_up) + w(&l.ffn_down)
            })
            .sum();
        per_layer + w(&self.output)
    }

    /// One token against the recurrent state and the KV cache.
    ///
    /// **Single token only, deliberately, for now.** The delta rule is a
    /// sequential scan — token `t`'s state update feeds token `t+1` — so a
    /// batched prefill is a different algorithm (llama.cpp has a whole chunked
    /// path for it), not a loop tightening. Prefill therefore runs this once
    /// per prompt token, which is correct and slow, and the chunked form is a
    /// later optimization rather than a correctness question.
    ///
    /// That is a real departure from [`super::Qwen3::forward`], where one
    /// function serving both phases is what makes the cache acceptance test
    /// exact. Here the equivalent property comes for free: there is only one
    /// path, so prefill and decode cannot disagree.
    pub fn forward<O: Ops>(
        &self,
        ops: &O,
        token: u32,
        pos: usize,
        kv: &mut KvCache,
        rs: &mut RecurrentState,
        ctx: &mut Ctx<'_>,
    ) -> Result<Vec<f32>> {
        let c = &self.cfg;
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
        if pos >= kv.n_ctx() {
            return Err(Error::ContextOverflow {
                pos,
                n_ctx: kv.n_ctx(),
            });
        }

        let step = ctx.prof.begin_step();
        ops.begin_pass(1);

        let nd = c.n_embd;
        let mut x = vec![0.0f32; nd];
        self.embed(token, &mut x)?;
        ops.host_wrote(&x);
        ctx.trace("inp_embd", 0, &x);

        // One allocation per pass, not per layer. See `Scratch`.
        let mut s = Scratch::new(c);

        for il in 0..c.n_main_layer() {
            let layer = &self.layers[il];
            let t_mix = ctx.prof.layer_begin();

            ops.rms_norm(&x, &layer.attn_norm, c.rms_eps, &mut s.normed);
            ctx.trace("attn_norm", il, &s.normed);

            match &layer.mixer {
                Mixer::Attn { .. } => self.attention(ops, layer, il, pos, kv, &mut s, ctx)?,
                Mixer::Delta { .. } => self.gated_delta(ops, layer, il, rs, &mut s, ctx)?,
            }
            ops.add_assign(&mut x, &s.mixed);
            ctx.trace("attn_residual", il, &x);
            ctx.prof.layer_end(t_mix, step, il, Part::Attn);

            let t_ffn = ctx.prof.layer_begin();
            ops.rms_norm(&x, &layer.ffn_norm, c.rms_eps, &mut s.normed);
            ops.matmul(&layer.ffn_gate, &s.normed, &mut s.gate);
            ops.matmul(&layer.ffn_up, &s.normed, &mut s.up);
            ops.silu_mul(&mut s.gate, &s.up);
            ops.matmul(&layer.ffn_down, &s.gate, &mut s.ffn_out);
            ops.add_assign(&mut x, &s.ffn_out);
            ctx.trace("post_ffn", il, &x);
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
        kv.commit(pos + 1);

        ops.rms_norm(&x, &self.output_norm, c.rms_eps, &mut s.normed);
        ctx.trace("result_norm", 0, &s.normed);

        let mut logits = vec![0.0f32; c.n_vocab];
        ops.matmul(&self.output, &s.normed, &mut logits);
        ops.end_pass();
        ops.host_needs(&mut logits);
        ctx.trace("result_output", 0, &logits);
        Ok(logits)
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
        pos: usize,
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
        let qd = hd * c.n_head;

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

        ops.rope_neox(&mut s.q, pos, hd, c.n_rot, c.n_head, c.rope_theta);
        ctx.trace("Qcur", il, &s.q);
        ops.rope_neox(&mut s.k, pos, hd, c.n_rot, c.n_head_kv, c.rope_theta);
        ctx.trace("Kcur", il, &s.k);

        let slot = self.kv_slot[il];
        ops.kv_write(kv.k_layer_mut(slot), pos * kd, &s.k);
        ops.kv_write(kv.v_layer_mut(slot), pos * kd, &s.v);

        let a = Attn {
            q: &s.q,
            k: kv.k_layer(slot),
            v: kv.v_layer(slot),
            kv_dim: kd,
            n_pos: pos + 1,
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
