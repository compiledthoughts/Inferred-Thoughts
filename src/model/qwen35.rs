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
//! way GQA is: value head `h` reads key/query head `h / 2`.
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
//! RoPE on the attention layers is **mRoPE** with sections `{11, 11, 10, 0}`
//! over `rope.dimension_count = 64` of the 256-wide head — partial rotation,
//! the rest passed through.

use crate::error::{Error, Result};
use crate::gguf::GgufFile;

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
