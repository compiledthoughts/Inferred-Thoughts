//! The `qwen3` architecture — dense, GQA, QK-norm, SwiGLU.
//!
//! Ported from `llama_model_qwen3::graph` in llama.cpp's
//! `src/models/qwen3.cpp`. Every hyperparameter is read from GGUF metadata;
//! nothing here is hardcoded per model.
//!
//! Six details that would otherwise produce plausible-looking wrong numbers,
//! all confirmed against the reference:
//!
//! 1. `head_dim` comes from `attention.key_length`, **not** `n_embd / n_head`.
//!    For the 0.6B those are 128 and 64 respectively.
//! 2. RoPE is NEOX-style: dimension `i` pairs with `i + head_dim/2`.
//!    `LLM_ARCH_QWEN3` sits under llama.cpp's "pairs of head values are offset
//!    by n_rot/2" group.
//! 3. QK-norm is applied **before** RoPE, per head over `head_dim`.
//! 4. GQA: key/value head `h` serves query heads `h * n_head / n_head_kv` ..
//! 5. RMSNorm multiplies by the weight with no `+1` — that is Gemma's variant.
//! 6. The attention scale is `1/sqrt(head_dim)`, not `1/sqrt(n_embd)`.
//!
//! The LM head is tied when the file has no `output.weight`, which is how
//! llama.cpp's loader behaves and is the case for Qwen3-0.6B.

use crate::error::{Error, Result};
use crate::gguf::GgufFile;
use crate::ops::{Ops, Weights};
use crate::quant::dequantize_into;
use crate::quant::half::{f16_to_f32, f32_to_f16};

use super::{matrix, tensor, vector};

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_vocab: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
}

impl Config {
    /// Query heads per key/value head.
    pub fn gqa_group(&self) -> usize {
        self.n_head / self.n_head_kv
    }

    /// Width of the concatenated query projection.
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// Width of the concatenated key or value projection.
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    pub fn from_gguf(f: &GgufFile) -> Result<Self> {
        let md = &f.metadata;
        let arch = md.architecture()?;
        if arch != "qwen3" {
            return Err(Error::UnsupportedArchitecture {
                arch: arch.to_string(),
                supported: "qwen3",
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
        // The reference asserts n_embd_head_k == n_embd_head_v and uses one
        // head_dim throughout; a file violating that would break silently.
        if v_len != head_dim {
            return Err(Error::InconsistentArchitecture {
                what: "head dimension",
                detail: format!("key_length {head_dim} != value_length {v_len}"),
            });
        }

        // Vocabulary comes from the embedding tensor rather than the token list,
        // so the forward pass stays consistent even if they disagree.
        let n_vocab = tensor(f, "token_embd.weight")?
            .dims
            .get(1)
            .copied()
            .ok_or_else(|| Error::InconsistentArchitecture {
                what: "token_embd.weight",
                detail: "expected 2 dimensions".to_string(),
            })? as usize;

        Ok(Self {
            n_layer: md.get_arch_u32("block_count")? as usize,
            n_embd: md.get_arch_u32("embedding_length")? as usize,
            n_ff: md.get_arch_u32("feed_forward_length")? as usize,
            n_head,
            n_head_kv,
            head_dim,
            n_vocab,
            rope_theta: md.get_arch_f32("rope.freq_base")?,
            rms_eps: md.get_arch_f32("attention.layer_norm_rms_epsilon")?,
        })
    }
}

struct Layer<'a> {
    attn_norm: Vec<f32>,
    wq: Weights<'a>,
    wk: Weights<'a>,
    wv: Weights<'a>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    wo: Weights<'a>,
    ffn_norm: Vec<f32>,
    ffn_gate: Weights<'a>,
    ffn_up: Weights<'a>,
    ffn_down: Weights<'a>,
}

pub struct Qwen3<'a> {
    pub cfg: Config,
    tok_embd: Weights<'a>,
    output_norm: Vec<f32>,
    /// The LM head. Points at `tok_embd` when the file has no `output.weight`.
    output: Weights<'a>,
    layers: Vec<Layer<'a>>,
}

impl<'a> Qwen3<'a> {
    pub fn load(f: &'a GgufFile) -> Result<Self> {
        let cfg = Config::from_gguf(f)?;
        let (n_embd, n_ff) = (cfg.n_embd, cfg.n_ff);

        let tok_embd = matrix(f, "token_embd.weight", n_embd, cfg.n_vocab)?;

        // llama.cpp falls back to the embedding matrix when output.weight is
        // absent (models/qwen3.cpp, TENSOR_DUPLICATED). Qwen3-0.6B ties them.
        let output = match f.tensor("output.weight") {
            Some(_) => matrix(f, "output.weight", n_embd, cfg.n_vocab)?,
            None => tok_embd,
        };

        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = |name: &str| format!("blk.{i}.{name}");
            layers.push(Layer {
                attn_norm: vector(f, &p("attn_norm.weight"), n_embd)?,
                wq: matrix(f, &p("attn_q.weight"), n_embd, cfg.q_dim())?,
                wk: matrix(f, &p("attn_k.weight"), n_embd, cfg.kv_dim())?,
                wv: matrix(f, &p("attn_v.weight"), n_embd, cfg.kv_dim())?,
                q_norm: vector(f, &p("attn_q_norm.weight"), cfg.head_dim)?,
                k_norm: vector(f, &p("attn_k_norm.weight"), cfg.head_dim)?,
                wo: matrix(f, &p("attn_output.weight"), cfg.q_dim(), n_embd)?,
                ffn_norm: vector(f, &p("ffn_norm.weight"), n_embd)?,
                ffn_gate: matrix(f, &p("ffn_gate.weight"), n_embd, n_ff)?,
                ffn_up: matrix(f, &p("ffn_up.weight"), n_embd, n_ff)?,
                ffn_down: matrix(f, &p("ffn_down.weight"), n_ff, n_embd)?,
            });
        }

        Ok(Self {
            cfg,
            tok_embd,
            output_norm: vector(f, "output_norm.weight", n_embd)?,
            output,
            layers,
        })
    }

    /// Embedding lookup: row `id` of `token_embd`, dequantized.
    fn embed(&self, id: u32, out: &mut [f32]) -> Result<()> {
        if id as usize >= self.cfg.n_vocab {
            return Err(Error::TokenOutOfRange {
                id,
                vocab_size: self.cfg.n_vocab,
            });
        }
        dequantize_into(self.tok_embd.row(id as usize), self.tok_embd.ty, out)
    }

    /// Run the whole prompt and return logits for the final position.
    ///
    /// Stage 4 has no KV cache: the full sequence is recomputed every call, as
    /// `PROMPTS.md` specifies. `trace` receives every named intermediate so the
    /// acceptance test can diff layer by layer against `llama-eval-callback`.
    pub fn forward<O: Ops>(
        &self,
        ops: &O,
        tokens: &[u32],
        trace: &mut dyn FnMut(&str, usize, &[f32]),
    ) -> Result<Vec<f32>> {
        let c = &self.cfg;
        let n = tokens.len();
        if n == 0 {
            return Err(Error::InconsistentArchitecture {
                what: "forward",
                detail: "no tokens supplied".to_string(),
            });
        }

        // Residual stream, one row of n_embd per token.
        let mut x = vec![0.0f32; n * c.n_embd];
        for (t, &id) in tokens.iter().enumerate() {
            self.embed(id, &mut x[t * c.n_embd..(t + 1) * c.n_embd])?;
        }
        trace("inp_embd", 0, &x);

        // Full-sequence buffers. Sized for every token rather than reused per
        // token so that each `trace` call hands out a whole tensor, matching
        // what `llama-eval-callback` prints at the same point.
        let (nd, qd, kd, nf) = (c.n_embd, c.q_dim(), c.kv_dim(), c.n_ff);
        let mut normed = vec![0.0f32; n * nd];
        let mut q = vec![0.0f32; n * qd];
        let mut k = vec![0.0f32; n * kd];
        let mut v = vec![0.0f32; n * kd];
        let mut attn = vec![0.0f32; n * qd];
        let mut kqv = vec![0.0f32; n * nd];
        let mut gate = vec![0.0f32; n * nf];
        let mut up = vec![0.0f32; n * nf];
        let mut ffn_out = vec![0.0f32; n * nd];
        let mut scores = vec![0.0f32; n];

        for (il, layer) in self.layers.iter().enumerate() {
            for t in 0..n {
                ops.rms_norm(
                    &x[t * nd..(t + 1) * nd],
                    &layer.attn_norm,
                    c.rms_eps,
                    &mut normed[t * nd..(t + 1) * nd],
                );
            }
            trace("attn_norm", il, &normed);

            for t in 0..n {
                let inp = &normed[t * nd..(t + 1) * nd];
                ops.matmul(&layer.wq, inp, &mut q[t * qd..(t + 1) * qd]);
                ops.matmul(&layer.wk, inp, &mut k[t * kd..(t + 1) * kd]);
                ops.matmul(&layer.wv, inp, &mut v[t * kd..(t + 1) * kd]);
            }
            trace("Vcur", il, &v);

            // QK-norm strictly before RoPE, per head over head_dim.
            for t in 0..n {
                ops.rms_norm_heads(&mut q[t * qd..(t + 1) * qd], &layer.q_norm, c.head_dim, c.rms_eps);
            }
            trace("Qcur_normed", il, &q);
            for t in 0..n {
                ops.rms_norm_heads(&mut k[t * kd..(t + 1) * kd], &layer.k_norm, c.head_dim, c.rms_eps);
            }
            trace("Kcur_normed", il, &k);

            for t in 0..n {
                ops.rope_neox(&mut q[t * qd..(t + 1) * qd], t, c.head_dim, c.n_head, c.rope_theta);
            }
            trace("Qcur", il, &q);
            for t in 0..n {
                ops.rope_neox(&mut k[t * kd..(t + 1) * kd], t, c.head_dim, c.n_head_kv, c.rope_theta);
            }
            trace("Kcur", il, &k);

            // llama.cpp writes K and V into an f16 KV cache and reads them back
            // for attention, so its scores are computed on f16-rounded values.
            // That is a semantic difference worth ~3e-3 of tensor magnitude,
            // not rounding noise -- and an f16 cache is what we want regardless,
            // since it halves the VRAM the cache takes from the expert pool.
            // Stage 5 will do this at the cache boundary instead.
            for val in k.iter_mut().chain(v.iter_mut()) {
                *val = f16_to_f32(f32_to_f16(*val));
            }

            let scale = 1.0 / (c.head_dim as f32).sqrt();
            let group = c.gqa_group();

            for t in 0..n {
                for h in 0..c.n_head {
                    let h_kv = h / group;
                    let qh = &q[t * qd + h * c.head_dim..][..c.head_dim];

                    // Causal mask: positions 0..=t only.
                    for (s, score) in scores[..=t].iter_mut().enumerate() {
                        let kh = &k[s * kd + h_kv * c.head_dim..][..c.head_dim];
                        *score = qh.iter().zip(kh).map(|(a, b)| a * b).sum::<f32>() * scale;
                    }
                    ops.softmax(&mut scores[..=t]);

                    let out = &mut attn[t * qd + h * c.head_dim..][..c.head_dim];
                    out.fill(0.0);
                    for (s, &w) in scores[..=t].iter().enumerate() {
                        let vh = &v[s * kd + h_kv * c.head_dim..][..c.head_dim];
                        for (o, &vi) in out.iter_mut().zip(vh) {
                            *o += w * vi;
                        }
                    }
                }
            }

            // The reference names the concatenated head output "kqv_out",
            // before the output projection -- its dims are {q_dim, n_tokens}.
            // For token 0 this equals V[0], since it can only attend to itself.
            trace("kqv_out", il, &attn);

            for t in 0..n {
                ops.matmul(&layer.wo, &attn[t * qd..(t + 1) * qd], &mut kqv[t * nd..(t + 1) * nd]);
            }

            for t in 0..n {
                let (row, add) = (&mut x[t * nd..(t + 1) * nd], &kqv[t * nd..(t + 1) * nd]);
                ops.add_assign(row, add);
            }
            trace("ffn_inp", il, &x);

            for t in 0..n {
                ops.rms_norm(
                    &x[t * nd..(t + 1) * nd],
                    &layer.ffn_norm,
                    c.rms_eps,
                    &mut normed[t * nd..(t + 1) * nd],
                );
            }
            trace("ffn_norm", il, &normed);

            for t in 0..n {
                let inp = &normed[t * nd..(t + 1) * nd];
                ops.matmul(&layer.ffn_gate, inp, &mut gate[t * nf..(t + 1) * nf]);
                ops.matmul(&layer.ffn_up, inp, &mut up[t * nf..(t + 1) * nf]);
            }
            trace("ffn_gate", il, &gate);
            trace("ffn_up", il, &up);

            for t in 0..n {
                let (g, u) = (&mut gate[t * nf..(t + 1) * nf], &up[t * nf..(t + 1) * nf]);
                ops.silu_mul(g, u);
            }
            trace("ffn_swiglu", il, &gate);

            for t in 0..n {
                ops.matmul(&layer.ffn_down, &gate[t * nf..(t + 1) * nf], &mut ffn_out[t * nd..(t + 1) * nd]);
            }
            trace("ffn_out", il, &ffn_out);

            for t in 0..n {
                let (row, add) = (&mut x[t * nd..(t + 1) * nd], &ffn_out[t * nd..(t + 1) * nd]);
                ops.add_assign(row, add);
            }
            trace("l_out", il, &x);
        }

        // Only the final position's logits are needed.
        let last = &x[(n - 1) * nd..];
        let mut final_norm = vec![0.0f32; nd];
        ops.rms_norm(last, &self.output_norm, c.rms_eps, &mut final_norm);
        trace("result_norm", 0, &final_norm);
        let normed = final_norm;

        let mut logits = vec![0.0f32; c.n_vocab];
        ops.matmul(&self.output, &normed, &mut logits);
        trace("result_output", 0, &logits);

        Ok(logits)
    }

    /// Greedy pick from a logit vector.
    pub fn argmax(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        best as u32
    }
}
