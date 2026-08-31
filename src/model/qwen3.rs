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

use crate::cache::KvCache;
use crate::error::{Error, Result};
use crate::gguf::GgufFile;
use crate::ops::{Attn, Ops, Weights};
use crate::profile::{Ctx, Part};
use crate::quant::dequantize_into;

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

    /// Bytes of quantized weight one full forward pass reads.
    ///
    /// Derived from the tensor types and shapes rather than counted during
    /// execution, per rule 3 in [`crate::profile`]: every matmul reads its
    /// whole weight exactly once per pass, so a runtime counter would only
    /// recompute a constant — while contending across threads to do it.
    ///
    /// The embedding lookup is excluded: it touches one row, not the tensor.
    /// When the LM head is tied it *is* `token_embd`, and a pass still reads it
    /// once, so counting it once is right either way.
    pub fn weight_bytes_per_pass(&self) -> u64 {
        let w = |m: &Weights<'_>| m.ty.n_bytes(m.n_in as u64) * m.n_out as u64;
        let per_layer: u64 = self
            .layers
            .iter()
            .map(|l| {
                w(&l.wq)
                    + w(&l.wk)
                    + w(&l.wv)
                    + w(&l.wo)
                    + w(&l.ffn_gate)
                    + w(&l.ffn_up)
                    + w(&l.ffn_down)
            })
            .sum();
        per_layer + w(&self.output)
    }

    /// Run `tokens` starting at absolute position `start_pos`, append their K
    /// and V to `cache`, and return logits for the final position.
    ///
    /// **One function serves both phases.** Prefill is the whole prompt at
    /// `start_pos = 0` against a fresh cache; decode is a single token at
    /// `start_pos = cache.len()`. Keeping them one code path is what makes the
    /// acceptance test exact: incremental decode must produce *bit-identical*
    /// logits to a full recompute, because it is the same arithmetic in the
    /// same order over the same f16-rounded K and V. Any position or slot
    /// indexing error breaks that equality immediately and unambiguously,
    /// which a token-for-token comparison against `llama-cli` cannot do — the
    /// ~1% logit drift documented in `CLAUDE.md` would flip an argmax
    /// somewhere in 250 greedy decisions regardless of whether the cache is
    /// correct.
    ///
    /// `ctx` carries the Stage 4 tensor tracer and the profiler. Tracer calls
    /// still hand out whole batch-local tensors, so the `llama-eval-callback`
    /// comparison keeps working unchanged.
    pub fn forward<O: Ops>(
        &self,
        ops: &O,
        tokens: &[u32],
        start_pos: usize,
        cache: &mut KvCache,
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
        if cache.kv_dim() != c.kv_dim() {
            return Err(Error::InconsistentArchitecture {
                what: "kv cache",
                detail: format!(
                    "cache holds {} lanes per position, model needs {}",
                    cache.kv_dim(),
                    c.kv_dim()
                ),
            });
        }
        // Checked up front so the cache is never left half-written.
        if start_pos + n > cache.n_ctx() {
            return Err(Error::ContextOverflow {
                pos: start_pos + n - 1,
                n_ctx: cache.n_ctx(),
            });
        }

        let step = ctx.prof.begin_step();

        // A device backend keys its copies on host addresses, and the buffers
        // below are allocated fresh each pass, so last pass's addresses must
        // not be trusted. No-op on the CPU backends.
        ops.begin_pass();

        // Residual stream, one row of n_embd per token in this batch.
        let mut x = vec![0.0f32; n * c.n_embd];
        for (t, &id) in tokens.iter().enumerate() {
            self.embed(id, &mut x[t * c.n_embd..(t + 1) * c.n_embd])?;
        }
        // Written here rather than by an op, so a device copy would be stale.
        // Per row, because that is the granularity the ops below work at and
        // therefore the granularity a device backend keys its copies on.
        for t in 0..n {
            ops.host_wrote(&x[t * c.n_embd..(t + 1) * c.n_embd]);
        }
        ctx.trace("inp_embd", 0, &x);

        // Batch-local buffers: one slot per token being processed now, which in
        // decode is exactly one. Sized for the whole batch rather than reused
        // per token so each trace call hands out a whole tensor, matching what
        // `llama-eval-callback` prints at the same point.
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

        for (il, layer) in self.layers.iter().enumerate() {
            let t_attn = ctx.prof.layer_begin();

            for t in 0..n {
                ops.rms_norm(
                    &x[t * nd..(t + 1) * nd],
                    &layer.attn_norm,
                    c.rms_eps,
                    &mut normed[t * nd..(t + 1) * nd],
                );
            }
            ctx.trace("attn_norm", il, &normed);

            for t in 0..n {
                let inp = &normed[t * nd..(t + 1) * nd];
                ops.matmul(&layer.wq, inp, &mut q[t * qd..(t + 1) * qd]);
                ops.matmul(&layer.wk, inp, &mut k[t * kd..(t + 1) * kd]);
                ops.matmul(&layer.wv, inp, &mut v[t * kd..(t + 1) * kd]);
            }
            ctx.trace("Vcur", il, &v);

            // QK-norm strictly before RoPE, per head over head_dim.
            for t in 0..n {
                ops.rms_norm_heads(
                    &mut q[t * qd..(t + 1) * qd],
                    &layer.q_norm,
                    c.head_dim,
                    c.rms_eps,
                );
            }
            ctx.trace("Qcur_normed", il, &q);
            for t in 0..n {
                ops.rms_norm_heads(
                    &mut k[t * kd..(t + 1) * kd],
                    &layer.k_norm,
                    c.head_dim,
                    c.rms_eps,
                );
            }
            ctx.trace("Kcur_normed", il, &k);

            // RoPE at the ABSOLUTE position. Using the batch index `t` here is
            // the classic KV cache bug: it is invisible during prefill, where
            // the two are equal, and wrong for every token decoded after.
            for t in 0..n {
                ops.rope_neox(
                    &mut q[t * qd..(t + 1) * qd],
                    start_pos + t,
                    c.head_dim,
                    c.n_head,
                    c.rope_theta,
                );
            }
            ctx.trace("Qcur", il, &q);
            for t in 0..n {
                ops.rope_neox(
                    &mut k[t * kd..(t + 1) * kd],
                    start_pos + t,
                    c.head_dim,
                    c.n_head_kv,
                    c.rope_theta,
                );
            }
            ctx.trace("Kcur", il, &k);

            // Publish the whole batch before attending: within a prefill, token
            // t attends to tokens start_pos..=start_pos+t, which includes rows
            // written by this same call. The cache rounds to f16 on the way in
            // — that is where llama.cpp's f16 KV semantics now live, replacing
            // the explicit round-trip Stage 4 did here.
            // The cache is host memory and `store` reads these directly, so
            // this is one of the two points in the pass where a device copy has
            // to come home. The other is the logits.
            for t in 0..n {
                ops.host_needs(&mut k[t * kd..(t + 1) * kd]);
                ops.host_needs(&mut v[t * kd..(t + 1) * kd]);
            }
            for t in 0..n {
                cache.store(
                    il,
                    start_pos + t,
                    &k[t * kd..(t + 1) * kd],
                    &v[t * kd..(t + 1) * kd],
                )?;
            }

            let scale = 1.0 / (c.head_dim as f32).sqrt();

            // Attention now goes through the ops seam rather than a loop here,
            // so a backend can thread over heads. Model code stays unaware of
            // which backend it is talking to, per CLAUDE.md.
            for t in 0..n {
                let pos = start_pos + t;
                let a = Attn {
                    q: &q[t * qd..(t + 1) * qd],
                    k: cache.k_layer(il),
                    v: cache.v_layer(il),
                    kv_dim: kd,
                    // Inclusive of this token, which is what masks the future.
                    n_pos: pos + 1,
                    head_dim: c.head_dim,
                    n_head: c.n_head,
                    n_head_kv: c.n_head_kv,
                    scale,
                };
                ops.attend(&a, &mut attn[t * qd..(t + 1) * qd]);
            }

            // The reference names the concatenated head output "kqv_out",
            // before the output projection -- its dims are {q_dim, n_tokens}.
            // For token 0 this equals V[0], since it can only attend to itself.
            ctx.trace("kqv_out", il, &attn);

            for t in 0..n {
                ops.matmul(
                    &layer.wo,
                    &attn[t * qd..(t + 1) * qd],
                    &mut kqv[t * nd..(t + 1) * nd],
                );
            }

            for t in 0..n {
                let (row, add) = (&mut x[t * nd..(t + 1) * nd], &kqv[t * nd..(t + 1) * nd]);
                ops.add_assign(row, add);
            }
            ctx.trace("ffn_inp", il, &x);
            ctx.prof.layer_end(t_attn, step, il, Part::Attn);

            let t_ffn = ctx.prof.layer_begin();

            for t in 0..n {
                ops.rms_norm(
                    &x[t * nd..(t + 1) * nd],
                    &layer.ffn_norm,
                    c.rms_eps,
                    &mut normed[t * nd..(t + 1) * nd],
                );
            }
            ctx.trace("ffn_norm", il, &normed);

            for t in 0..n {
                let inp = &normed[t * nd..(t + 1) * nd];
                ops.matmul(&layer.ffn_gate, inp, &mut gate[t * nf..(t + 1) * nf]);
                ops.matmul(&layer.ffn_up, inp, &mut up[t * nf..(t + 1) * nf]);
            }
            ctx.trace("ffn_gate", il, &gate);
            ctx.trace("ffn_up", il, &up);

            for t in 0..n {
                let (g, u) = (&mut gate[t * nf..(t + 1) * nf], &up[t * nf..(t + 1) * nf]);
                ops.silu_mul(g, u);
            }
            ctx.trace("ffn_swiglu", il, &gate);

            for t in 0..n {
                ops.matmul(
                    &layer.ffn_down,
                    &gate[t * nf..(t + 1) * nf],
                    &mut ffn_out[t * nd..(t + 1) * nd],
                );
            }
            ctx.trace("ffn_out", il, &ffn_out);

            for t in 0..n {
                let (row, add) = (&mut x[t * nd..(t + 1) * nd], &ffn_out[t * nd..(t + 1) * nd]);
                ops.add_assign(row, add);
            }
            ctx.trace("l_out", il, &x);
            ctx.prof.layer_end(t_ffn, step, il, Part::Ffn);
        }

        // Every layer wrote its rows, so the positions are now real.
        cache.commit(start_pos + n);

        // KV traffic, derived rather than counted (rule 3 in `crate::profile`).
        // These are *distinct* bytes: a GQA group of query heads shares one kv
        // head, so counting per query head would report re-reads of the same
        // cache lines as bus traffic. The read term grows with position, which
        // is what makes attention's quadratic term visible next to the FFN's
        // flat one.
        let per_pos = cache.bytes_per_position();
        ctx.prof.kv_write_bytes += n as u64 * per_pos;
        ctx.prof.kv_read_bytes += (start_pos..start_pos + n)
            .map(|p| (p as u64 + 1) * per_pos)
            .sum::<u64>();

        // Only the final position's logits are needed.
        let last = &x[(n - 1) * nd..];
        let mut final_norm = vec![0.0f32; nd];
        ops.rms_norm(last, &self.output_norm, c.rms_eps, &mut final_norm);
        ctx.trace("result_norm", 0, &final_norm);
        let normed = final_norm;

        let mut logits = vec![0.0f32; c.n_vocab];
        ops.matmul(&self.output, &normed, &mut logits);
        ops.host_needs(&mut logits);
        ctx.trace("result_output", 0, &logits);

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
