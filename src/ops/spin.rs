//! The threaded backend, on a spinning pool instead of rayon.
//!
//! Same relationship to [`super::naive`] as [`super::par`]: it reuses the
//! oracle's kernels verbatim and only changes *which thread* computes an output
//! row, never how a row is computed. So it is held to the same standard —
//! **bit-identical** results, not "within a tolerance".
//!
//! What changes is the dispatch mechanism, and that is the whole point.
//! [`super::par`] can only afford to thread the LM head, because a rayon
//! parallel region costs ~430 us here and a decode step has ~196 matmuls.
//! [`super::pool`] measures 0.40 us for the same barrier, so this backend can
//! thread **every** matmul, which is where ~74% of a token's weight traffic
//! lives.
//!
//! That matters beyond arithmetic: a single core cannot saturate dual-channel
//! DDR5, because it can only keep so many cache misses in flight. Our decode
//! moves ~12 GB/s where llama.cpp moves ~41 GB/s, and 12 GB/s is about what one
//! core's memory-level parallelism buys. Spreading the loads across cores is
//! how that gap closes — not wider instructions.

use super::naive::{self, Naive};
use crate::quant::{Q8KRow, kquant};
use super::pool::{Pool, Rows};
use super::{Attn, Delta, Ops, Weights};
use crate::gguf::GgmlType;

/// Below this many output rows, run serially.
///
/// Far lower than [`super::par`]'s 8192, because the dispatch it guards against
/// is ~1000x cheaper here. It exists only so that genuinely tiny matmuls do not
/// pay a barrier for a few hundred bytes of work.
const MIN_ROWS: usize = 64;

/// Below this many cached positions, attention runs serially.
const MIN_POS: usize = 32;

/// Tokens whose quantized activations are held resident while one weight row is
/// reused across them.
///
/// **Measured to be worth almost nothing, on both models, and that is the
/// useful part.** The tile exists so a weight row is fetched from DRAM once and
/// dotted against several tokens while it sits in L1, rather than once per
/// token. Sweeping it 32x moved prefill by under 10%:
///
/// | `TOKEN_TILE` | 1 | 8 | 32 | 128 |
/// |---|---|---|---|---|
/// | Qwen3-0.6B, 841 prompt tokens | 106.5 | 108.7 | 107.3 | 107.7 |
/// | Qwen3.5-9B, 217 prompt tokens | 10.5 | 11.2 | 11.6 | |
///
/// So **weight-row reuse is not what bounds CPU prefill**, on a model whose
/// per-layer working set fits in this machine's 32 MB L3 (the 0.6B, ~15 MB) or
/// on one whose does not (the 9B, ~300 MB). The candidate explanation that the
/// small model was simply being served from L3 does not survive the second row.
///
/// What does bound it is not settled here. The arithmetic is a plausible
/// suspect — a pass does ~429M Q8_0 MACs on the 0.6B and 8 threads clear them
/// at ~1.15 MACs/cycle/core, which is scalar-code territory on a machine with
/// AVX512-VNNI — but that is an inference from a rate, not a measurement, and
/// this file's own history is a warning about those. It is recorded as an open
/// question rather than a conclusion.
///
/// Note the contrast with the GPU, where the same idea is the whole win:
/// `MM_TOK` in `ops/cuda/ops.rs` moved the 9B's prefill 52.6 -> 105.2 tok/s
/// going from 1 to 4. Register reuse across a batch is decisive on a card
/// limited by weight bandwidth and nearly free on a CPU that is limited by
/// something else.
///
/// It is kept at 8 because it costs nothing measured and is the right loop
/// order for a vectorized kernel, which is the thing that would actually
/// collect this if arithmetic is indeed the bound. Note what that would imply:
/// `CLAUDE.md` retired CPU SIMD on the grounds that decode is DDR5-bound and
/// SIMD width buys little there. That argument holds for decode and does not
/// automatically transfer to prefill, which is a different regime.
const TOKEN_TILE: usize = 8;

/// The activation, in whichever form the weight's `vec_dot_type` calls for.
///
/// ggml pairs Q8_0 weights with a Q8_0 activation and the k-quants with Q8_K,
/// and quantizing is a function of the activation alone — so it happens once
/// per matmul and every worker shares it read-only, exactly as the single-token
/// path has always shared its Q8_0 row.
enum Act {
    Q80(naive::QuantizedRow),
    Q8k(Q8KRow),
    /// F32 and F16 weights, which dot against the activation directly.
    Raw,
}

impl Act {
    fn of(ty: GgmlType, x: &[f32]) -> Self {
        match ty {
            GgmlType::Q8_0 => Act::Q80(naive::QuantizedRow::from_f32(x)),
            GgmlType::Q5K | GgmlType::Q6K | GgmlType::Iq4Xs => Act::Q8k(Q8KRow::from_f32(x)),
            _ => Act::Raw,
        }
    }

    fn dot(&self, ty: GgmlType, row: &[u8], x: &[f32]) -> f32 {
        match self {
            Act::Q80(q) => naive::dot_q8_0_q8_0(row, q),
            Act::Q8k(q) => kquant::dot_row_q8_k(ty, row, q),
            Act::Raw => naive::dot_row(ty, row, x),
        }
    }
}

pub struct Spin {
    pool: Pool,
}

impl Spin {
    /// One token against every weight row: the path decode takes.
    ///
    /// Kept separate rather than folded into the batched loop because decode is
    /// the measured path, and a contiguous per-worker output span is both what
    /// `Rows` was designed around and what every published number here was
    /// taken on. At `n == 1` the batched loop would be the same arithmetic
    /// through a narrower write.
    fn matmul_one(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) {
        // Quantized once and shared read-only across every worker, exactly as
        // the serial path shares it across every row. Doing it per worker would
        // be both slower and a different function.
        let qx = Act::of(w.ty, x);

        let rows = Rows::new(out);
        self.pool.run(|index, n| {
            let (start, end) = Rows::range(rows.len(), index, n);
            if start == end {
                return;
            }
            // SAFETY: `Rows::range` gives disjoint spans for distinct `index`
            // at a fixed `n`, so no two workers touch the same element; and
            // `out` outlives this `run` call, which does not return until every
            // worker has finished.
            let mine = unsafe { rows.slice(start, end) };
            for (i, o) in mine.iter_mut().enumerate() {
                *o = qx.dot(w.ty, w.row(start + i), x);
            }
        });
    }

    /// `threads` counts the calling thread, so `1` is fully serial.
    pub fn new(threads: usize) -> Self {
        Self {
            pool: Pool::new(threads),
        }
    }

    pub fn threads(&self) -> usize {
        self.pool.threads()
    }
}

impl Ops for Spin {
    fn matmul(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) {
        let n = x.len() / w.n_in;
        debug_assert_eq!(x.len() % w.n_in, 0);
        debug_assert_eq!(out.len(), n * w.n_out);

        if w.n_out < MIN_ROWS || self.pool.threads() == 1 {
            return Naive.matmul(w, x, out);
        }
        if n == 1 {
            return self.matmul_one(w, x, out);
        }

        // Quantized once per token and shared read-only across every worker,
        // exactly as the serial path shares one row across every weight row.
        // Doing it per worker would be both slower and a different function.
        let qx: Vec<Act> = x.chunks_exact(w.n_in).map(|r| Act::of(w.ty, r)).collect();

        // **The weight row is the outer loop and the token the inner one.**
        // That is the whole reason a batch is faster on a DDR5-bound CPU: a row
        // is fetched from DRAM once and dotted against every token in the tile
        // while it sits in L1, instead of being re-fetched once per token.
        // Tokens are tiled because at `n` in the hundreds their quantized
        // activations would not stay resident together.
        let n_out = w.n_out;
        let rows = Rows::new(out);
        self.pool.run(|index, nw| {
            let (start, end) = Rows::range(n_out, index, nw);
            if start == end {
                return;
            }
            for lo in (0..n).step_by(TOKEN_TILE) {
                let hi = (lo + TOKEN_TILE).min(n);
                for j in start..end {
                    let row = w.row(j);
                    for t in lo..hi {
                        let v = qx[t].dot(w.ty, row, &x[t * w.n_in..(t + 1) * w.n_in]);
                        // SAFETY: workers own disjoint weight-row bands, from
                        // `Rows::range` over `n_out` at a fixed `nw`, so for
                        // every `t` the element `t * n_out + j` belongs to
                        // exactly one worker. `out` outlives this `run` call,
                        // which does not return until every worker finishes.
                        let at = t * n_out + j;
                        let o = unsafe { rows.slice(at, at + 1) };
                        o[0] = v;
                    }
                }
            }
        });
    }

    /// Threaded over key/value heads, which are already disjoint in `out`.
    ///
    /// Query heads served by one key/value head are contiguous, so the output
    /// splits into `n_head_kv` independent `group * head_dim` runs. Workers take
    /// a contiguous span of those, which keeps `Rows::range`'s disjointness
    /// argument intact at head granularity rather than element granularity.
    fn attend(&self, a: &Attn<'_>, out: &mut [f32]) {
        let n_q = a.n_q();
        debug_assert_eq!(out.len(), n_q * a.n_head * a.head_dim);

        if a.n_pos < MIN_POS || (a.n_head_kv < 2 && n_q < 2) || self.pool.threads() == 1 {
            return Naive.attend(a, out);
        }

        let per_kv = a.group() * a.head_dim;
        let per_row = a.n_head * a.head_dim;
        let n_kv = a.n_head_kv;
        // `(query row, kv head)` is the unit of independent work. In decode
        // that is `n_head_kv` units, which the pool can outnumber -- 8 kv heads
        // against 8 threads is why `-t 8` was never faster than `-t 4` here. A
        // prefill batch multiplies the grid by `n_q`, so this is the first time
        // attention has more parallelism available than the pool has threads.
        let units = n_q * n_kv;
        let rows = Rows::new(out);
        self.pool.run(|index, n| {
            let (first, last) = Rows::range(units, index, n);
            if first == last {
                return;
            }
            let mut sc = naive::Scratch::for_attn(a);
            for u in first..last {
                let (t, h_kv) = (u / n_kv, u % n_kv);
                // SAFETY: `range` partitions the `units` grid, and unit
                // `(t, h_kv)` owns exactly the `per_kv` contiguous outputs at
                // `t * per_row + h_kv * per_kv`, which is injective in `u`. So
                // no two workers touch the same element, and `out` outlives
                // this call.
                let at = t * per_row + h_kv * per_kv;
                let chunk = unsafe { rows.slice(at, at + per_kv) };
                naive::attend_kv_head(a, t, h_kv, chunk, &mut sc);
            }
        });
    }

    // The rest are vector ops over at most a few thousand elements, well under
    // the point where a barrier pays for itself. They are the oracle's code.
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        Naive.rms_norm(x, weight, eps, out)
    }

    fn rms_norm_heads(&self, x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
        Naive.rms_norm_heads(x, weight, head_dim, eps)
    }

    fn rope_neox(
        &self,
        x: &mut [f32],
        pos: usize,
        head_dim: usize,
        n_rot: usize,
        n_heads: usize,
        theta: f32,
    ) {
        Naive.rope_neox(x, pos, head_dim, n_rot, n_heads, theta)
    }

    fn softmax(&self, x: &mut [f32]) {
        Naive.softmax(x)
    }

    fn silu_mul(&self, gate: &mut [f32], up: &[f32]) {
        Naive.silu_mul(gate, up)
    }

    // GatedDeltaNet's three primitives forward to the oracle for now. Each is
    // either tiny (l2_norm_heads, ssm_conv over 4 taps) or a sequential scan
    // whose parallel decomposition is over value heads -- worth threading only
    // once the 9B says these show up in a profile. `PARALLEL_THRESHOLD` records
    // why guessing at that is a losing move here.
    fn l2_norm_heads(&self, x: &mut [f32], head_dim: usize, eps: f32) {
        Naive.l2_norm_heads(x, head_dim, eps)
    }

    fn ssm_conv(
        &self,
        state: &mut [f32],
        x: &[f32],
        weight: &[f32],
        kernel: usize,
        out: &mut [f32],
    ) {
        Naive.ssm_conv(state, x, weight, kernel, out)
    }

    fn gather_chunks(
        &self,
        src: &[f32],
        chunk: usize,
        stride: usize,
        offset: usize,
        out: &mut [f32],
    ) {
        Naive.gather_chunks(src, chunk, stride, offset, out)
    }

    fn sigmoid_mul(&self, x: &mut [f32], g: &[f32]) {
        Naive.sigmoid_mul(x, g)
    }

    fn delta_rule(&self, d: &Delta<'_>, state: &mut [f32], out: &mut [f32]) {
        Naive.delta_rule(d, state, out)
    }

    fn add_assign(&self, a: &mut [f32], b: &[f32]) {
        Naive.add_assign(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::half::f32_to_f16;

    fn fixture(ty: GgmlType, n_in: usize, n_out: usize) -> (Vec<u8>, Vec<f32>) {
        let val = |i: usize| ((i as f32) * 0.7391).sin() * 2.0 + ((i as f32) * 0.113).cos();
        let mut bytes = Vec::new();
        match ty {
            GgmlType::F32 => {
                for i in 0..n_in * n_out {
                    bytes.extend_from_slice(&val(i).to_le_bytes());
                }
            }
            GgmlType::F16 => {
                for i in 0..n_in * n_out {
                    bytes.extend_from_slice(&f32_to_f16(val(i)).to_le_bytes());
                }
            }
            GgmlType::Q8_0 => {
                for row in 0..n_out {
                    for b in 0..n_in / 32 {
                        let block: Vec<f32> =
                            (0..32).map(|k| val(row * n_in + b * 32 + k)).collect();
                        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                        let d = amax / 127.0;
                        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
                        bytes.extend_from_slice(&f32_to_f16(d).to_le_bytes());
                        for v in block {
                            bytes.push(((v * id).round() as i8) as u8);
                        }
                    }
                }
            }
            other => panic!("no fixture for {}", other.name()),
        }
        let x: Vec<f32> = (0..n_in).map(|i| val(i + 9871)).collect();
        (bytes, x)
    }

    /// The test the whole backend rests on. Identical bits, not a tolerance:
    /// `Spin` reorders which thread computes a row, never how one is computed.
    fn assert_matmul_identical(ty: GgmlType, n_in: usize, n_out: usize, threads: usize) {
        let (bytes, x) = fixture(ty, n_in, n_out);
        let w = Weights {
            data: &bytes,
            ty,
            n_in,
            n_out,
        };
        let mut serial = vec![0.0f32; n_out];
        let mut threaded = vec![0.0f32; n_out];
        Naive.matmul(&w, &x, &mut serial);
        Spin::new(threads).matmul(&w, &x, &mut threaded);
        for j in 0..n_out {
            assert_eq!(
                serial[j].to_bits(),
                threaded[j].to_bits(),
                "{} row {j} at {threads} threads: {} vs {}",
                ty.name(),
                serial[j],
                threaded[j]
            );
        }
    }

    #[test]
    fn q8_0_matmul_is_bit_identical_to_the_oracle() {
        for threads in [1, 2, 3, 4, 8] {
            // Rows that divide evenly, and rows that leave a ragged tail.
            assert_matmul_identical(GgmlType::Q8_0, 128, 512, threads);
            assert_matmul_identical(GgmlType::Q8_0, 96, 517, threads);
        }
    }

    #[test]
    fn f32_and_f16_matmul_are_bit_identical_to_the_oracle() {
        for threads in [1, 3, 8] {
            assert_matmul_identical(GgmlType::F32, 64, 300, threads);
            assert_matmul_identical(GgmlType::F16, 64, 300, threads);
        }
    }

    /// More workers than rows must not corrupt or drop anything.
    #[test]
    fn more_threads_than_rows_is_safe() {
        assert_matmul_identical(GgmlType::Q8_0, 64, MIN_ROWS + 1, 16);
    }

    #[test]
    fn below_the_row_threshold_falls_back_to_serial() {
        assert_matmul_identical(GgmlType::Q8_0, 64, MIN_ROWS - 1, 8);
    }

    /// Attention, the other threaded op. Same standard.
    #[test]
    fn attend_is_bit_identical_to_the_oracle() {
        let (head_dim, n_head, n_head_kv) = (16usize, 8usize, 4usize);
        let kv_dim = n_head_kv * head_dim;
        for n_pos in [MIN_POS, MIN_POS + 7, 100] {
            let q: Vec<f32> = (0..n_head * head_dim)
                .map(|i| ((i as f32) * 0.31).sin())
                .collect();
            let k: Vec<u16> = (0..n_pos * kv_dim)
                .map(|i| f32_to_f16(((i as f32) * 0.017).cos()))
                .collect();
            let v: Vec<u16> = (0..n_pos * kv_dim)
                .map(|i| f32_to_f16(((i as f32) * 0.029).sin()))
                .collect();
            let a = Attn {
                q: &q,
                k: &k,
                v: &v,
                kv_dim,
                n_pos,
                head_dim,
                n_head,
                n_head_kv,
                scale: 1.0 / (head_dim as f32).sqrt(),
            };
            let mut serial = vec![0.0f32; n_head * head_dim];
            Naive.attend(&a, &mut serial);
            for threads in [1, 2, 3, 4, 8] {
                let mut threaded = vec![0.0f32; n_head * head_dim];
                Spin::new(threads).attend(&a, &mut threaded);
                for i in 0..serial.len() {
                    assert_eq!(
                        serial[i].to_bits(),
                        threaded[i].to_bits(),
                        "n_pos {n_pos}, {threads} threads, element {i}"
                    );
                }
            }
        }
    }

    /// Work stealing does not exist here, but the barrier still has to be
    /// deterministic across repeated runs.
    #[test]
    fn repeated_runs_agree_with_each_other() {
        let (bytes, x) = fixture(GgmlType::Q8_0, 128, 1024);
        let w = Weights {
            data: &bytes,
            ty: GgmlType::Q8_0,
            n_in: 128,
            n_out: 1024,
        };
        let ops = Spin::new(4);
        let mut first = vec![0.0f32; 1024];
        ops.matmul(&w, &x, &mut first);
        for _ in 0..16 {
            let mut again = vec![0.0f32; 1024];
            ops.matmul(&w, &x, &mut again);
            assert!(
                first.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits()),
                "spin matmul is not deterministic"
            );
        }
    }
}
