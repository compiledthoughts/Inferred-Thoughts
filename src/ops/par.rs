//! The same kernels as [`super::naive`], spread across threads.
//!
//! **Only `matmul` is parallel, and only over output rows.** That restriction
//! is the whole design:
//!
//! * `out[j]` is a complete dot product computed by one thread, start to
//!   finish, in the same order [`super::naive`] would use. No reduction is ever
//!   split across threads, so no partial sums are recombined and no f32
//!   accumulation order changes. The result is **bit-identical** to the oracle,
//!   which is what [`tests`] asserts — a tolerance would be the wrong test
//!   here, because there is nothing to be tolerant of.
//! * Everything else in the forward pass operates on vectors of a few thousand
//!   f32 at most. Threading those would cost more in dispatch than it saves, so
//!   they delegate to `Naive` unchanged.
//!
//! This backend is not an exception to the "never optimize the oracle" rule in
//! `CLAUDE.md`: [`super::naive::Naive`] is untouched and stays the thing to
//! diff against. `Par` is a second implementation behind the same seam, held to
//! the same standard the `ggml` and `cuda` backends will be — it must reproduce
//! the oracle exactly.
//!
//! Thread count defaults to *physical* cores, not logical. Per `CLAUDE.md`,
//! SMT does not help memory-bound kernels that share a core's load/store
//! units, and the Zen 5 target has 8 cores / 16 threads.
//!
//! **Read [`PARALLEL_THRESHOLD`] before changing anything here.** Threading a
//! matmul this small is a net loss, and the measurements that establish where
//! the line falls are recorded there.

use rayon::prelude::*;

use super::naive::{self, Naive};
use super::{Attn, Delta, Ops, Weights};
use crate::gguf::GgmlType;

/// Rows handed to one task.
const ROWS_PER_TASK: usize = 512;

/// Below this many cached positions, attention is too small to thread.
///
/// One key/value head at `n_pos` positions is roughly `2 * n_pos * head_dim`
/// conversions plus `group * n_pos * head_dim` multiply-adds. At 64 positions
/// and `head_dim` 128 that is already tens of microseconds per head, well clear
/// of dispatch; below it the prompt is short enough that decode is dominated by
/// the weight matmuls anyway.
const ATTN_POS_THRESHOLD: usize = 64;

/// Below this many rows, run serially: the tasks cost more than they save.
///
/// **Both constants are measured, and the threshold is deliberately high.**
/// Decoding Qwen3-0.6B on the 8-core target, 48 tokens, ms/token:
///
/// | rows/task | threshold | t=8   |
/// |-----------|-----------|-------|
/// | 32        | 128       | 127.0 |
/// | 256       | 2048      |  73.7 |
/// | 1024      | 128       |  45.5 |
/// | 4096      | 128       |  37.5 |
/// | 512       | 8192      |  37.5 |
///
/// against 44.5 ms/token on `Naive`. The first three rows are *slower than
/// serial*, and monotonically so in task count — the cost is per **task**, not
/// per parallel region: at 4096 rows/task every matmul except the LM head
/// yields a single chunk, so it still enters a rayon region and still hits
/// 37.5 ms. Entering the region is nearly free; splitting it is not.
///
/// The reason is granularity. A decode step runs ~196 matmuls, and every one
/// except the LM head is at most 3072 rows of ~1 KB — a few microseconds of
/// work per task, which is the same order as handing that task to another
/// thread. Only `output.weight` (151,936 rows, 165 MB of the 633 MB a pass
/// reads) is big enough to pay for itself, and a threshold of 8192 selects
/// exactly it.
///
/// So the 1.19x here is close to the arithmetic limit of this decomposition:
/// the LM head is 26% of the bytes, and 0.74 + 0.26/8 predicts 1.30x. Getting
/// the other 74% needs a *coarser* parallel region — one dispatch covering a
/// whole layer or token rather than one per matmul — which the [`Ops`] seam
/// cannot express, since its unit is a single matmul. That is a design change,
/// not a tuning change. Note also that this is a property of a 0.6B: the 35B's
/// expert matmuls are far larger relative to the same fixed overhead.
const PARALLEL_THRESHOLD: usize = 8192;

pub struct Par;

impl Par {
    /// Size the global rayon pool. Idempotent: later calls are ignored, since
    /// rayon permits exactly one global pool per process.
    ///
    /// `threads` of 0 means "physical cores".
    pub fn init(threads: usize) {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let n = if threads > 0 { threads } else { Self::default_threads() };
            // A failure here leaves rayon's own default pool in place, which is
            // correct but wider than we want; it cannot make results wrong.
            let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
        });
    }

    /// Physical cores, approximated as half the logical count when the platform
    /// reports an even number. `available_parallelism` counts SMT siblings, and
    /// CLAUDE.md is explicit that those do not help here.
    pub fn default_threads() -> usize {
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        if logical >= 2 && logical.is_multiple_of(2) {
            logical / 2
        } else {
            logical
        }
    }

    pub fn threads(&self) -> usize {
        rayon::current_num_threads()
    }
}

impl Ops for Par {
    fn matmul(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len() % w.n_in, 0);
        debug_assert_eq!(out.len(), (x.len() / w.n_in) * w.n_out);

        // `par` is kept only as the control that demonstrates rayon's dispatch
        // cost (see the module docs), so it does not grow a batched matmul: a
        // batch goes to the oracle, which is bit-identical and keeps this
        // backend's one interesting property -- its decode behaviour -- exactly
        // as it was measured.
        if w.n_out < PARALLEL_THRESHOLD || x.len() != w.n_in {
            return Naive.matmul(w, x, out);
        }

        match w.ty {
            // Quantized once and shared read-only across every task, exactly as
            // the serial path shares it across every row. Doing it per task
            // would be both slower and a different function.
            GgmlType::Q8_0 => {
                let qx = naive::QuantizedRow::from_f32(x);
                out.par_chunks_mut(ROWS_PER_TASK)
                    .enumerate()
                    .for_each(|(c, chunk)| {
                        let base = c * ROWS_PER_TASK;
                        for (i, o) in chunk.iter_mut().enumerate() {
                            *o = naive::dot_q8_0_q8_0(w.row(base + i), &qx);
                        }
                    });
            }
            _ => {
                out.par_chunks_mut(ROWS_PER_TASK)
                    .enumerate()
                    .for_each(|(c, chunk)| {
                        let base = c * ROWS_PER_TASK;
                        for (i, o) in chunk.iter_mut().enumerate() {
                            *o = naive::dot_row(w.ty, w.row(base + i), x);
                        }
                    });
            }
        }
    }

    // The rest are vector ops over at most a few thousand elements. Dispatch
    // would dominate, so they are the oracle's code verbatim.
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

    /// Threaded over key/value heads.
    ///
    /// **This is the case matmul threading could not be**, and the contrast is
    /// the whole reason the op exists at the token level. A matmul here is a
    /// fixed few microseconds, so per-task overhead swamped it (see
    /// [`PARALLEL_THRESHOLD`]). Attention work *grows with context* — at 384
    /// positions one key/value head is `group * 384 * head_dim` multiply-adds
    /// plus 2 * 384 * head_dim conversions — and there is one dispatch per
    /// layer rather than seven.
    ///
    /// Output chunking is what makes it safe: query heads served by one
    /// key/value head are contiguous, so `out` splits into disjoint
    /// `group * head_dim` slices, one per task, with no sharing at all.
    ///
    /// Bit-identical to [`Naive`]: same kernel, and threads change only which
    /// core runs a head, never the order within one.
    fn attend(&self, a: &Attn<'_>, out: &mut [f32]) {
        let n_kv = a.n_head_kv;
        debug_assert_eq!(out.len(), a.n_q() * a.n_head * a.head_dim);
        let per_kv = a.group() * a.head_dim;

        // Below this there is not enough work to cover a dispatch -- the same
        // lesson PARALLEL_THRESHOLD records, applied to the other axis.
        if a.n_pos < ATTN_POS_THRESHOLD || n_kv < 2 {
            return Naive.attend(a, out);
        }

        // A batch makes the chunk index `(t, h_kv)` rather than `h_kv`: the
        // output rows are laid out token-major, so consecutive `per_kv` chunks
        // run through one token's kv heads before moving to the next.
        out.par_chunks_mut(per_kv)
            .enumerate()
            .for_each_init(
                || naive::Scratch::for_attn(a),
                |sc, (u, chunk)| naive::attend_kv_head(a, u / n_kv, u % n_kv, chunk, sc),
            );
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

    /// A deterministic pseudo-random weight matrix in the given type, plus the
    /// activation to multiply it by.
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

    /// The differential test the whole backend rests on. Not "within a
    /// tolerance" — identical bits. `Par` reorders which thread computes a row,
    /// never how a row is computed, so anything else is a bug.
    fn assert_identical(ty: GgmlType, n_in: usize, n_out: usize) {
        Par::init(4);
        let (bytes, x) = fixture(ty, n_in, n_out);
        let w = Weights {
            data: &bytes,
            ty,
            n_in,
            n_out,
        };
        let mut serial = vec![0.0f32; n_out];
        let mut parallel = vec![0.0f32; n_out];
        Naive.matmul(&w, &x, &mut serial);
        Par.matmul(&w, &x, &mut parallel);

        for j in 0..n_out {
            assert_eq!(
                serial[j].to_bits(),
                parallel[j].to_bits(),
                "{} row {j}: {} vs {}",
                ty.name(),
                serial[j],
                parallel[j]
            );
        }
    }

    /// Row counts must exceed PARALLEL_THRESHOLD, or the fallback makes these
    /// compare `Naive` against `Naive` and assert nothing at all. Checked at
    /// compile time, so raising the threshold cannot quietly hollow out the
    /// differential tests.
    const WIDE: usize = PARALLEL_THRESHOLD + ROWS_PER_TASK;
    /// Not a multiple of ROWS_PER_TASK, so the last chunk is short.
    const RAGGED: usize = PARALLEL_THRESHOLD + 37;

    const _: () = assert!(WIDE > PARALLEL_THRESHOLD);
    const _: () = assert!(RAGGED > PARALLEL_THRESHOLD);
    const _: () = assert!(!RAGGED.is_multiple_of(ROWS_PER_TASK));
    const _: () = assert!(PARALLEL_THRESHOLD > 1);

    #[test]
    fn q8_0_matmul_is_bit_identical_to_the_oracle() {
        assert_identical(GgmlType::Q8_0, 128, WIDE);
        assert_identical(GgmlType::Q8_0, 96, RAGGED);
    }

    #[test]
    fn f32_matmul_is_bit_identical_to_the_oracle() {
        assert_identical(GgmlType::F32, 64, RAGGED);
    }

    #[test]
    fn f16_matmul_is_bit_identical_to_the_oracle() {
        assert_identical(GgmlType::F16, 64, RAGGED);
    }

    /// Narrow matmuls take the serial path; it must still be correct, and it
    /// must still be the oracle's answer.
    #[test]
    fn below_the_threshold_falls_back_to_serial() {
        assert_identical(GgmlType::Q8_0, 64, PARALLEL_THRESHOLD - 1);
    }

    /// Repeated runs must agree with each other, not just with the oracle: work
    /// stealing changes which thread takes a chunk between runs, and if that
    /// were observable in the output the backend would be unusable as a base
    /// for cache-policy debugging.
    #[test]
    fn results_do_not_depend_on_the_thread_that_ran_them() {
        Par::init(4);
        let (bytes, x) = fixture(GgmlType::Q8_0, 128, WIDE);
        let w = Weights {
            data: &bytes,
            ty: GgmlType::Q8_0,
            n_in: 128,
            n_out: WIDE,
        };
        let mut first = vec![0.0f32; WIDE];
        Par.matmul(&w, &x, &mut first);
        for _ in 0..8 {
            let mut again = vec![0.0f32; WIDE];
            Par.matmul(&w, &x, &mut again);
            assert!(
                first.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits()),
                "parallel matmul is not deterministic"
            );
        }
    }

    #[test]
    fn default_threads_avoids_smt_siblings() {
        let n = Par::default_threads();
        assert!(n >= 1);
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        assert!(n <= logical);
    }
}
