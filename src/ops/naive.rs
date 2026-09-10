//! Scalar f32 reference implementation.
//!
//! **This is not the engine; it is the oracle we own.** When a cache or offload
//! policy later produces wrong output, this is what to diff against —
//! something whose semantics we control, rather than llama.cpp's very different
//! execution path.
//!
//! So: no SIMD, no threads, no `unsafe`, and no cleverness. Correctness and
//! legibility only. Per `CLAUDE.md`, never optimize this module.
//!
//! Its dot products are `pub(crate)` so [`super::par`] can call the *same*
//! kernels across threads rather than growing a second copy that could drift.
//! The parallel backend changes only which thread runs a row, never how a row
//! is computed, which is what lets its differential test demand bit equality
//! instead of a tolerance.

// The "never optimize the oracle" rule in CLAUDE.md, enforced by the compiler
// rather than by convention. Threading, SIMD and CUDA all live beside this
// module, never inside it.
#![forbid(unsafe_code)]

use super::{Attn, Delta, Ops, Weights};
use crate::gguf::GgmlType;
use crate::quant::half::{f16_to_f32, f32_to_f16};

pub struct Naive;

/// Elements per Q8_0 block (`QK8_0` in ggml-common.h).
pub(crate) const QK8_0: usize = 32;

impl Naive {
    /// One token against every weight row — the body the batched
    /// [`Ops::matmul`] loops over.
    ///
    /// Split out rather than inlined so the batch loop reads as "do the
    /// single-token thing `n` times", which is exactly the claim the
    /// differential tests check.
    fn matmul_row(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), w.n_in);
        debug_assert_eq!(out.len(), w.n_out);

        match w.ty {
            // ggml does not multiply Q8_0 weights by f32 activations. Its
            // `type_traits_cpu[GGML_TYPE_Q8_0].vec_dot_type` is Q8_0, so the
            // activation is quantized first and the dot product is integer.
            // That is a different function from exact f32 arithmetic, and
            // matching it is what makes this a usable oracle. It is also what
            // a SIMD backend will do, so this stays a faithful scalar model.
            GgmlType::Q8_0 => {
                // Quantized once per token and shared across all rows, as in
                // ggml_compute_forward_mul_mat.
                let qx = QuantizedRow::from_f32(x);
                for j in 0..w.n_out {
                    out[j] = dot_q8_0_q8_0(w.row(j), &qx);
                }
            }
            // The k-quants, which the 35B needs everywhere: IQ4_XS on every
            // routed expert, Q5_K on `attn_output`, Q6_K on `attn_q` and the LM
            // head. ggml pairs all three with a **Q8_K** activation rather than
            // f32, so the row is quantized once and shared across every weight
            // row, exactly as the Q8_0 arm above does with its own format.
            GgmlType::Q5K | GgmlType::Q6K | GgmlType::Iq4Xs => {
                let qx = crate::quant::Q8KRow::from_f32(x);
                for j in 0..w.n_out {
                    out[j] = crate::quant::kquant::dot_row_q8_k(w.ty, w.row(j), &qx);
                }
            }
            _ => {
                for j in 0..w.n_out {
                    out[j] = dot_row(w.ty, w.row(j), x);
                }
            }
        }
    }
    /// One token's delta-rule update — the body the batched [`Ops::delta_rule`]
    /// applies in order.
    fn delta_rule_row(&self, d: &Delta<'_>, state: &mut [f32], out: &mut [f32]) {
        debug_assert_eq!(out.len(), d.n_v_heads * d.head_v_dim);
        debug_assert_eq!(d.v.len(), d.n_v_heads * d.head_v_dim);
        debug_assert_eq!(d.q.len(), d.n_k_heads * d.head_k_dim);

        let (sk, sv) = (d.head_k_dim, d.head_v_dim);
        let q_scale = d.scale();

        for h in 0..d.n_v_heads {
            let kh = d.key_head(h);
            let q = &d.q[kh * sk..(kh + 1) * sk];
            let k = &d.k[kh * sk..(kh + 1) * sk];
            let v = &d.v[h * sv..(h + 1) * sv];
            let s = &mut state[h * sk * sv..(h + 1) * sk * sv];

            // exp(softplus(alpha + dt) * ssm_a). The 20.0 cutoff is the
            // reference's (ggml_compute_softplus_f32), not a guard added here:
            // above it, log(1 + exp(x)) is x to f32 precision anyway.
            let a = d.alpha[h] + d.dt_bias[h];
            let softplus = if a > 20.0 { a } else { (1.0 + a.exp()).ln() };
            let g = (softplus * d.ssm_a[h]).exp();
            let beta = 1.0 / (1.0 + (-d.beta[h]).exp());

            for x in s.iter_mut() {
                *x *= g;
            }

            // Row j of the state holds the value axis; the key axis is
            // contiguous within it, which makes all three loops below
            // sequential reads and matches ggml's `ne[0]` being the axis that
            // `sum_rows` contracts over.
            for j in 0..sv {
                let row = &mut s[j * sk..(j + 1) * sk];
                let mut pred = 0.0f32;
                for i in 0..sk {
                    pred += row[i] * k[i];
                }
                let delta = beta * (v[j] - pred);
                for i in 0..sk {
                    row[i] += k[i] * delta;
                }
                let mut o = 0.0f32;
                for i in 0..sk {
                    o += row[i] * (q[i] * q_scale);
                }
                out[h * sv + j] = o;
            }
        }
    
    }
}

impl Ops for Naive {
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        let nd = weight.len();
        debug_assert_eq!(x.len() % nd, 0);
        debug_assert_eq!(x.len(), out.len());

        // Each row normalizes against its own mean. Batching cannot change a
        // bit: no accumulation crosses a row boundary.
        for (row, o) in x.chunks_exact(nd).zip(out.chunks_exact_mut(nd)) {
            let scale = rms_scale(row, eps);
            for i in 0..nd {
                o[i] = row[i] * scale * weight[i];
            }
        }
    }

    fn rms_norm_heads(&self, x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
        debug_assert_eq!(weight.len(), head_dim);
        debug_assert_eq!(x.len() % head_dim, 0);

        for head in x.chunks_exact_mut(head_dim) {
            let scale = rms_scale(head, eps);
            for i in 0..head_dim {
                head[i] = head[i] * scale * weight[i];
            }
        }
    }

    fn matmul(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len() % w.n_in, 0);
        debug_assert_eq!(out.len(), (x.len() / w.n_in) * w.n_out);

        // The oracle walks the batch token by token, so every dot product is
        // the one the single-token path computed, in the same order. A backend
        // that instead reads each weight row once for all tokens is still
        // bit-identical -- it reorders which outputs are computed together, not
        // how any one of them accumulates.
        for (xt, ot) in x.chunks_exact(w.n_in).zip(out.chunks_exact_mut(w.n_out)) {
            self.matmul_row(w, xt, ot);
        }
    }

    fn rope_neox(
        &self,
        x: &mut [f32],
        pos: usize,
        head_dim: usize,
        n_rot: usize,
        n_heads: usize,
        theta_base: f32,
    ) {
        let per_row = head_dim * n_heads;
        debug_assert_eq!(x.len() % per_row, 0);
        debug_assert_eq!(n_rot % 2, 0);
        debug_assert!(n_rot <= head_dim);

        // Only the first `n_rot` of each head rotate; the rest pass through.
        // The frequency divides by `n_rot`, not `head_dim` -- ggml's rope_yarn
        // takes `theta_scale = powf(freq_base, -2/n_dims)` with n_dims = n_rot.
        let half = n_rot / 2;
        // `pos` is row 0's absolute position; rows are consecutive. Rotating
        // every row at `pos` would be the classic KV cache bug, invisible in a
        // prefill from zero and wrong for everything decoded after.
        for (t, row) in x.chunks_exact_mut(per_row).enumerate() {
            let pos = pos + t;
            for head in row.chunks_exact_mut(head_dim) {
            for i in 0..half {
                let freq = (theta_base as f64).powf(-2.0 * i as f64 / n_rot as f64);
                let theta = pos as f64 * freq;
                let (sin, cos) = theta.sin_cos();
                let (sin, cos) = (sin as f32, cos as f32);

                // NEOX pairs i with i + head_dim/2.
                let x0 = head[i];
                let x1 = head[i + half];
                head[i] = x0 * cos - x1 * sin;
                head[i + half] = x0 * sin + x1 * cos;
            }
            }
        }
    }

    fn softmax(&self, x: &mut [f32], row: usize) {
        if row == 0 {
            return;
        }
        for r in x.chunks_mut(row) {
            softmax_in_place(r);
        }
    }

    fn attend(&self, a: &Attn<'_>, out: &mut [f32]) {
        debug_assert_eq!(out.len(), a.n_q() * a.n_head * a.head_dim);
        let per_kv = a.group() * a.head_dim;
        // One scratch for the whole call, not one per head or per row.
        let mut sc = Scratch::for_attn(a);
        // Query rows are independent given the cache — the batch is published
        // before any of this runs — so `(t, h_kv)` is a grid of independent
        // outputs. A threaded backend gets a wider one during prefill than it
        // ever has in decode.
        for t in 0..a.n_q() {
            let row = &mut out[t * a.n_head * a.head_dim..][..a.n_head * a.head_dim];
            for (h_kv, chunk) in row.chunks_mut(per_kv).enumerate() {
                attend_kv_head(a, t, h_kv, chunk, &mut sc);
            }
        }
    }

    fn silu_mul(&self, gate: &mut [f32], up: &[f32]) {
        debug_assert_eq!(gate.len(), up.len());
        for i in 0..gate.len() {
            let g = gate[i];
            gate[i] = g / (1.0 + (-g).exp()) * up[i];
        }
    }

    fn l2_norm_heads(&self, x: &mut [f32], head_dim: usize, eps: f32) {
        debug_assert_eq!(x.len() % head_dim, 0);

        for head in x.chunks_exact_mut(head_dim) {
            // Transcribed from ggml_compute_forward_l2_norm_f32. Two details
            // that a glance at RMSNorm would get wrong: the sum is not divided
            // by n, and eps clamps the norm from below instead of being added
            // under the root. The f64 accumulator is the reference's
            // `ggml_float`, and `sqrtf` takes a float, so the narrowing before
            // the root is the reference's too and not an accident here.
            let mut sum = 0.0f64;
            for &v in head.iter() {
                sum += f64::from(v * v);
            }
            let scale = 1.0f32 / (sum as f32).sqrt().max(eps);
            for v in head.iter_mut() {
                *v *= scale;
            }
        }
    }

    fn ssm_conv(
        &self,
        state: &mut [f32],
        x: &[f32],
        weight: &[f32],
        kernel: usize,
        out: &mut [f32],
    ) {
        let keep = kernel - 1;
        let nc = weight.len() / kernel;
        debug_assert_eq!(state.len(), nc * keep);
        debug_assert_eq!(x.len(), out.len());
        debug_assert_eq!(x.len() % nc, 0);

        // Sequential in the batch: token `t` convolves over the window token
        // `t-1` advanced, so the tokens cannot be reordered.
        for (xt, ot) in x.chunks_exact(nc).zip(out.chunks_exact_mut(nc)) {
        for (c, o) in ot.iter_mut().enumerate() {
            let past = &mut state[c * keep..(c + 1) * keep];

            // f32, not f64. ggml_compute_forward_ssm_conv_f32 says outright
            // that it avoids ggml_vec_dot_f32 "because its sum is in double
            // precision", so accumulating wider here would make the oracle
            // disagree with the reference it exists to reproduce.
            //
            // The window is the stored samples oldest-first, then this token,
            // so tap `keep` is always the newest sample and never comes from
            // the state.
            let mut sum = 0.0f32;
            for t in 0..keep {
                sum += past[t] * weight[c * kernel + t];
            }
            sum += xt[c] * weight[c * kernel + keep];
            *o = sum / (1.0 + (-sum).exp()); // silu, fused in as the reference does

            // Advance: drop the oldest, append this token.
            past.rotate_left(1);
            past[keep - 1] = xt[c];
        }
        }
    }

    fn gather_chunks(
        &self,
        src: &[f32],
        chunk: usize,
        stride: usize,
        offset: usize,
        out: &mut [f32],
    ) {
        debug_assert_eq!(out.len() % chunk, 0);
        for (c, dst) in out.chunks_exact_mut(chunk).enumerate() {
            let at = c * stride + offset;
            dst.copy_from_slice(&src[at..at + chunk]);
        }
    }

    fn sigmoid_mul(&self, x: &mut [f32], g: &[f32]) {
        debug_assert_eq!(x.len(), g.len());
        for i in 0..x.len() {
            x[i] *= 1.0 / (1.0 + (-g[i]).exp());
        }
    }

    fn delta_rule(&self, d: &Delta<'_>, state: &mut [f32], out: &mut [f32]) {
        debug_assert_eq!(state.len(), d.n_v_heads * d.state_per_head());
        debug_assert_eq!(out.len(), d.n_tokens() * d.n_v_heads * d.head_v_dim);

        // The one op with no batched form: token `t`'s rank-1 correction is
        // token `t+1`'s stored state. `Delta::row` is shared with every other
        // backend so the striding cannot drift from the oracle's.
        let per_token = d.n_v_heads * d.head_v_dim;
        for t in 0..d.n_tokens() {
            self.delta_rule_row(&d.row(t), state, &mut out[t * per_token..(t + 1) * per_token]);
        }
    }

    fn add_assign(&self, a: &mut [f32], b: &[f32]) {
        debug_assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            a[i] += b[i];
        }
    }
}

/// Numerically stable softmax. Shared so [`attend_kv_head`] runs the same code
/// the trait method does, rather than a second copy that could drift.
pub(crate) fn softmax_in_place(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    // Subtract the max before exponentiating, or a large logit overflows.
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Reusable buffers for one key/value head's attention.
///
/// Sized once and reused across positions. `par` keeps one per worker thread,
/// so a threaded run allocates per thread rather than per head.
pub(crate) struct Scratch {
    /// One score row per query head in the group: `group * n_pos`.
    scores: Vec<f32>,
    /// One position's key or value head, converted to f32: `head_dim`.
    conv: Vec<f32>,
}

impl Scratch {
    pub(crate) fn for_attn(a: &Attn<'_>) -> Self {
        Self {
            scores: vec![0.0; a.group() * a.n_pos],
            conv: vec![0.0; a.head_dim],
        }
    }

    /// Grow to fit; context only ever increases within a run.
    fn fit(&mut self, a: &Attn<'_>) {
        self.scores.resize(a.group() * a.n_pos, 0.0);
        self.conv.resize(a.head_dim, 0.0);
    }
}

/// Attention for the `group` query heads served by key/value head `h_kv`.
///
/// **Structured around the f16 conversion, which was the bottleneck.** The old
/// inline loop converted K and V one element at a time inside the dot product,
/// once per *query* head — so with GQA it converted the same key data `group`
/// times, ~44 million branchy scalar conversions per token at 384 positions.
/// Here each position is converted once into [`Scratch::conv`] and reused
/// across the group, over a contiguous run the compiler can vectorize.
///
/// **This does not change a single output bit.** f16 -> f32 is exact — every
/// f16 value is representable in f32, with no rounding — so hoisting the
/// conversion cannot alter a value. The dot product still accumulates serially
/// in index order, and the weighted sum still walks positions outermost, which
/// is what keeps `par` bit-identical to this. Breaking the accumulator
/// dependency chain *would* change the order, and is deliberately not done
/// here.
pub(crate) fn attend_kv_head(
    a: &Attn<'_>,
    t: usize,
    h_kv: usize,
    out: &mut [f32],
    sc: &mut Scratch,
) {
    sc.fit(a);
    // Query row `t` sees only its own history: the causal mask *is* this
    // number. In decode `n_q == 1` and it is `a.n_pos`, unchanged.
    let (hd, group, n_pos) = (a.head_dim, a.group(), a.n_pos_of(t));
    let off = h_kv * hd;
    let q_row = t * a.n_head * hd;
    debug_assert_eq!(out.len(), group * hd);

    // Pass 1: scores. Convert each position's key once, score it against every
    // query head in the group.
    for s in 0..n_pos {
        let key = &a.k[s * a.kv_dim + off..][..hd];
        for (dst, &bits) in sc.conv.iter_mut().zip(key) {
            *dst = f16_to_f32(bits);
        }
        for g in 0..group {
            let q = &a.q[q_row + (h_kv * group + g) * hd..][..hd];
            let dot: f32 = q.iter().zip(&sc.conv).map(|(x, y)| x * y).sum();
            sc.scores[g * n_pos + s] = dot * a.scale;
        }
    }
    for g in 0..group {
        softmax_in_place(&mut sc.scores[g * n_pos..(g + 1) * n_pos]);
    }

    // Pass 2: weighted sum of values, same conversion trick.
    out.fill(0.0);
    for s in 0..n_pos {
        let val = &a.v[s * a.kv_dim + off..][..hd];
        for (dst, &bits) in sc.conv.iter_mut().zip(val) {
            *dst = f16_to_f32(bits);
        }
        for g in 0..group {
            let w = sc.scores[g * n_pos + s];
            for (o, &vi) in out[g * hd..(g + 1) * hd].iter_mut().zip(&sc.conv) {
                *o += w * vi;
            }
        }
    }
}

/// The `1/sqrt(mean(x^2) + eps)` factor of RMSNorm.
///
/// Transcribed from `ggml_compute_forward_rms_norm_f32` in
/// `ggml/src/ggml-cpu/ops.cpp`:
///
/// ```c
/// ggml_float sum = 0.0;                            // ggml_float is double
/// for (i00) sum += (ggml_float)(x[i00] * x[i00]);  // f32 square, f64 accumulate
/// const float mean  = sum/ne00;
/// const float scale = 1.0f/sqrtf(mean + eps);
/// ```
///
/// The **double accumulator is load-bearing**. Summing 1024 squares in f32
/// shifts the scale by ~1e-5 relative, which is invisible in a printed tensor
/// but is enough to move activations across Q8_0 quantization boundaries in
/// every matmul downstream. The square itself stays f32, as in the reference.
fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let mut sum = 0.0f64;
    for &v in x {
        sum += f64::from(v * v);
    }
    let mean = (sum / x.len() as f64) as f32;
    1.0f32 / (mean + eps).sqrt()
}

/// An activation vector quantized to Q8_0, one scale per 32 elements.
///
/// Mirrors `quantize_row_q8_0_ref` in `ggml/src/ggml-quants.c`:
/// `d = amax/127`, `q = roundf(x/d)`, with `d` stored as f16 and read back —
/// so `scales` holds the f16-rounded value, not the exact one.
pub(crate) struct QuantizedRow {
    scales: Vec<f32>,
    quants: Vec<i8>,
}

impl QuantizedRow {
    pub(crate) fn from_f32(x: &[f32]) -> Self {
        let n_blocks = x.len() / QK8_0;
        let mut scales = Vec::with_capacity(n_blocks);
        let mut quants = Vec::with_capacity(x.len());

        for block in x.chunks_exact(QK8_0) {
            let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };

            // The scale round-trips through f16 before use, exactly as the
            // reference stores and reloads it.
            scales.push(f16_to_f32(f32_to_f16(d)));

            for &v in block {
                // roundf is round-half-away-from-zero, which is also what
                // Rust's f32::round does.
                quants.push((v * id).round() as i8);
            }
        }

        Self { scales, quants }
    }
}

/// `ggml_vec_dot_q8_0_q8_0`: per block, an integer sum of products scaled by
/// the product of the two f16 scales, accumulated in f32.
pub(crate) fn dot_q8_0_q8_0(row: &[u8], x: &QuantizedRow) -> f32 {
    let mut sumf = 0.0f32;
    for (i, block) in row.chunks_exact(2 + QK8_0).enumerate() {
        let dw = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qs = &x.quants[i * QK8_0..(i + 1) * QK8_0];

        let mut sumi = 0i32;
        for (&w, &a) in block[2..].iter().zip(qs) {
            sumi += (w as i8) as i32 * a as i32;
        }
        sumf += sumi as f32 * (dw * x.scales[i]);
    }
    sumf
}

/// Dot product of one quantized weight row with an f32 activation.
///
/// Used for the unquantized types, where ggml also works directly in f32.
/// Dequantizes a block at a time rather than materializing the row.
pub(crate) fn dot_row(ty: GgmlType, row: &[u8], x: &[f32]) -> f32 {
    match ty {
        GgmlType::F32 => row
            .chunks_exact(4)
            .zip(x)
            .map(|(c, &xi)| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * xi)
            .sum(),

        GgmlType::F16 => row
            .chunks_exact(2)
            .zip(x)
            .map(|(c, &xi)| f16_to_f32(u16::from_le_bytes([c[0], c[1]])) * xi)
            .sum(),

        // Unreachable in v0: the model loader rejects unsupported types when it
        // builds the weight views, so this cannot be hit from the forward pass.
        other => panic!("matmul over unsupported type {}", other.name()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q8_0_row(values: &[f32]) -> (Vec<u8>, Vec<f32>) {
        // Quantize like ggml: scale = max|x| / 127, round to nearest.
        let mut bytes = Vec::new();
        let mut exact = Vec::new();
        for block in values.chunks(QK8_0) {
            let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d == 0.0 { 0.0 } else { 1.0 / d };
            // Store the f16 scale, then round-trip it so the expected values
            // reflect the precision actually stored.
            let d_f16 = f32_to_f16(d);
            let d_actual = f16_to_f32(d_f16);
            bytes.extend_from_slice(&d_f16.to_le_bytes());
            for &v in block {
                let q = (v * id).round().clamp(-127.0, 127.0) as i8;
                bytes.push(q as u8);
                exact.push(q as f32 * d_actual);
            }
        }
        (bytes, exact)
    }

    #[test]
    fn rms_norm_matches_definition() {
        let x = [3.0f32, 4.0];
        let w = [1.0f32, 1.0];
        let mut out = [0.0f32; 2];
        Naive.rms_norm(&x, &w, 0.0, &mut out);
        // rms = sqrt((9+16)/2) = 3.5355...
        let rms = ((9.0f32 + 16.0) / 2.0).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-6);
        assert!((out[1] - 4.0 / rms).abs() < 1e-6);
    }

    #[test]
    fn rms_norm_applies_weight_without_adding_one() {
        // Gemma uses (1 + w); Qwen does not. A weight of 0 must zero the output.
        let x = [1.0f32, 2.0];
        let mut out = [9.0f32; 2];
        Naive.rms_norm(&x, &[0.0, 0.0], 1e-6, &mut out);
        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn rms_norm_heads_normalizes_each_head_independently() {
        // Two heads of 2, with very different magnitudes. If normalization
        // leaked across heads, the second would not come back to the same
        // values as the first.
        let mut x = [3.0f32, 4.0, 300.0, 400.0];
        Naive.rms_norm_heads(&mut x, &[1.0, 1.0], 2, 0.0);
        assert!((x[0] - x[2]).abs() < 1e-4, "{x:?}");
        assert!((x[1] - x[3]).abs() < 1e-4, "{x:?}");
    }

    #[test]
    fn rope_neox_pairs_across_the_half_boundary() {
        // head_dim 4 => pairs are (0,2) and (1,3). At i=0 the frequency is 1,
        // so pos=1 rotates (x0, x2) by exactly 1 radian.
        let mut x = [1.0f32, 0.0, 0.0, 0.0];
        Naive.rope_neox(&mut x, 1, 4, 4, 1, 10000.0);
        assert!((x[0] - 1.0f32.cos()).abs() < 1e-6, "{x:?}");
        assert!((x[2] - 1.0f32.sin()).abs() < 1e-6, "{x:?}");
        // If this were the adjacent-pair variant, index 1 would have moved.
        assert_eq!(x[1], 0.0);
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let before = x;
        Naive.rope_neox(&mut x, 0, 8, 8, 1, 1_000_000.0);
        for (a, b) in x.iter().zip(&before) {
            assert!((a - b).abs() < 1e-6, "{x:?}");
        }
    }

    #[test]
    fn rope_preserves_pair_magnitude() {
        // Rotation is orthogonal, so each pair's norm must be unchanged.
        let mut x: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01 - 0.5).collect();
        let before = x.clone();
        Naive.rope_neox(&mut x, 37, 128, 128, 1, 1_000_000.0);
        for i in 0..64 {
            let n0 = before[i].hypot(before[i + 64]);
            let n1 = x[i].hypot(x[i + 64]);
            assert!((n0 - n1).abs() < 1e-5, "pair {i}: {n0} vs {n1}");
        }
    }

    #[test]
    fn rope_treats_heads_independently() {
        let mut two = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        Naive.rope_neox(&mut two, 5, 4, 4, 2, 10000.0);
        assert_eq!(two[0..4], two[4..8]);
    }

    #[test]
    fn softmax_sums_to_one_and_survives_large_inputs() {
        let mut x = [1.0f32, 2.0, 3.0];
        let n = x.len();
        Naive.softmax(&mut x, n);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[2] > x[1] && x[1] > x[0]);

        // Without the max subtraction this overflows to NaN.
        let mut big = [1000.0f32, 1001.0];
        let n = big.len();
        Naive.softmax(&mut big, n);
        assert!(big.iter().all(|v| v.is_finite()), "{big:?}");
        assert!((big.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_norm_divides_by_the_norm_not_the_rms() {
        // The whole reason this op exists separately from rms_norm. For [3, 4]
        // the L2 norm is 5, so the result is [0.6, 0.8] and has unit norm.
        // RMSNorm would divide by sqrt(mean) = 3.5355 and give [0.849, 1.13],
        // a vector of norm sqrt(2). Confusing the two produces plausible
        // numbers that are wrong by exactly sqrt(n).
        let mut x = [3.0f32, 4.0];
        Naive.l2_norm_heads(&mut x, 2, 0.0);
        assert!((x[0] - 0.6).abs() < 1e-6, "{x:?}");
        assert!((x[1] - 0.8).abs() < 1e-6, "{x:?}");
        assert!((x[0] * x[0] + x[1] * x[1] - 1.0).abs() < 1e-6, "unit norm");
    }

    #[test]
    fn l2_norm_eps_clamps_the_norm_from_below() {
        // eps bounds the norm itself, so a tiny vector is scaled by 1/eps
        // rather than exploding. RMSNorm puts eps under the root instead and
        // would not produce this value.
        let mut x = [1e-6f32, 0.0];
        Naive.l2_norm_heads(&mut x, 2, 1e-3);
        assert!((x[0] - 1e-3).abs() < 1e-9, "{x:?}");
    }

    #[test]
    fn l2_norm_treats_each_head_independently() {
        let mut x = [3.0f32, 4.0, 30.0, 40.0];
        Naive.l2_norm_heads(&mut x, 2, 0.0);
        assert!((x[0] - x[2]).abs() < 1e-6, "{x:?}");
        assert!((x[1] - x[3]).abs() < 1e-6, "{x:?}");
    }

    #[test]
    fn ssm_conv_is_depthwise_and_reads_oldest_first() {
        // Two channels, kernel 3. Channel 0 has window [1,2,3] oldest-first
        // against weights [100,10,1], giving 100+20+3 = 123. That pins the tap
        // order: newest-first would give 1+20+300 = 321.
        //
        // Channel 1 is all zeros, so any cross-channel mixing would show up as
        // an output other than silu(0) = 0.
        // State is [c0: 1,2 | c1: 0,0] and this token is [3, 0], so the
        // windows are [1,2,3] and [0,0,0].
        let mut state = [1.0f32, 2.0, 0.0, 0.0];
        let x = [3.0f32, 0.0];
        let weight = [100.0f32, 10.0, 1.0, 100.0, 10.0, 1.0];
        let mut out = [0.0f32; 2];
        Naive.ssm_conv(&mut state, &x, &weight, 3, &mut out);
        let silu = |x: f32| x / (1.0 + (-x).exp());
        assert!((out[0] - silu(123.0)).abs() < 1e-3, "{out:?}");
        assert_eq!(out[1], 0.0, "depthwise: channel 1 saw only its own zeros");

        // And the state advanced: oldest dropped, this token appended.
        assert_eq!(state, [2.0, 3.0, 0.0, 0.0], "state must advance in place");
    }

    /// A single-head [`Delta`] for tests to perturb.
    fn delta_fixture<'a>(
        q: &'a [f32],
        k: &'a [f32],
        v: &'a [f32],
        alpha: &'a [f32],
        beta: &'a [f32],
        ssm_a: &'a [f32],
        dt: &'a [f32],
        dim: usize,
    ) -> Delta<'a> {
        Delta {
            q,
            k,
            v,
            alpha,
            beta,
            ssm_a,
            dt_bias: dt,
            head_k_dim: dim,
            head_v_dim: dim,
            n_k_heads: 1,
            n_v_heads: 1,
        }
    }

    #[test]
    fn delta_rule_writes_a_value_it_can_read_back() {
        // From an empty state with beta = 1 and no decay (ssm_a = 0, so the
        // gate is exp(0) = 1), one step stores v at key k. Reading with q = k
        // returns v scaled by |k|^2 / sqrt(d), which for a unit k and d = 4 is
        // v/2.
        //
        // This is the test that catches a transposed state: writing at [i][j]
        // and reading at [j][i] returns zero here rather than a scaled v.
        let k = [0.5f32, 0.5, 0.5, 0.5];
        let v = [1.0f32, 2.0, 3.0, 4.0];
        let (alpha, beta, ssm_a, dt) = ([0.0f32], [40.0f32], [0.0f32], [0.0f32]);
        let d = delta_fixture(&k, &k, &v, &alpha, &beta, &ssm_a, &dt, 4);
        let mut state = vec![0.0f32; 16];
        let mut out = vec![0.0f32; 4];
        Naive.delta_rule(&d, &mut state, &mut out);
        for j in 0..4 {
            assert!((out[j] - v[j] * 0.5).abs() < 1e-4, "{out:?}");
        }
    }

    #[test]
    fn delta_rule_forgets_at_the_rate_the_gate_says() {
        // Store, then take a step whose beta is ~0 so nothing new is written
        // and whose gate is exp(softplus(0) * -1) = exp(-ln 2) = 0.5. The
        // readout must halve.
        let k = [0.5f32, 0.5, 0.5, 0.5];
        let v = [1.0f32, 2.0, 3.0, 4.0];
        let mut state = vec![0.0f32; 16];
        let mut out = vec![0.0f32; 4];

        let (a0, b0, s0, dt) = ([0.0f32], [40.0f32], [0.0f32], [0.0f32]);
        let write = delta_fixture(&k, &k, &v, &a0, &b0, &s0, &dt, 4);
        Naive.delta_rule(&write, &mut state, &mut out);
        let first = out.clone();

        let (a1, b1, s1) = ([0.0f32], [-40.0f32], [-1.0f32]);
        let zeros = [0.0f32; 4];
        let decay = delta_fixture(&k, &k, &zeros, &a1, &b1, &s1, &dt, 4);
        Naive.delta_rule(&decay, &mut state, &mut out);
        for j in 0..4 {
            assert!((out[j] - first[j] * 0.5).abs() < 1e-3, "{out:?} vs {first:?}");
        }
    }

    #[test]
    fn delta_rule_groups_value_heads_by_modulo_not_division() {
        // Four value heads over two key heads. Modulo maps 0,1,2,3 to key
        // 0,1,0,1; division would map them to 0,0,1,1. The two disagree on
        // heads 1 and 2, which is what this pins.
        let q = [1.0f32, 0.0, 0.0, 1.0];
        let v = [1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let alpha = [0.0f32; 4];
        let beta = [40.0f32; 4];
        let ssm_a = [0.0f32; 4];
        let dt = [0.0f32; 4];
        let d = Delta {
            q: &q,
            k: &q,
            v: &v,
            alpha: &alpha,
            beta: &beta,
            ssm_a: &ssm_a,
            dt_bias: &dt,
            head_k_dim: 2,
            head_v_dim: 2,
            n_k_heads: 2,
            n_v_heads: 4,
        };
        assert_eq!(
            (0..4).map(|h| d.key_head(h)).collect::<Vec<_>>(),
            vec![0, 1, 0, 1],
            "value to key head must tile, matching ggml_repeat and iv1 % neq1"
        );
        let mut state = vec![0.0f32; 4 * 4];
        let mut out = vec![0.0f32; 8];
        Naive.delta_rule(&d, &mut state, &mut out);
        assert!(out[2] > 0.0, "head 1 read back nothing: {out:?}");
        assert!((out[0] - out[2]).abs() < 1e-6, "heads 0 and 1 are symmetric");
    }

    #[test]
    fn delta_rule_corrects_rather_than_accumulates() {
        // Write v twice at the same key with beta = 1 and no decay. A pure
        // accumulator would double the stored value; the delta rule subtracts
        // what it already predicts, so the second write is a no-op.
        let k = [0.5f32, 0.5, 0.5, 0.5];
        let v = [1.0f32, 2.0, 3.0, 4.0];
        let (alpha, beta, ssm_a, dt) = ([0.0f32], [40.0f32], [0.0f32], [0.0f32]);
        let d = delta_fixture(&k, &k, &v, &alpha, &beta, &ssm_a, &dt, 4);
        let mut state = vec![0.0f32; 16];
        let (mut a, mut b) = (vec![0.0f32; 4], vec![0.0f32; 4]);
        Naive.delta_rule(&d, &mut state, &mut a);
        Naive.delta_rule(&d, &mut state, &mut b);
        for j in 0..4 {
            assert!((a[j] - b[j]).abs() < 1e-4, "second write moved it: {a:?} {b:?}");
        }
    }

    #[test]
    fn silu_mul_matches_definition() {
        let mut gate = [1.0f32, -1.0];
        Naive.silu_mul(&mut gate, &[2.0, 2.0]);
        let silu = |g: f32| g / (1.0 + (-g).exp());
        assert!((gate[0] - silu(1.0) * 2.0).abs() < 1e-6);
        assert!((gate[1] - silu(-1.0) * 2.0).abs() < 1e-6);
    }

    #[test]
    fn matmul_f32_is_row_times_vector() {
        // Two rows of length 3: [1,2,3] and [4,5,6]; x = [1,1,1].
        let vals: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let w = Weights {
            data: &bytes,
            ty: GgmlType::F32,
            n_in: 3,
            n_out: 2,
            pooled: false,
        };
        let mut out = [0.0f32; 2];
        Naive.matmul(&w, &[1.0, 1.0, 1.0], &mut out);
        assert_eq!(out, [6.0, 15.0]);
    }

    /// The activation is quantized to Q8_0 before the dot product, so the
    /// result must match a hand-model of ggml's algorithm — not exact f32.
    #[test]
    fn matmul_q8_0_matches_ggml_semantics() {
        let vals: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect();
        let (bytes, _) = q8_0_row(&vals);
        let x: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.11).cos()).collect();

        let w = Weights {
            data: &bytes,
            ty: GgmlType::Q8_0,
            n_in: 64,
            n_out: 1,
            pooled: false,
        };
        let mut out = [0.0f32];
        Naive.matmul(&w, &x, &mut out);

        // Independent restatement of ggml_vec_dot_q8_0_q8_0 over a
        // quantize_row_q8_0_ref activation.
        let mut expected = 0.0f32;
        for (b, (wblock, xblock)) in bytes
            .chunks_exact(2 + QK8_0)
            .zip(x.chunks_exact(QK8_0))
            .enumerate()
        {
            let _ = b;
            let amax = xblock.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let dx = f16_to_f32(f32_to_f16(amax / 127.0));
            let id = if dx != 0.0 { 1.0 / (amax / 127.0) } else { 0.0 };
            let dw = f16_to_f32(u16::from_le_bytes([wblock[0], wblock[1]]));

            let mut sumi = 0i32;
            for (&wq, &xv) in wblock[2..].iter().zip(xblock) {
                sumi += (wq as i8) as i32 * ((xv * id).round() as i8) as i32;
            }
            expected += sumi as f32 * (dw * dx);
        }

        assert!(
            (out[0] - expected).abs() < 1e-5,
            "got {}, expected {}",
            out[0],
            expected
        );
    }

    /// Documents the relationship to exact arithmetic: quantizing the
    /// activation costs about a percent, and that gap is expected rather than a
    /// bug. If this ever tightens to zero, the activation quantization was
    /// silently dropped.
    #[test]
    fn matmul_q8_0_differs_from_exact_f32_by_quantization_error() {
        let vals: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.37).sin() * 3.0).collect();
        let (bytes, exact_w) = q8_0_row(&vals);
        let x: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.11).cos()).collect();

        let w = Weights {
            data: &bytes,
            ty: GgmlType::Q8_0,
            n_in: 64,
            n_out: 1,
            pooled: false,
        };
        let mut out = [0.0f32];
        Naive.matmul(&w, &x, &mut out);

        let exact: f32 = exact_w.iter().zip(&x).map(|(a, b)| a * b).sum();
        let rel = (out[0] - exact).abs() / exact.abs().max(1.0);
        assert!(rel > 1e-7, "activation quantization appears to have been dropped");
        assert!(rel < 0.05, "quantization error {rel:.3e} is larger than expected");
    }

    #[test]
    fn matmul_reads_the_right_row() {
        // Row 1 is all zeros except one element, so a stride error shows up as
        // the wrong output being non-zero.
        let mut vals = vec![0.0f32; 96];
        vals[32 + 5] = 1.0;
        let mut bytes = Vec::new();
        for v in &vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let w = Weights {
            data: &bytes,
            ty: GgmlType::F32,
            n_in: 32,
            n_out: 3,
            pooled: false,
        };
        let mut x = vec![0.0f32; 32];
        x[5] = 2.0;
        let mut out = [0.0f32; 3];
        Naive.matmul(&w, &x, &mut out);
        assert_eq!(out, [0.0, 2.0, 0.0]);
    }
}
