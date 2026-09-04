//! The k-quant dot products, and the Q8_K activation format they consume.
//!
//! **ggml does not multiply k-quant weights by f32 activations.** Its
//! `type_traits_cpu[T].vec_dot_type` is `GGML_TYPE_Q8_K` for Q5_K, Q6_K and
//! IQ4_XS, so the activation is quantized to Q8_K and the dot is an *integer*
//! sum scaled per sub-block. Matching that is what makes this a usable oracle,
//! exactly as `ops::naive`'s Q8_0 path already argues for its own format.
//!
//! Every function here is transcribed from `ggml/src/ggml-cpu/quants.c`'s
//! `*_generic` paths and `ggml/src/ggml-quants.c`, and checked bit-for-bit
//! against ggml itself by `tests/kquant_dots.rs` -- the fixtures are produced by
//! calling the reference through `ctypes`, not by re-deriving it.
//!
//! The accumulation structure is load-bearing and is the thing to preserve if
//! any of this is ever rewritten: the reference keeps **eight** f32 lanes
//! across all super-blocks and sums them only at the very end. A single serial
//! accumulator would be a different function, and the difference shows up as a
//! last-bit disagreement rather than as anything obviously wrong.
//!
//! # The FMA contraction, which is the uncomfortable part
//!
//! Bit-exactness here depends on **the reference's C compiler**, per function,
//! and not on anything in the C source:
//!
//! | | `acc += a * b` is | so we use |
//! |---|---|---|
//! | `vec_dot_q5_K_q8_K` | contracted, both sites | `mul_add` |
//! | `vec_dot_q6_K_q8_K` | *not* contracted | plain `+` and `*` |
//! | `vec_dot_iq4_xs_q8_K` | *not* contracted | plain `+` and `*` |
//!
//! Q5_K and Q6_K contain the *same expression* — `sums[l] += d * aux32[l]` —
//! and the compiler fuses one and not the other. Rust never contracts unless
//! asked, so matching the reference means writing `mul_add` exactly where the
//! reference's build happened to fuse.
//!
//! This was found by measurement, not by reading: a Python model of the C
//! agreed with our Rust and both sat ~12 ulps from ggml, and toggling the two
//! candidate fusions took Q5_K from 4/16 rows exact to 16/16. Trying the same
//! fusion on the other two broke them.
//!
//! **What this costs.** The fixtures encode a compiler decision, so rebuilding
//! llama.cpp with different flags (`-ffp-contract=off`, another GCC) could
//! change the expected values and require the `mul_add`s to move. That is
//! recorded rather than hidden because it bounds what "bit-exact" can mean for
//! a k-quant: unlike Q8_0, whose 32-element block sum is an integer and
//! therefore order-free, these have no implementation-independent answer. Note
//! also that llama.cpp on this machine runs the *SIMD* kernels, which differ
//! from `_generic` by ~1e-6 anyway — so being exact here is being exact against
//! ggml's portable definition, not against what llama.cpp actually computes.

use crate::gguf::GgmlType;
use crate::quant::half::f16_to_f32;

/// Elements per k-quant super-block (`QK_K` in ggml-common.h).
pub const QK_K: usize = 256;

/// An activation vector quantized to Q8_K — what the k-quants dot against.
///
/// **Not an alternative to [`QuantizedRow`]; a different pairing.** ggml's
/// `type_traits_cpu[T].vec_dot_type` is `GGML_TYPE_Q8_0` for Q8_0 weights and
/// `GGML_TYPE_Q8_K` for Q5_K, Q6_K and IQ4_XS. Reproducing llama.cpp on the 35B
/// therefore means quantizing the activation *this* way for those three
/// formats, and the old way for Q8_0.
///
/// Mirrors `quantize_row_q8_K_ref` in `ggml/src/ggml-quants.c`. Four details
/// there are easy to get wrong and all of them are silent:
///
/// 1. **The scale divides by the signed extreme, not the magnitude.** The
///    reference tracks `max` = the element with the largest *absolute* value,
///    keeping its sign, and sets `iscale = -127 / max`. So `iscale` is negative
///    for a positive extreme, `d = 1/iscale` is negative too, and the extreme
///    element quantizes to exactly -127.
/// 2. **Rounding is half-to-even, not half-away-from-zero.** `nearest_int` adds
///    12582912.0 and reads the mantissa back, which rides IEEE f32 addition's
///    round-to-nearest-even. Rust's `f32::round` is half-away-from-zero and
///    would differ on exact halves — note [`QuantizedRow`] above *does* use
///    `round`, because Q8_0's reference uses `roundf`. The two formats round
///    differently and that is not a mistake in either place.
/// 3. **Only the upper side is clamped.** The reference writes `MIN(127, v)`
///    with no lower bound, which is safe because the extreme maps to -127 by
///    construction, and load-bearing because clamping symmetrically would be a
///    different function.
/// 4. **`bsums` are sums of 16 consecutive quants**, and Q5_K's dot multiplies
///    them by `dmin` *without* `d`. Omitting them leaves Q6_K and IQ4_XS
///    correct and Q5_K subtly wrong, which is the worst possible failure shape.
///
/// One deliberate divergence: for an all-zero block the reference zeroes `qs`
/// and `d` but leaves `bsums` untouched — its `memset` covers only `qs`. We
/// zero them, because reading uninitialized memory is not a semantic we can
/// reproduce and a zero block contributes nothing through `d` anyway. Real
/// activations do not produce a fully-zero 256-element block, so this has no
/// effect on any comparison against the reference; it is recorded because it is
/// the one place we knowingly differ.
pub struct Q8KRow {
    /// One `d` per super-block. **May be negative**, per detail 1.
    scales: Vec<f32>,
    /// `QK_K` quants per super-block.
    quants: Vec<i8>,
    /// `QK_K/16` sums per super-block, each over 16 consecutive quants.
    bsums: Vec<i16>,
}

impl Q8KRow {
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub fn quants(&self) -> &[i8] {
        &self.quants
    }

    pub fn bsums(&self) -> &[i16] {
        &self.bsums
    }

    pub fn from_f32(x: &[f32]) -> Self {
        debug_assert_eq!(x.len() % QK_K, 0);
        let nb = x.len() / QK_K;
        let mut scales = Vec::with_capacity(nb);
        let mut quants = Vec::with_capacity(x.len());
        let mut bsums = Vec::with_capacity(nb * (QK_K / 16));

        for block in x.chunks_exact(QK_K) {
            // The signed value at the largest magnitude, not the magnitude.
            let mut amax = 0.0f32;
            let mut max = 0.0f32;
            for &v in block {
                let ax = v.abs();
                if ax > amax {
                    amax = ax;
                    max = v;
                }
            }

            if amax == 0.0 {
                scales.push(0.0);
                quants.extend(std::iter::repeat_n(0i8, QK_K));
                bsums.extend(std::iter::repeat_n(0i16, QK_K / 16));
                continue;
            }

            let iscale = -127.0f32 / max;
            let base = quants.len();
            for &v in block {
                quants.push(nearest_int(iscale * v).min(127) as i8);
            }
            for g in 0..QK_K / 16 {
                let sum: i32 = quants[base + g * 16..base + (g + 1) * 16]
                    .iter()
                    .map(|&q| q as i32)
                    .sum();
                bsums.push(sum as i16);
            }
            scales.push(1.0 / iscale);
        }

        Self {
            scales,
            quants,
            bsums,
        }
    }
}

/// Round to nearest, ties to even — `nearest_int` in `ggml/src/ggml-quants.c`.
///
/// ```c
/// float val = fval + 12582912.f;              // 1.5 * 2^23
/// int i; memcpy(&i, &val, sizeof(int));
/// return (i & 0x007fffff) - 0x00400000;
/// ```
///
/// Adding 1.5 * 2^23 forces the fractional bits out of an f32's mantissa, and
/// IEEE addition resolves the tie to even while doing it. The mantissa then
/// holds the rounded integer biased by 2^22, which the mask and subtract undo.
///
/// **Transcribed rather than replaced by `f32::round`**, which rounds halves
/// away from zero and would disagree on every exact .5 — a difference that
/// moves a quant and is invisible in any single value.
pub fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4_194_303.0);
    let val = fval + 12_582_912.0;
    let i = val.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}


/// Serialize an activation into ggml's `block_q8_K` byte layout.
///
/// Exists so `tests/kquant_dots.rs` can compare our quantization against the
/// reference *before* any dot product runs — all three formats share this
/// input, so one wrong byte here fails three kernels and looks like three bugs.
/// Nothing on the forward path calls it; [`Q8KRow`] is the compute form.
pub fn q8_k_blocks(x: &[f32]) -> Vec<u8> {
    let q = Q8KRow::from_f32(x);
    let nb = x.len() / QK_K;
    let mut out = Vec::with_capacity(nb * (4 + QK_K + 2 * (QK_K / 16)));
    for b in 0..nb {
        out.extend_from_slice(&q.scales[b].to_le_bytes());
        for &v in &q.quants[b * QK_K..(b + 1) * QK_K] {
            out.push(v as u8);
        }
        for &v in &q.bsums[b * (QK_K / 16)..(b + 1) * (QK_K / 16)] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Dot one quantized weight row against an f32 activation.
///
/// A convenience that quantizes on every call, for tests. The forward path
/// quantizes once per token and calls [`dot_row_q8_k`].
pub fn vec_dot_q8_k(ty: GgmlType, w: &[u8], x: &[f32]) -> f32 {
    dot_row_q8_k(ty, w, &Q8KRow::from_f32(x))
}

/// Dot one quantized weight row against an already-quantized activation.
pub fn dot_row_q8_k(ty: GgmlType, w: &[u8], qx: &Q8KRow) -> f32 {
    match ty {
        GgmlType::Q6K => dot_q6_k(w, qx),
        GgmlType::Q5K => dot_q5_k(w, qx),
        GgmlType::Iq4Xs => dot_iq4_xs(w, qx),
        other => panic!("no Q8_K dot product for {}", other.name()),
    }
}

/// `block_q6_K`: `{ uint8 ql[128]; uint8 qh[64]; int8 scales[16]; f16 d; }`.
const Q6K_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;

/// Transcribed from `ggml_vec_dot_q6_K_q8_K_generic`.
///
/// Six bits per weight, split across a low nibble in `ql` and a high pair in
/// `qh`, biased by **-32** so the stored unsigned 0..63 becomes -32..31. The
/// 16 sub-block scales are plain `int8`, not packed — which is what makes this
/// the simplest of the three and the right one to validate `Q8KRow` against.
fn dot_q6_k(w: &[u8], qx: &Q8KRow) -> f32 {
    let nb = w.len() / Q6K_BYTES;
    debug_assert_eq!(w.len() % Q6K_BYTES, 0);
    debug_assert_eq!(qx.scales.len(), nb);

    // Eight f32 lanes carried across every super-block, summed once at the end.
    let mut sums = [0.0f32; 8];
    let mut a = [0i8; QK_K];

    for i in 0..nb {
        let blk = &w[i * Q6K_BYTES..(i + 1) * Q6K_BYTES];
        let ql = &blk[..QK_K / 2];
        let qh = &blk[QK_K / 2..QK_K / 2 + QK_K / 4];
        let scales = &blk[QK_K / 2 + QK_K / 4..QK_K / 2 + QK_K / 4 + QK_K / 16];
        let d16 = u16::from_le_bytes([blk[Q6K_BYTES - 2], blk[Q6K_BYTES - 1]]);

        // Unpack 256 six-bit weights, in the reference's interleaving: two
        // 128-element halves, each drawing its four 32-element runs from the
        // low and high nibbles of `ql` and successive bit-pairs of `qh`.
        for j in 0..2 {
            let (q4, qhh, out) = (&ql[j * 64..], &qh[j * 32..], j * 128);
            for l in 0..32 {
                let h = qhh[l];
                a[out + l] = (((q4[l] & 0xF) | (((h >> 0) & 3) << 4)) as i8).wrapping_sub(32);
                a[out + l + 32] = (((q4[l + 32] & 0xF) | (((h >> 2) & 3) << 4)) as i8).wrapping_sub(32);
                a[out + l + 64] = (((q4[l] >> 4) | (((h >> 4) & 3) << 4)) as i8).wrapping_sub(32);
                a[out + l + 96] = (((q4[l + 32] >> 4) | (((h >> 6) & 3) << 4)) as i8).wrapping_sub(32);
            }
        }

        let q8 = &qx.quants[i * QK_K..(i + 1) * QK_K];
        let mut aux32 = [0i32; 8];
        for j in 0..QK_K / 16 {
            let scale = scales[j] as i8 as i32;
            // Two halves of eight, exactly as the reference walks them: the
            // eight lanes are a fixed position within a 16-element sub-block.
            for half in 0..2 {
                let at = j * 16 + half * 8;
                for l in 0..8 {
                    // int16 in the reference; the product of an int8 quant and
                    // a -32..31 weight cannot leave its range.
                    let aux16 = (q8[at + l] as i32 * a[at + l] as i32) as i16;
                    aux32[l] += scale * aux16 as i32;
                }
            }
        }

        let d = f16_to_f32(d16) * qx.scales[i];
        for l in 0..8 {
            // **Not** `mul_add`, though Q5_K's identical-looking line is. The
            // reference's compiler contracts that one and not this one; see the
            // module docs.
            sums[l] += d * aux32[l] as f32;
        }
    }

    let mut sumf = 0.0f32;
    for l in 0..8 {
        sumf += sums[l];
    }
    sumf
}

/// `block_q5_K`: `{ f16 d; f16 dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128] }`.
const Q5K_BYTES: usize = 2 + 2 + 12 + QK_K / 8 + QK_K / 2;

/// Transcribed from `ggml_vec_dot_q5_K_q8_K_generic`.
///
/// Five bits per weight: four in a nibble of `qs`, the fifth as a bit-plane in
/// `qh`, giving an **unsigned** 0..31 with no bias — unlike Q6_K's -32. The
/// offset instead lives in a per-sub-block `min`, subtracted through the
/// activation's `bsums`, which is the only reason [`Q8KRow`] carries them.
///
/// Two orderings must be preserved or the last bits move: the mins term
/// accumulates into `sumf` *inside* the super-block loop while the eight lanes
/// accumulate separately and join only at the end; and the twelve scale bytes
/// unpack into 8 scales and 8 mins through a fixed shuffle that is easier to
/// transcribe than to re-derive.
fn dot_q5_k(w: &[u8], qx: &Q8KRow) -> f32 {
    const KMASK1: u32 = 0x3f3f_3f3f;
    const KMASK2: u32 = 0x0f0f_0f0f;
    const KMASK3: u32 = 0x0303_0303;

    let nb = w.len() / Q5K_BYTES;
    debug_assert_eq!(w.len() % Q5K_BYTES, 0);

    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;
    let mut a = [0i8; QK_K];

    for i in 0..nb {
        let blk = &w[i * Q5K_BYTES..(i + 1) * Q5K_BYTES];
        let d16 = u16::from_le_bytes([blk[0], blk[1]]);
        let dmin16 = u16::from_le_bytes([blk[2], blk[3]]);
        let sc12 = &blk[4..16];
        let qh = &blk[16..16 + QK_K / 8];
        let qs = &blk[16 + QK_K / 8..];

        // Low nibble then high nibble of each 32-byte run, each lifted by 16
        // where the matching `qh` bit is set. `m` walks one bit per run.
        let mut m = 1u8;
        for j in 0..QK_K / 64 {
            let q4 = &qs[j * 32..(j + 1) * 32];
            for half in 0..2 {
                let out = j * 64 + half * 32;
                for l in 0..32 {
                    let base = if half == 0 { q4[l] & 0xF } else { q4[l] >> 4 };
                    a[out + l] = base as i8 + if qh[l] & m != 0 { 16 } else { 0 };
                }
                m <<= 1;
            }
        }

        // 12 bytes -> 8 six-bit scales and 8 six-bit mins.
        let mut utmp = [0u32; 4];
        for k in 0..3 {
            utmp[k] = u32::from_le_bytes(sc12[k * 4..k * 4 + 4].try_into().unwrap_or([0; 4]));
        }
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        let mut scales = [0u8; 8];
        scales[..4].copy_from_slice(&utmp[0].to_le_bytes());
        scales[4..].copy_from_slice(&utmp[1].to_le_bytes());
        let mut mins = [0u8; 8];
        mins[..4].copy_from_slice(&utmp[2].to_le_bytes());
        mins[4..].copy_from_slice(&utmp[3].to_le_bytes());

        let q8 = &qx.quants[i * QK_K..(i + 1) * QK_K];
        let bs = &qx.bsums[i * (QK_K / 16)..(i + 1) * (QK_K / 16)];

        // The offset term, through the activation's per-16 sums.
        let mut sumi = 0i32;
        for j in 0..QK_K / 16 {
            sumi += bs[j] as i32 * mins[j / 2] as i32;
        }

        let mut aux32 = [0i32; 8];
        for j in 0..QK_K / 32 {
            let scale = scales[j] as i32;
            for quarter in 0..4 {
                let at = j * 32 + quarter * 8;
                for l in 0..8 {
                    let aux16 = (q8[at + l] as i32 * a[at + l] as i32) as i16;
                    aux32[l] += scale * aux16 as i32;
                }
            }
        }

        let d = f16_to_f32(d16) * qx.scales[i];
        for l in 0..8 {
            sums[l] = d.mul_add(aux32[l] as f32, sums[l]);
        }
        // Inside the loop, before the lanes are folded in. Order matters, and
        // so does the fusion: this is `fma(-dmin, sumi, sumf)` in the compiled
        // reference, not a multiply followed by a subtract.
        let dmin = f16_to_f32(dmin16) * qx.scales[i];
        sumf = (-dmin).mul_add(sumi as f32, sumf);
    }

    for l in 0..8 {
        sumf += sums[l];
    }
    sumf
}

/// `kvalues_iq4nl` from `ggml/src/ggml-common.h`.
///
/// **A non-uniform grid**, which is what the "IQ" in IQ4_XS means: the sixteen
/// reachable values are spaced to fit a normal-ish weight distribution rather
/// than evenly. A linear dequantization would be a different format.
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// `block_iq4_xs`: `{ f16 d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128] }`.
const IQ4XS_BYTES: usize = 2 + 2 + QK_K / 64 + QK_K / 2;

/// Transcribed from `ggml_vec_dot_iq4_xs_q8_K_generic`.
///
/// The odd one of the three. Its accumulation is a **single** `sumf`, not eight
/// lanes — each 32-element sub-block contributes `d * (sumi1 + sumi2)` directly
/// — so its rounding behaviour differs from Q5_K's and Q6_K's by construction,
/// not by accident.
///
/// Each sub-block's 6-bit scale is split across two places: four low bits in
/// `scales_l`, two high bits marching through `scales_h`, and the result is
/// biased by -32.
fn dot_iq4_xs(w: &[u8], qx: &Q8KRow) -> f32 {
    let nb = w.len() / IQ4XS_BYTES;
    debug_assert_eq!(w.len() % IQ4XS_BYTES, 0);

    let mut sumf = 0.0f32;
    for ibl in 0..nb {
        let blk = &w[ibl * IQ4XS_BYTES..(ibl + 1) * IQ4XS_BYTES];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let mut h = u16::from_le_bytes([blk[2], blk[3]]);
        let scales_l = &blk[4..4 + QK_K / 64];
        let qs = &blk[4 + QK_K / 64..];

        let d4d8 = d * qx.scales[ibl];
        let q8 = &qx.quants[ibl * QK_K..(ibl + 1) * QK_K];

        let mut ib = 0usize;
        while ib < QK_K / 32 {
            let l = scales_l[ib / 2];
            let ls1 = (l & 0xf) | (((h << 4) & 0x30) as u8);
            let ls2 = (l >> 4) | (((h << 2) & 0x30) as u8);
            h >>= 4;
            let d1 = d4d8 * (ls1 as i32 - 32) as f32;
            let d2 = d4d8 * (ls2 as i32 - 32) as f32;

            for (half, dh) in [d1, d2].into_iter().enumerate() {
                let qo = ib * 16 + half * 16;
                let ao = ib * 32 + half * 32;
                let (mut s1, mut s2) = (0i32, 0i32);
                for j in 0..16 {
                    s1 += q8[ao + j] as i32 * KVALUES_IQ4NL[(qs[qo + j] & 0xf) as usize] as i32;
                    s2 += q8[ao + 16 + j] as i32 * KVALUES_IQ4NL[(qs[qo + j] >> 4) as usize] as i32;
                }
                // **Not** `mul_add` here, unlike Q5_K and Q6_K. The reference's
                // compiler contracts those two and not this one, and matching
                // that is the difference between exact and a few ulps out.
                sumf += dh * (s1 + s2) as f32;
            }
            ib += 2;
        }
    }
    sumf
}
