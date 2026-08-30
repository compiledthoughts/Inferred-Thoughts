//! IEEE-754 half and bfloat16 to f32.
//!
//! Rust has no stable `f16`, so this is written out. Every f16 value is exactly
//! representable in f32 (wider exponent and mantissa), so a correct conversion
//! is bit-exact against `GGML_FP16_TO_FP32` in the reference — there is no
//! rounding to disagree about, including for subnormals.

/// Convert an IEEE-754 binary16 bit pattern to f32.
///
/// **Branch-free on purpose.** Attention converts K and V a block at a time
/// (see `ops::naive::attend_kv_head`), tens of millions of elements per token,
/// and a `match` on the exponent plus a normalization loop stops LLVM
/// vectorizing that block. Masks instead of branches let it emit
/// `vcvtph2ps`-shaped code over a contiguous run.
///
/// The shape is the standard one: shift the exponent and mantissa into f32
/// position, rebias, then correct the two special ranges. Subnormals are
/// renormalized by a float subtraction against a magic constant rather than a
/// shift loop — `2^-14 * (1 + m/2^10) - 2^-14` is exactly `m * 2^-24`, which is
/// the value a half subnormal denotes.
///
/// Both corrections are computed unconditionally and selected with masks. The
/// discarded one is harmless: the subtraction cannot trap, and its result is
/// masked away for inputs that do not need it.
///
/// This is exact, as [`f16_to_f32`]'s caller relies on: every f16 value is
/// representable in f32, so there is no rounding, and `exhaustive_agreement`
/// below proves it against the readable implementation over all 65,536 inputs.
pub fn f16_to_f32(h: u16) -> f32 {
    /// Exponent rebias, 15 -> 127.
    const EXP_ADJUST: u32 = (127 - 15) << 23;
    /// A second rebias, applied only to infinities and NaNs.
    const INF_NAN_ADJUST: u32 = (128 - 16) << 23;
    /// `2^-14`, the smallest normal half. Subtracting it renormalizes.
    const MAGIC: u32 = 113 << 23;
    /// The half exponent field, shifted into f32 position.
    const SHIFTED_EXP: u32 = 0x7c00 << 13;

    let h = h as u32;
    let sign = (h & 0x8000) << 16;
    let shifted = (h & 0x7fff) << 13;
    let exp = shifted & SHIFTED_EXP;
    let normal = shifted + EXP_ADJUST;

    // All ones or all zeros, with no branch.
    let is_inf_nan = 0u32.wrapping_sub((exp == SHIFTED_EXP) as u32);
    let is_subnormal = 0u32.wrapping_sub((exp == 0) as u32);

    let inf_nan = normal.wrapping_add(INF_NAN_ADJUST);
    let subnormal =
        (f32::from_bits(normal.wrapping_add(1 << 23)) - f32::from_bits(MAGIC)).to_bits();

    // The two masks are mutually exclusive, so this is a three-way select.
    let bits = (normal & !(is_inf_nan | is_subnormal))
        | (inf_nan & is_inf_nan)
        | (subnormal & is_subnormal);

    f32::from_bits(bits | sign)
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Convert f32 to an IEEE-754 binary16 bit pattern, round-to-nearest-even.
///
/// Needed because quantizing an activation to Q8_0 stores the block scale as
/// f16 (`GGML_FP32_TO_FP16` in `quantize_row_q8_0_ref`), and the dot product
/// then reads it back. Getting the rounding mode wrong here shifts every
/// quantized matmul by an ulp of the scale.
pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    // Inf or NaN. A NaN keeps a non-zero mantissa so it stays a NaN.
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }

    let e = exp - 127 + 15;

    // Overflow saturates to infinity, as IEEE round-to-nearest does.
    if e >= 0x1f {
        return sign | 0x7c00;
    }

    if e <= 0 {
        // Too small even for a subnormal.
        if e < -10 {
            return sign;
        }
        // Subnormal: restore the implicit leading 1 and shift into place.
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let mut half = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && (half & 1) == 1) {
            half += 1;
        }
        return sign | half as u16;
    }

    // Normal. Rounding may carry into the exponent, which is correct: it
    // promotes to the next binade, and from the top binade to infinity.
    let mut h = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The readable implementation this replaced, kept as the reference the
    /// branch-free one is proved against. Transcribed from the original: a
    /// `match` on the exponent, with a shift loop to renormalize subnormals.
    fn reference(h: u16) -> f32 {
        let sign = ((h >> 15) & 1) as u32;
        let exp = ((h >> 10) & 0x1f) as u32;
        let mant = (h & 0x03ff) as u32;

        let bits = match exp {
            0 => {
                if mant == 0 {
                    sign << 31
                } else {
                    let mut m = mant;
                    let mut k = 0u32;
                    while m & 0x0400 == 0 {
                        m <<= 1;
                        k += 1;
                    }
                    (sign << 31) | ((113 - k) << 23) | ((m & 0x03ff) << 13)
                }
            }
            0x1f => (sign << 31) | (0xff << 23) | (mant << 13),
            _ => (sign << 31) | ((exp + 112) << 23) | (mant << 13),
        };
        f32::from_bits(bits)
    }

    /// **The whole input domain is 65,536 values, so this is a proof rather
    /// than a sample.** Bits are compared, not values, so NaN payloads and the
    /// sign of zero are held to account too — `==` would call every NaN
    /// unequal and both zeros equal, and neither is what we want here.
    #[test]
    fn exhaustive_agreement_with_the_readable_implementation() {
        for h in 0..=u16::MAX {
            let (fast, slow) = (f16_to_f32(h), reference(h));
            assert_eq!(
                fast.to_bits(),
                slow.to_bits(),
                "h = {h:#06x}: {fast:?} ({:#010x}) vs {slow:?} ({:#010x})",
                fast.to_bits(),
                slow.to_bits()
            );
        }
    }

    /// Every class in one place, so a failure names which range broke.
    #[test]
    fn covers_every_exponent_class() {
        let mut classes = [0usize; 4];
        for h in 0..=u16::MAX {
            let exp = (h >> 10) & 0x1f;
            let mant = h & 0x3ff;
            classes[match (exp, mant) {
                (0, 0) => 0,      // zeros
                (0, _) => 1,      // subnormals
                (0x1f, _) => 2,   // inf and NaN
                _ => 3,           // normals
            }] += 1;
        }
        assert_eq!(classes[0], 2, "two zeros");
        assert_eq!(classes[1], 2 * 1023, "subnormals");
        assert_eq!(classes[2], 2 * 1024, "inf and NaN");
        assert_eq!(classes[3], 2 * 30 * 1024, "normals");
    }

    #[test]
    fn exact_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        // Largest finite half: 65504.
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
    }

    #[test]
    fn subnormals() {
        // Smallest positive subnormal: 2^-24.
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        // Largest subnormal: 1023 * 2^-24.
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2.0f32.powi(-24));
        // Smallest positive normal: 2^-14.
        assert_eq!(f16_to_f32(0x0400), 2.0f32.powi(-14));
        // The subnormal/normal boundary must be continuous.
        assert_eq!(
            f16_to_f32(0x0400) - f16_to_f32(0x03ff),
            2.0f32.powi(-24)
        );
    }

    #[test]
    fn infinities_and_nan() {
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    /// Every one of the 65536 half values must round-trip through f32 back to
    /// the same bit pattern, which pins both conversions exactly.
    #[test]
    fn all_finite_halves_round_trip() {
        for h in 0u32..=0xffff {
            let h = h as u16;
            let f = f16_to_f32(h);
            if f.is_nan() {
                continue;
            }
            let back = f32_to_f16(f);
            assert_eq!(back, h, "half 0x{h:04x} -> {f} -> 0x{back:04x}");
        }
    }

    #[test]
    fn f32_to_f16_rounds_to_nearest_even() {
        // Halfway between two representable halves must go to the even one,
        // not always up. 1.0 and the next half up differ by 2^-10.
        let one = f16_to_f32(0x3c00); // 1.0, mantissa even
        let next = f16_to_f32(0x3c01); // mantissa odd
        let mid = (one + next) / 2.0;
        assert_eq!(f32_to_f16(mid), 0x3c00, "should round down to even");

        let after = f16_to_f32(0x3c02); // mantissa even
        let mid2 = (next + after) / 2.0;
        assert_eq!(f32_to_f16(mid2), 0x3c02, "should round up to even");
    }

    #[test]
    fn f32_to_f16_saturates_and_underflows() {
        assert_eq!(f32_to_f16(1e30), 0x7c00, "overflow to +inf");
        assert_eq!(f32_to_f16(-1e30), 0xfc00, "overflow to -inf");
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert!(f32_to_f16(f32::NAN) & 0x03ff != 0, "NaN must stay NaN");
        assert_eq!(f32_to_f16(1e-30), 0x0000, "underflow to +0");
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        // Smallest subnormal survives.
        assert_eq!(f32_to_f16(2.0f32.powi(-24)), 0x0001);
    }

    #[test]
    fn bf16_is_the_top_half_of_an_f32() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xbf80), -1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
    }
}
