//! IEEE-754 half and bfloat16 to f32.
//!
//! Rust has no stable `f16`, so this is written out. Every f16 value is exactly
//! representable in f32 (wider exponent and mantissa), so a correct conversion
//! is bit-exact against `GGML_FP16_TO_FP32` in the reference — there is no
//! rounding to disagree about, including for subnormals.

/// Convert an IEEE-754 binary16 bit pattern to f32.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;

    let bits = match exp {
        // Zero or subnormal.
        0 => {
            if mant == 0 {
                sign << 31
            } else {
                // A half subnormal is mant * 2^-24. Shift until bit 10 is set,
                // so the value reads as 1.f x 2^(-14-k); then exp32 = 113 - k.
                let mut m = mant;
                let mut k = 0u32;
                while m & 0x0400 == 0 {
                    m <<= 1;
                    k += 1;
                }
                (sign << 31) | ((113 - k) << 23) | ((m & 0x03ff) << 13)
            }
        }
        // Infinity or NaN: exponent saturates, mantissa is carried across so a
        // signalling/quiet distinction survives.
        0x1f => (sign << 31) | (0xff << 23) | (mant << 13),
        // Normal: rebias the exponent from 15 to 127.
        _ => (sign << 31) | ((exp + 112) << 23) | (mant << 13),
    };

    f32::from_bits(bits)
}

/// Convert a bfloat16 bit pattern to f32. bf16 is the top 16 bits of an f32.
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
