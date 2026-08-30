//! Codepoint classification, mirroring llama.cpp's `unicode_cpt_flags`.

use super::unicode_data as data;

/// Category flags for one codepoint. A default (all-zero) value means "outside
/// the span being examined" -- llama.cpp's splitters return `unicode_cpt_flags{}`
/// for out-of-range positions and then test `flags.as_uint()`, so zero has to
/// stay distinguishable from `UNDEFINED`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Flags(pub u16);

impl Flags {
    pub const NONE: Flags = Flags(0);

    pub fn is_number(self) -> bool {
        self.0 & data::NUMBER != 0
    }
    pub fn is_letter(self) -> bool {
        self.0 & data::LETTER != 0
    }
    pub fn is_accent_mark(self) -> bool {
        self.0 & data::ACCENT_MARK != 0
    }
    pub fn is_whitespace(self) -> bool {
        self.0 & data::WHITESPACE != 0
    }
    /// True when any flag is set. Distinguishes "a real codepoint" from the
    /// zero value used for positions outside the span.
    pub fn any(self) -> bool {
        self.0 != 0
    }
}

const MAX_CODEPOINT: u32 = 0x110000;

/// Classify a codepoint. Out-of-range codepoints are `UNDEFINED`, matching
/// `unicode_cpt_flags_from_cpt`.
pub fn cpt_flags(cpt: u32) -> Flags {
    if cpt >= MAX_CODEPOINT {
        return Flags(data::UNDEFINED);
    }

    // Each table entry covers [start, next_start). Find the last start <= cpt.
    let idx = match data::RANGES_FLAGS.binary_search_by_key(&cpt, |&(start, _)| start) {
        Ok(i) => i,
        Err(0) => return Flags(data::UNDEFINED),
        Err(i) => i - 1,
    };
    let mut flags = data::RANGES_FLAGS[idx].1;

    // Whitespace is a separate set in the reference, not part of the ranges.
    if data::WHITESPACE_CPTS.binary_search(&cpt).is_ok() {
        flags |= data::WHITESPACE;
    }

    Flags(flags)
}

/// Simple lowercase mapping. Used only for the case-insensitive contraction
/// rule (`'s`, `'re`, ...) in the pre-tokenizer.
pub fn to_lower(cpt: u32) -> u32 {
    match data::LOWERCASE_MAP.binary_search_by_key(&cpt, |&(upper, _)| upper) {
        Ok(i) => data::LOWERCASE_MAP[i].1,
        Err(_) => cpt,
    }
}

/// Decode UTF-8 into codepoints. Invalid bytes are not expected: input comes
/// from a Rust `&str`, which is guaranteed well-formed.
pub fn cpts_from_str(s: &str) -> Vec<u32> {
    s.chars().map(|c| c as u32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_classification() {
        assert!(cpt_flags('a' as u32).is_letter());
        assert!(cpt_flags('Z' as u32).is_letter());
        assert!(cpt_flags('7' as u32).is_number());
        assert!(cpt_flags(' ' as u32).is_whitespace());
        assert!(cpt_flags('\n' as u32).is_whitespace());
        assert!(cpt_flags('\t' as u32).is_whitespace());
        assert!(!cpt_flags('!' as u32).is_letter());
        assert!(!cpt_flags('!' as u32).is_whitespace());
        assert!(cpt_flags('!' as u32).any());
    }

    #[test]
    fn non_ascii_classification() {
        assert!(cpt_flags('é' as u32).is_letter());
        assert!(cpt_flags('中' as u32).is_letter());
        assert!(cpt_flags('α' as u32).is_letter());
        // U+0301 COMBINING ACUTE ACCENT -- the \p{M} case that separates the
        // qwen35 splitter from the qwen2 one.
        assert!(cpt_flags(0x0301).is_accent_mark());
        assert!(!cpt_flags(0x0301).is_letter());
        // U+00A0 NO-BREAK SPACE is whitespace in the reference's set.
        assert!(cpt_flags(0x00A0).is_whitespace());
    }

    #[test]
    fn out_of_range_is_undefined_not_zero() {
        let f = cpt_flags(0x200000);
        assert_eq!(f.0, data::UNDEFINED);
        assert!(f.any(), "UNDEFINED must be distinguishable from the zero value");
        assert!(!Flags::NONE.any());
    }

    #[test]
    fn lowercase_mapping() {
        assert_eq!(to_lower('A' as u32), 'a' as u32);
        assert_eq!(to_lower('S' as u32), 's' as u32);
        assert_eq!(to_lower('a' as u32), 'a' as u32);
        assert_eq!(to_lower('É' as u32), 'é' as u32);
        // Unmapped codepoints pass through.
        assert_eq!(to_lower('中' as u32), '中' as u32);
    }

    #[test]
    fn ranges_table_is_searchable() {
        // The binary search assumes strict ordering; the generator checks this
        // too, but a stale committed table would slip past that.
        for w in data::RANGES_FLAGS.windows(2) {
            assert!(w[0].0 < w[1].0, "ranges not strictly increasing at {:#x}", w[0].0);
        }
        for w in data::WHITESPACE_CPTS.windows(2) {
            assert!(w[0] < w[1]);
        }
        for w in data::LOWERCASE_MAP.windows(2) {
            assert!(w[0].0 < w[1].0);
        }
    }
}
