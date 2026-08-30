//! Pre-tokenizer splitting, ported from llama.cpp.
//!
//! Both our models declare a `tokenizer.ggml.pre` that llama.cpp handles with a
//! hand-written splitter rather than a regex engine:
//! `unicode_regex_split_custom_qwen2` and `..._qwen35` in `src/unicode.cpp`.
//! Porting those directly means we match the oracle by construction, and it is
//! why this crate needs no regex dependency -- the published regexes use a
//! negative lookahead that Rust's `regex` cannot express anyway.
//!
//! The two functions are identical except that qwen35 folds combining marks
//! (`\p{M}`) into letter runs and excludes them from the punctuation class:
//!
//!   qwen2:  `[^\r\n\p{L}\p{N}]?\p{L}+`        ` ?[^\s\p{L}\p{N}]+[\r\n]*`
//!   qwen35: `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`
//!
//! so one implementation parameterized by `marks_are_letters` covers both.

use super::unicode::{Flags, cpt_flags};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreType {
    /// `tokenizer.ggml.pre == "qwen2"` (Qwen3-0.6B).
    Qwen2,
    /// `tokenizer.ggml.pre == "qwen35"` (Qwen3.5-9B).
    Qwen35,
}

impl PreType {
    fn marks_are_letters(self) -> bool {
        matches!(self, PreType::Qwen35)
    }
}

const OUT_OF_RANGE: u32 = 0xFFFF_FFFF;

/// Split codepoints into pre-tokenizer words, returned as `[start, end)` index
/// ranges into `cpts`.
pub fn split(cpts: &[u32], pre: PreType) -> Vec<(usize, usize)> {
    let marks = pre.marks_are_letters();
    let n = cpts.len();

    let get_cpt = |pos: usize| -> u32 {
        if pos < n { cpts[pos] } else { OUT_OF_RANGE }
    };
    // Positions outside the text yield all-zero flags, not UNDEFINED, matching
    // the reference's `unicode_cpt_flags{}`. The `.any()` tests below depend on
    // that distinction.
    let get_flags = |pos: usize| -> Flags {
        if pos < n { cpt_flags(cpts[pos]) } else { Flags::NONE }
    };
    // `[\p{L}]` for qwen2, `[\p{L}\p{M}]` for qwen35.
    let letterish = |f: Flags| -> bool { f.is_letter() || (marks && f.is_accent_mark()) };

    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut prev_end = 0usize;

    let mut pos = 0usize;
    while pos < n {
        let cpt = get_cpt(pos);
        let flags = get_flags(pos);

        // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if cpt == '\'' as u32 && pos + 1 < n {
            let next = super::unicode::to_lower(get_cpt(pos + 1));
            if next == 's' as u32 || next == 't' as u32 || next == 'm' as u32 || next == 'd' as u32
            {
                pos += add_token(&mut out, &mut prev_end, pos + 2);
                continue;
            }
            if pos + 2 < n {
                let after = super::unicode::to_lower(get_cpt(pos + 2));
                if (next == 'r' as u32 && after == 'e' as u32)
                    || (next == 'v' as u32 && after == 'e' as u32)
                    || (next == 'l' as u32 && after == 'l' as u32)
                {
                    pos += add_token(&mut out, &mut prev_end, pos + 3);
                    continue;
                }
            }
        }

        // regex: [^\r\n\p{L}\p{N}]? <letters>+
        if !(cpt == '\r' as u32 || cpt == '\n' as u32 || flags.is_number())
            && (letterish(flags) || letterish(get_flags(pos + 1)))
        {
            pos += 1;
            while letterish(get_flags(pos)) {
                pos += 1;
            }
            add_token(&mut out, &mut prev_end, pos);
            continue;
        }

        // regex: \p{N}  -- one digit at a time, deliberately: there is no `+`.
        if flags.is_number() {
            pos += 1;
            add_token(&mut out, &mut prev_end, pos);
            continue;
        }

        // regex: <space>? [^\s<letters>\p{N}]+ [\r\n]*
        let mut flags2 = if cpt == ' ' as u32 { get_flags(pos + 1) } else { flags };
        if !(flags2.is_whitespace() || letterish(flags2) || flags2.is_number()) && flags.any() {
            if cpt == ' ' as u32 {
                pos += 1;
            }
            while !(flags2.is_whitespace() || letterish(flags2) || flags2.is_number())
                && flags2.any()
            {
                pos += 1;
                flags2 = get_flags(pos);
            }
            let mut cpt2 = get_cpt(pos);
            while cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                pos += 1;
                cpt2 = get_cpt(pos);
            }
            add_token(&mut out, &mut prev_end, pos);
            continue;
        }

        let mut num_whitespaces = 0usize;
        let mut last_end_r_or_n = 0usize;
        while get_flags(pos + num_whitespaces).is_whitespace() {
            let cpt2 = get_cpt(pos + num_whitespaces);
            if cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                last_end_r_or_n = pos + num_whitespaces + 1;
            }
            num_whitespaces += 1;
        }

        // regex: \s*[\r\n]+
        if last_end_r_or_n > 0 {
            pos = last_end_r_or_n;
            add_token(&mut out, &mut prev_end, pos);
            continue;
        }

        // regex: \s+(?!\S) -- trailing run keeps its last space for the next word
        if num_whitespaces > 1 && get_cpt(pos + num_whitespaces) != OUT_OF_RANGE {
            pos += num_whitespaces - 1;
            add_token(&mut out, &mut prev_end, pos);
            continue;
        }

        // regex: \s+
        if num_whitespaces > 0 {
            pos += num_whitespaces;
            add_token(&mut out, &mut prev_end, pos);
            continue;
        }

        // no matches
        pos += 1;
        add_token(&mut out, &mut prev_end, pos);
    }

    out
}

/// Emit `[prev_end, end)` if non-empty and advance. Returns the length, which
/// the contraction branches add to `pos` exactly as the reference does.
fn add_token(out: &mut Vec<(usize, usize)>, prev_end: &mut usize, end: usize) -> usize {
    let len = end - *prev_end;
    if len > 0 {
        out.push((*prev_end, end));
    }
    *prev_end = end;
    len
}

/// Convenience wrapper returning the words as strings.
pub fn split_str(text: &str, pre: PreType) -> Vec<String> {
    let cpts = super::unicode::cpts_from_str(text);
    split(&cpts, pre)
        .into_iter()
        .map(|(a, b)| cpts[a..b].iter().filter_map(|&c| char::from_u32(c)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q2(text: &str) -> Vec<String> {
        split_str(text, PreType::Qwen2)
    }

    #[test]
    fn splits_words_with_leading_space() {
        assert_eq!(q2("Hello world"), vec!["Hello", " world"]);
    }

    #[test]
    fn digits_split_individually() {
        // The regex is `\p{N}`, not `\p{N}+`.
        assert_eq!(q2("abc123"), vec!["abc", "1", "2", "3"]);
    }

    #[test]
    fn contractions_are_case_insensitive() {
        assert_eq!(q2("it's"), vec!["it", "'s"]);
        assert_eq!(q2("IT'S"), vec!["IT", "'S"]);
        assert_eq!(q2("they're"), vec!["they", "'re"]);
        assert_eq!(q2("we'll"), vec!["we", "'ll"]);
    }

    #[test]
    fn trailing_whitespace_run_yields_last_space_to_next_word() {
        // `\s+(?!\S)` keeps all but the final space, which then leads the word.
        assert_eq!(q2("a   b"), vec!["a", "  ", " b"]);
    }

    #[test]
    fn whitespace_at_end_of_text_is_kept_whole() {
        assert_eq!(q2("a   "), vec!["a", "   "]);
    }

    #[test]
    fn newlines_group_with_preceding_whitespace() {
        assert_eq!(q2("a  \n\nb"), vec!["a", "  \n\n", "b"]);
    }

    #[test]
    fn combining_marks_are_the_only_difference_between_the_two() {
        // "e" + U+0301 combining acute + "x".
        let text = "e\u{0301}x";

        // qwen35 folds marks into letter runs: one word.
        assert_eq!(split_str(text, PreType::Qwen35), vec!["e\u{0301}x"]);

        // qwen2 does not, so the mark cannot extend "e". It is instead absorbed
        // by the optional `[^\r\n\p{L}\p{N}]?` prefix of the *next* letter run,
        // which is why the accent attaches to "x" rather than standing alone.
        assert_eq!(split_str(text, PreType::Qwen2), vec!["e", "\u{0301}x"]);
    }

    #[test]
    fn a_mark_with_no_following_letter_stands_alone() {
        // With no letter run to prefix, the mark falls through to the
        // punctuation branch under qwen2 but is a letter run under qwen35.
        assert_eq!(split_str("e\u{0301}", PreType::Qwen2), vec!["e", "\u{0301}"]);
        assert_eq!(split_str("e\u{0301}", PreType::Qwen35), vec!["e\u{0301}"]);
    }

    #[test]
    fn empty_input() {
        assert!(q2("").is_empty());
    }

    #[test]
    fn spans_cover_the_input_exactly() {
        for text in ["Hello, world!\n\n  foo 42", "  ", "中文テスト", "a'b'c"] {
            let cpts = super::super::unicode::cpts_from_str(text);
            let spans = split(&cpts, PreType::Qwen2);
            let mut at = 0;
            for (a, b) in &spans {
                assert_eq!(*a, at, "gap or overlap in {text:?}");
                assert!(b > a);
                at = *b;
            }
            assert_eq!(at, cpts.len(), "spans do not cover {text:?}");
        }
    }
}
