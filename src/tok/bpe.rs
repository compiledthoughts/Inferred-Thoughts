//! GPT-2 byte encoding and the BPE merge loop.
//!
//! Ported from `llm_tokenizer_bpe_session::tokenize` and `add_new_bigram` in
//! llama.cpp's `src/llama-vocab.cpp`.
//!
//! Both our models use `byte_encode = true`: each *byte* of a pre-tokenized
//! word is mapped to a printable codepoint before merging, so the vocabulary
//! never has to contain raw control bytes. Merges then operate on those
//! encoded strings.

use std::collections::HashMap;

/// The GPT-2 byte<->codepoint bijection.
///
/// Bytes that are already printable ASCII or printable Latin-1 map to
/// themselves; the remaining 68 map to U+0100 upward in order. This is why a
/// space appears as `Ġ` (U+0120 = 0x100 + 32) in merge rules.
pub struct ByteEncoder {
    to_char: [char; 256],
    from_char: HashMap<char, u8>,
}

impl Default for ByteEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ByteEncoder {
    pub fn new() -> Self {
        let printable = |b: u8| {
            (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b)
        };

        let mut to_char = ['\0'; 256];
        let mut next = 0u32;
        for b in 0..=255u8 {
            let c = if printable(b) {
                char::from_u32(b as u32).expect("byte is a valid codepoint")
            } else {
                let c = char::from_u32(256 + next).expect("in range");
                next += 1;
                c
            };
            to_char[b as usize] = c;
        }

        let from_char = to_char
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();

        Self { to_char, from_char }
    }

    /// Encode raw bytes into the printable codepoint alphabet.
    pub fn encode(&self, bytes: &[u8]) -> String {
        bytes.iter().map(|&b| self.to_char[b as usize]).collect()
    }

    /// Reverse the mapping. Returns `None` on a codepoint outside the alphabet,
    /// which means the token text was not byte-encoded.
    pub fn decode(&self, s: &str) -> Option<Vec<u8>> {
        s.chars().map(|c| self.from_char.get(&c).copied()).collect()
    }
}

/// One node of the doubly-linked list of symbols within a word. `len == 0`
/// marks a symbol that has been merged away.
#[derive(Debug, Clone, Copy)]
struct Symbol {
    start: usize,
    len: usize,
    prev: i32,
    next: i32,
}

/// A candidate merge. Ordered so that `BinaryHeap` (a max-heap) yields the
/// lowest rank first, breaking ties toward the leftmost position -- matching
/// llama.cpp's `comparator`, which reports `l` as lower priority when
/// `l.rank > r.rank || (l.rank == r.rank && l.left > r.left)`.
#[derive(Debug, Clone, Eq, PartialEq)]
struct Bigram {
    left: i32,
    right: i32,
    rank: u32,
    text: Vec<u8>,
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Apply BPE merges to one pre-tokenized, byte-encoded word.
///
/// Returns the merged pieces in order, as byte slices of `word`.
pub fn merge_word(word: &[u8], ranks: &HashMap<(Vec<u8>, Vec<u8>), u32>) -> Vec<(usize, usize)> {
    let mut symbols: Vec<Symbol> = Vec::new();

    // Initial symbols are single UTF-8 characters.
    let mut offset = 0usize;
    let mut index = 0i32;
    while offset < word.len() {
        let char_len = utf8_len(word[offset]).min(word.len() - offset);
        offset += char_len;
        symbols.push(Symbol {
            start: offset - char_len,
            len: char_len,
            prev: index - 1,
            next: if offset == word.len() { -1 } else { index + 1 },
        });
        index += 1;
    }

    if symbols.is_empty() {
        return Vec::new();
    }

    let mut queue: std::collections::BinaryHeap<Bigram> = std::collections::BinaryHeap::new();
    for i in 1..symbols.len() {
        push_bigram(&mut queue, &symbols, word, ranks, i as i32 - 1, i as i32);
    }

    while let Some(bigram) = queue.pop() {
        let left = symbols[bigram.left as usize];
        let right = symbols[bigram.right as usize];

        if left.len == 0 || right.len == 0 {
            continue;
        }
        // The queue holds entries that may have been invalidated by an earlier
        // merge; the recorded text is how the reference detects that.
        let joined = [
            &word[left.start..left.start + left.len],
            &word[right.start..right.start + right.len],
        ]
        .concat();
        if joined != bigram.text {
            continue;
        }

        symbols[bigram.left as usize].len += right.len;
        symbols[bigram.right as usize].len = 0;

        symbols[bigram.left as usize].next = right.next;
        if right.next >= 0 {
            symbols[right.next as usize].prev = bigram.left;
        }

        let l = symbols[bigram.left as usize];
        push_bigram(&mut queue, &symbols, word, ranks, l.prev, bigram.left);
        push_bigram(&mut queue, &symbols, word, ranks, bigram.left, l.next);
    }

    symbols
        .iter()
        .filter(|s| s.len > 0)
        .map(|s| (s.start, s.start + s.len))
        .collect()
}

fn push_bigram(
    queue: &mut std::collections::BinaryHeap<Bigram>,
    symbols: &[Symbol],
    word: &[u8],
    ranks: &HashMap<(Vec<u8>, Vec<u8>), u32>,
    left: i32,
    right: i32,
) {
    if left == -1 || right == -1 {
        return;
    }
    let l = symbols[left as usize];
    let r = symbols[right as usize];
    let lt = word[l.start..l.start + l.len].to_vec();
    let rt = word[r.start..r.start + r.len].to_vec();

    let Some(&rank) = ranks.get(&(lt.clone(), rt.clone())) else {
        return;
    };

    let mut text = lt;
    text.extend_from_slice(&rt);
    queue.push(Bigram {
        left,
        right,
        rank,
        text,
    });
}

/// Length in bytes of the UTF-8 sequence starting with `first`, matching
/// llama.cpp's `unicode_len_utf8`.
fn utf8_len(first: u8) -> usize {
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    LOOKUP[(first >> 4) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_encoder_is_a_bijection() {
        let e = ByteEncoder::new();
        let mut seen = std::collections::HashSet::new();
        for b in 0..=255u8 {
            assert!(seen.insert(e.to_char[b as usize]), "duplicate for byte {b}");
        }
        for b in 0..=255u8 {
            let s = e.encode(&[b]);
            assert_eq!(e.decode(&s).unwrap(), vec![b]);
        }
    }

    #[test]
    fn byte_encoder_known_values() {
        let e = ByteEncoder::new();
        // Space is the first non-printable byte, so it maps to U+0100 + 32.
        assert_eq!(e.encode(b" "), "\u{0120}");
        assert_eq!(e.encode(b"\n"), "\u{010A}");
        // Printable ASCII is identity.
        assert_eq!(e.encode(b"Hello"), "Hello");
        assert_eq!(e.encode(b"!~"), "!~");
    }

    #[test]
    fn utf8_len_matches_lead_byte() {
        assert_eq!(utf8_len(b'a'), 1);
        assert_eq!(utf8_len(0xC4), 2);
        assert_eq!(utf8_len(0xE4), 3);
        assert_eq!(utf8_len(0xF0), 4);
    }

    fn ranks(pairs: &[(&str, &str)]) -> HashMap<(Vec<u8>, Vec<u8>), u32> {
        pairs
            .iter()
            .enumerate()
            .map(|(i, (a, b))| ((a.as_bytes().to_vec(), b.as_bytes().to_vec()), i as u32))
            .collect()
    }

    fn pieces(word: &str, r: &HashMap<(Vec<u8>, Vec<u8>), u32>) -> Vec<String> {
        merge_word(word.as_bytes(), r)
            .into_iter()
            .map(|(a, b)| String::from_utf8(word.as_bytes()[a..b].to_vec()).unwrap())
            .collect()
    }

    #[test]
    fn merges_in_rank_order() {
        // "ab" ranks before "bc", so "abc" becomes "ab" + "c".
        let r = ranks(&[("a", "b"), ("b", "c")]);
        assert_eq!(pieces("abc", &r), vec!["ab", "c"]);
        // Reversing the ranks reverses the outcome.
        let r = ranks(&[("b", "c"), ("a", "b")]);
        assert_eq!(pieces("abc", &r), vec!["a", "bc"]);
    }

    #[test]
    fn merges_cascade() {
        let r = ranks(&[("a", "b"), ("ab", "c")]);
        assert_eq!(pieces("abc", &r), vec!["abc"]);
    }

    #[test]
    fn no_ranks_leaves_single_characters() {
        let r = ranks(&[]);
        assert_eq!(pieces("abc", &r), vec!["a", "b", "c"]);
    }

    #[test]
    fn ties_break_leftmost() {
        // Both pairs are the same rank; the leftmost must win.
        let mut r = HashMap::new();
        r.insert((b"a".to_vec(), b"a".to_vec()), 0u32);
        assert_eq!(pieces("aaa", &r), vec!["aa", "a"]);
    }

    #[test]
    fn multibyte_characters_are_single_symbols() {
        let r = ranks(&[]);
        assert_eq!(pieces("中文", &r), vec!["中", "文"]);
    }

    #[test]
    fn empty_word() {
        assert!(merge_word(b"", &ranks(&[])).is_empty());
    }
}
