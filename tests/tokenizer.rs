//! Stage 3 acceptance: our token ids must match `llama-tokenize` exactly.
//!
//! Fixtures under `tests/fixtures/tokenizer_*.txt` are produced by
//! `scripts/dump_tokenizer_fixture.py`, which runs the reference binary over a
//! corpus covering ASCII, CJK, emoji, combining marks, whitespace runs,
//! newlines, digits, control characters, code, and the chat template's special
//! tokens.
//!
//! Unlike the dequantization fixtures these cannot be self-contained: building
//! the tokenizer needs the model's 150k-entry vocabulary and merge list. So a
//! fixture is skipped when its model is absent, but the suite fails if *no*
//! model is present, so it can never quietly degrade to testing nothing.
//! Override paths with `INFERRED_MODEL_DIR`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use inferred_thoughts::GgufFile;
use inferred_thoughts::tok::Tokenizer;

struct Case {
    text: String,
    expected: Vec<u32>,
}

struct FixtureSet {
    fixture: PathBuf,
    model: PathBuf,
    cases: Vec<Case>,
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// The fixture records the absolute model path used to generate it. Honour it,
/// but allow relocation via `INFERRED_MODEL_DIR` so the tests are not welded to
/// one machine's layout.
fn resolve_model(recorded: &str) -> PathBuf {
    let recorded = PathBuf::from(recorded);
    if let Ok(dir) = std::env::var("INFERRED_MODEL_DIR") {
        if let Some(name) = recorded.file_name() {
            let candidate = Path::new(&dir).join(name);
            if candidate.exists() {
                return candidate;
            }
        }
    }
    recorded
}

fn load_fixtures() -> Vec<FixtureSet> {
    let dir = fixture_dir();
    let mut sets = Vec::new();

    let entries = fs::read_dir(&dir).unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()));
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("tokenizer_") && n.ends_with(".txt"))
        })
        .collect();
    paths.sort();

    for path in paths {
        let body = fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {path:?}: {e}"));
        let mut model = None;
        let mut cases = Vec::new();

        for line in body.lines() {
            if let Some(rest) = line.strip_prefix("# model: ") {
                model = Some(rest.trim().to_string());
                continue;
            }
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let (hex, ids) = line
                .split_once('\t')
                .unwrap_or_else(|| panic!("malformed line in {path:?}: {line:?}"));
            let bytes = decode_hex(hex);
            let text = String::from_utf8(bytes)
                .unwrap_or_else(|e| panic!("fixture text is not utf-8 in {path:?}: {e}"));
            let expected = ids
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse::<u32>().expect("token id"))
                .collect();
            cases.push(Case { text, expected });
        }

        let model = model.unwrap_or_else(|| panic!("{path:?} has no `# model:` header"));
        sets.push(FixtureSet {
            fixture: path,
            model: resolve_model(&model),
            cases,
        });
    }

    assert!(!sets.is_empty(), "no tokenizer fixtures in {}", dir.display());
    sets
}

fn decode_hex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "odd-length hex: {s:?}");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex digit"))
        .collect()
}

/// Show a string with control characters visible, so a failure message is
/// readable when the input is a run of newlines or a zero-width space.
fn show(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c if (c as u32) >= 0x80 && !c.is_alphanumeric() => {
                out.push_str(&format!("\\u{{{:04x}}}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[test]
fn matches_llama_tokenize() {
    let sets = load_fixtures();
    let mut ran = 0usize;
    let mut skipped: Vec<String> = Vec::new();

    for set in &sets {
        if !set.model.exists() {
            skipped.push(format!(
                "{} (model not found at {})",
                set.fixture.file_name().unwrap().to_string_lossy(),
                set.model.display()
            ));
            continue;
        }

        let f = GgufFile::open(&set.model).expect("open model");
        let tk = Tokenizer::from_metadata(&f.metadata).expect("build tokenizer");

        let mut failures: Vec<String> = Vec::new();
        for case in &set.cases {
            let got = tk.encode(&case.text, true, true);
            if got != case.expected {
                // Report where the two sequences first diverge, and decode the
                // surrounding tokens, rather than dumping two id lists.
                let at = got
                    .iter()
                    .zip(&case.expected)
                    .position(|(a, b)| a != b)
                    .unwrap_or(got.len().min(case.expected.len()));
                failures.push(format!(
                    "  input {}\n    expected {:?}\n    got      {:?}\n    first difference at index {at}\n\
                     \x20   expected[{at}..] as text: {}\n    got[{at}..] as text:      {}",
                    show(&case.text),
                    case.expected,
                    got,
                    show(&tk.decode(&case.expected[at.min(case.expected.len())..], true).unwrap_or_default()),
                    show(&tk.decode(&got[at.min(got.len())..], true).unwrap_or_default()),
                ));
            }
        }

        assert!(
            failures.is_empty(),
            "{} of {} cases differ from llama-tokenize for {}:\n{}",
            failures.len(),
            set.cases.len(),
            set.model.display(),
            failures
                .iter()
                .take(10)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );

        println!(
            "ok  {} cases match llama-tokenize  [{}]",
            set.cases.len(),
            set.model.file_name().unwrap().to_string_lossy()
        );
        ran += 1;
    }

    for s in &skipped {
        println!("skipped {s}");
    }
    assert!(
        ran > 0,
        "every tokenizer fixture was skipped -- no model files found. \
         Set INFERRED_MODEL_DIR to the directory holding them.\nSkipped: {skipped:?}"
    );
}

/// Encoding then decoding must return the original text. This catches byte
/// encoder errors that happen to be self-consistent during encoding.
#[test]
fn round_trips_through_decode() {
    for set in &load_fixtures() {
        if !set.model.exists() {
            continue;
        }
        let f = GgufFile::open(&set.model).expect("open model");
        let tk = Tokenizer::from_metadata(&f.metadata).expect("build tokenizer");

        for case in &set.cases {
            // Text containing a lone surrogate or unpaired control could not
            // round-trip, but the corpus has none; every case must survive.
            let ids = tk.encode(&case.text, false, true);
            let back = tk.decode(&ids, true).expect("decode");
            assert_eq!(
                back,
                case.text,
                "round trip failed for {}\n  ids: {:?}",
                show(&case.text),
                ids
            );
        }
        println!(
            "ok  round trip  [{}]",
            set.model.file_name().unwrap().to_string_lossy()
        );
    }
}

/// The corpus must actually cover the categories it claims to, so that thinning
/// it later is a visible change rather than a silent loss of coverage.
#[test]
fn corpus_covers_the_hard_cases() {
    let sets = load_fixtures();
    let all: Vec<&str> = sets
        .iter()
        .flat_map(|s| s.cases.iter().map(|c| c.text.as_str()))
        .collect();

    let mut coverage: BTreeMap<&str, bool> = BTreeMap::new();
    coverage.insert("cjk", all.iter().any(|s| s.contains('中')));
    coverage.insert("emoji", all.iter().any(|s| s.contains('🙂')));
    coverage.insert("combining mark", all.iter().any(|s| s.contains('\u{0301}')));
    coverage.insert("newline run", all.iter().any(|s| s.contains("\n\n")));
    coverage.insert("space run", all.iter().any(|s| s.contains("   ")));
    coverage.insert("tab", all.iter().any(|s| s.contains('\t')));
    coverage.insert("leading space", all.iter().any(|s| s.starts_with(' ')));
    coverage.insert("trailing space", all.iter().any(|s| s.ends_with(' ')));
    coverage.insert("special token", all.iter().any(|s| s.contains("<|im_start|>")));
    coverage.insert("contraction", all.iter().any(|s| s.contains("'s")));
    coverage.insert("digits", all.iter().any(|s| s.contains("1234567890")));
    coverage.insert("control char", all.iter().any(|s| s.contains('\u{0}')));
    coverage.insert("rtl", all.iter().any(|s| s.contains('ع')));
    coverage.insert("long input", all.iter().any(|s| s.len() > 400));

    let missing: Vec<&str> = coverage
        .iter()
        .filter(|(_, present)| !**present)
        .map(|(k, _)| *k)
        .collect();
    assert!(missing.is_empty(), "corpus is missing coverage for: {missing:?}");

    let n = sets.iter().map(|s| s.cases.len()).max().unwrap_or(0);
    assert!(n >= 200, "corpus has only {n} cases; the stage calls for 200+");
}
