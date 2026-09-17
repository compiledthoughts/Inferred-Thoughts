//! Binary dumps for offline analysis, each behind its own environment variable.
//!
//! Unlike [`crate::profile`], these **do** perturb a run: a dump may read device
//! state back mid-pass, which turns CUDA graphs off. They exist to answer a design
//! question from a real run — which prefetch predictor to build (TODO.md, tier 3)
//! — and are read by `scripts/prefetch_recall.py`. A run with a dump switched on
//! is a measurement of *what* the model did, never of how fast.
//!
//! Every record starts with a four-byte tag and is little-endian throughout.
//! A write error is reported once on stderr and the dump stops; the run goes on.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Mutex, OnceLock};

/// A dump file, opened on first use if its variable names a path.
pub struct Dump {
    out: Mutex<Option<BufWriter<File>>>,
    name: &'static str,
}

impl Dump {
    fn open(var: &'static str) -> Option<Dump> {
        let path = std::env::var(var).ok()?;
        match File::create(&path) {
            Ok(f) => {
                eprintln!("dump: {var} -> {path} (a measurement of what ran, not of speed)");
                Some(Dump { out: Mutex::new(Some(BufWriter::new(f))), name: var })
            }
            Err(e) => {
                eprintln!("dump: {var}: cannot create {path}: {e}; not dumping");
                None
            }
        }
    }

    /// Append one record, given as its parts in order.
    pub fn record(&self, parts: &[&[u8]]) {
        let Ok(mut guard) = self.out.lock() else { return };
        let Some(w) = guard.as_mut() else { return };
        let r = parts.iter().try_for_each(|p| w.write_all(p)).and_then(|()| w.flush());
        if let Err(e) = r {
            eprintln!("dump: {}: write failed ({e}); stopping this dump", self.name);
            *guard = None;
        }
    }
}

/// `INFERRED_ROUTER_DUMP`: each MoE layer's router input, per pass.
///
/// Record: `b"RDMP"`, layer `u32`, start position `u64`, rows `u32`, width `u32`,
/// then `rows * width` `f32`.
pub fn router() -> Option<&'static Dump> {
    static D: OnceLock<Option<Dump>> = OnceLock::new();
    D.get_or_init(|| Dump::open("INFERRED_ROUTER_DUMP")).as_ref()
}

/// `INFERRED_EXPERT_LOG`: each expert resolve while oversubscribed.
///
/// Record: `b"ELOG"`, pass `u64`, tensor key `u64`, rows `u32`, picks per row
/// `u32`, experts in the tensor `u32`, then `rows * picks` `i32` ids and as many
/// `u8` flags, 1 where the pick was cold (fetched from the file) at that moment.
pub fn experts() -> Option<&'static Dump> {
    static D: OnceLock<Option<Dump>> = OnceLock::new();
    D.get_or_init(|| Dump::open("INFERRED_EXPERT_LOG")).as_ref()
}

/// Bytes of a slice of plain numbers, little-endian on this target.
pub fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: `T` is `Copy` and only ever a primitive number here (f32, i32, u8),
    // with no padding; the slice is read as its own bytes for its own lifetime.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
