//! inferredThoughts — a from-scratch inference engine for GGUF models.
//!
//! v0 is correctness only. See `CLAUDE.md` for the rules this crate is written
//! under and `HANDOFF.md` for why the project exists.

pub mod error;
pub mod gguf;
pub mod model;
pub mod ops;
pub mod quant;
pub mod tok;

pub use error::{Error, Result};
pub use gguf::GgufFile;
pub use model::Qwen3;
pub use ops::{Ops, naive::Naive};
pub use tok::Tokenizer;
