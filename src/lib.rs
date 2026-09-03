//! inferredThoughts — a from-scratch inference engine for GGUF models.
//!
//! Stages 1-5. See `CLAUDE.md` for the rules this crate is written
//! under and `HANDOFF.md` for why the project exists.

pub mod cache;
pub mod engine;
pub mod error;
pub mod gguf;
pub mod model;
pub mod ops;
pub mod profile;
pub mod quant;
pub mod serve;
pub mod tok;

pub use cache::{KvCache, RecurrentState};
pub use engine::Engine;
pub use error::{Error, Result};
pub use gguf::GgufFile;
pub use model::{Model, Qwen3, Qwen35};
pub use ops::{Ops, naive::Naive, par::Par, spin::Spin};
#[cfg(feature = "cuda")]
pub use ops::cuda::{Cuda, DeviceBench, DeviceStats, Resident};
pub use profile::{Ctx, Profile};
pub use tok::Tokenizer;
