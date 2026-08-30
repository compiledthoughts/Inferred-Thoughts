//! The KV cache.
//!
//! Without one, `forward` recomputes every past token's K and V on every step,
//! which makes generation quadratic: ~430 ms/token at 12 tokens of context and
//! ~15 s/token by 500. That is the single largest speedup available in v0 and
//! it is not a kernel problem.
//!
//! **Storage is f16, and that is a semantic choice rather than a space
//! optimization.** llama.cpp writes K and V into an f16 cache and computes
//! attention on the values it reads back, so its scores are over f16-rounded
//! K and V. Stage 4 reproduced that with an explicit round-trip immediately
//! before attention (see the comment it replaced in `model/qwen3.rs`); doing it
//! at the cache boundary is the same arithmetic in the honest place. The gap is
//! worth ~3e-3 of tensor magnitude — a real semantic difference, not rounding
//! noise, so an f32 cache would *increase* divergence from the reference.
//!
//! It also halves the memory, which matters later: on the MoE model every byte
//! the KV cache takes is a byte the expert cache does not get.
//!
//! Layout is `[layer][position][kv_dim]`, contiguous in that order. Attention
//! walks positions for a fixed head, so it strides by `kv_dim` — the same
//! access pattern the Stage 4 code had over its full-sequence buffers.

use crate::error::{Error, Result};
use crate::quant::half::{f16_to_f32, f32_to_f16};

/// Per-layer key and value history, stored as raw f16 bits.
#[derive(Debug)]
pub struct KvCache {
    n_layer: usize,
    kv_dim: usize,
    n_ctx: usize,
    /// Positions filled, i.e. the next free slot.
    len: usize,
    k: Vec<u16>,
    v: Vec<u16>,
}

impl KvCache {
    pub fn new(n_layer: usize, kv_dim: usize, n_ctx: usize) -> Self {
        let cells = n_layer * n_ctx * kv_dim;
        Self {
            n_layer,
            kv_dim,
            n_ctx,
            len: 0,
            k: vec![0; cells],
            v: vec![0; cells],
        }
    }

    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Number of positions currently held.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total resident bytes, both tensors, whether or not they are filled.
    pub fn capacity_bytes(&self) -> u64 {
        (self.k.len() + self.v.len()) as u64 * 2
    }

    /// Bytes one position occupies across all layers — what a decode step
    /// writes, and what each additional position adds to every later read.
    pub fn bytes_per_position(&self) -> u64 {
        (self.n_layer * self.kv_dim * 2 * 2) as u64
    }

    /// Forget everything. Buffers are kept so a second run does not reallocate.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    fn offset(&self, il: usize, pos: usize) -> usize {
        (il * self.n_ctx + pos) * self.kv_dim
    }

    /// Write one position's K and V for one layer, rounding to f16 on the way
    /// in. Returns an error rather than truncating when the context is full,
    /// because silently dropping positions produces plausible-looking wrong
    /// text instead of a failure.
    pub fn store(&mut self, il: usize, pos: usize, k: &[f32], v: &[f32]) -> Result<()> {
        if pos >= self.n_ctx {
            return Err(Error::ContextOverflow {
                pos,
                n_ctx: self.n_ctx,
            });
        }
        debug_assert!(il < self.n_layer);
        debug_assert_eq!(k.len(), self.kv_dim);
        debug_assert_eq!(v.len(), self.kv_dim);

        let at = self.offset(il, pos);
        for i in 0..self.kv_dim {
            self.k[at + i] = f32_to_f16(k[i]);
            self.v[at + i] = f32_to_f16(v[i]);
        }
        Ok(())
    }

    /// Mark positions `0..len` as valid. Called once per `forward` after every
    /// layer has stored its K and V, so a failed pass cannot leave the cache
    /// claiming positions it never wrote.
    pub fn commit(&mut self, len: usize) {
        debug_assert!(len <= self.n_ctx);
        self.len = len;
    }

    /// A whole layer's key slab, `n_ctx * kv_dim` f16 bits, position-major.
    ///
    /// Handed to [`crate::ops::Ops::attend`] so the backend can walk positions
    /// itself and thread over heads. The ops layer sees raw bits and a stride,
    /// never this type.
    #[inline]
    pub fn k_layer(&self, il: usize) -> &[u16] {
        let at = self.offset(il, 0);
        &self.k[at..at + self.n_ctx * self.kv_dim]
    }

    #[inline]
    pub fn v_layer(&self, il: usize) -> &[u16] {
        let at = self.offset(il, 0);
        &self.v[at..at + self.n_ctx * self.kv_dim]
    }

    /// One head's slice of K at a position, still as f16 bits.
    ///
    /// Callers convert per element inside the dot product rather than
    /// materializing an f32 copy — see the memory rule in `CLAUDE.md`.
    #[inline]
    pub fn k_head(&self, il: usize, pos: usize, head_off: usize, head_dim: usize) -> &[u16] {
        let at = self.offset(il, pos) + head_off;
        &self.k[at..at + head_dim]
    }

    #[inline]
    pub fn v_head(&self, il: usize, pos: usize, head_off: usize, head_dim: usize) -> &[u16] {
        let at = self.offset(il, pos) + head_off;
        &self.v[at..at + head_dim]
    }

    /// The f32 value attention will actually see for a stored element. Exposed
    /// so tests can assert the round-trip explicitly rather than inferring it.
    #[inline]
    pub fn read(bits: u16) -> f32 {
        f16_to_f32(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled() -> KvCache {
        // 2 layers, kv_dim 4, room for 3 positions.
        let mut c = KvCache::new(2, 4, 3);
        for il in 0..2 {
            for pos in 0..3 {
                let base = (il * 10 + pos) as f32;
                let k: Vec<f32> = (0..4).map(|i| base + i as f32 * 0.5).collect();
                let v: Vec<f32> = (0..4).map(|i| -(base + i as f32 * 0.5)).collect();
                c.store(il, pos, &k, &v).unwrap();
            }
        }
        c.commit(3);
        c
    }

    #[test]
    fn layers_and_positions_do_not_alias() {
        let c = filled();
        // Layer 1 position 0 must not read back as layer 0 position 1, which is
        // what a wrong stride would produce.
        let l0p1 = KvCache::read(c.k_head(0, 1, 0, 4)[0]);
        let l1p0 = KvCache::read(c.k_head(1, 0, 0, 4)[0]);
        assert_eq!(l0p1, 1.0);
        assert_eq!(l1p0, 10.0);
    }

    #[test]
    fn head_offset_selects_within_a_position() {
        let c = filled();
        // kv_dim 4 as two heads of 2: head 1 starts at offset 2.
        let head1 = c.k_head(0, 0, 2, 2);
        assert_eq!(KvCache::read(head1[0]), 1.0);
        assert_eq!(KvCache::read(head1[1]), 1.5);
    }

    #[test]
    fn k_and_v_are_separate() {
        let c = filled();
        assert_eq!(KvCache::read(c.k_head(1, 2, 0, 4)[3]), 13.5);
        assert_eq!(KvCache::read(c.v_head(1, 2, 0, 4)[3]), -13.5);
    }

    /// The f16 round-trip is the point of the cache, not an accident of it, so
    /// it gets an explicit test: a value f16 cannot represent must come back
    /// rounded, and must come back *identically* every time it is read.
    #[test]
    fn storage_rounds_through_f16() {
        let mut c = KvCache::new(1, 1, 1);
        let exact = 1.0f32 / 3.0;
        c.store(0, 0, &[exact], &[exact]).unwrap();
        let got = KvCache::read(c.k_head(0, 0, 0, 1)[0]);
        assert_ne!(got, exact, "an f32 cache would defeat the point");
        assert!((got - exact).abs() < 1e-3);
        assert_eq!(got, f16_to_f32(f32_to_f16(exact)));
    }

    #[test]
    fn overflow_is_an_error_not_a_truncation() {
        let mut c = KvCache::new(1, 2, 2);
        assert!(c.store(0, 1, &[0.0, 0.0], &[0.0, 0.0]).is_ok());
        let e = c.store(0, 2, &[0.0, 0.0], &[0.0, 0.0]).unwrap_err();
        assert!(matches!(e, Error::ContextOverflow { pos: 2, n_ctx: 2 }));
    }

    #[test]
    fn reset_forgets_positions_but_keeps_capacity() {
        let mut c = filled();
        let bytes = c.capacity_bytes();
        c.reset();
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
        assert_eq!(c.capacity_bytes(), bytes);
    }

    #[test]
    fn byte_accounting_matches_the_layout() {
        let c = KvCache::new(28, 1024, 4096);
        // 28 layers * 4096 positions * 1024 lanes * 2 bytes * (K and V)
        assert_eq!(c.capacity_bytes(), 28 * 4096 * 1024 * 2 * 2);
        assert_eq!(c.bytes_per_position(), 28 * 1024 * 2 * 2);
    }
}
