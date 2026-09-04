//! Bounded VRAM residency for the MoE expert pool.
//!
//! **This is the project's subject.** Everything else in the crate exists to
//! hold one question still while it is answered: which expert bytes live in
//! VRAM, and when do they move. Up to here the CUDA backend answered it by not
//! answering — `Cuda::resident` uploads a tensor on first touch and keeps it
//! forever, which is right for a model that fits and impossible for one that
//! does not. Measured on the 35B: 63.5% of the pool resident by token 282,
//! still growing 35 tensors a token, and the card exhausted around token 426.
//!
//! # Why a cache rather than putting layers on the CPU
//!
//! Layer-wise CPU placement also makes the model fit — 11 layers' experts in
//! host RAM leaves 11.56 GiB against 14.80 free — and it is what llama.cpp's
//! `-ncmoe 11` does. It is the wrong trade here. **Host RAM is 43.4 GB/s at its
//! best and VRAM is 448**, so a byte handed to the CPU costs about ten times a
//! byte read on the GPU, and this model is only 3.5 GiB short. Per-token floors:
//! 15.9 ms with 11 CPU layers on our current CPU kernel, 7.95 ms with a perfect
//! one, and **5.20 ms streaming the ~22% that does not fit**. `HANDOFF.md`'s
//! 05-09 (late) entry has the derivation.
//!
//! PCIe is the binding term in that last figure — 5.20 ms of transfer against
//! 4.43 ms of compute — so this cache is not a way to avoid moving bytes. It is
//! a way to move fewer of them, and eventually to move them *early*.
//!
//! # The slot table
//!
//! Every expert tensor in this model is **exactly the same size**: gate and up
//! are `{2048, 512}` and down is `{512, 2048}`, and IQ4_XS makes both 557,056
//! bytes. So the arena is a single slab of uniform slots — no fragmentation, no
//! per-expert allocation, and a slot index is a device address by arithmetic.
//!
//! That last property is why this shape was chosen over a `HashMap` of
//! individual `DeviceBuffer`s, which is what the mirror does today and what
//! costs ~550 `cuMemAlloc` calls a token. A slot *index* is something a kernel
//! can be handed, which is what expert selection needs before it can move onto
//! the device — and moving it there is what removes the two host syncs per
//! layer per token and lets the decode step be a CUDA graph again. One
//! structure, three problems.
//!
//! # The policy is CLOCK, and it is meant to be replaced
//!
//! Second-chance: a ring of slots each carrying one reference bit, set on use.
//! A miss advances a hand, clearing bits as it goes, and takes the first slot
//! whose bit was already clear. O(1) amortized, one bit of state per slot, and
//! a good approximation of LRU.
//!
//! Exact LRU was not chosen because it costs an intrusive list threaded through
//! ~24,000 slots touched ~960 times a token, and because the measurement that
//! would justify the difference does not exist yet. Routing on this model looks
//! close to uniform — 63.5% of the pool touched inside 282 tokens, with no small
//! hot set — and policies converge on uniform traffic. **The hit rate is
//! counted so that claim can be checked rather than repeated**; if CLOCK turns
//! out to leave something on the table, this is the one type that has to change.

use std::collections::HashMap;

/// VRAM held back from the expert slab, in bytes.
///
/// The slab is sized from *free* VRAM at the moment the first expert is asked
/// for, which is partway through layer 0 — so the permanent weights are only
/// partly uploaded and several allocations have not happened yet. This covers
/// them: the rest of the non-expert weights (~1.6 GiB on the 35B), the KV cache
/// at full context, activation mirrors at the configured batch, the recurrent
/// state, and the driver's own working set.
///
/// **Deliberately generous rather than tuned.** Getting it wrong upward costs a
/// few percent of hit rate; getting it wrong downward means a later allocation
/// fails in the middle of a run, which is a far worse failure. `--expert-cache`
/// overrides it when the budget is actually known.
pub const DEFAULT_RESERVE: usize = 3 << 30;

use super::{DeviceBuffer, ffi};
use crate::error::{Error, Result};

/// What the cache did, so the policy can be judged instead of assumed.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExpertStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Bytes uploaded on misses. The PCIe term of the per-token floor, observed.
    pub filled_bytes: u64,
    pub slots: u64,
    pub slot_bytes: u64,
    /// Distinct tensors the cache has ever been asked for. Against `slots`,
    /// this says whether the working set fits or the cache is thrashing.
    pub distinct: u64,
}

impl ExpertStats {
    pub fn lookups(&self) -> u64 {
        self.hits + self.misses
    }

    pub fn hit_rate(&self) -> f64 {
        let n = self.lookups();
        if n == 0 { 0.0 } else { self.hits as f64 / n as f64 }
    }

    /// Bytes the slab holds when full.
    pub fn capacity_bytes(&self) -> u64 {
        self.slots * self.slot_bytes
    }
}

/// A fixed slab of uniform slots, with a CLOCK eviction policy.
pub(super) struct ExpertCache {
    slab: DeviceBuffer,
    stride: usize,
    /// Host mmap pointer of a resident tensor -> the slot holding it.
    ///
    /// Keyed on the address rather than on `(layer, expert)` because the
    /// backend never learns those: `Experts::expert(e)` hands out a borrow of
    /// the mapping, and the mapping outlives the backend, so the address is a
    /// stable identity we get for free. It also means the three tensors of one
    /// expert are three independent entries, which is deliberate — they are
    /// always used together, so a policy that keeps them together falls out,
    /// and one that does not can be measured.
    map: HashMap<usize, u32>,
    /// Slot -> the key it holds, if any.
    owner: Vec<Option<usize>>,
    /// CLOCK's reference bit, set on every hit and on fill.
    referenced: Vec<bool>,
    hand: usize,
    stats: ExpertStats,
}

impl ExpertCache {
    /// Allocate a slab of `slots` slots of `stride` bytes.
    ///
    /// Halves the request and retries rather than failing outright: the
    /// budget is computed from free VRAM at a moment when not every permanent
    /// weight has been uploaded, so it can be optimistic by a few hundred MiB,
    /// and dying there would be a worse answer than being slightly smaller.
    pub fn new(stride: usize, mut slots: usize) -> Result<Self> {
        if stride == 0 {
            return Err(Error::Cuda {
                what: "expert cache",
                detail: "zero-sized expert tensor".to_string(),
            });
        }
        let slab = loop {
            if slots == 0 {
                return Err(Error::Cuda {
                    what: "expert cache",
                    detail: "no VRAM left for even one expert slot".to_string(),
                });
            }
            match DeviceBuffer::new(slots * stride) {
                Ok(b) => break b,
                Err(_) if slots > 1 => slots /= 2,
                Err(e) => return Err(e),
            }
        };
        Ok(Self {
            slab,
            stride,
            map: HashMap::with_capacity(slots * 2),
            owner: vec![None; slots],
            referenced: vec![false; slots],
            hand: 0,
            stats: ExpertStats {
                slots: slots as u64,
                slot_bytes: stride as u64,
                ..Default::default()
            },
        })
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn stats(&self) -> ExpertStats {
        self.stats
    }

    pub fn resident_bytes(&self) -> u64 {
        self.slab.len_bytes() as u64
    }

    fn slot_ptr(&self, slot: u32) -> ffi::CUdeviceptr {
        self.slab.ptr + (slot as usize * self.stride) as ffi::CUdeviceptr
    }

    /// The device address of `src`, filling a slot from host memory on a miss.
    ///
    /// `src` is the tensor's bytes in the mmap; `key` is its address, which is
    /// its identity for the life of the run.
    pub fn get_or_fill(&mut self, key: usize, src: &[u8]) -> Result<ffi::CUdeviceptr> {
        if src.len() != self.stride {
            return Err(Error::Cuda {
                what: "expert cache",
                detail: format!(
                    "expert tensor is {} bytes but the slab's slot is {}; the pool is \
                     not uniform, which this arena assumes",
                    src.len(),
                    self.stride
                ),
            });
        }

        if let Some(&slot) = self.map.get(&key) {
            self.referenced[slot as usize] = true;
            self.stats.hits += 1;
            return Ok(self.slot_ptr(slot));
        }

        self.stats.misses += 1;
        self.stats.distinct += 1;
        let slot = self.evict_one();
        if let Some(old) = self.owner[slot as usize].take() {
            self.map.remove(&old);
            self.stats.evictions += 1;
        }

        // The fill. This is the PCIe term of the per-token floor, and the whole
        // point of a prefetcher later is that this copy should already have
        // happened by the time the layer runs.
        self.slab.write_at(slot as usize * self.stride, src)?;
        self.stats.filled_bytes += src.len() as u64;

        self.map.insert(key, slot);
        self.owner[slot as usize] = Some(key);
        self.referenced[slot as usize] = true;
        Ok(self.slot_ptr(slot))
    }

    /// CLOCK: advance the hand, clearing reference bits, and take the first
    /// slot whose bit was already clear.
    ///
    /// Terminates in at most two laps — a full lap clears every bit, so the
    /// second cannot find one set. The empty-slot case falls out of the same
    /// loop because a never-filled slot has its bit clear.
    fn evict_one(&mut self) -> u32 {
        let n = self.owner.len();
        for _ in 0..2 * n {
            let at = self.hand;
            self.hand = (self.hand + 1) % n;
            if self.referenced[at] {
                self.referenced[at] = false;
            } else {
                return at as u32;
            }
        }
        // Unreachable by the argument above; taking the hand is still correct.
        self.hand as u32
    }
}

#[cfg(test)]
mod tests {
    //! The policy is testable without a device. Only `get_or_fill` touches
    //! CUDA, so the ring is exercised through `evict_one` directly — which is
    //! the part that can be wrong in a way no output would reveal.

    /// A standalone CLOCK ring, identical to [`super::ExpertCache`]'s, so the
    /// policy can be tested where the slab cannot be allocated.
    struct Clock {
        referenced: Vec<bool>,
        hand: usize,
    }

    impl Clock {
        fn new(n: usize) -> Self {
            Self { referenced: vec![false; n], hand: 0 }
        }
        fn evict(&mut self) -> usize {
            let n = self.referenced.len();
            for _ in 0..2 * n {
                let at = self.hand;
                self.hand = (self.hand + 1) % n;
                if self.referenced[at] {
                    self.referenced[at] = false;
                } else {
                    return at;
                }
            }
            self.hand
        }
    }

    /// Every slot is handed out once before any is reused. A ring that returned
    /// the same slot twice would look like a working cache with a terrible hit
    /// rate, which is exactly the failure that would be blamed on the policy.
    #[test]
    fn a_cold_ring_fills_every_slot_before_evicting_any() {
        let mut c = Clock::new(8);
        let got: Vec<usize> = (0..8).map(|_| c.evict()).collect();
        assert_eq!(got, (0..8).collect::<Vec<_>>());
    }

    /// A referenced slot survives exactly one pass of the hand, which is what
    /// "second chance" means and what makes this an LRU approximation rather
    /// than FIFO.
    #[test]
    fn a_referenced_slot_gets_one_second_chance_and_no_more() {
        let mut c = Clock::new(4);
        for _ in 0..4 {
            c.evict();
        }
        c.referenced = vec![true, false, false, false];
        // Slot 0 is protected this lap, so slot 1 goes first.
        assert_eq!(c.evict(), 1);
        assert!(!c.referenced[0], "the hand must clear the bit as it passes");
        // Its bit is now clear, so the next lap takes it.
        assert_eq!(c.evict(), 2);
        assert_eq!(c.evict(), 3);
        assert_eq!(c.evict(), 0);
    }

    /// With every bit set the hand still terminates, clearing a whole lap
    /// first. An implementation that scanned for a clear bit without clearing
    /// would spin forever here, and it would only ever happen on a full cache
    /// under load.
    #[test]
    fn a_fully_referenced_ring_still_terminates() {
        let mut c = Clock::new(6);
        c.referenced = vec![true; 6];
        assert_eq!(c.evict(), 0);
        assert!(c.referenced.iter().all(|r| !r));
    }
}
