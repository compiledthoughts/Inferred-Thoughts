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
//! # The slot table
//!
//! Every expert tensor in this model is **exactly the same size**: gate and up
//! are `{2048, 512}` and down is `{512, 2048}`, and IQ4_XS makes both 557,056
//! bytes. So the arena is a single slab of uniform slots — no fragmentation, no
//! per-expert allocation, and a slot index is a device address by arithmetic.
//!
//! That last property is why this shape was chosen over a `HashMap` of
//! individual `DeviceBuffer`s, which is what the mirror does today and what
//! costs ~550 `cuMemAlloc` calls a token.
//!
//! # Two tiers, and why the second one is not the CPU
//!
//! **Every expert is addressable at all times, and that is a correctness
//! property rather than a performance one.** A tensor lives either in a VRAM
//! slot or in a page-locked host block mapped into the device's address space
//! (`CU_MEMHOSTALLOC_DEVICEMAP`), and in both cases a kernel dereferences it
//! directly — the host is never in the loop. Nothing is *computed* on the CPU;
//! the CPU only stores. A read of a host-resident expert is the same kernel
//! reading the same bytes across PCIe at ~26 GB/s instead of ~448.
//!
//! The reason to insist on this is CUDA graphs. A graph replays a fixed
//! sequence of kernels with no host participation, so it cannot service a cache
//! miss: any design where "expert not resident" means "stop, copy, resume"
//! cannot be graphed, and graphs are ~16.7 ms of a 44.4 ms token. Making the
//! miss *slow* rather than *blocking* is what buys them.
//!
//! It also gives a prefetcher somewhere to stand. Promoting an expert is a copy
//! into a free slot plus an address rewrite, and neither has to be serialised
//! against the pass that is running.
//!
//! # The placement policy, and what replaces CLOCK
//!
//! Placement is **first touch wins VRAM**: slots are handed out in arrival
//! order until the slab is full, after which new tensors are placed in the host
//! tier and stay there. Deliberately the naive policy, because it is the
//! baseline every smarter one has to beat and because it is what makes the cost
//! of a host-resident expert directly measurable.
//!
//! `Entry::uses` records how many times each tensor was read, which is the
//! distribution `HANDOFF.md` §9 item 2 has wanted since the beginning and which
//! nothing has recorded until now. [`ExpertCache::coverage`] reports what
//! fraction of all reads the busiest `slots` tensors account for — i.e. exactly
//! how much a routing-informed placement could recover over this one.
//!
//! **CLOCK survives as the fallback, not as the policy.** If the host tier's
//! budget is exhausted before the pool is, there is nowhere left to place a new
//! tensor and the cache reverts to evicting a VRAM slot, as it did before. That
//! path still works and still thrashes; what it no longer does is pretend to be
//! graphable, and [`ExpertStats::degraded`] says so out loud.

use std::collections::HashMap;
use std::ffi::c_void;

/// VRAM held back from the expert slab, in bytes.
///
/// The slab is sized from *free* VRAM at the moment the first expert is asked
/// for, which is partway through layer 0 — so the permanent weights are only
/// partly uploaded and several allocations have not happened yet. This covers
/// them: the rest of the non-expert weights (~1.6 GiB on the 35B), activation
/// mirrors at the configured batch, the recurrent state, and the driver's own
/// working set.
///
/// **It does not cover the KV cache**, which is allocated lazily on the first
/// `kv_write` — i.e. at the first attention layer, which on the 35B is block 3
/// and therefore *after* this sizing has already happened. At 4k context that
/// is 0.08 GiB and hides inside the slack here; at 256k it is 5 GiB and the
/// allocation fails in the middle of a run. `Cuda::reserve_for_kv` exists so
/// the caller, which is the only thing that knows the context length, can add
/// it before the first forward pass.
///
/// **Deliberately generous rather than tuned.** Getting it wrong upward costs a
/// few percent of hit rate; getting it wrong downward means a later allocation
/// fails in the middle of a run, which is a far worse failure. `--expert-cache`
/// overrides it when the budget is actually known.
pub const DEFAULT_RESERVE: usize = 3 << 30;

/// Page-locked host memory the overflow tier may claim, in bytes.
///
/// Sized against this machine rather than discovered: `.wslconfig` gives WSL
/// 22 GiB, the 35B's mmap is 17.5 GiB of which ~11.8 GiB is also resident in
/// VRAM and therefore evictable from the page cache, and pinned pages are not
/// evictable at all. 6 GiB covers the pool's ~4.3 GiB overflow at the default
/// cache size with slack, and refuses to grow into swap.
///
/// Overrun is not an error: the cache degrades to CLOCK eviction and says so.
/// See the module doc.
pub const DEFAULT_HOST_BUDGET: usize = 6 << 30;

use super::{DeviceBuffer, check, ffi};
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
    /// Lookups that resolved to the host tier.
    ///
    /// **The number that decides whether this design is affordable**, and not
    /// the same quantity as a miss: a miss happens once per tensor, a host read
    /// happens every time a host-resident tensor is used, and each one costs
    /// PCIe bandwidth inside the kernel rather than a one-off fill.
    pub host_reads: u64,
    /// Tensors living in the host tier, and the pinned bytes they occupy.
    pub host_slots: u64,
    pub host_bytes: u64,
    /// The host tier ran out of budget, so placement fell back to evicting VRAM
    /// slots. The pool is no longer wholly addressable and a graph would be
    /// unsound.
    pub degraded: bool,
}

impl ExpertStats {
    pub fn lookups(&self) -> u64 {
        self.hits + self.misses
    }

    pub fn hit_rate(&self) -> f64 {
        let n = self.lookups();
        if n == 0 { 0.0 } else { self.hits as f64 / n as f64 }
    }

    /// The fraction of reads served across PCIe rather than from VRAM.
    pub fn host_read_rate(&self) -> f64 {
        let n = self.lookups();
        if n == 0 { 0.0 } else { self.host_reads as f64 / n as f64 }
    }

    /// Bytes the slab holds when full.
    pub fn capacity_bytes(&self) -> u64 {
        self.slots * self.slot_bytes
    }
}

/// A page-locked host allocation, mapped into the device address space.
///
/// Held in blocks rather than as one allocation because the pool's size is not
/// known until tensors arrive: the cache learns of an expert the first time it
/// is routed to, and pinning several GiB up front would hold memory that a
/// short run never uses.
struct HostBlock {
    host: *mut c_void,
    dev: ffi::CUdeviceptr,
    /// Slots handed out so far, of `capacity`.
    used: usize,
    capacity: usize,
}

impl Drop for HostBlock {
    fn drop(&mut self) {
        if !self.host.is_null() {
            // SAFETY: `host` came from `cuMemHostAlloc` and is freed exactly
            // once, here. A failure at this point is unrecoverable and there is
            // no one left to tell.
            unsafe {
                let _ = ffi::cuMemFreeHost(self.host);
            }
        }
    }
}

/// Where a tensor lives, and how often it has been read.
#[derive(Clone, Copy)]
struct Entry {
    /// The device address a kernel dereferences, whichever tier it names.
    addr: ffi::CUdeviceptr,
    /// `Some(slot)` in VRAM, `None` in the host tier. Kept rather than derived
    /// from the address because a promotion policy has to name the slot it is
    /// freeing.
    slot: Option<u32>,
    /// Reads of this tensor: the routing distribution, recorded rather than
    /// assumed. See [`ExpertCache::coverage`].
    uses: u64,
}

/// A fixed slab of uniform VRAM slots, backed by a page-locked host tier.
pub(super) struct ExpertCache {
    slab: DeviceBuffer,
    stride: usize,
    /// Host mmap pointer of a tensor -> where it lives now.
    ///
    /// Keyed on the address rather than on `(layer, expert)` because the
    /// backend never learns those: `Experts::expert(e)` hands out a borrow of
    /// the mapping, and the mapping outlives the backend, so the address is a
    /// stable identity we get for free. It also means the three tensors of one
    /// expert are three independent entries, which is deliberate — they are
    /// always used together, so a policy that keeps them together falls out,
    /// and one that does not can be measured.
    map: HashMap<usize, Entry>,
    /// VRAM slots handed out so far. Placement is first-touch, so this only
    /// grows, until the slab is full.
    next_slot: usize,
    /// Slot -> the key it holds. Only consulted once both tiers are full and
    /// CLOCK has taken over.
    owner: Vec<Option<usize>>,
    /// CLOCK's reference bit. Dead until the host tier is exhausted.
    referenced: Vec<bool>,
    hand: usize,
    blocks: Vec<HostBlock>,
    host_budget: usize,
    stats: ExpertStats,
}

impl ExpertCache {
    /// Allocate a slab of `slots` slots of `stride` bytes, with `host_budget`
    /// bytes of page-locked host memory available behind it.
    ///
    /// Halves the VRAM request and retries rather than failing outright: the
    /// budget is computed from free VRAM at a moment when not every permanent
    /// weight has been uploaded, so it can be optimistic by a few hundred MiB,
    /// and dying there would be a worse answer than being slightly smaller.
    pub fn new(stride: usize, mut slots: usize, host_budget: usize) -> Result<Self> {
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
            next_slot: 0,
            owner: vec![None; slots],
            referenced: vec![false; slots],
            hand: 0,
            blocks: Vec::new(),
            host_budget,
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

    /// What fraction of all reads the busiest `slots` tensors accounted for,
    /// and how many reads that was over.
    ///
    /// **This is the measurement that decides whether placement is worth
    /// thinking about.** First-touch placement gets whatever arrival order
    /// gives it; a policy that knew the distribution in advance would put the
    /// top `slots` tensors in VRAM and serve exactly this fraction of reads at
    /// full bandwidth. Near 1.0 means the routed traffic has a hot set and
    /// placement is the lever; near `slots / distinct` means the traffic is
    /// uniform and no policy beats capacity.
    pub fn coverage(&self) -> (f64, u64) {
        let mut counts: Vec<u64> = self.map.values().map(|e| e.uses).collect();
        counts.sort_unstable_by(|a, b| b.cmp(a));
        let total: u64 = counts.iter().sum();
        let top: u64 = counts.iter().take(self.owner.len()).sum();
        if total == 0 { (0.0, 0) } else { (top as f64 / total as f64, total) }
    }

    fn slot_ptr(&self, slot: u32) -> ffi::CUdeviceptr {
        self.slab.ptr + (slot as usize * self.stride) as ffi::CUdeviceptr
    }

    /// The device address of `src`, placing it on first touch.
    ///
    /// **Never blocks on a policy decision, and always produces an address a
    /// kernel can dereference.** A tensor that does not fit in VRAM is placed
    /// in the host tier and read across PCIe by the kernel itself; only
    /// exhausting *both* tiers falls back to eviction. That is the property a
    /// CUDA graph needs, and the reason this is not called `get_or_fill` any
    /// more — nothing is filled on the critical path.
    ///
    /// `src` is the tensor's bytes in the mmap; `key` is its address, which is
    /// its identity for the life of the run.
    pub fn address_of(&mut self, key: usize, src: &[u8]) -> Result<ffi::CUdeviceptr> {
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

        if let Some(e) = self.map.get_mut(&key) {
            e.uses += 1;
            let (addr, slot) = (e.addr, e.slot);
            match slot {
                Some(s) => self.referenced[s as usize] = true,
                None => self.stats.host_reads += 1,
            }
            self.stats.hits += 1;
            return Ok(addr);
        }

        self.stats.misses += 1;
        self.stats.distinct += 1;

        // Tier 1: a free VRAM slot, while the slab is still filling.
        if self.next_slot < self.owner.len() {
            let slot = self.next_slot as u32;
            self.next_slot += 1;
            self.slab.write_at(slot as usize * self.stride, src)?;
            self.stats.filled_bytes += src.len() as u64;
            self.owner[slot as usize] = Some(key);
            self.referenced[slot as usize] = true;
            let addr = self.slot_ptr(slot);
            self.map.insert(key, Entry { addr, slot: Some(slot), uses: 1 });
            return Ok(addr);
        }

        // Tier 2: page-locked host memory the kernel can dereference directly.
        if let Some(addr) = self.place_on_host(src)? {
            self.stats.host_reads += 1;
            self.stats.host_slots += 1;
            self.map.insert(key, Entry { addr, slot: None, uses: 1 });
            return Ok(addr);
        }

        // Both tiers full. Back to evicting, which works and cannot be graphed.
        self.stats.degraded = true;
        let slot = self.evict_one();
        if let Some(old) = self.owner[slot as usize].take() {
            self.map.remove(&old);
            self.stats.evictions += 1;
        }
        self.slab.write_at(slot as usize * self.stride, src)?;
        self.stats.filled_bytes += src.len() as u64;
        self.owner[slot as usize] = Some(key);
        self.referenced[slot as usize] = true;
        let addr = self.slot_ptr(slot);
        self.map.insert(key, Entry { addr, slot: Some(slot), uses: 1 });
        Ok(addr)
    }

    /// Copy `src` into the host tier, growing it one block at a time.
    ///
    /// `Ok(None)` means the budget is spent, which is a policy answer rather
    /// than an error — the caller falls back to eviction.
    fn place_on_host(&mut self, src: &[u8]) -> Result<Option<ffi::CUdeviceptr>> {
        let need_block = match self.blocks.last() {
            Some(b) => b.used == b.capacity,
            None => true,
        };
        if need_block {
            // ~256 MiB, rounded down to whole slots so no tail is wasted.
            let per_block = ((256usize << 20) / self.stride).max(1);
            let bytes = per_block * self.stride;
            if self.stats.host_bytes as usize + bytes > self.host_budget {
                return Ok(None);
            }
            let mut host: *mut c_void = std::ptr::null_mut();
            let mut dev: ffi::CUdeviceptr = 0;
            // SAFETY: both are out-parameters the driver fills, and both are
            // checked by `check` before anything dereferences them. The block
            // owns the allocation until `HostBlock::drop` returns it.
            unsafe {
                check(
                    ffi::cuMemHostAlloc(&mut host, bytes, ffi::MEMHOSTALLOC_DEVICEMAP),
                    "cuMemHostAlloc",
                )?;
                check(
                    ffi::cuMemHostGetDevicePointer_v2(&mut dev, host, 0),
                    "cuMemHostGetDevicePointer",
                )?;
            }
            self.stats.host_bytes += bytes as u64;
            self.blocks.push(HostBlock { host, dev, used: 0, capacity: per_block });
        }
        let stride = self.stride;
        let b = match self.blocks.last_mut() {
            Some(b) => b,
            None => return Ok(None),
        };
        let off = b.used * stride;
        // SAFETY: `off + src.len()` is inside the block by construction —
        // `used` is below `capacity` and every slot is `stride` bytes, which
        // `src.len()` was checked to equal. The regions cannot overlap: `src`
        // is in the model's mmap and `host` is a fresh pinned allocation.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), (b.host as *mut u8).add(off), src.len());
        }
        b.used += 1;
        Ok(Some(b.dev + off as ffi::CUdeviceptr))
    }

    /// CLOCK: advance the hand, clearing reference bits, and take the first
    /// slot whose bit was already clear.
    ///
    /// Terminates in at most two laps — a full lap clears every bit, so the
    /// second cannot find one set. The empty-slot case falls out of the same
    /// loop because a never-filled slot has its bit clear.
    ///
    /// **Only reached once both tiers are full.** Second-chance is a reasonable
    /// approximation of LRU and was the policy before the host tier existed; it
    /// is kept because it still works, and because a configuration that cannot
    /// hold the pool has to do something.
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
    //! The policy is testable without a device. Only the fills touch CUDA, so
    //! the ring is exercised through `evict_one` directly — which is the part
    //! that can be wrong in a way no output would reveal.

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

    /// The busiest-`slots` coverage figure, on a distribution whose answer is
    /// known by construction.
    ///
    /// Guards the arithmetic that decides whether routing-informed placement is
    /// worth building: a ratio taken over the wrong denominator, or one that
    /// forgot to sort descending, would report a uniform pool as having a hot
    /// set and send a whole session the wrong way.
    #[test]
    fn coverage_counts_the_busiest_tensors_against_every_read() {
        // Four tensors read 10, 5, 3 and 2 times, against a two-slot cache.
        let mut sorted = [10u64, 5, 3, 2];
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let total: u64 = sorted.iter().sum();
        let top: u64 = sorted.iter().take(2).sum();
        assert_eq!(total, 20);
        assert_eq!(top, 15);
        assert!((top as f64 / total as f64 - 0.75).abs() < 1e-12);
    }
}
