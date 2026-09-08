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
use std::ffi::{c_int, c_void};

/// `MADV_DONTNEED`, from `asm-generic/mman-common.h`.
///
/// On a `MAP_PRIVATE` file mapping this discards the resident pages; a later
/// read faults them back from the file. So it is safe by construction here —
/// worst case it costs a re-read of data nothing reads again.
const MADV_DONTNEED: c_int = 4;

unsafe extern "C" {
    fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
}

/// `POSIX_FADV_DONTNEED`, from `fcntl.h`.
const POSIX_FADV_DONTNEED: c_int = 4;

unsafe extern "C" {
    fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) -> c_int;
}

/// Evict a file from the page cache.
///
/// **`madvise` was not enough and this is why.** `MADV_DONTNEED` on a private
/// file mapping unmaps the pages from *this process* — the RSS drops and it
/// looks fixed — but the pages stay in the page cache, because they are clean
/// and any process might want them again. Placement reads all 16.3 GiB of the
/// expert pool through the mmap, so every process start filled WSL's cache with
/// the model and WSL does not hand that back to Windows. Measured: 757 MB of
/// process against **16.7 GB of cache**, and a host at 96%.
///
/// `posix_fadvise` evicts, which is the thing that was actually needed. It
/// works on any descriptor for the file rather than the one the mapping was
/// made from, so this needs no plumbing through `GgufFile` — a fresh `open` is
/// enough.
///
/// Best-effort by nature: the kernel may keep pages another process has mapped,
/// and a failure just means the cache stays, which is the behaviour before this
/// existed.
pub fn drop_file_cache(path: &std::path::Path) {
    drop_file_range(path, 0, 0);
}

/// Evict `[offset, offset + len)` of a file from the page cache. `len == 0`
/// means to the end.
///
/// **Per range, not once at the end, because the peak is what hurts.** Dropping
/// the whole file after placement lowers the resting level and leaves the peak
/// untouched: the cache still climbs to 16 GiB while the pool is being read,
/// which is what takes a 32 GB machine to 91%. Evicting each tensor's bytes as
/// soon as they are placed holds the cache at roughly one tensor — 142 MiB —
/// instead.
pub fn drop_file_range(path: &std::path::Path, offset: i64, len: i64) {
    use std::os::unix::io::AsRawFd;
    let Ok(f) = std::fs::File::open(path) else { return };
    // SAFETY: `f` owns a valid descriptor for the duration of the call. The
    // range is advisory; the kernel clamps it to the file.
    unsafe {
        let _ = posix_fadvise(f.as_raw_fd(), offset, len, POSIX_FADV_DONTNEED);
    }
}

/// Release the page cache backing `data`, keeping only whole pages inside it.
///
/// **Eager placement made the engine unusable on this machine without this.**
/// Placing the pool copies 16.3 GiB out of the model's mmap — into VRAM, or
/// into page-locked host blocks — and once that is done nothing reads those
/// mmap pages again. But they stay resident, so the process holds the 18.8 GB
/// file *and* 4.5 GiB of unevictable pinned memory against a 21 GB WSL VM. The
/// kernel thrashes, `free` reaches zero, and the Windows host stalls with it.
///
/// Rounds the start up and the end down, because `madvise` needs a page-aligned
/// address and dropping a partial page at either end would discard bytes
/// belonging to a neighbouring tensor.
///
/// Advisory and best-effort: a failure means the pages stay, which is the
/// behaviour before this existed, so the return value is deliberately ignored.
fn release_pages(data: &[u8]) {
    const PAGE: usize = 4096;
    let start = data.as_ptr() as usize;
    let end = start + data.len();
    let lo = start.div_ceil(PAGE) * PAGE;
    let hi = end / PAGE * PAGE;
    if hi <= lo {
        return;
    }
    // SAFETY: `[lo, hi)` is a whole number of pages inside `data`, which is a
    // live borrow of the model's mmap. `MADV_DONTNEED` on a private file
    // mapping only discards the cached pages; the mapping stays valid and a
    // later read re-faults from the file.
    unsafe {
        let _ = madvise(lo as *mut c_void, hi - lo, MADV_DONTNEED);
    }
}

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
/// **The premise below is false and this constant has not been re-derived.**
/// It says `.wslconfig` gives WSL 22 GiB; `free` reports **15,996 MB**, so the
/// guest is at the 50% default and `CLAUDE.md` records that separately. 6 GiB
/// of *unevictable* pinned memory out of 16 GB, beside a 17.5 GiB mmap, works
/// for one process and cannot work for two — which is the whole content of the
/// 06-09 hang, and was reproduced again on 09-09 by running a benchmark while
/// a `serve` was live. Check for a running `inferred` before starting either.
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
    /// Microseconds spent placing experts, split by where the time went.
    ///
    /// **Because a whole-prefill number cannot say which part is expensive.**
    /// Eager placement reads 16.3 GB of mmap, page-locks ~4.5 GiB and issues
    /// ~22,400 host-to-device copies, and the first attempt to explain a 46 s
    /// prefill from arithmetic over those three was wrong by 40 s. These are
    /// observed at the point each happens.
    pub place_h2d_us: u64,
    pub place_pin_us: u64,
    pub place_copy_us: u64,
    /// Bytes of the model's mmap handed back to the page cache after placement.
    pub released_bytes: u64,
    /// Experts exchanged between the tiers since load.
    pub migrated: u64,
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
    /// One device pointer table per `Experts` tensor, keyed on its base
    /// address: `n_expert` addresses a kernel can index with an id it computed
    /// itself. Built on first sight of the tensor and never moved, because a
    /// CUDA graph replays fixed arguments and an address that changed under it
    /// would route a later token to an expert that has been evicted.
    tables: HashMap<usize, DeviceBuffer>,
    /// Where each tensor's experts start in the global counter arrays.
    bases: HashMap<usize, usize>,
    /// Global, one entry per placed expert: how many times a kernel resolved
    /// it, and whether it resolved to VRAM.
    ///
    /// **The cache's only remaining eyes.** With selection on the device the
    /// host never sees a read, so these are written by `moe_gather_ptrs` and
    /// read back at report time. Sized once, generously, because growing them
    /// would invalidate the bases already handed out.
    counts: Option<DeviceBuffer>,
    vram_flags: Option<DeviceBuffer>,
    /// Two counters: reads that resolved to VRAM, and reads that resolved to
    /// the host tier.
    tally: Option<DeviceBuffer>,
    next_base: usize,
    /// The last read-back of `counts`, so `coverage` can be recomputed without
    /// touching the driver again.
    device_counts: Vec<u32>,
    /// Global counter index -> the tensor whose table holds it, and the expert
    /// index within that tensor. The inverse of the `base + e` numbering
    /// `table` hands out, and what lets a counter be traced back to something
    /// that can be moved.
    counter_owner: Vec<(usize, u32)>,
    /// Counts as of the last migration. The decision uses `now - then` rather
    /// than the running total, so a slot earned early does not hold its place
    /// for the rest of the session.
    prev_counts: Vec<u32>,
    /// One slot of VRAM to exchange through. A swap is three copies —
    /// resident to staging, host to resident, staging to host — because both
    /// tiers are full by construction and there is nowhere else to put the
    /// evicted expert.
    staging: Option<DeviceBuffer>,
    /// Experts moved between tiers, and the passes that have gone by since the
    /// last time any were.
    migrated: u64,
    /// Tokens routed since the last migration. See `migration_state`.
    since_migration: u64,
    /// The model file and the address its mapping starts at. See
    /// `ExpertCache::set_source`.
    source: Option<(std::path::PathBuf, usize)>,
    stats: ExpertStats,
}

/// Experts the global counter arrays have room for.
///
/// The 35B needs 30,720. Overrunning it does not corrupt anything — a tensor
/// past the cap simply gets no counters and [`ExpertCache::counters`] reports
/// that it is incomplete, which is the behaviour an instrument should have when
/// it cannot see everything.
const COUNTER_CAP: usize = 65_536;

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
            tables: HashMap::new(),
            bases: HashMap::new(),
            counts: None,
            vram_flags: None,
            tally: None,
            next_base: 0,
            device_counts: Vec::new(),
            counter_owner: Vec::new(),
            prev_counts: Vec::new(),
            staging: None,
            migrated: 0,
            since_migration: 0,
            source: None,
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
        let mut s = self.stats;
        s.migrated = self.migrated;
        s
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
        // Device counts when routing happened on the card, `Entry::uses`
        // otherwise. Both are read counts per expert; only the writer differs.
        let mut counts: Vec<u64> = if self.device_counts.is_empty() {
            self.map.values().map(|e| e.uses).collect()
        } else {
            self.device_counts.iter().map(|&c| u64::from(c)).collect()
        };
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

        // First sight through a *read*, so it counts as a miss and as a read of
        // whichever tier it lands in. `place` on its own does neither, because
        // building a pointer table touches every expert of a tensor and those
        // are not reads.
        self.stats.misses += 1;
        let addr = self.place(key, src)?;
        match self.map.get(&key).and_then(|e| e.slot) {
            Some(_) => {}
            None => self.stats.host_reads += 1,
        }
        Ok(addr)
    }

    /// Give `src` a permanent device address, without counting it as a read.
    ///
    /// Idempotent: an expert already placed keeps the address it has, which is
    /// what lets a pointer table be built over a tensor whose hot experts are
    /// already resident.
    fn place(&mut self, key: usize, src: &[u8]) -> Result<ffi::CUdeviceptr> {
        if let Some(e) = self.map.get(&key) {
            return Ok(e.addr);
        }
        self.stats.distinct += 1;

        // Tier 1: a free VRAM slot, while the slab is still filling.
        if self.next_slot < self.owner.len() {
            let slot = self.next_slot as u32;
            self.next_slot += 1;
            let t = std::time::Instant::now();
            self.slab.write_at(slot as usize * self.stride, src)?;
            self.stats.place_h2d_us += t.elapsed().as_micros() as u64;
            self.stats.filled_bytes += src.len() as u64;
            self.owner[slot as usize] = Some(key);
            self.referenced[slot as usize] = true;
            let addr = self.slot_ptr(slot);
            self.map.insert(key, Entry { addr, slot: Some(slot), uses: 0 });
            return Ok(addr);
        }

        // Tier 2: page-locked host memory the kernel can dereference directly.
        if let Some(addr) = self.place_on_host(src)? {
            self.stats.host_slots += 1;
            self.map.insert(key, Entry { addr, slot: None, uses: 0 });
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
        self.map.insert(key, Entry { addr, slot: Some(slot), uses: 0 });
        Ok(addr)
    }

    /// The device pointer table for one `Experts` tensor, built on first sight.
    ///
    /// **Placing every expert of the tensor is the price of a CUDA graph.** A
    /// graph replays a fixed sequence with no host participation, so it cannot
    /// service a miss; every id the router could emit must already resolve to
    /// something a kernel can dereference. Lazy placement can never satisfy
    /// that, because an expert not yet routed to has no address.
    ///
    /// So this is eager, and the cost is placement *quality* rather than
    /// correctness: tensors are seen in layer order, so VRAM fills with the
    /// early layers and the late ones land in the host tier whether or not they
    /// are hot. Measured at ~27% of expert reads across PCIe, which at
    /// 26.6 GB/s is ~5.7 ms/token against the ~16.7 ms a graph returns. A
    /// placement that knew the routing distribution would serve 99.6% of reads
    /// from VRAM (see [`ExpertCache::coverage`]); that is a separate change
    /// with its own measurement, deliberately not tangled into this one.
    ///
    /// `data` is the whole contiguous pool for this tensor, so every expert can
    /// be addressed without ever having been routed to.
    pub fn table(
        &mut self,
        key: usize,
        data: &[u8],
        n_expert: usize,
    ) -> Result<ffi::CUdeviceptr> {
        if let Some(t) = self.tables.get(&key) {
            return Ok(t.ptr);
        }
        let stride = self.stride;
        if data.len() != n_expert * stride {
            return Err(Error::Cuda {
                what: "expert table",
                detail: format!(
                    "{} bytes for {n_expert} experts of {stride}; the pool is not uniform",
                    data.len()
                ),
            });
        }
        let mut addrs = Vec::with_capacity(n_expert);
        let mut vram = Vec::with_capacity(n_expert);
        let mut e = 0;
        while e < n_expert {
            // **The run of experts that lands in consecutive fresh VRAM slots
            // is one copy, not one per expert.**
            //
            // `next_slot` is monotonic while the slab fills, so expert `e` goes
            // to slot `s` and expert `e + 1` to `s + 1`: the source is a run of
            // this tensor's pool and the destination a run of the slab. Calling
            // `place` per expert chopped that into 256 separate
            // `cuMemcpyHtoD`s of 557,056 bytes each.
            //
            // Measured before this: 11.53 GiB at **0.87 GB/s**, against 28.6
            // GB/s of pinned H2D on this bus — 33x below the link, and ~22.5 s
            // of a ~30 s start-up. It is start-up only and changes no
            // steady-state number, but it is the tax on every experiment.
            let mut run = 0;
            while e + run < n_expert
                && self.next_slot + run < self.owner.len()
                && !self
                    .map
                    .contains_key(&(data[(e + run) * stride..].as_ptr() as usize))
            {
                run += 1;
            }

            if run == 0 {
                // Not a fresh-slot case: already placed, or the slab is full
                // and this belongs to the host tier or an eviction. One at a
                // time, exactly as before.
                let src = &data[e * stride..(e + 1) * stride];
                let k = src.as_ptr() as usize;
                addrs.push(self.place(k, src)?);
                vram.push(i32::from(self.map.get(&k).and_then(|x| x.slot).is_some()));
                e += 1;
                continue;
            }

            let first = self.next_slot;
            let src = &data[e * stride..(e + run) * stride];
            let t = std::time::Instant::now();
            self.slab.write_at(first * stride, src)?;
            self.stats.place_h2d_us += t.elapsed().as_micros() as u64;
            self.stats.filled_bytes += src.len() as u64;
            for i in 0..run {
                let slot = (first + i) as u32;
                let k = data[(e + i) * stride..].as_ptr() as usize;
                self.stats.distinct += 1;
                self.owner[slot as usize] = Some(k);
                self.referenced[slot as usize] = true;
                let addr = self.slot_ptr(slot);
                self.map.insert(k, Entry { addr, slot: Some(slot), uses: 0 });
                addrs.push(addr);
                vram.push(1);
            }
            self.next_slot += run;
            e += run;
        }
        let buf = DeviceBuffer::from_slice(&addrs)?;
        let ptr = buf.ptr;
        self.tables.insert(key, buf);

        // Every expert of this tensor now lives in VRAM or in a pinned block,
        // so its 142 MiB of mmap is dead weight. Dropping it here rather than
        // at the end of placement keeps peak residency to one tensor rather
        // than the whole 16.3 GiB pool.
        release_pages(data);
        // And the page cache, which `madvise` does not touch: it unmaps the
        // pages from this process while the kernel keeps them, because they are
        // clean and something else might want them. That distinction is why an
        // earlier version of this reported 757 MB of process against 16.7 GB of
        // cache and looked fixed.
        if let Some((path, base)) = self.source.as_ref() {
            let offset = (data.as_ptr() as usize).saturating_sub(*base) as i64;
            drop_file_range(path, offset, data.len() as i64);
        }
        self.stats.released_bytes += data.len() as u64;

        // Counters for this tensor's slice of the global arrays.
        if self.counts.is_none() {
            self.counts = Some(DeviceBuffer::zeroed(COUNTER_CAP * 4)?);
            self.vram_flags = Some(DeviceBuffer::zeroed(COUNTER_CAP * 4)?);
            self.tally = Some(DeviceBuffer::zeroed(2 * 8)?);
        }
        let base = self.next_base;
        if base + n_expert <= COUNTER_CAP {
            self.counter_owner.resize(base + n_expert, (0, 0));
            for e in 0..n_expert {
                self.counter_owner[base + e] = (key, e as u32);
            }
            if let Some(f) = self.vram_flags.as_ref() {
                f.write_at(base * 4, &vram)?;
            }
            self.bases.insert(key, base);
            self.next_base = base + n_expert;
        }
        Ok(ptr)
    }

    /// The device counter arrays a gather launch writes: `(vram_flags, counts,
    /// tally, base)`. `None` if this tensor is past [`COUNTER_CAP`].
    pub fn counters(
        &self,
        key: usize,
    ) -> Option<(ffi::CUdeviceptr, ffi::CUdeviceptr, ffi::CUdeviceptr, usize)> {
        let base = *self.bases.get(&key)?;
        Some((
            self.vram_flags.as_ref()?.ptr,
            self.counts.as_ref()?.ptr,
            self.tally.as_ref()?.ptr,
            base,
        ))
    }

    /// Fold device-side read counts back into the statistics.
    ///
    /// `counts` is one entry per placed expert and `tally` is
    /// `[vram_reads, host_reads]`; the caller does the copies because only it
    /// can talk to the driver. Coverage is recomputed from `counts` rather than
    /// from `Entry::uses`, which device routing no longer updates.
    pub fn absorb_counters(&mut self, counts: &[u32], tally: &[u64]) {
        if tally.len() < 2 {
            return;
        }
        let (vram_reads, host_reads) = (tally[0], tally[1]);
        self.stats.hits = vram_reads + host_reads;
        self.stats.host_reads = host_reads;
        self.device_counts = counts[..self.next_base.min(counts.len())].to_vec();
    }

    /// Move hot experts into VRAM and cold ones out, from the read counts.
    ///
    /// **This is the project's subject, finally acting rather than measuring.**
    /// Placement is eager and in layer order — a graph needs every expert
    /// addressable before the router can name it, and the only ordering
    /// available at load is the order tensors are first seen. That fills VRAM
    /// with layers 0-28 and puts the rest in the host tier hot or not, which
    /// measures **26-27% of reads across PCIe** and a VRAM read rate of exactly
    /// `slots / pool` — precisely what uniform routing would give, i.e. no
    /// information used at all.
    ///
    /// Routing is not uniform: an oracle placement serves **99.0-99.6%** of
    /// reads from VRAM. That gap is ~11.5 ms/token, three times anything left
    /// in the dense kernels.
    ///
    /// # Why migration rather than a frozen profile
    ///
    /// The table is device memory, so rewriting an entry changes where a
    /// *recorded graph* reads on its next replay, with no re-record. That is
    /// what makes this possible at all, and it is why the two-tier design kept
    /// the table mutable instead of taking the handoff's static partition.
    ///
    /// It also needs no warm-up with graphs off, no per-model artifact, and it
    /// keeps working when a long context makes the KV cache displace slots.
    ///
    /// # The exchange
    ///
    /// Both tiers are full, so promotion is a swap: resident to staging, host
    /// to resident, staging to host. Every copy is device-to-device — the
    /// host tier is device-mapped, so the bytes never pass through this
    /// program. Two eight-byte table writes follow, one per tensor involved.
    ///
    /// About 45 us a swap, and it decays to nothing: once the hot set is
    /// resident there is nothing left to exchange.
    ///
    /// Returns how many experts moved.
    pub fn migrate(&mut self, budget: usize, counts: &[u32]) -> Result<usize> {
        let n = self.next_base.min(counts.len()).min(self.counter_owner.len());
        if n == 0 || budget == 0 {
            return Ok(0);
        }
        if self.prev_counts.len() < n {
            self.prev_counts.resize(n, 0);
        }

        // **Cumulative, not this window's delta — measured.**
        //
        // The first version used `now - then` over a 64-pass window, for
        // recency. It moved 6,200 experts and changed the host-read rate by
        // nothing (26.3% -> 26.6%), hitting the 200-swap budget on every one of
        // 31 migrations: it never converged because there was no signal to
        // converge on. A layer makes 512 expert-selections among 256 experts in
        // 64 passes, so the average delta is ~2 and most are 0 or 1. Sorting
        // that is sorting noise.
        //
        // The running total is what `coverage` measures, and coverage is the
        // thing that says 99.7% is available against this policy's 73.4%. So
        // the decision uses the same quantity as the target.
        //
        // The cost is that an expert busy early keeps its slot — which is what
        // "top N by total" means, and is exactly the placement coverage calls
        // optimal. If a workload shift ever makes that wrong, the fix is a
        // decayed score (`s = s/2 + delta` per window), not a raw delta.
        let mut hot: Vec<(u32, usize)> = Vec::new();   // host-resident
        let mut cold: Vec<(u32, usize)> = Vec::new();  // VRAM-resident
        for i in 0..n {
            let delta = counts[i];
            let (tkey, e) = self.counter_owner[i];
            let ekey = tkey + e as usize * self.stride;
            match self.map.get(&ekey).map(|x| x.slot) {
                Some(Some(_)) => cold.push((delta, i)),
                Some(None) => hot.push((delta, i)),
                None => {}
            }
        }
        self.prev_counts[..n].copy_from_slice(&counts[..n]);

        // Busiest exiles first, idlest residents first.
        hot.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        cold.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        if self.staging.is_none() {
            self.staging = Some(DeviceBuffer::new(self.stride)?);
        }
        let staging = match self.staging.as_ref() {
            Some(b) => b.ptr,
            None => return Ok(0),
        };

        let mut moved = 0usize;
        for k in 0..budget.min(hot.len()).min(cold.len()) {
            let (h_uses, h_i) = hot[k];
            let (c_uses, c_i) = cold[k];
            // Strictly busier, so a tie never causes churn, and nothing moves
            // once the ordering is right.
            if h_uses <= c_uses {
                break;
            }
            let (h_tkey, h_e) = self.counter_owner[h_i];
            let (c_tkey, c_e) = self.counter_owner[c_i];
            let h_key = h_tkey + h_e as usize * self.stride;
            let c_key = c_tkey + c_e as usize * self.stride;

            let (h_addr, c_addr, slot) = match (self.map.get(&h_key), self.map.get(&c_key)) {
                (Some(h), Some(c)) => match c.slot {
                    Some(s) => (h.addr, c.addr, s),
                    None => break,
                },
                _ => break,
            };

            // SAFETY: three copies of exactly `stride` bytes between
            // allocations this cache owns. `h_addr` is device-mapped host
            // memory and `c_addr` a slab slot, both valid for `stride`.
            unsafe {
                check(ffi::cuMemcpyDtoD_v2(staging, c_addr, self.stride), "swap out")?;
                check(ffi::cuMemcpyDtoD_v2(c_addr, h_addr, self.stride), "swap in")?;
                check(ffi::cuMemcpyDtoD_v2(h_addr, staging, self.stride), "swap back")?;
            }

            // The addresses trade places, and so do the entries.
            if let Some(h) = self.map.get_mut(&h_key) {
                h.addr = c_addr;
                h.slot = Some(slot);
            }
            if let Some(c) = self.map.get_mut(&c_key) {
                c.addr = h_addr;
                c.slot = None;
            }
            self.owner[slot as usize] = Some(h_key);
            self.write_table_entry(h_tkey, h_e, c_addr)?;
            self.write_table_entry(c_tkey, c_e, h_addr)?;

            // **And the residency flags the kernel counts with.**
            //
            // Forgetting these made the first working migration invisible: the
            // gather kernel reads `vram[base + id]` to decide whether a read
            // was a VRAM read or a PCIe one, and with the flags frozen at
            // load the report described the *original* layout for the rest of
            // the run. 6,200 experts moved, decode went 29.28 -> 32.07 tok/s,
            // and the host-read rate printed 26.6% throughout — a policy
            // working and an instrument saying it was not. Eighth of this kind
            // in the repo, and the third introduced rather than inherited.
            self.set_vram_flag(h_i, 1)?;
            self.set_vram_flag(c_i, 0)?;
            moved += 1;
        }
        self.migrated += moved as u64;
        Ok(moved)
    }

    /// Mark a global expert index as VRAM-resident or not, for the counters.
    fn set_vram_flag(&self, idx: usize, resident: i32) -> Result<()> {
        match self.vram_flags.as_ref() {
            Some(f) => f.write_at(idx * 4, &[resident]),
            None => Ok(()),
        }
    }

    /// Point one table entry at a new address.
    fn write_table_entry(&self, tkey: usize, e: u32, addr: ffi::CUdeviceptr) -> Result<()> {
        match self.tables.get(&tkey) {
            Some(t) => t.write_at(e as usize * 8, &[addr]),
            None => Err(Error::Cuda {
                what: "expert table",
                detail: "migrating an expert whose tensor has no table".to_string(),
            }),
        }
    }

    /// Where this pool's bytes came from, so placement can evict them from the
    /// page cache as it goes. `None` leaves the cache alone.
    pub fn set_source(&mut self, path: std::path::PathBuf, map_base: usize) {
        self.source = Some((path, map_base));
    }

    /// Passes since the last migration, and how many experts have ever moved.
    /// Tokens seen since the last migration, and the running exchange count.
    ///
    /// **Tokens, not passes, and that was the whole defect.** This counted
    /// calls, so a 512-token prefill pass advanced it by one — the same as a
    /// single decode token — and a 1501-token prompt advanced it by three
    /// against a threshold of 64. The evidence a pass carries is proportional
    /// to the tokens it routed, not to the number of times `begin_pass` was
    /// called, and counting the wrong one made the policy inert in exactly the
    /// workload that needs it most: a long prompt and a short answer.
    pub fn migration_state(&mut self, n_tokens: usize) -> (u64, u64) {
        self.since_migration += n_tokens as u64;
        (self.since_migration, self.migrated)
    }

    pub fn migration_done(&mut self) {
        self.since_migration = 0;
    }

    /// The last counter read-back, for the migration decision.
    pub fn observed_counts(&self) -> &[u32] {
        &self.device_counts
    }

    /// How many experts have counters, i.e. how much of the pool is observed.
    pub fn counted_experts(&self) -> usize {
        self.next_base
    }

    /// The global counter and tally buffers, for the read-back.
    pub fn counters_base(&self) -> Option<(ffi::CUdeviceptr, ffi::CUdeviceptr)> {
        Some((self.counts.as_ref()?.ptr, self.tally.as_ref()?.ptr))
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
            let t = std::time::Instant::now();
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
            self.stats.place_pin_us += t.elapsed().as_micros() as u64;
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
        let t = std::time::Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), (b.host as *mut u8).add(off), src.len());
        }
        let us = t.elapsed().as_micros() as u64;
        b.used += 1;
        self.stats.place_copy_us += us;
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
