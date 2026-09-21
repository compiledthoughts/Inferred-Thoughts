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
//! # When both tiers are full: cold experts, fetched on demand (tier 3's MVP)
//!
//! If the host tier's budget runs out before the pool does, the rest of the
//! experts are **cold** (SSD-TIER.md D12). A cold expert has no slot; its table
//! entry points at a zeroed sentinel, and [`ExpertCache::resolve`] fetches it
//! from the model file into a VRAM slot when a layer picks it — between the
//! table lookup and the gather that reads it, so a kernel never sees a cold
//! entry. Fetching evicts an expert the current layer did not pick (the lease),
//! and **repoints the evicted expert's own table entry at the sentinel first**.
//!
//! That last step is the one this replaced. The old fallback evicted a VRAM slot
//! and wrote the new expert there but left the victim's entry pointing at the
//! slot — presumably correct while every read resolved through
//! [`ExpertCache::address_of`] on the host, and wrong once the pointer table moved
//! to the device. Measured 14-09-2026: every oversubscribed run of the
//! expert-cap sweep generated garbage, worse as evictions rose, and a chat prompt
//! at 4 + 4 GiB emitted EOS as its first token. The acceptance test for the
//! replacement is `the_35b_generates_identically_with_the_expert_pool_oversubscribed`.
//!
//! While oversubscribed the picks are read back mid-pass, so graphs are off
//! (D13) and migration is paused; both return in later steps.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
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



unsafe extern "C" {
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
    crate::platform::release_range(path, offset, len);
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

/// What a reserve holds beyond the model's permanent weights: activation mirrors,
/// recurrent state and the driver's working set. Used by `Cuda::reserve_for_weights`,
/// which raises the reserve to `dense weights + this` when that exceeds
/// [`DEFAULT_RESERVE`].
///
/// **Derived so the 35B is untouched.** `DEFAULT_RESERVE` was set on the 35B, whose
/// dense weights are 1.31 GiB (IQ4_XS) and 1.68 GiB (NVFP4) by
/// `Model::dense_weight_bytes`; 3 − 1.68 = 1.32 GiB, rounded down, so both files
/// still get exactly `DEFAULT_RESERVE`. The 125B's 4.44 GiB of dense weights take
/// its reserve to 5.69 GiB — the case `DEFAULT_RESERVE` never covered, since the
/// slab is sized inside block 0, before the later blocks upload theirs.
pub const NON_WEIGHT_RESERVE: usize = 5 << 28;

/// The automatic expert slab's ceiling, in bytes: 12 GiB (15-09-2026, the user's
/// call). An explicit `--expert-cache` is not capped by it. On this card the 35B's
/// automatic slab is ~11.7 GiB, under the cap; the 125B's is bounded by its
/// reserve well before it.
pub const DEFAULT_SLAB_CAP: usize = 12 << 30;

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
/// Overrun is not an error: the rest of the pool goes cold and is fetched from
/// the model file when a layer picks it. See the module doc.
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
    /// Both tiers filled before the pool did, so some experts are **cold**: not
    /// resident anywhere a kernel can read. Their table entries point at a
    /// zeroed sentinel, and [`ExpertCache::resolve`] fetches each from the model
    /// file into a VRAM slot when a layer picks it. SSD-TIER.md D12.
    ///
    /// Replaced `degraded`, which marked the eviction fallback that computed with
    /// the wrong experts. Being oversubscribed now costs speed, not correctness.
    pub oversubscribed: bool,
    /// Cold experts fetched from the model file when a layer picked them, and
    /// the bytes that took. Every fetch evicts one VRAM slot, so `evictions`
    /// counts the same events once the slab is full.
    pub fetched: u64,
    pub fetch_bytes: u64,
    /// Experts given a cold table entry when their tensor's table was built:
    /// counted in `distinct`, placed in neither tier. `distinct - host_slots -
    /// cold_at_load` were placed in VRAM.
    ///
    /// **Kept apart because `distinct` alone lied.** The 2 + 2 GiB CLI run
    /// printed "placed at load (30720 tensors)" with 7,060 placed.
    pub cold_at_load: u64,
    /// Host-to-device copies this cache issued, and their bytes: slab fills,
    /// pointer tables, table entries and residency flags, each counted where it
    /// is issued. [`ExpertCache::take_uploads`] hands the backend's crossing
    /// counters what they have not yet seen.
    ///
    /// **Counted per copy because the callers could not see them.** Callers
    /// added one crossing per call that filled anything, so a fetch — the bytes
    /// plus four eight-byte writes — read as a fifth of itself at best, and the
    /// 2 + 2 GiB run printed 6,110 up against ~125,000 copies.
    pub up_calls: u64,
    pub up_bytes: u64,
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
    /// Microseconds of the tier-3 fetch path, split where the time goes: reading
    /// cold experts from the file, uploading them into their slots, the small
    /// table-entry and flag writes around each (the victim's two, the fetched
    /// expert's two), and the picks readback that starts each resolve.
    /// `readbacks` counts those.
    ///
    /// **Measured before parallelizing the reads**, because the 125B's decode rate
    /// tracked serial disk throughput and nothing said how much of a 300 ms token
    /// the disk actually was.
    ///
    /// **The readback is two costs, and was one counter until 18-09.** Its
    /// `synchronize` waits out every kernel queued ahead of it, so that part is
    /// GPU time, not overhead; the downloads after it are `n_tok * n_used * 4`
    /// bytes of picks (40 at decode) plus the hint logits (~2 KB), which cannot
    /// account for the ~32 ms a 125B token spent here. `readback_wait_us` is the
    /// synchronize alone — the host waiting for the device — and
    /// `readback_copy_us` the downloads and the host-side top-k. **Only the
    /// second is overhead a faster path could remove**; the first shrinks only by
    /// giving the GPU less to do, or by not waiting for it.
    pub fetch_read_us: u64,
    pub fetch_upload_us: u64,
    pub fetch_writes_us: u64,
    pub readback_wait_us: u64,
    pub readback_copy_us: u64,
    pub readbacks: u64,
    /// Table-entry and flag writes queued during resolves, and the flushes that
    /// sent them: one copy and one `apply_patches` launch each.
    pub patches: u64,
    pub patch_flushes: u64,
    /// Lookahead prefetch (SSD-TIER.md D20): cold experts read ahead of the layer
    /// that picked them, those a resolve then used, and those dropped unused.
    pub prefetch_reads: u64,
    pub prefetch_used: u64,
    pub prefetch_wasted: u64,
    /// Whether the parallel reads run with `O_DIRECT`: asked for by default, and
    /// false if refused — `INFERRED_FETCH_DIRECT=0`, a staging buffer that is not
    /// page-aligned, or a file system that rejects the flag.
    pub fetch_direct: bool,
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
    /// GCLOCK's heat counter, 0..=[`HEAT_MAX`]. Dead until the host tier is
    /// exhausted. See [`clock_pick`] for the policy and why it is a count rather
    /// than the reference bit it was until 19-09.
    heat: Vec<u8>,
    /// [`heat_max`], read once at construction.
    ///
    /// **Not called per touch.** It is a `OnceLock`, so reading it is an atomic
    /// load, and the touch below it runs on every resident pick — roughly 19M
    /// times in the 35B's 19,706-token standard prefill (tokens x top-8 x 3
    /// tensors x 40 layers). That atomic cost the standard run **0.9% of
    /// prefill** against the reference bit it replaced, on a model where nothing
    /// is ever cold and the eviction policy itself never runs. SSD-TIER D24.
    heat_cap: u8,
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
    /// read back at report time. Sized once, to `counter_capacity`, because
    /// growing them would invalidate the bases already handed out.
    counts: Option<DeviceBuffer>,
    /// Experts the counter arrays hold: the model's whole pool when setup
    /// declared it, else [`DEFAULT_COUNTERS`].
    counter_capacity: usize,
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
    /// Reused across tensors so placement does not allocate and zero 142 MiB a
    /// hundred and twenty times. Held rather than local for that reason alone.
    stage: Vec<u8>,
    /// Experts not resident anywhere a kernel can read. Their table entries
    /// point at `sentinel`; [`ExpertCache::resolve`] fetches them from the model
    /// file when a layer picks them. SSD-TIER.md D12.
    cold: HashSet<usize>,
    /// Experts picked by the layer being resolved. None may be evicted to make
    /// room for another, so a layer's gate, up and down all stay put until the
    /// next routing decision clears it — the lease colibri's expert store names.
    lease: HashSet<usize>,
    /// Expert key -> (its tensor's key, its index in that tensor), so an evicted
    /// expert's *own* table entry can be found and repointed. The eviction this
    /// replaced never did that, and computed with the wrong experts.
    home: HashMap<usize, (usize, u32)>,
    /// One zeroed slot every cold table entry points at. A correct pass never
    /// reads it; a stray read gets zeros rather than another expert's weights.
    sentinel: Option<DeviceBuffer>,
    /// The model file, opened on the first fetch.
    file: Option<std::fs::File>,
    /// Threads that read a resolve's cold experts concurrently, started on the
    /// first fetch. See `fetch`.
    readers: Option<super::fetch::ReadPool>,
    /// Page-locked staging for those reads, [`FETCH_CHUNK`] experts at a time.
    pinned: Option<super::fetch::Pinned>,
    /// Recorded after a batch of queued uploads, so their staging can be reused
    /// once it has fired. Created on first use; `None` when uploads are
    /// blocking. See [`ExpertCache::drain_uploads`].
    upload_event: Option<ffi::CUevent>,
    /// Whether [`Self::upload_event`] has been recorded since the last drain.
    uploads_pending: bool,
    /// Lookahead reads in flight or done, and their staging. Declared after
    /// `readers` on purpose: fields drop in order, and the pool finishes every
    /// queued read before its threads exit, so the staging outlives them.
    prefetch: Prefetch,
    /// Whether [`ExpertCache::start_prefetch`] reads anything.
    /// `INFERRED_PREFETCH=0` turns it off; [`ExpertCache::set_prefetch`] too.
    prefetch_on: bool,
    /// The group opener that followed each opener last time: layer L's gate ->
    /// layer L+1's. Learned as `groups` is. See [`ExpertCache::group_for`].
    next_opener: HashMap<usize, usize>,
    /// Experts per tensor at the last resolve, for a prefetch's bounds.
    n_expert_seen: usize,
    /// Read threads for a fetch; 1 keeps the serial path. `INFERRED_FETCH_THREADS`.
    fetch_threads: usize,
    /// Whether the read pool asks for `O_DIRECT`; `INFERRED_FETCH_DIRECT=0` turns it
    /// off. What the pool actually got is `ExpertStats::fetch_direct`.
    fetch_direct: bool,
    /// Whether a layer's tensors are fetched as one batch (SSD-TIER.md D20):
    /// `INFERRED_FETCH_GROUP=0` goes back to one batch per tensor.
    group_fetch: bool,
    /// Tensors resolved under one routing decision, keyed by the first: a layer's
    /// gate with its up and down. Learned from the resolves themselves, since the
    /// pool is declared only as a count. See [`ExpertCache::group_for`].
    groups: HashMap<usize, Vec<usize>>,
    /// The routing decision being learned, and the tensor that opened it.
    learning: Option<(u64, usize)>,
    /// `INFERRED_EXPERT_LOG`: the routing decision whose tensors were already
    /// logged, and which, so a tensor fetched early is logged once.
    logged: (u64, Vec<usize>),
    /// While set, table-entry and residency-flag writes queue in `patches`
    /// instead of each being its own copy; [`ExpertCache::flush_patches`] sends
    /// them up together. Set only for the length of a resolve.
    deferring: Cell<bool>,
    /// Queued writes: `[device address, value, width in bytes]`.
    patches: RefCell<Vec<[u64; 3]>>,
    /// Where a flush puts the queued writes for `apply_patches` to read.
    patch_buf: Option<DeviceBuffer>,
    /// Host-to-device copies issued, and bytes: `Cell`s because table-entry and
    /// flag writes happen from `&self`. See [`ExpertStats::up_calls`].
    uploads: Cell<(u64, u64)>,
    /// How much of `uploads` [`ExpertCache::take_uploads`] has handed out.
    uploads_reported: Cell<(u64, u64)>,
    stats: ExpertStats,
}

/// Experts the global counter arrays hold when no pool size was declared.
///
/// **Setup declares the real one** (`Cuda::set_expert_pool`, from
/// `Model::expert_pool`): 30,720 on the 35B, 73,728 on the 125B. This fallback is
/// for callers that drive expert ops without an engine — the per-op tests.
///
/// **Overrunning it is not harmless**, whatever this comment used to say. A tensor
/// past the capacity has no counter slice, `moe_gather_ptrs` refuses to launch,
/// and the expert op returns before its matmul: the layer's routed output is
/// never computed. The 125B's first run hit it at the 129th tensor — layer 42's
/// `down` and all of layers 43–47 — and still produced llama.cpp's text.
/// The refusal stays, so an overrun is a reported error rather than a blind layer.
const DEFAULT_COUNTERS: usize = 65_536;

/// Threads reading cold experts from the model file. `INFERRED_FETCH_THREADS=1`
/// restores the serial reads timed at `7d7a3a2` (152.8 ms of a 315 ms 125B token).
const DEFAULT_FETCH_THREADS: usize = 8;

/// Experts read per batch, and so the page-locked staging a fetch holds: 32 x
/// 0.88 MiB on the 125B. A decode layer boundary fetches a handful; a prefill
/// resolve that needs more goes in batches.
const FETCH_CHUNK: usize = 32;

/// Lookahead reads at once: the most a layer's guesses start. The 125B's top-16
/// over three tensors is 48 candidates, of which the cold ones are ~a third.
const PREFETCH_SLOTS: usize = 32;

/// A lookahead read's state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Prefetched {
    InFlight,
    Ready,
    Failed,
}

/// Lookahead reads and the page-locked staging they land in (SSD-TIER.md D20).
///
/// **The invariant that makes the raw pointers sound**: a slot is in `free` only
/// when no read targets it. A read's slot leaves `free` when it is submitted and
/// returns only when the pool has reported it — through [`Prefetch::collect`] —
/// whether its expert was used, dropped or never looked at.
struct Prefetch {
    staging: Option<super::fetch::Pinned>,
    slot_len: usize,
    free: Vec<usize>,
    /// Expert key -> (slot, where the expert starts in it, state).
    entries: HashMap<usize, (usize, usize, Prefetched)>,
    /// Slots with a read in flight: true while their key is still wanted, false
    /// once dropped (the slot is freed when the read reports).
    busy: HashMap<usize, bool>,
    done: super::fetch::Done,
    reports: std::sync::mpsc::Receiver<(usize, std::io::Result<()>)>,
}

impl Prefetch {
    fn new() -> Self {
        let (done, reports) = std::sync::mpsc::channel();
        Self {
            staging: None,
            slot_len: 0,
            free: Vec::new(),
            entries: HashMap::new(),
            busy: HashMap::new(),
            done,
            reports,
        }
    }

    /// Staging for [`PREFETCH_SLOTS`] slots of `slot_len`, allocated once.
    fn ensure_staging(&mut self, slot_len: usize) -> Result<()> {
        if self.staging.is_some() {
            if slot_len != self.slot_len {
                return Err(Error::Cuda {
                    what: "expert prefetch",
                    detail: format!("staging slots are {} bytes, a read needs {slot_len}", self.slot_len),
                });
            }
            return Ok(());
        }
        self.staging = Some(super::fetch::Pinned::new(PREFETCH_SLOTS * slot_len)?);
        self.slot_len = slot_len;
        self.free = (0..PREFETCH_SLOTS).rev().collect();
        Ok(())
    }

    fn slot_ptr(&mut self, slot: usize) -> *mut u8 {
        match self.staging.as_mut() {
            // In bounds: `slot < PREFETCH_SLOTS`, and the staging holds that many.
            Some(p) => p.as_mut_slice()[slot * self.slot_len..].as_mut_ptr(),
            None => std::ptr::null_mut(),
        }
    }

    /// Take the pool's reports: all that have arrived, and, with `until`, block
    /// until that slot's read has reported.
    fn collect(&mut self, until: Option<usize>) {
        loop {
            let waiting = until.is_some_and(|s| self.busy.contains_key(&s));
            let report = if waiting {
                match self.reports.recv() {
                    Ok(r) => r,
                    Err(_) => return,
                }
            } else {
                match self.reports.try_recv() {
                    Ok(r) => r,
                    Err(_) => return,
                }
            };
            let (slot, result) = report;
            match self.busy.remove(&slot) {
                Some(true) => {
                    let state = if result.is_ok() { Prefetched::Ready } else { Prefetched::Failed };
                    if let Some(e) = self.entries.values_mut().find(|e| e.0 == slot) {
                        e.2 = state;
                    }
                }
                // Dropped while in flight: the slot is free only now.
                Some(false) => self.free.push(slot),
                None => {}
            }
        }
    }

    /// Forget a guess. Its slot is freed now if its read is done, or when the
    /// read reports.
    fn discard(&mut self, key: usize) {
        let Some((slot, _, state)) = self.entries.remove(&key) else { return };
        if state == Prefetched::InFlight {
            self.busy.insert(slot, false);
        } else {
            self.free.push(slot);
        }
    }
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
            heat: vec![0; slots],
            heat_cap: heat_max(),
            hand: 0,
            blocks: Vec::new(),
            host_budget,
            tables: HashMap::new(),
            bases: HashMap::new(),
            counts: None,
            counter_capacity: DEFAULT_COUNTERS,
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
            stage: Vec::new(),
            cold: HashSet::new(),
            lease: HashSet::new(),
            home: HashMap::new(),
            sentinel: None,
            file: None,
            readers: None,
            pinned: None,
            upload_event: None,
            uploads_pending: false,
            prefetch: Prefetch::new(),
            prefetch_on: std::env::var("INFERRED_PREFETCH").map_or(true, |v| v != "0"),
            next_opener: HashMap::new(),
            n_expert_seen: 0,
            fetch_threads: std::env::var("INFERRED_FETCH_THREADS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(DEFAULT_FETCH_THREADS)
                .max(1),
            fetch_direct: std::env::var("INFERRED_FETCH_DIRECT").map_or(true, |v| v != "0"),
            group_fetch: std::env::var("INFERRED_FETCH_GROUP").map_or(true, |v| v != "0"),
            groups: HashMap::new(),
            learning: None,
            logged: (u64::MAX, Vec::new()),
            deferring: Cell::new(false),
            patches: RefCell::new(Vec::new()),
            patch_buf: None,
            uploads: Cell::new((0, 0)),
            uploads_reported: Cell::new((0, 0)),
            stats: ExpertStats {
                slots: slots as u64,
                slot_bytes: stride as u64,
                ..Default::default()
            },
        })
    }

    pub fn stats(&self) -> ExpertStats {
        let mut s = self.stats;
        s.migrated = self.migrated;
        (s.up_calls, s.up_bytes) = self.uploads.get();
        s
    }

    /// Host-to-device copies issued since the last call, and their bytes, for
    /// the backend's crossing counters.
    pub fn take_uploads(&self) -> (u64, u64) {
        let (calls, bytes) = self.uploads.get();
        let (seen_calls, seen_bytes) = self.uploads_reported.replace((calls, bytes));
        (calls - seen_calls, bytes - seen_bytes)
    }

    /// Copy `data` into `buf` at `offset_bytes`, counted. Every host-to-device
    /// copy this cache issues goes through here.
    fn upload<T: Copy>(&self, buf: &DeviceBuffer, offset_bytes: usize, data: &[T]) -> Result<()> {
        buf.write_at(offset_bytes, data)?;
        let (calls, bytes) = self.uploads.get();
        self.uploads.set((calls + 1, bytes + std::mem::size_of_val(data) as u64));
        Ok(())
    }

    /// [`Self::upload`], queued instead of waited on.
    ///
    /// **`data` must be page-locked and must not be rewritten until
    /// [`Self::drain_uploads`] has run.** Only the two fetch paths call this,
    /// and both hold their staging across the call.
    fn upload_async<T: Copy>(&self, buf: &DeviceBuffer, offset_bytes: usize, data: &[T]) -> Result<()> {
        buf.write_at_async(offset_bytes, data)?;
        let (calls, bytes) = self.uploads.get();
        self.uploads.set((calls + 1, bytes + std::mem::size_of_val(data) as u64));
        Ok(())
    }

    /// Record that queued uploads are outstanding, so a later
    /// [`Self::drain_uploads`] knows what to wait for.
    fn note_uploads_issued(&mut self) -> Result<()> {
        if !async_upload_on() {
            return Ok(());
        }
        if self.upload_event.is_none() {
            let mut e: ffi::CUevent = std::ptr::null_mut();
            // SAFETY: out-parameter; the event is destroyed in `Drop`.
            unsafe { super::check(ffi::cuEventCreate(&mut e, 0), "cuEventCreate")? };
            self.upload_event = Some(e);
        }
        if let Some(e) = self.upload_event {
            // SAFETY: `e` was created above and the null stream is always valid.
            unsafe { super::check(ffi::cuEventRecord(e, std::ptr::null_mut()), "cuEventRecord")? };
            self.uploads_pending = true;
        }
        Ok(())
    }

    /// Wait for queued uploads before their staging is reused.
    ///
    /// **An event, not `cuCtxSynchronize`.** The drain at `start_prefetch` runs
    /// with the layer's kernels already queued; a context sync would block on
    /// those too, which is precisely the wait this change exists to remove. The
    /// event was recorded when the GPU held nothing but these copies, so waiting
    /// on it waits for them alone.
    ///
    /// Ordering against later kernels needs no drain at all: the copies and the
    /// kernels share the null stream, which runs in order.
    fn drain_uploads(&mut self) -> Result<()> {
        if !self.uploads_pending {
            return Ok(());
        }
        if let Some(e) = self.upload_event {
            // SAFETY: `e` was created and recorded by `note_uploads_issued`.
            unsafe { super::check(ffi::cuEventSynchronize(e), "cuEventSynchronize")? };
        }
        self.uploads_pending = false;
        Ok(())
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
    /// in the host tier and read across PCIe by the kernel itself. Only once
    /// *both* tiers are full does a first touch fetch into a VRAM slot, evicting
    /// an unleased expert — the one case in which something is filled on the
    /// critical path, and the reason it cannot be graphed.
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
                Some(s) => self.heat[s as usize] = self.heat[s as usize].saturating_add(1).min(self.heat_cap),
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
        if self.cold.contains(&key) {
            // Both tiers were full, so `place` left it cold. This is a read, so
            // it must come back holding the bytes: fetch it into a VRAM slot now,
            // from the bytes the caller already holds. SSD-TIER.md D12.
            return self.make_resident(key, src, false);
        }
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
        if self.cold.contains(&key) {
            return self.sentinel_ptr();
        }
        self.stats.distinct += 1;

        // Tier 1: a free VRAM slot, while the slab is still filling.
        if self.next_slot < self.owner.len() {
            let slot = self.next_slot as u32;
            self.next_slot += 1;
            let t = std::time::Instant::now();
            self.upload(&self.slab, slot as usize * self.stride, src)?;
            self.stats.place_h2d_us += t.elapsed().as_micros() as u64;
            self.stats.filled_bytes += src.len() as u64;
            self.owner[slot as usize] = Some(key);
            self.heat[slot as usize] = HEAT_NEW;
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

        // Both tiers full: the expert is **cold**. It gets no slot; its table
        // entry points at the zeroed sentinel, and `resolve` fetches it from the
        // model file when a layer picks it (SSD-TIER.md D12).
        //
        // This replaced an eviction here that wrote the new expert into a VRAM
        // slot without repointing the evicted expert's own table entry, so every
        // oversubscribed run computed with the wrong experts.
        self.stats.oversubscribed = true;
        self.stats.cold_at_load += 1;
        self.cold.insert(key);
        self.sentinel_ptr()
    }

    /// Queue table-entry and residency-flag writes from here until
    /// [`ExpertCache::flush_patches`], instead of issuing a copy for each.
    pub fn defer_writes(&mut self) {
        self.deferring.set(true);
    }

    /// Stop queueing, and send the queued writes to the device in one copy.
    /// Returns where they went and how many there are, for `apply_patches`, or
    /// `None` when nothing was queued.
    ///
    /// **Last write per address wins, as the serial writes did.** A resolve can
    /// write one entry twice: gate's fetches can evict an `up` expert that `up`'s
    /// resolve, later in the same call, fetches straight back. The kernel applies
    /// patches concurrently, so duplicates are dropped here, keeping the last.
    pub fn flush_patches(&mut self) -> Result<Option<(ffi::CUdeviceptr, usize)>> {
        self.deferring.set(false);
        let queued = std::mem::take(&mut *self.patches.borrow_mut());
        if queued.is_empty() {
            return Ok(None);
        }
        let t = std::time::Instant::now();
        let mut last: HashMap<u64, usize> = HashMap::with_capacity(queued.len());
        for (i, p) in queued.iter().enumerate() {
            last.insert(p[0], i);
        }
        let flat: Vec<u64> = queued
            .iter()
            .enumerate()
            .filter(|&(i, p)| last.get(&p[0]) == Some(&i))
            .flat_map(|(_, p)| p.iter().copied())
            .collect();
        let n = flat.len() / 3;
        if self.patch_buf.as_ref().is_none_or(|b| b.len_bytes() < flat.len() * 8) {
            self.patch_buf = Some(DeviceBuffer::new((flat.len() * 8).next_power_of_two().max(4096))?);
        }
        let ptr = match self.patch_buf.as_ref() {
            Some(b) => {
                self.upload(b, 0, &flat)?;
                b.ptr
            }
            None => {
                return Err(Error::Cuda { what: "expert cache", detail: "the patch buffer vanished".to_string() });
            }
        };
        self.stats.patches += queued.len() as u64;
        self.stats.patch_flushes += 1;
        self.stats.fetch_writes_us += t.elapsed().as_micros() as u64;
        Ok(Some((ptr, n)))
    }

    /// Count one picks readback: `wait` is its synchronize (GPU time), `copy` the
    /// downloads and the host-side top-k that follow it.
    pub fn note_readback(&mut self, wait: u64, copy: u64) {
        self.stats.readback_wait_us += wait;
        self.stats.readback_copy_us += copy;
        self.stats.readbacks += 1;
    }

    /// Whether any expert is cold. See [`ExpertStats::oversubscribed`].
    pub fn oversubscribed(&self) -> bool {
        self.stats.oversubscribed
    }

    /// End the current lease: a new routing decision means the layer that held it
    /// is done, so its experts may be evicted again.
    pub fn clear_lease(&mut self) {
        self.lease.clear();
    }

    /// Make every expert `ids` picks from one tensor resident, and lease them.
    ///
    /// **Tier 3's MVP** (SSD-TIER.md D12, D13). Called between a tensor's table
    /// lookup and the gather that reads it, while oversubscribed. Every pick is
    /// leased *before* any is fetched, so resolving one cannot evict another, and
    /// the lease persists across this layer's gate, up and down until the next
    /// routing decision clears it.
    ///
    /// A cold pick is fetched into a VRAM slot taken from an expert the layer did
    /// not pick. The order is what makes it safe: the victim's entry is pointed at
    /// the sentinel *before* its slot is overwritten, and the fetched expert's
    /// entry is pointed at the slot only *after* its bytes are in it. The caller
    /// must have synchronized, so no kernel is still reading the victim's slot.
    ///
    /// `tkey` is the tensor's key, its address in the mapping, and `data` the
    /// tensor's bytes there, read only when no model file is known. Returns the
    /// bytes fetched.
    /// For each pick in `ids`, 1 if that expert of tensor `tkey` is cold now —
    /// what [`ExpertCache::resolve`] would fetch — else 0. For
    /// `INFERRED_EXPERT_LOG`; changes nothing.
    pub fn cold_flags(&self, tkey: usize, n_expert: usize, ids: &[i32]) -> Vec<u8> {
        ids.iter()
            .map(|&e| match usize::try_from(e) {
                Ok(e) if e < n_expert => u8::from(self.cold.contains(&(tkey + e * self.stride))),
                _ => 0,
            })
            .collect()
    }

    pub fn resolve(&mut self, tkey: usize, data: &[u8], n_expert: usize, ids: &[i32]) -> Result<u64> {
        let stride = self.stride;
        let mut picks: Vec<usize> =
            ids.iter().filter_map(|&e| usize::try_from(e).ok()).filter(|&e| e < n_expert).collect();
        picks.sort_unstable();
        picks.dedup();

        for &e in &picks {
            self.lease.insert(tkey + e * stride);
        }

        let mut fetched = 0u64;
        if self.parallel_fetch() {
            return self.resolve_many(&[tkey], n_expert, ids);
        }
        for &e in &picks {
            let key = tkey + e * stride;
            if !self.cold.contains(&key) {
                if let Some(slot) = self.map.get(&key).and_then(|x| x.slot) {
                    self.heat[slot as usize] = self.heat[slot as usize].saturating_add(1).min(self.heat_cap);
                }
                continue;
            }
            let mut stage = std::mem::take(&mut self.stage);
            if stage.len() < stride {
                stage.resize(stride, 0);
            }
            let t = std::time::Instant::now();
            let read = self.read_expert(&mut stage[..stride], data, key, e);
            self.stats.fetch_read_us += t.elapsed().as_micros() as u64;
            let placed = match read {
                Ok(()) => self.make_resident(key, &stage[..stride], false),
                Err(err) => Err(err),
            };
            self.stage = stage;
            placed?;
            fetched += stride as u64;
        }
        Ok(fetched)
    }

    /// Whether cold experts are read by the parallel pool — the path
    /// [`ExpertCache::resolve_many`] batches.
    pub fn parallel_fetch(&self) -> bool {
        self.fetch_threads > 1 && self.source.is_some()
    }

    /// Whether a layer's tensors are fetched as one batch; `INFERRED_FETCH_GROUP=0`
    /// turns it off.
    pub fn group_fetch(&self) -> bool {
        self.group_fetch
    }

    /// [`ExpertCache::resolve`] for several tensors picked by the same routing
    /// decision, **their cold experts read as one batch** (SSD-TIER.md D20). Each
    /// tensor used to be its own batch — gate, then up, then down at its own
    /// matmul — so a 125B layer issued three sequential batches of ~2 reads while
    /// most of the eight threads idled: 75 ms of a 181 ms token. Every pick of
    /// every tensor is leased first, so fetching down's experts with gate's
    /// cannot evict anything this layer uses. Parallel path only.
    pub fn resolve_many(&mut self, tkeys: &[usize], n_expert: usize, ids: &[i32]) -> Result<u64> {
        let stride = self.stride;
        let mut picks: Vec<usize> =
            ids.iter().filter_map(|&e| usize::try_from(e).ok()).filter(|&e| e < n_expert).collect();
        picks.sort_unstable();
        picks.dedup();
        let mut cold = Vec::new();
        for &tkey in tkeys {
            for &e in &picks {
                let key = tkey + e * stride;
                self.lease.insert(key);
                if self.cold.contains(&key) {
                    cold.push(key);
                } else if let Some(slot) = self.map.get(&key).and_then(|x| x.slot) {
                    self.heat[slot as usize] = self.heat[slot as usize].saturating_add(1).min(self.heat_cap);
                }
            }
        }
        self.n_expert_seen = n_expert;
        let mut fetched = 0u64;
        // Read ahead already: wait for any still in flight, then publish from the
        // prefetch staging. The wait is a read the layer blocked on, so it is
        // timed as one.
        let mut rest = Vec::with_capacity(cold.len());
        for key in cold {
            match self.prefetch.entries.get(&key).map(|e| e.0) {
                Some(slot) => {
                    let t = std::time::Instant::now();
                    self.prefetch.collect(Some(slot));
                    self.stats.fetch_read_us += t.elapsed().as_micros() as u64;
                    if self.publish_prefetched(key)? {
                        fetched += stride as u64;
                    } else {
                        rest.push(key);
                    }
                }
                None => rest.push(key),
            }
        }
        for batch in rest.chunks(FETCH_CHUNK) {
            self.fetch_batch(batch)?;
            fetched += (batch.len() * stride) as u64;
        }
        Ok(fetched)
    }

    /// Turn lookahead prefetch on or off; on by default, `INFERRED_PREFETCH=0`
    /// off. For tests that compare both in one process.
    pub fn set_prefetch(&mut self, on: bool) {
        self.prefetch_on = on;
    }

    /// Whether lookahead prefetch is on.
    pub fn prefetch_on(&self) -> bool {
        self.prefetch_on
    }

    /// **Lookahead prefetch** (SSD-TIER.md D20): start reading the cold experts
    /// that the *next* routing decision is likely to pick — `ids`, the next
    /// layer's router's top-k on this layer's input — for every tensor of the
    /// group that follows the one just resolved. The reads run on the pool while
    /// the GPU computes; [`ExpertCache::resolve_many`] takes what finished.
    ///
    /// Guesses left from the previous decision are dropped first. At most
    /// [`PREFETCH_SLOTS`] reads are started; the rest wait for the real resolve.
    /// Changes nothing a kernel reads: bytes stay in host staging until a resolve
    /// publishes them.
    pub fn start_prefetch(&mut self, ids: &[i32]) -> Result<()> {
        if !self.prefetch_on || !self.parallel_fetch() {
            return Ok(());
        }
        // **Guard 2 of 3.** Slots freed by this layer's publishes go back on the
        // free list, and the reads started below would overwrite staging whose
        // queued upload has not landed. Waits on an event, not the context, so
        // the layer's kernels — already queued by now — are not waited on.
        self.drain_uploads()?;
        self.prefetch.collect(None);
        let stale: Vec<usize> = self.prefetch.entries.keys().copied().collect();
        for key in stale {
            self.prefetch.discard(key);
            self.stats.prefetch_wasted += 1;
        }
        let Some((_, opener)) = self.learning else { return Ok(()) };
        let Some(&next) = self.next_opener.get(&opener) else { return Ok(()) };
        let mut tensors = vec![next];
        tensors.extend(self.groups.get(&next).into_iter().flatten().copied());

        let n_expert = self.n_expert_seen;
        let stride = self.stride;
        let mut picks: Vec<usize> =
            ids.iter().filter_map(|&e| usize::try_from(e).ok()).filter(|&e| e < n_expert).collect();
        picks.sort_unstable();
        picks.dedup();
        let mut want = Vec::new();
        for &tkey in &tensors {
            for &e in &picks {
                let key = tkey + e * stride;
                if self.cold.contains(&key) && !self.prefetch.entries.contains_key(&key) {
                    want.push(key);
                }
            }
        }
        if want.is_empty() {
            return Ok(());
        }
        self.ensure_readers()?;
        let (align, slot_len) = self.staging_geometry();
        self.prefetch.ensure_staging(slot_len)?;
        let Some((_, base)) = self.source.clone() else { return Ok(()) };
        for key in want {
            let Some(slot) = self.prefetch.free.pop() else { break };
            let pre = key.saturating_sub(base) % align;
            let start = (key.saturating_sub(base) - pre) as u64;
            let len = (pre + stride).div_ceil(align) * align;
            let dst = self.prefetch.slot_ptr(slot);
            let submitted = match self.readers.as_ref() {
                // SAFETY: `slot` was free, so nothing else reads, writes or
                // publishes its memory until the pool reports on `done` for it:
                // `Prefetch` returns a slot to `free` only on that report, and the
                // staging is freed only after the pool, which finishes every
                // queued read first (field order). `len <= slot_len`.
                Some(pool) => unsafe { pool.submit(start, dst, len, pre + stride, slot, &self.prefetch.done) },
                None => Err(Error::Cuda { what: "expert prefetch", detail: "no read pool".to_string() }),
            };
            match submitted {
                Ok(()) => {
                    self.prefetch.entries.insert(key, (slot, pre, Prefetched::InFlight));
                    self.prefetch.busy.insert(slot, true);
                    self.stats.prefetch_reads += 1;
                }
                Err(e) => {
                    self.prefetch.free.push(slot);
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// Publish a finished prefetch of `key` through `make_resident`, freeing its
    /// slot. False when there was none to publish (it failed), so the caller
    /// reads it the ordinary way.
    fn publish_prefetched(&mut self, key: usize) -> Result<bool> {
        let Some((slot, pre, state)) = self.prefetch.entries.remove(&key) else { return Ok(false) };
        if state != Prefetched::Ready {
            self.prefetch.free.push(slot);
            return Ok(false);
        }
        let Some(staging) = self.prefetch.staging.take() else {
            return Err(Error::Cuda { what: "expert prefetch", detail: "the staging buffer vanished".to_string() });
        };
        let at = slot * self.prefetch.slot_len + pre;
        let published = self.make_resident(key, &staging.as_slice()[at..at + self.stride], true).map(|_| ());
        self.prefetch.staging = Some(staging);
        self.prefetch.free.push(slot);
        published?;
        // The slot just went back on the free list, so the next `start_prefetch`
        // may read into it. Guard 2 of 3 is the drain there; this only records
        // what that drain waits for.
        self.note_uploads_issued()?;
        self.stats.prefetch_used += 1;
        Ok(true)
    }

    /// The read pool and its staging, opened on first use.
    fn ensure_readers(&mut self) -> Result<()> {
        use super::fetch::{DIRECT_ALIGN, Pinned, ReadPool};
        if self.readers.is_some() {
            return Ok(());
        }
        let Some((path, _)) = self.source.clone() else {
            return Err(Error::Cuda {
                what: "expert fetch",
                detail: "a parallel fetch needs the model file".to_string(),
            });
        };
        let mut direct = self.fetch_direct;
        let pinned = Pinned::new(FETCH_CHUNK * self.slot_len_for(DIRECT_ALIGN))?;
        // Stage 0 found `cuMemHostAlloc` page-aligned; checked, not assumed.
        if pinned.as_slice().as_ptr() as usize % DIRECT_ALIGN != 0 {
            direct = false;
        }
        let file = match super::fetch::open(&path, direct) {
            Ok(f) => f,
            Err(_) if direct => {
                direct = false;
                super::fetch::open(&path, false).map_err(|err| Error::Cuda {
                    what: "expert fetch",
                    detail: format!("opening {}: {err}", path.display()),
                })?
            }
            Err(err) => {
                return Err(Error::Cuda { what: "expert fetch", detail: format!("opening {}: {err}", path.display()) });
            }
        };
        self.fetch_direct = direct;
        self.stats.fetch_direct = direct;
        self.pinned = Some(pinned);
        self.readers = Some(ReadPool::new(file, self.fetch_threads)?);
        Ok(())
    }

    /// One staging slot per expert. With `O_DIRECT` a read covers the expert's
    /// enclosing page-aligned range — at most `align - 1` bytes before it and the
    /// rest of its last page after — so a slot is that range's largest size, and
    /// the expert sits `pre` bytes into it (SSD-TIER.md D4).
    fn slot_len_for(&self, align: usize) -> usize {
        (self.stride + align - 1).div_ceil(align) * align
    }

    /// The alignment reads use now, and the staging slot that fits one.
    fn staging_geometry(&self) -> (usize, usize) {
        let align = if self.fetch_direct { super::fetch::DIRECT_ALIGN } else { 1 };
        (align, self.slot_len_for(align))
    }

    /// The tensors to resolve now for a call on `tkeys` under routing decision
    /// `route`: `tkeys` themselves, then — when grouping is on and this call opens
    /// the decision — the tensors learned to follow it (a layer's down, after its
    /// gate and up). Learning: every tensor resolved under the decision a call
    /// opened joins that call's group, so the first pass teaches the second.
    pub fn group_for(&mut self, route: u64, tkeys: &[usize]) -> Vec<usize> {
        let mut all = tkeys.to_vec();
        let Some(&first) = tkeys.first() else { return all };
        match self.learning {
            Some((g, opener)) if g == route => {
                let group = self.groups.entry(opener).or_default();
                for &k in tkeys {
                    if k != opener && !group.contains(&k) {
                        group.push(k);
                    }
                }
            }
            _ => {
                if let Some((_, prev)) = self.learning
                    && prev != first
                {
                    self.next_opener.insert(prev, first);
                }
                self.learning = Some((route, first));
                let group = self.groups.entry(first).or_default();
                for &k in &tkeys[1..] {
                    if !group.contains(&k) {
                        group.push(k);
                    }
                }
                if self.group_fetch {
                    all.extend(group.iter().copied().filter(|k| !tkeys.contains(k)));
                }
            }
        }
        all
    }

    /// `INFERRED_EXPERT_LOG`: whether `tkey` is still to be logged under routing
    /// decision `route`, marking it logged. A tensor fetched early with its group is
    /// logged there, with its cold flags as they were, and not again.
    pub fn log_once(&mut self, route: u64, tkey: usize) -> bool {
        if self.logged.0 != route {
            self.logged = (route, Vec::new());
        }
        if self.logged.1.contains(&tkey) {
            return false;
        }
        self.logged.1.push(tkey);
        true
    }

    /// Fetch up to [`FETCH_CHUNK`] cold experts: read them all from the model file
    /// at once, then publish each through [`ExpertCache::make_resident`] in order.
    ///
    /// Reading first changes no device state, so the ordering `resolve` relies on
    /// is untouched: every pick is leased already, and each expert still evicts,
    /// repoints its victim, uploads and repoints itself one at a time, in the same
    /// sequence as the serial path — which is why the two give the same evictions.
    fn fetch_batch(&mut self, keys: &[usize]) -> Result<()> {
        let stride = self.stride;
        self.ensure_readers()?;
        let Some((_, base)) = self.source.clone() else {
            return Err(Error::Cuda {
                what: "expert fetch",
                detail: "a parallel fetch needs the model file".to_string(),
            });
        };
        let (align, slot) = self.staging_geometry();
        // **Guard 1 of 3.** The reads below overwrite the staging a previous
        // batch's queued uploads may still be copying out of. Only bites when a
        // resolve needs more than `FETCH_CHUNK` experts, which decode does not
        // and prefill does.
        self.drain_uploads()?;
        let mut pinned = self.pinned.take().ok_or_else(|| Error::Cuda {
            what: "expert fetch",
            detail: "the staging buffer vanished".to_string(),
        })?;

        // Where each expert starts in its slot.
        let pres: Vec<usize> = keys.iter().map(|&key| key.saturating_sub(base) % align).collect();
        let t = std::time::Instant::now();
        let read = {
            let mut reads: Vec<(u64, &mut [u8], usize)> = pinned
                .as_mut_slice()
                .chunks_exact_mut(slot)
                .zip(keys.iter().zip(&pres))
                .map(|(dst, (&key, &pre))| {
                    let start = (key.saturating_sub(base) - pre) as u64;
                    let len = (pre + stride).div_ceil(align) * align;
                    (start, &mut dst[..len], pre + stride)
                })
                .collect();
            match self.readers.as_ref() {
                Some(pool) => pool.read_all(&mut reads),
                None => Err(Error::Cuda { what: "expert fetch", detail: "no read pool".to_string() }),
            }
        };
        self.stats.fetch_read_us += t.elapsed().as_micros() as u64;

        let published = read.and_then(|()| {
            keys.iter().zip(&pres).enumerate().try_for_each(|(i, (&key, &pre))| {
                let at = i * slot + pre;
                self.make_resident(key, &pinned.as_slice()[at..at + stride], true).map(|_| ())
            })
        });
        self.pinned = Some(pinned);
        self.note_uploads_issued()?;
        published
    }

    /// Put `bytes` — expert `key`'s — into a VRAM slot, evicting an unleased
    /// expert to make room, and point `key`'s table entry at the slot.
    /// `src_pinned` says the caller's `bytes` are page-locked and will outlive a
    /// queued copy — only then may the upload be asynchronous. The serial fetch
    /// stages through an ordinary `Vec` and the first-sight path uses the
    /// caller's own slice, so both pass `false`.
    fn make_resident(&mut self, key: usize, bytes: &[u8], src_pinned: bool) -> Result<ffi::CUdeviceptr> {
        let slot = self.evict_unleased()?;
        if let Some(victim) = self.owner[slot as usize].take() {
            let t = std::time::Instant::now();
            self.make_cold(victim)?;
            self.stats.fetch_writes_us += t.elapsed().as_micros() as u64;
        }
        let t = std::time::Instant::now();
        if src_pinned && async_upload_on() {
            self.upload_async(&self.slab, slot as usize * self.stride, bytes)?;
        } else {
            self.upload(&self.slab, slot as usize * self.stride, bytes)?;
        }
        self.stats.fetch_upload_us += t.elapsed().as_micros() as u64;
        self.owner[slot as usize] = Some(key);
        // **A fetched expert enters cold, not hot** (19-09). 27.3% of a 125B
        // token's expert accesses are cold and 40% of the hot set turns over
        // inside one run, so most arrivals here are one-offs. Entering at
        // `HEAT_MAX` — which the reference bit effectively did — let every
        // transient flush a stable resident. `HEAT_NEW` makes an arrival
        // survive one hand pass and no more unless it is picked again.
        self.heat[slot as usize] = HEAT_NEW;
        let addr = self.slot_ptr(slot);
        self.map.insert(key, Entry { addr, slot: Some(slot), uses: 0 });
        self.cold.remove(&key);
        if let Some(&(tkey, e)) = self.home.get(&key) {
            let t = std::time::Instant::now();
            self.write_table_entry(tkey, e, addr)?;
            if let Some(&base) = self.bases.get(&tkey) {
                self.set_vram_flag(base + e as usize, 1)?;
            }
            self.stats.fetch_writes_us += t.elapsed().as_micros() as u64;
        }
        self.stats.fetched += 1;
        self.stats.fetch_bytes += bytes.len() as u64;
        Ok(addr)
    }

    /// Give up `victim`'s VRAM residency: point its table entry at the sentinel,
    /// so nothing can read the slot it is about to lose.
    ///
    /// A victim with no table was placed through [`ExpertCache::address_of`] and
    /// has no entry to repoint; its address was handed to a caller that used it
    /// at once, with graphs off.
    fn make_cold(&mut self, victim: usize) -> Result<()> {
        self.map.remove(&victim);
        self.cold.insert(victim);
        self.stats.evictions += 1;
        if let Some(&(tkey, e)) = self.home.get(&victim) {
            let sentinel = self.sentinel_ptr()?;
            self.write_table_entry(tkey, e, sentinel)?;
            if let Some(&base) = self.bases.get(&tkey) {
                self.set_vram_flag(base + e as usize, 0)?;
            }
        }
        Ok(())
    }

    /// Read expert `e`'s bytes — key `key` — into `dst`: from the model file with
    /// `pread` when one is known (SSD-TIER.md D10), else from the mapping.
    fn read_expert(&mut self, dst: &mut [u8], data: &[u8], key: usize, e: usize) -> Result<()> {
        let stride = self.stride;
        match self.source.clone() {
            Some((path, base)) => {
                if self.file.is_none() {
                    let f = std::fs::File::open(&path).map_err(|err| Error::Cuda {
                        what: "expert fetch",
                        detail: format!("opening {}: {err}", path.display()),
                    })?;
                    self.file = Some(f);
                }
                let offset = key.saturating_sub(base) as u64;
                match self.file.as_ref() {
                    Some(f) => crate::platform::read_exact_at(f, dst, offset).map_err(|err| Error::Cuda {
                        what: "expert fetch",
                        detail: format!("{} bytes at offset {offset} of {}: {err}", dst.len(), path.display()),
                    }),
                    None => Err(Error::Cuda {
                        what: "expert fetch",
                        detail: "the model file did not stay open".to_string(),
                    }),
                }
            }
            None => match data.get(e * stride..(e + 1) * stride) {
                Some(s) => {
                    dst.copy_from_slice(s);
                    Ok(())
                }
                None => Err(Error::Cuda {
                    what: "expert fetch",
                    detail: format!("expert {e} is past the tensor's {} bytes", data.len()),
                }),
            },
        }
    }

    /// The zeroed slot cold entries point at, allocated on first need.
    fn sentinel_ptr(&mut self) -> Result<ffi::CUdeviceptr> {
        if self.sentinel.is_none() {
            self.sentinel = Some(DeviceBuffer::zeroed(self.stride)?);
        }
        match self.sentinel.as_ref() {
            Some(b) => Ok(b.ptr),
            None => Err(Error::Cuda {
                what: "expert cache",
                detail: "the sentinel slot was not allocated".to_string(),
            }),
        }
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
        // **Read the tensor with `pread` rather than faulting it out of the
        // mmap.** Measured on this machine, same file, both cold: demand
        // faulting the mapping gives **0.83 GB/s** and a sequential `pread`
        // gives **2.00 GB/s** — 2.4x, and the whole of what open item 0b has
        // left to give.
        //
        // `9a7cbcb` established that placement is bound by getting the bytes
        // out of the file, not by the bus or the copy granularity: the device
        // copy ran at 0.80 GB/s and a plain CPU memcpy at 0.85, and those have
        // no reason to agree unless the shared source binds them.
        //
        // `MADV_WILLNEED` was tried first and does nothing — 0.82 against 0.86
        // GB/s on cold disjoint ranges. It appeared to give 14.7x only because
        // the arms shared pages, which `posix_fadvise` cannot evict while a
        // live mapping holds them; that is the same distinction `drop_file_cache`
        // documents, walked into from the other side.
        //
        // The keys stay the **mmap** addresses. `map`, `owner` and every table
        // are keyed on where the tensor lives in the mapping, so only the bytes
        // come from the staging buffer.
        let mut stage = std::mem::take(&mut self.stage);
        // **Off by default, because it trades 6% of every prefill for 20 s of
        // start-up.** Measured, interleaved, one sitting, both pairs agreeing:
        //
        //   pread   setup 5.0 / 4.8 s    prefill 311.8 / 311.2
        //   mmap    setup 24.5 / 24.7 s  prefill 331.2 / 331.6
        //
        // A start-up path has no business moving a steady-state number and the
        // cause is not yet known — the staging buffer is 142 MiB of anonymous
        // memory held for the life of the process, beside a multi-GiB pinned
        // host tier, and physical fragmentation of those pinned blocks is the
        // leading suspect rather than a demonstrated cause.
        //
        // A server starts once and runs for hours, so the default keeps the
        // prefill. `INFERRED_PREAD=1` takes the fast start-up instead, which is
        // the better trade for one-shot `generate` runs and for iterating on
        // anything that is not prefill throughput.
        let use_pread = std::env::var("INFERRED_PREAD").is_ok();
        let staged = match self.source.as_ref().filter(|_| use_pread) {
            Some((path, base)) => {
                let offset = (data.as_ptr() as usize).saturating_sub(*base) as u64;
                if stage.len() < data.len() {
                    stage.resize(data.len(), 0);
                }
                std::fs::File::open(path)
                    .and_then(|f| crate::platform::read_exact_at(&f, &mut stage[..data.len()], offset))
                    .is_ok()
            }
            None => false,
        };
        // Falls back to the mapping if anything about the read did not hold,
        // which keeps this an optimisation rather than a new failure mode.
        let src_all: &[u8] = if staged { &stage[..data.len()] } else { data };

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
                let src = &src_all[e * stride..(e + 1) * stride];
                let k = data[e * stride..].as_ptr() as usize;
                self.home.insert(k, (key, e as u32));
                addrs.push(self.place(k, src)?);
                vram.push(i32::from(self.map.get(&k).and_then(|x| x.slot).is_some()));
                e += 1;
                continue;
            }

            let first = self.next_slot;
            let src = &src_all[e * stride..(e + run) * stride];
            let t = std::time::Instant::now();
            self.upload(&self.slab, first * stride, src)?;
            self.stats.place_h2d_us += t.elapsed().as_micros() as u64;
            self.stats.filled_bytes += src.len() as u64;
            for i in 0..run {
                let slot = (first + i) as u32;
                let k = data[(e + i) * stride..].as_ptr() as usize;
                self.home.insert(k, (key, (e + i) as u32));
                self.stats.distinct += 1;
                self.owner[slot as usize] = Some(k);
                self.heat[slot as usize] = HEAT_NEW;
                let addr = self.slot_ptr(slot);
                self.map.insert(k, Entry { addr, slot: Some(slot), uses: 0 });
                addrs.push(addr);
                vram.push(1);
            }
            self.next_slot += run;
            e += run;
        }
        self.stage = stage;

        let buf = DeviceBuffer::new(std::mem::size_of_val(addrs.as_slice()))?;
        self.upload(&buf, 0, &addrs)?;
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
            self.counts = Some(DeviceBuffer::zeroed(self.counter_capacity * 4)?);
            self.vram_flags = Some(DeviceBuffer::zeroed(self.counter_capacity * 4)?);
            self.tally = Some(DeviceBuffer::zeroed(2 * 8)?);
        }
        let base = self.next_base;
        if base + n_expert <= self.counter_capacity {
            self.counter_owner.resize(base + n_expert, (0, 0));
            for e in 0..n_expert {
                self.counter_owner[base + e] = (key, e as u32);
            }
            if let Some(f) = self.vram_flags.as_ref() {
                self.upload(f, base * 4, &vram)?;
            }
            self.bases.insert(key, base);
            self.next_base = base + n_expert;
        }
        Ok(ptr)
    }

    /// The device counter arrays a gather launch writes: `(vram_flags, counts,
    /// tally, base)`. `None` if this tensor is past the counter capacity.
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
        // **Off while oversubscribed, for the MVP.** A swap here assumes every
        // expert is resident in one tier or the other; cold experts break that,
        // and fetches already move experts into VRAM on demand. Migration and
        // streaming meet in a later step. SSD-TIER.md D12.
        if self.stats.oversubscribed {
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
            Some(f) if self.deferring.get() => {
                self.patches.borrow_mut().push([f.ptr + (idx * 4) as u64, resident as u32 as u64, 4]);
                Ok(())
            }
            Some(f) => self.upload(f, idx * 4, &[resident]),
            None => Ok(()),
        }
    }

    /// Point one table entry at a new address.
    fn write_table_entry(&self, tkey: usize, e: u32, addr: ffi::CUdeviceptr) -> Result<()> {
        match self.tables.get(&tkey) {
            Some(t) if self.deferring.get() => {
                self.patches.borrow_mut().push([t.ptr + (e as usize * 8) as u64, addr, 8]);
                Ok(())
            }
            Some(t) => self.upload(t, e as usize * 8, &[addr]),
            None => Err(Error::Cuda {
                what: "expert table",
                detail: "migrating an expert whose tensor has no table".to_string(),
            }),
        }
    }

    /// Size the counter arrays for `n_experts`, the model's whole pool. Only
    /// before the first table: the arrays are allocated there, once.
    pub fn set_counter_capacity(&mut self, n_experts: usize) {
        if self.counts.is_none() && n_experts > 0 {
            self.counter_capacity = n_experts;
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

    /// A VRAM slot to fetch into: CLOCK over the slab, never taking a slot whose
    /// expert the layer being resolved picked. See [`clock_pick`].
    fn evict_unleased(&mut self) -> Result<u32> {
        let (owner, lease) = (&self.owner, &self.lease);
        match clock_pick(&mut self.heat, &mut self.hand, |at| {
            owner[at].is_some_and(|k| lease.contains(&k))
        }) {
            Some(at) => Ok(at as u32),
            None => Err(Error::Cuda {
                what: "expert cache",
                detail: format!(
                    "every one of the {} VRAM slots holds an expert the layer being resolved \
                     picked; the slab is smaller than one layer's picks",
                    self.owner.len()
                ),
            }),
        }
    }
}

/// Heat a slot can reach. The hand may need `heat_max() + 1` laps to find a
/// victim, so this trades scan length for how long a hot expert is protected.
///
/// **Deliberately small.** 40% of the 125B's hot set turns over inside one run
/// (`skew.py`, 19-09), so a cache that holds on too hard cannot follow the drift
/// — stickiness is the failure mode here, not the goal.
const HEAT_MAX: u8 = 3;

/// [`HEAT_MAX`], or `INFERRED_HEAT_MAX` when set; read once.
///
/// **`INFERRED_HEAT_MAX=1` is the control, and in the engine it is exact.** Heat
/// only ever originates two ways — an arrival at [`HEAT_NEW`] (1) and a touch at
/// `min(h + 1, heat_max())` — so at 1 every slot is 0 or 1, a touch sets it, a
/// pass clears it, and the loop bound falls back to `2n + 1`. That is precisely
/// the reference bit this replaced.
///
/// It is *not* exact for the unit tests below, which seed `heat` directly and so
/// can hold values this cap never produced. Those were falsified instead by
/// forcing the constant to 1, where the three behavioural tests fail with
/// `Some(0)` against `Some(1)` — the hand taking whichever slot it met first.
fn heat_max() -> u8 {
    static H: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *H.get_or_init(|| {
        std::env::var("INFERRED_HEAT_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .map(|v: u8| v.max(HEAT_NEW))
            .unwrap_or(HEAT_MAX)
    })
}

/// Heat an expert enters a slot with, below [`HEAT_MAX`] so it must be picked
/// again to be protected. This is the admission rule; see the call in
/// `make_resident`.
const HEAT_NEW: u8 = 1;

/// Whether a fetched expert's upload is queued rather than waited on.
/// `INFERRED_ASYNC_UPLOAD=0` restores the blocking copies.
///
/// **What it is worth**: a blocking 0.88 MiB upload measures 78 us against
/// 32 us of bytes at this bus's 28.6 GB/s, so roughly 46 us of each is the host
/// waiting while the copy engine idles between copies. Queued, a layer's ~7
/// fetches issue in ~20 us and the engine runs them back to back
/// (SSD-TIER.md D21).
fn async_upload_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("INFERRED_ASYNC_UPLOAD").map(|v| v != "0").unwrap_or(true))
}

/// GCLOCK with leases: advance the hand, decaying heat, and take the first slot
/// that is not leased and whose heat has reached zero. Leased slots are passed
/// over without decaying.
///
/// **Counters, not a reference bit, since 19-09.** The bit made this pure
/// recency, and recency alone is what a CLOCK cache gets wrong on this
/// workload: measured on a 125B run, the resident quarter of the pool served
/// 72.7% of accesses where an oracle of the same size served 87.9%. A count
/// adds frequency — the classic LRU-K/GCLOCK direction — so an expert picked
/// repeatedly outlives one picked once. Paired with `HEAT_NEW` admission, which
/// is the half that stops one-off arrivals flushing the stable set.
///
/// Terminates within `HEAT_MAX + 1` laps plus one step whenever any slot is
/// unleased: each lap drops every unleased slot's heat by at least one, so by
/// the last one some unleased slot is at zero. `None` only when every slot is
/// leased.
///
/// A free function rather than a method so the policy is testable without a
/// device, which is where it can be wrong in a way no output would reveal.
// **There is deliberately no `Drop` for `ExpertCache`, and one here segfaulted
// every async run (20-09).** `Cuda::drop` calls `cuCtxDestroy_v2` in its body,
// and a struct's fields drop *after* that body returns — so the cache's `Drop`
// runs against a context that no longer exists, and `cuEventSynchronize` on a
// dead context is a SIGSEGV, not an error code. Every arm with queued uploads
// exited 139; the blocking arms never create the event and never crashed.
//
// Teardown does not need the drain anyway: `cuCtxDestroy` blocks until the
// device has finished the context's work, and it runs before `Pinned` frees the
// page-locked staging a queued copy reads. The event handle goes with the
// context.
//
// What is left unguarded is a cache dropped *while* the context lives — a reset
// or a model swap. No such path exists today. One that appears needs an
// explicit drain at the call site, where the context is known good; it must not
// be a `Drop`.

fn clock_pick(heat: &mut [u8], hand: &mut usize, leased: impl Fn(usize) -> bool) -> Option<usize> {
    let n = heat.len();
    if n == 0 {
        return None;
    }
    for _ in 0..(heat_max() as usize + 1) * n + 1 {
        let at = *hand;
        *hand = (*hand + 1) % n;
        if leased(at) {
            continue;
        }
        if heat[at] > 0 {
            heat[at] -= 1;
        } else {
            return Some(at);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    //! The policy is testable without a device. Only the fills touch CUDA, so
    //! the ring is exercised through [`super::clock_pick`] directly — which is
    //! the part that can be wrong in a way no output would reveal.

    use super::{HEAT_MAX, HEAT_NEW, clock_pick};

    /// A GCLOCK ring over the real [`super::clock_pick`], with nothing leased.
    /// It used to be a hand-written copy "identical to" the cache's; it now
    /// calls the function the cache calls, so the two cannot drift.
    struct Clock {
        heat: Vec<u8>,
        hand: usize,
    }

    impl Clock {
        fn new(n: usize) -> Self {
            Self { heat: vec![0; n], hand: 0 }
        }
        fn evict(&mut self) -> usize {
            clock_pick(&mut self.heat, &mut self.hand, |_| false).unwrap_or(usize::MAX)
        }
    }

    /// An arrival must be able to lose to an established resident, which is the
    /// whole of the admission rule. If these ever became equal the policy would
    /// silently go back to being recency-only.
    #[test]
    fn a_new_arrival_starts_below_the_ceiling() {
        assert!(HEAT_NEW < HEAT_MAX, "HEAT_NEW {HEAT_NEW} must leave room to climb to {HEAT_MAX}");
        assert!(HEAT_NEW > 0, "an arrival at zero heat would be evicted before it is used again");
    }

    /// **The property this policy exists for, and the one the reference bit
    /// could not express**: the hand passes over a hot slot to take a colder one
    /// it meets later. Under the old bit both slots read "referenced" and the
    /// hand took whichever it reached first — here that would be slot 0, so this
    /// test fails on the policy it replaced.
    #[test]
    fn the_hand_passes_a_hot_slot_to_take_a_cold_one() {
        let mut heat = vec![HEAT_MAX, HEAT_NEW];
        let mut hand = 0;
        assert_eq!(clock_pick(&mut heat, &mut hand, |_| false), Some(1));
    }

    /// A one-off arrival does not flush an expert the layer keeps picking. With
    /// 27.3% of a 125B token's expert accesses cold, arrivals are the common
    /// case, so this is the ordinary path rather than an edge.
    #[test]
    fn an_arrival_is_evicted_before_an_expert_picked_again() {
        let mut heat = vec![HEAT_NEW; 2];
        // Slot 0 is picked again while resident; slot 1 is a fresh fetch.
        for _ in 0..HEAT_MAX {
            heat[0] = heat[0].saturating_add(1).min(HEAT_MAX);
        }
        let mut hand = 0;
        assert_eq!(clock_pick(&mut heat, &mut hand, |_| false), Some(1));
    }

    /// A slot survives exactly as many passes of the hand as it has heat — the
    /// graded version of "second chance". Leasing the other slots makes every
    /// step of the hand a pass over the one being measured.
    #[test]
    fn a_slot_survives_exactly_its_heat_in_passes() {
        let mut heat = vec![2u8, HEAT_MAX, HEAT_MAX];
        let mut hand = 0;
        assert_eq!(clock_pick(&mut heat, &mut hand, |at| at != 0), Some(0));
        assert_eq!(heat[0], 0, "two passes must have spent both units of heat");
    }

    /// **A leased slot is never taken**, even when its reference bit is clear and
    /// the hand is standing on it. This is the property whose absence made
    /// resolving a layer's `up` able to evict the `gate` it had just resolved.
    #[test]
    fn a_leased_slot_is_never_taken() {
        let mut heat = vec![0u8; 4];
        let mut hand = 0;
        let leased = |at: usize| at == 0 || at == 2;
        let got: Vec<usize> =
            (0..6).map(|_| clock_pick(&mut heat, &mut hand, leased).unwrap_or(usize::MAX)).collect();
        assert!(got.iter().all(|&s| s == 1 || s == 3), "took a leased slot: {got:?}");
    }

    /// A leased slot's heat is left alone as the hand passes, so the lease does
    /// not cost it the protection it earned once the lease ends.
    #[test]
    fn passing_a_leased_slot_leaves_its_heat_alone() {
        let mut heat = vec![HEAT_MAX, 0, 0];
        let mut hand = 0;
        assert_eq!(clock_pick(&mut heat, &mut hand, |at| at == 0), Some(1));
        assert_eq!(heat[0], HEAT_MAX, "the hand decayed a leased slot's heat");
    }

    /// With every slot leased there is nothing to take, and the answer is `None`
    /// rather than a spin or a leased slot.
    #[test]
    fn a_fully_leased_ring_yields_nothing() {
        let mut heat = vec![0, HEAT_MAX, 0];
        let mut hand = 1;
        assert_eq!(clock_pick(&mut heat, &mut hand, |_| true), None);
    }

    /// One unleased slot among many hot, leased ones is still found:
    /// termination must not depend on the leased slots' heat.
    #[test]
    fn a_single_unleased_slot_is_found_through_a_full_ring() {
        let mut heat = vec![HEAT_MAX; 8];
        let mut hand = 3;
        let got = clock_pick(&mut heat, &mut hand, |at| at != 6);
        assert_eq!(got, Some(6));
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

    /// A slot with one unit of heat survives exactly one pass of the hand, which
    /// is the "second chance" the reference bit used to give and what makes this
    /// an LRU approximation rather than FIFO. `HEAT_NEW` is that case, so this
    /// also pins what a fresh arrival is worth.
    #[test]
    fn a_slot_at_one_heat_gets_one_second_chance_and_no_more() {
        let mut c = Clock::new(4);
        for _ in 0..4 {
            c.evict();
        }
        c.heat = vec![HEAT_NEW, 0, 0, 0];
        // Slot 0 is protected this lap, so slot 1 goes first.
        assert_eq!(c.evict(), 1);
        assert_eq!(c.heat[0], 0, "the hand must spend the heat as it passes");
        // Its heat is gone, so the next lap takes it.
        assert_eq!(c.evict(), 2);
        assert_eq!(c.evict(), 3);
        assert_eq!(c.evict(), 0);
    }

    /// With every bit set the hand still terminates, clearing a whole lap
    /// first. An implementation that scanned for a cold slot without decaying
    /// would spin forever here, and it would only ever happen on a full cache
    /// under load. The loop bound grew with `HEAT_MAX`, so this is also the
    /// guard on that arithmetic.
    #[test]
    fn a_ring_at_full_heat_still_terminates() {
        let mut c = Clock::new(6);
        c.heat = vec![HEAT_MAX; 6];
        assert_eq!(c.evict(), 0);
        assert!(c.heat.iter().all(|&h| h == 0), "every slot should have decayed to zero: {:?}", c.heat);
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
