//! Expert placement: the two-tier slab behind every pooled weight, its
//! budgets, and migration by the read counts the gather kernel keeps.

use crate::error::{Error, Result};
use crate::ops::{Experts, Weights};
use crate::ops::cuda::{Cuda, experts, ffi};

impl Cuda {
    /// What the expert cache did. `None` if the model has no pooled weights.
    ///
    /// **The measurement the whole design is waiting on.** `HANDOFF.md` §9
    /// item 2 proposed getting the hit rate from an offline replay of a routing
    /// trace; this is better, because it is the policy actually running against
    /// the traffic actually generated.
    pub fn expert_stats(&self) -> Option<experts::ExpertStats> {
        self.absorb_expert_counters();
        self.experts.borrow().as_ref().map(|c| c.stats())
    }

    /// Move hot experts into VRAM, from the counts the gather kernel writes.
    ///
    /// **Between passes, never inside one.** A recorded graph replays a fixed
    /// kernel sequence, so nothing may be inserted mid-pass — but the table it
    /// reads is ordinary device memory, and rewriting an entry here changes
    /// where the *next* replay looks with no re-record. That is the whole
    /// reason the two-tier design kept the table mutable.
    ///
    /// Every `MIGRATE_EVERY` passes, so the counter read-back (which costs a
    /// synchronize) is amortized, and bounded at `MIGRATE_BUDGET` exchanges so
    /// a single pass boundary cannot stall. Converges and then stops: the
    /// exchange only happens while some host-resident expert is busier than
    /// some resident one.
    pub(super) fn migrate_experts(&self, n_tokens: usize) {
        /// Tokens between migrations. The counter read-back costs a
        /// synchronize, so this is not free; 64 puts it under 1% of a decode
        /// pass while still converging inside a few hundred tokens.
        const MIGRATE_EVERY: u64 = 64;
        /// Exchanges per 64 tokens of evidence. At ~45 us each this is ~9 ms,
        /// which lands at a pass boundary rather than inside a token.
        const MIGRATE_BUDGET: usize = 200;
        /// Ceiling on one boundary's exchanges, so a large prefill pass cannot
        /// turn its boundary into a visible stall. 2,000 at ~45 us is ~90 ms
        /// against a 512-token pass that takes seconds.
        const MIGRATE_CAP: usize = 2_000;

        let since = match self.experts.borrow_mut().as_mut() {
            Some(c) => c.migration_state(n_tokens).0,
            None => return,
        };
        if since < MIGRATE_EVERY {
            return;
        }
        // **Budget scales with the evidence, because the trigger now can.** A
        // 512-token pass arrives with eight windows' worth of routing counts at
        // once; spending one window's budget on it would take eight prefills to
        // do what one could. Capped so a single boundary stays bounded.
        let budget = (MIGRATE_BUDGET * (since / MIGRATE_EVERY) as usize).min(MIGRATE_CAP);
        self.absorb_expert_counters();
        let counts = match self.experts.borrow().as_ref() {
            Some(c) => c.observed_counts().to_vec(),
            None => return,
        };
        let r = match self.experts.borrow_mut().as_mut() {
            Some(c) => {
                let r = c.migrate(budget, &counts);
                c.migration_done();
                r
            }
            None => return,
        };
        self.count_expert_uploads();
        if let Err(e) = r {
            self.note(Err::<(), _>(e));
        }
    }

    /// Add the expert cache's host-to-device copies since the last call to the
    /// crossing counters. Called after every cache operation that can copy.
    ///
    /// **Per copy, not per call.** The callers used to add one crossing for any
    /// call that filled bytes, which undercounted a fetch — the bytes and four
    /// eight-byte writes — and missed migration's table writes entirely.
    /// SSD-TIER.md, "The first run through the CLI".
    pub(super) fn count_expert_uploads(&self) {
        let (calls, bytes) = match self.experts.borrow().as_ref() {
            Some(c) => c.take_uploads(),
            None => return,
        };
        if calls > 0 {
            self.bump(|st| {
                st.h2d_calls += calls;
                st.h2d_bytes += bytes;
            });
        }
    }

    /// Re-place the whole expert pool once, by every read counted so far, with
    /// no swap budget. Returns the experts moved.
    ///
    /// **For `serve --warmup`, and only between passes.** Migration is bounded
    /// per boundary so that no pass stalls, which is right inside a session and
    /// wrong once at start-up, where nobody is waiting and the pool still sits in
    /// the layer order `ExpertCache::table` placed it in. Here the exchange runs
    /// to completion: afterwards every VRAM expert has been read at least as
    /// often as every host-tier one. Measured on the 35B at the default budget
    /// after a 733-token warm-up: 5,205 swaps in 1.83 s, ~350 us a swap, which is
    /// about eight times the ~45 us the migration comments assume.
    ///
    /// **The counts are kept.** They are the evidence later migration ranks by,
    /// and without them the session's first 64 tokens — a handful of reads per
    /// expert — would swap out experts the warm-up found busy but the session
    /// has not reached yet.
    pub fn replace_experts_by_counts(&self) -> Result<usize> {
        self.absorb_expert_counters();
        let counts = match self.experts.borrow().as_ref() {
            Some(c) => c.observed_counts().to_vec(),
            None => return Ok(0),
        };
        let moved = {
            let mut cache = self.experts.borrow_mut();
            let Some(c) = cache.as_mut() else { return Ok(0) };
            let moved = c.migrate(usize::MAX, &counts);
            c.migration_done();
            moved
        };
        self.count_expert_uploads();
        moved
    }

    /// Bring the device-side read counters home and fold them into the stats.
    ///
    /// **Without this the cache is unobservable.** Routing on the device means
    /// the host never learns which experts a token read, so `hits`,
    /// `host_reads` and the coverage distribution would all report zero — a
    /// working cache saying nothing, which is worse than a broken one saying
    /// so. Called from the two accessors rather than per token, because a
    /// counter read costs a synchronize.
    fn absorb_expert_counters(&self) {
        let (counts_ptr, tally_ptr, n) = {
            let b = self.experts.borrow();
            let Some(c) = b.as_ref() else { return };
            let n = c.counted_experts();
            match c.counters_base() {
                Some((counts, tally)) if n > 0 => (counts, tally, n),
                _ => return,
            }
        };
        if self.sync().is_err() {
            return;
        }
        let mut counts = vec![0u32; n];
        let mut tally = [0u64; 2];
        if self.d2h(&mut counts, counts_ptr).is_err() || self.d2h(&mut tally, tally_ptr).is_err() {
            return;
        }
        if let Some(c) = self.experts.borrow_mut().as_mut() {
            c.absorb_counters(&counts, &tally);
        }
    }

    /// Fix the expert slab at `slots` slots, whatever the budget says; `None`
    /// restores sizing by budget. For tests that must push a small model into
    /// tier 3. Call before the first expert is placed.
    pub fn set_expert_slots(&self, slots: Option<usize>) {
        self.expert_slots.set(slots);
    }

    /// Turn lookahead prefetch on or off for this backend's expert cache, once it
    /// exists (`INFERRED_PREFETCH=0` sets the default off). For tests comparing
    /// both.
    pub fn set_prefetch(&self, on: bool) {
        if let Some(c) = self.experts.borrow_mut().as_mut() {
            c.set_prefetch(on);
        }
    }

    /// Cap the expert slab, in bytes. Zero restores the automatic budget.
    pub fn set_expert_budget(&self, bytes: usize) {
        // Expressed as a reserve because that is what the sizing code has to
        // work with: total free VRAM minus what everything else will need.
        let (free, _) = self.mem_info().unwrap_or((0, 0));
        if bytes == 0 {
            self.expert_reserve.set(experts::DEFAULT_RESERVE);
            self.expert_cap.set(Some(experts::DEFAULT_SLAB_CAP));
        } else {
            self.expert_reserve.set(free.saturating_sub(bytes));
            self.expert_cap.set(None);
        }
    }

    /// Leave room for the model's permanent weights when sizing the automatic slab.
    ///
    /// **The hole this closes is the one `reserve_for_kv` closed for the cache.**
    /// The slab is sized at the first pooled tensor, inside block 0, when almost
    /// none of the permanent weights are up yet. `DEFAULT_RESERVE` covered the
    /// 35B's 1.68 GiB of them; the 125B has 4.44 GiB, so a default slab took VRAM
    /// its later blocks then failed to get. The reserve becomes the larger of
    /// what it is and `dense + NON_WEIGHT_RESERVE`, which leaves both 35B files at
    /// `DEFAULT_RESERVE` exactly.
    ///
    /// No effect after an explicit `set_expert_budget`, whose number is the
    /// caller's. Call before `reserve_for_kv`, which adds to the result.
    pub fn reserve_for_weights(&self, dense_bytes: usize) {
        if self.expert_cap.get().is_none() {
            return;
        }
        let want = dense_bytes.saturating_add(experts::NON_WEIGHT_RESERVE);
        self.expert_reserve.set(self.expert_reserve.get().max(want));
    }

    /// Declare the model's whole expert pool — the sum of `n_expert` over every
    /// expert tensor — so the read counters are sized for all of it.
    ///
    /// **Setup's job, before the first pass.** The counters are one flat device
    /// array handed out a slice per tensor in first-seen order, allocated once
    /// because growing it would move the slices already given out. A fixed size
    /// was right for the 35B's 30,720 and wrong for the 125B's 73,728, where the
    /// last 16 tensors got no slice and their layers' experts never ran.
    pub fn set_expert_pool(&self, n_experts: usize) {
        self.expert_pool.set((n_experts > 0).then_some(n_experts));
    }

    /// Slab bytes from free VRAM at sizing time: free minus the reserve, under
    /// the automatic cap when there is one.
    fn slab_budget(&self, free: usize) -> usize {
        let budget = free.saturating_sub(self.expert_reserve.get());
        self.expert_cap.get().map_or(budget, |cap| budget.min(cap))
    }

    /// Cap the page-locked host tier behind the expert slab, in bytes.
    ///
    /// Zero restores [`experts::DEFAULT_HOST_BUDGET`]. Must be called before
    /// the first pooled tensor, since the cache reads it once at construction.
    pub fn set_expert_host_budget(&self, bytes: usize) {
        self.expert_host_budget
            .set(if bytes == 0 { experts::DEFAULT_HOST_BUDGET } else { bytes });
    }

    /// Hold `bytes` of VRAM back for the KV cache.
    ///
    /// **Fixes a sizing hole rather than tuning one.** The expert slab is sized
    /// from free VRAM at the first pooled tensor, which is inside block 0,
    /// while KV slabs are allocated at the first `kv_write` — block 3 on the
    /// 35B. So without this the slab takes VRAM the KV cache is going to need,
    /// and at a long context the KV allocation fails partway through the first
    /// prefill. At 4k it hides inside `DEFAULT_RESERVE`'s slack; at 256k it is
    /// 5 GiB and does not.
    ///
    /// Additive, because the caller knows the context length and this file
    /// knows the rest.
    pub fn reserve_for_kv(&self, bytes: usize) {
        self.expert_reserve.set(self.expert_reserve.get().saturating_add(bytes));
    }

    /// What fraction of expert reads the busiest slab-many tensors accounted
    /// for. See [`experts::ExpertCache::coverage`].
    pub fn expert_coverage(&self) -> Option<(f64, u64)> {
        self.absorb_expert_counters();
        self.experts.borrow().as_ref().map(|c| c.coverage())
    }

    /// Tell the backend which file the weights came from, so its page cache
    /// can be dropped once every expert has been placed.
    ///
    /// Not discovered from the mapping because there is no portable way back
    /// from an address to a path; the caller opened the file and knows.
    pub fn set_model_path(&self, path: &std::path::Path) {
        *self.model_path.borrow_mut() = Some(path.to_path_buf());
    }

    /// Where the model's mapping starts, so a weight's address can be turned
    /// into a file offset and its page cache evicted once it is placed.
    pub fn set_map_base(&self, base: usize) {
        self.map_base.set(Some(base));
    }

    /// The device address of a weight, from whichever residency it belongs to.
    ///
    /// **The one branch that decides whether this engine can serve a model
    /// larger than VRAM.** Anything that fits goes in the permanent mirror, as
    /// it always has. A tensor marked `pooled` — an MoE expert, and only an MoE
    /// expert — goes in the bounded slab, where it may be evicted.
    ///
    /// The distinction is a fact about the tensor, carried on `Weights::pooled`,
    /// not a policy. The policy is [`experts::ExpertCache`]'s.
    ///
    /// Note the slab is two-tier: a pooled tensor that does not fit in VRAM
    /// gets a page-locked host address the kernel dereferences over PCIe, so
    /// this function always returns an address and never stalls on a fill.
    pub(super) fn expert_or_resident(&self, w: &Weights<'_>) -> Result<ffi::CUdeviceptr> {
        if !w.pooled {
            return self.resident(w.data);
        }
        let mut slot = self.experts.borrow_mut();
        if slot.is_none() {
            // Sized here rather than at construction, because "free VRAM" only
            // means something once the permanent weights are on their way up.
            // Nothing before the first expert of layer 0 is large.
            let (free, _) = self.mem_info()?;
            let budget = self.slab_budget(free);
            let slots = budget / w.data.len().max(1);
            let mut c = experts::ExpertCache::new(w.data.len(), slots, self.expert_host_budget.get())?;
            if let Some(n) = self.expert_pool.get() {
                c.set_counter_capacity(n);
            }
            *slot = Some(c);
        }
        let cache = match slot.as_mut() {
            Some(c) => c,
            None => {
                return Err(Error::Cuda {
                    what: "expert cache",
                    detail: "cache vanished between build and use".to_string(),
                });
            }
        };
        let ptr = cache.address_of(w.data.as_ptr() as usize, w.data);
        drop(slot);
        // A miss is a bus crossing and is counted; a hit moves nothing. Counted
        // rather than derived for the reason `DeviceBuffer::from_slice` taught
        // this session — an upload the counters cannot see reads as "3.9 MiB
        // up" against an actual 3111, and this is the exact traffic the whole
        // design is drawn against.
        self.count_expert_uploads();
        ptr
    }

    /// The device pointer table for one `Experts` tensor.
    ///
    /// Built on first sight, which places every one of its experts — see
    /// [`experts::ExpertCache::table`] for why eager placement is the price of
    /// a graph, and what it costs.
    pub(super) fn expert_table(&self, w: &Experts<'_>) -> Result<ffi::CUdeviceptr> {
        let key = w.data.as_ptr() as usize;
        let mut slot = self.experts.borrow_mut();
        if slot.is_none() {
            let (free, _) = self.mem_info()?;
            let budget = self.slab_budget(free);
            let stride = w.stride();
            let slots = self.expert_slots.get().unwrap_or(budget / stride.max(1));
            let mut c = experts::ExpertCache::new(stride, slots, self.expert_host_budget.get())?;
            if let (Some(p), Some(b)) = (self.model_path.borrow().as_ref(), self.map_base.get()) {
                c.set_source(p.clone(), b);
            }
            if let Some(n) = self.expert_pool.get() {
                c.set_counter_capacity(n);
            }
            *slot = Some(c);
        }
        let cache = match slot.as_mut() {
            Some(c) => c,
            None => {
                return Err(Error::Cuda {
                    what: "expert table",
                    detail: "cache vanished between build and use".to_string(),
                });
            }
        };
        let ptr = cache.table(key, w.data, w.n_expert);
        drop(slot);
        self.count_expert_uploads();
        ptr
    }
}
