//! A spin-waiting thread pool.
//!
//! **Why this exists, measured rather than assumed.** Entering one rayon
//! parallel region on this machine costs **~430 us** — about 100x what it costs
//! on bare metal, and independent of payload size (a 3M-element region and a
//! 3K-element one both pay it). A decode step runs ~196 matmuls, so threading
//! them through rayon costs ~84 ms/token in dispatch alone, against a ~44 ms
//! token. That is why `par` can only afford to thread the LM head.
//!
//! The cost is sleep/wake, not work distribution. Rayon parks idle workers, and
//! waking them goes through a futex and — under WSL2 — the Windows hypervisor's
//! vCPU scheduling. A barrier that *spins* instead measures **0.40 us** per
//! round trip on the same box: **1088x cheaper**, or 0.08 ms/token across all
//! 196 matmuls.
//!
//! This is not a novel trick. It is what ggml does, and it is why llama.cpp
//! saturates this CPU while we do not.
//!
//! # Safety
//!
//! This module is the only place in the crate that uses `unsafe`, and it uses
//! it for exactly one thing: handing persistent worker threads a pointer to a
//! job that lives on the caller's stack. Rust cannot prove the job outlives the
//! workers' use of it, because that fact is enforced by the barrier rather than
//! by lifetimes. [`Pool::run`] does not return until every worker has signalled
//! completion, which is what makes the assertion true. Each `unsafe` block
//! below carries the specific invariant it relies on.
//!
//! The oracle in [`super::naive`] is `#![forbid(unsafe_code)]`, so none of this
//! can leak into the thing we diff against.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Spin iterations before yielding the core back to the scheduler.
///
/// During generation, work arrives every few hundred microseconds, so workers
/// almost never reach this. It exists so an idle pool does not burn a core at
/// 100% while a user reads output — the same spin-then-back-off shape ggml
/// uses, and the reason `Pool` is polite to leave running.
const SPINS_BEFORE_YIELD: u32 = 4096;

/// A job, type-erased so the shared state does not need a generic parameter.
///
/// A trait object would be a fat pointer, which cannot live in a single atomic
/// cell, so the closure is split into a thin data pointer plus a monomorphized
/// trampoline that knows how to call it.
type Trampoline = fn(*const (), usize, usize);

struct Shared {
    /// Bumped once per job. Workers spin until it changes.
    epoch: AtomicUsize,
    /// Workers that have finished the current job.
    done: AtomicUsize,
    /// Asks workers to exit.
    stop: AtomicBool,
    /// The current job. Written only between jobs, read only after a worker
    /// observes a new `epoch`, which orders every access.
    job: std::cell::UnsafeCell<Option<(*const (), Trampoline)>>,
    /// Total participants, counting the calling thread as index 0.
    n: usize,
}

// SAFETY: `job` is written by the calling thread strictly before it publishes a
// new `epoch` with Release ordering, and read by workers strictly after they
// observe that `epoch` with Acquire ordering. The calling thread does not touch
// it again until every worker has incremented `done`. So there is never a
// concurrent read and write, and the release/acquire pair establishes the
// happens-before edge that makes the write visible.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

/// Persistent workers that spin rather than sleep between jobs.
pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    /// `n` total participants. The calling thread is one of them, so `n - 1`
    /// threads are spawned. `n <= 1` makes [`Pool::run`] purely serial, with no
    /// threads and no atomics on the hot path.
    pub fn new(n: usize) -> Self {
        let n = n.max(1);
        let shared = Arc::new(Shared {
            epoch: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            job: std::cell::UnsafeCell::new(None),
            n,
        });

        let mut workers = Vec::with_capacity(n.saturating_sub(1));
        for index in 1..n {
            let shared = Arc::clone(&shared);
            workers.push(std::thread::spawn(move || worker(shared, index)));
        }
        Self { shared, workers }
    }

    /// Participants, including the calling thread.
    pub fn threads(&self) -> usize {
        self.shared.n
    }

    /// Run `f(worker_index, n_workers)` on every participant and return once
    /// all have finished.
    ///
    /// `f` must be `Sync` because every worker holds a shared reference to it
    /// concurrently. It is *not* required to be `'static`: the barrier, not the
    /// borrow checker, is what keeps it alive long enough.
    pub fn run<F>(&self, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        let n = self.shared.n;
        if n == 1 {
            f(0, 1);
            return;
        }

        // Monomorphized for this specific `F`, so the cast inside is sound.
        fn call<F: Fn(usize, usize) + Sync>(p: *const (), index: usize, n: usize) {
            // SAFETY: `p` was produced from a `&F` in `run` below, and `run`
            // has not returned, so the referent is still alive and immutable.
            let f = unsafe { &*(p as *const F) };
            f(index, n)
        }

        self.shared.done.store(0, Ordering::Release);
        // SAFETY: no worker can be reading `job` right now. Workers only read
        // it after seeing an `epoch` newer than the one they last handled, and
        // the previous job's workers all incremented `done` before `run`
        // returned, which this thread waited for.
        unsafe {
            *self.shared.job.get() = Some((&f as *const F as *const (), call::<F> as Trampoline));
        }
        // Release: publishes both the store above and `done = 0`.
        self.shared.epoch.fetch_add(1, Ordering::Release);

        // The calling thread is participant 0 -- one fewer thread to wake, and
        // it would otherwise just be spinning.
        f(0, n);

        while self.shared.done.load(Ordering::Acquire) < n - 1 {
            std::hint::spin_loop();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        // Wake anyone spinning on `epoch` so they observe `stop`.
        self.shared.epoch.fetch_add(1, Ordering::Release);
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

fn worker(shared: Arc<Shared>, index: usize) {
    let mut seen = 0usize;
    loop {
        let mut spins = 0u32;
        loop {
            let epoch = shared.epoch.load(Ordering::Acquire);
            if epoch != seen {
                seen = epoch;
                break;
            }
            if shared.stop.load(Ordering::Relaxed) {
                return;
            }
            spins += 1;
            if spins >= SPINS_BEFORE_YIELD {
                spins = 0;
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }

        if shared.stop.load(Ordering::Acquire) {
            return;
        }

        // SAFETY: the Acquire load of `epoch` above pairs with the Release
        // store in `run`, so this read sees the job that was published for this
        // epoch, and `run` is still on the caller's stack waiting for us.
        let job = unsafe { *shared.job.get() };
        if let Some((data, call)) = job {
            call(data, index, shared.n);
        }
        shared.done.fetch_add(1, Ordering::Release);
    }
}

/// A disjoint mutable view of an output buffer, shareable across workers.
///
/// # Safety
///
/// Constructing one asserts that every worker will write only to indices no
/// other worker writes, and that the buffer outlives the [`Pool::run`] call.
/// [`Rows::range`] hands out the standard even split, which satisfies the first
/// half by construction; anything that does not use it must argue the point
/// itself.
pub struct Rows {
    ptr: *mut f32,
    len: usize,
}

// SAFETY: the type carries no interior mutability of its own; the invariant
// that makes concurrent use sound is the disjointness asserted at construction.
unsafe impl Send for Rows {}
unsafe impl Sync for Rows {}

impl Rows {
    pub fn new(out: &mut [f32]) -> Self {
        Self {
            ptr: out.as_mut_ptr(),
            len: out.len(),
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The half-open row range belonging to worker `index` of `n`.
    ///
    /// Splitting by a per-worker chunk rather than round-robin keeps each
    /// worker on a contiguous run of memory, and guarantees the ranges are
    /// disjoint and cover the whole buffer.
    pub fn range(len: usize, index: usize, n: usize) -> (usize, usize) {
        let per = len.div_ceil(n);
        let start = (index * per).min(len);
        let end = (start + per).min(len);
        (start, end)
    }

    /// The slice for worker `index`.
    ///
    /// # Safety
    ///
    /// The caller must ensure no other worker is given an overlapping range —
    /// use [`Rows::range`] with a consistent `n` — and that the underlying
    /// buffer is still alive.
    pub unsafe fn slice(&self, start: usize, end: usize) -> &mut [f32] {
        debug_assert!(start <= end && end <= self.len);
        // SAFETY: delegated to the caller, per the contract above.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), end - start) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn every_worker_runs_exactly_once_per_job() {
        let pool = Pool::new(4);
        let hits: Vec<AtomicU32> = (0..4).map(|_| AtomicU32::new(0)).collect();
        for _ in 0..100 {
            pool.run(|i, n| {
                assert_eq!(n, 4);
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
        }
        for (i, h) in hits.iter().enumerate() {
            assert_eq!(h.load(Ordering::Relaxed), 100, "worker {i}");
        }
    }

    #[test]
    fn a_single_thread_pool_runs_inline() {
        let pool = Pool::new(1);
        assert_eq!(pool.threads(), 1);
        let seen = AtomicU32::new(0);
        pool.run(|i, n| {
            assert_eq!((i, n), (0, 1));
            seen.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(seen.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn ranges_are_disjoint_and_cover_everything() {
        for len in [0usize, 1, 7, 64, 1000, 3072] {
            for n in 1..=9 {
                let mut covered = vec![0u8; len];
                for i in 0..n {
                    let (s, e) = Rows::range(len, i, n);
                    for c in &mut covered[s..e] {
                        *c += 1;
                    }
                }
                assert!(
                    covered.iter().all(|&c| c == 1),
                    "len {len} across {n} workers: {covered:?}"
                );
            }
        }
    }

    /// The whole point: workers write disjoint halves of one buffer with no
    /// synchronization beyond the barrier, and the result is correct.
    #[test]
    fn workers_fill_disjoint_slices_of_one_buffer() {
        let pool = Pool::new(4);
        let mut buf = vec![0.0f32; 1000];
        {
            let rows = Rows::new(&mut buf);
            pool.run(|i, n| {
                let (s, e) = Rows::range(rows.len(), i, n);
                // SAFETY: `range` gives disjoint spans for distinct `i`, and
                // `buf` outlives this `run` call.
                let mine = unsafe { rows.slice(s, e) };
                for (k, v) in mine.iter_mut().enumerate() {
                    *v = (s + k) as f32;
                }
            });
        }
        assert!(buf.iter().enumerate().all(|(i, &v)| v == i as f32));
    }

    /// Jobs must be independent: a second job must not observe the first's
    /// bookkeeping.
    #[test]
    fn consecutive_jobs_do_not_interfere() {
        let pool = Pool::new(3);
        let mut buf = vec![0.0f32; 300];
        for round in 1..=20u32 {
            let rows = Rows::new(&mut buf);
            pool.run(|i, n| {
                let (s, e) = Rows::range(rows.len(), i, n);
                // SAFETY: as above.
                let mine = unsafe { rows.slice(s, e) };
                for v in mine.iter_mut() {
                    *v = round as f32;
                }
            });
            assert!(buf.iter().all(|&v| v == round as f32), "round {round}");
        }
    }

    #[test]
    fn dropping_the_pool_stops_its_workers() {
        // If Drop did not signal, this would hang rather than fail.
        let pool = Pool::new(4);
        pool.run(|_, _| {});
        drop(pool);
    }
}
