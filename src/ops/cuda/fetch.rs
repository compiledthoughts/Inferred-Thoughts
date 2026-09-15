//! Tier 3's reads: cold experts read from the model file concurrently, into
//! page-locked memory, before the cache publishes them.
//!
//! **Measured first** (`7d7a3a2`): on the 125B the fetch path was ~84% of a
//! 315 ms decode token and its reads 152.8 ms of that — one `pread` of 0.88 MiB
//! at a time, 481 us each, the drive at 700–800 MB/s against 9.72 GB/s at queue
//! depth 32 (`TIERS.md`). A layer's misses are all known at its boundary, so
//! they can be in flight together. SSD-TIER.md D3, D10.
//!
//! Two pieces, both reused for the life of the cache:
//! - [`ReadPool`] — persistent threads blocked on a channel. Persistent because a
//!   125B token resolves ~87 times; spawning per resolve would cost milliseconds.
//! - [`Pinned`] — a page-locked staging buffer, so the upload that follows copies
//!   from pinned memory rather than through the driver's pageable bounce path.

use std::ffi::c_void;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

use super::{check, ffi};
use crate::error::{Error, Result};

/// Page-locked host memory owned for its lifetime. Not device-mapped: nothing
/// but a host-to-device copy ever reads it.
pub(crate) struct Pinned {
    ptr: *mut u8,
    len: usize,
}

impl Pinned {
    pub(crate) fn new(len: usize) -> Result<Self> {
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: `host` is an out-parameter the driver fills, checked before use.
        // `Drop` returns the allocation.
        unsafe { check(ffi::cuMemHostAlloc(&mut host, len, 0), "cuMemHostAlloc")? };
        Ok(Self { ptr: host as *mut u8, len })
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` owns `len` bytes for the life of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `&mut self` makes this the only view.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated by `cuMemHostAlloc` in `new` and freed once, here.
            unsafe {
                let _ = ffi::cuMemFreeHost(self.ptr as *mut c_void);
            }
        }
    }
}

/// One read: `len` bytes of the file at `offset` into the memory at `dst`.
struct Job {
    offset: u64,
    dst: usize,
    len: usize,
    done: mpsc::Sender<std::io::Result<()>>,
}

/// Threads that `pread` the model file on request.
pub(crate) struct ReadPool {
    jobs: Option<mpsc::Sender<Job>>,
    workers: Vec<JoinHandle<()>>,
}

impl ReadPool {
    /// `threads` workers sharing one handle to `file`. `pread` names its offset,
    /// so concurrent reads through one descriptor do not interfere.
    pub(crate) fn new(file: File, threads: usize) -> Result<Self> {
        let file = Arc::new(file);
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let mut workers = Vec::with_capacity(threads);
        for i in 0..threads.max(1) {
            let (rx, file) = (Arc::clone(&rx), Arc::clone(&file));
            let handle = std::thread::Builder::new()
                .name(format!("expert-read-{i}"))
                .spawn(move || {
                    loop {
                        let job = match rx.lock() {
                            Ok(guard) => guard.recv(),
                            Err(_) => break,
                        };
                        let Ok(job) = job else { break };
                        // SAFETY: `read_all` hands out `dst` from disjoint
                        // `&mut [u8]` slices and does not return until every job
                        // it sent has reported, so the memory outlives this use and
                        // no other thread touches it meanwhile.
                        let dst = unsafe { std::slice::from_raw_parts_mut(job.dst as *mut u8, job.len) };
                        let _ = job.done.send(file.read_exact_at(dst, job.offset));
                    }
                })
                .map_err(|e| Error::Cuda { what: "expert fetch", detail: format!("starting a read thread: {e}") })?;
            workers.push(handle);
        }
        Ok(Self { jobs: Some(tx), workers })
    }

    /// Fill every `(offset, dst)` from the file, concurrently. Returns once all
    /// have finished, with the first error if any failed.
    pub(crate) fn read_all(&self, reads: &mut [(u64, &mut [u8])]) -> Result<()> {
        let jobs = self.jobs.as_ref().ok_or_else(|| Error::Cuda {
            what: "expert fetch",
            detail: "the read pool is shut down".to_string(),
        })?;
        let (done_tx, done_rx) = mpsc::channel();
        let mut sent = 0usize;
        let mut first: Option<String> = None;
        for (offset, dst) in reads.iter_mut() {
            let job = Job { offset: *offset, dst: dst.as_mut_ptr() as usize, len: dst.len(), done: done_tx.clone() };
            match jobs.send(job) {
                Ok(()) => sent += 1,
                Err(_) => {
                    first.get_or_insert_with(|| "the read threads have exited".to_string());
                    break;
                }
            }
        }
        drop(done_tx);
        // Every job sent must report before this returns: that is what makes the
        // raw pointers in `Job` sound.
        for _ in 0..sent {
            match done_rx.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    first.get_or_insert_with(|| e.to_string());
                }
                // Every sender is gone, so every job sent has been dropped or done.
                Err(_) => {
                    first.get_or_insert_with(|| "a read thread exited mid-read".to_string());
                    break;
                }
            }
        }
        match first {
            None => Ok(()),
            Some(detail) => Err(Error::Cuda { what: "expert fetch", detail }),
        }
    }
}

impl Drop for ReadPool {
    fn drop(&mut self) {
        // Closing the channel ends each worker's `recv` loop.
        self.jobs = None;
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}
