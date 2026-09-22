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
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

use super::{check, ffi};
use crate::error::{Error, Result};

pub(crate) use crate::platform::DIRECT_ALIGN;

/// Open the model file for tier-3 reads, unbuffered when `direct`.
/// See [`crate::platform::open_read`] for what that costs and buys.
pub(crate) fn open(path: &std::path::Path, direct: bool) -> std::io::Result<File> {
    crate::platform::open_read(path, direct)
}

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

/// One read: up to `len` bytes of the file at `offset` into the memory at `dst`,
/// of which the first `need` must arrive. An aligned `O_DIRECT` read of the file's
/// last expert can run past the end of the file, and gets a short read there.
struct Job {
    offset: u64,
    dst: usize,
    len: usize,
    need: usize,
    /// Handed back with the result, so one channel can collect many jobs.
    tag: usize,
    done: mpsc::Sender<(usize, std::io::Result<()>)>,
}

/// Where a submitted read reports: its tag and its result.
pub(crate) type Done = mpsc::Sender<(usize, std::io::Result<()>)>;

/// Read from `offset` into `dst` until at least `need` bytes have arrived.
fn read_at_least(file: &File, dst: &mut [u8], offset: u64, need: usize) -> std::io::Result<()> {
    let mut got = 0usize;
    while got < need {
        match crate::platform::read_at(file, &mut dst[got..], offset + got as u64) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("{got} of {need} bytes at offset {offset}"),
                ));
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Threads that `pread` the model file on request.
pub(crate) struct ReadPool {
    jobs: Option<mpsc::Sender<Job>>,
    workers: Vec<JoinHandle<()>>,
}

impl ReadPool {
    /// One worker per handle in `files`, each reading only through its own.
    ///
    /// **One handle each, not one shared.** On Linux a shared descriptor would
    /// do: `pread` names its offset. On Windows every read through one
    /// synchronous handle is serialized on its file object, so the handles come
    /// from separate opens of the path — `try_clone` would share the file object
    /// and serialize just the same. Measured on the 125B natively, unbuffered:
    /// decode 5.93 -> 6.07 tok/s. It was not the large Windows cost; that was the
    /// model's mapping slowing unbuffered reads (`platform::DIRECT_BY_DEFAULT`).
    pub(crate) fn new(files: Vec<File>) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let mut workers = Vec::with_capacity(files.len());
        for (i, file) in files.into_iter().enumerate() {
            let rx = Arc::clone(&rx);
            let handle = std::thread::Builder::new()
                .name(format!("expert-read-{i}"))
                .spawn(move || {
                    loop {
                        let job = match rx.lock() {
                            Ok(guard) => guard.recv(),
                            Err(_) => break,
                        };
                        let Ok(job) = job else { break };
                        // SAFETY: `read_all` and `submit`'s callers hand out `dst`
                        // from disjoint memory that stays alive and untouched until
                        // this job has reported, which is their contract.
                        let dst = unsafe { std::slice::from_raw_parts_mut(job.dst as *mut u8, job.len) };
                        let _ = job.done.send((job.tag, read_at_least(&file, dst, job.offset, job.need)));
                    }
                })
                .map_err(|e| Error::Cuda { what: "expert fetch", detail: format!("starting a read thread: {e}") })?;
            workers.push(handle);
        }
        Ok(Self { jobs: Some(tx), workers })
    }

    /// Fill every `(offset, dst, need)` from the file, concurrently: at least `need`
    /// bytes into each `dst`. Returns once all have finished, with the first error
    /// if any failed.
    pub(crate) fn read_all(&self, reads: &mut [(u64, &mut [u8], usize)]) -> Result<()> {
        let jobs = self.jobs.as_ref().ok_or_else(|| Error::Cuda {
            what: "expert fetch",
            detail: "the read pool is shut down".to_string(),
        })?;
        let (done_tx, done_rx) = mpsc::channel();
        let mut sent = 0usize;
        let mut first: Option<String> = None;
        for (i, (offset, dst, need)) in reads.iter_mut().enumerate() {
            let job = Job {
                offset: *offset,
                dst: dst.as_mut_ptr() as usize,
                len: dst.len(),
                need: *need,
                tag: i,
                done: done_tx.clone(),
            };
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
                Ok((_, Ok(()))) => {}
                Ok((_, Err(e))) => {
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

impl ReadPool {
    /// Queue one read and return at once: at least `need` of `len` bytes of the
    /// file at `offset` into the memory at `dst`, reported on `done` with `tag`.
    /// For prefetch, whose reads run while the GPU computes (SSD-TIER.md D20).
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of `len` bytes, and no one else may read or
    /// write it or free it, until `(tag, _)` has been received from `done` — or
    /// until this pool has been dropped, which finishes every queued job first.
    pub(crate) unsafe fn submit(
        &self,
        offset: u64,
        dst: *mut u8,
        len: usize,
        need: usize,
        tag: usize,
        done: &Done,
    ) -> Result<()> {
        let jobs = self.jobs.as_ref().ok_or_else(|| Error::Cuda {
            what: "expert prefetch",
            detail: "the read pool is shut down".to_string(),
        })?;
        let job = Job { offset, dst: dst as usize, len, need, tag, done: done.clone() };
        jobs.send(job).map_err(|_| Error::Cuda {
            what: "expert prefetch",
            detail: "the read threads have exited".to_string(),
        })
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
