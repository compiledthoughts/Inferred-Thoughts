//! The three OS calls this engine makes that are not the same on Linux and
//! Windows, behind one seam.
//!
//! **There are only three, and that is the point.** Every dependency is
//! cross-platform — `memmap2`, `thiserror`, `clap`, `rayon`, `serde`, and no
//! `libc` — so "WSL2 only" was habit rather than architecture. What is genuinely
//! per-OS is positional reads, unbuffered opens, and a page-cache hint.
//!
//! Native Linux needs nothing from this module that it did not already have:
//! the Unix arm is the code that was inline before.

use std::fs::File;
use std::path::Path;

/// Read into `buf` from `offset` without moving the file cursor.
///
/// Positional so a caller never depends on a shared cursor; the read pool
/// (`super::ops::cuda::fetch`) gives each of its threads its own handle.
#[cfg(unix)]
pub fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buf, offset)
}

/// Windows' positional read. Unlike `pread`, `seek_read` *does* move the
/// cursor, and reads through one handle are serialized by the I/O manager, so
/// concurrent readers each need their own handle — which is how
/// `ops::cuda::fetch::ReadPool` is built.
#[cfg(windows)]
pub fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buf, offset)
}

/// Fill `buf` from `offset`, or fail.
///
/// **Windows has no `read_exact_at`**, only the `seek_read` primitive, which may
/// return short exactly as `read` does. So the loop is written once here rather
/// than being an ambient guarantee at two call sites — a short read that silently
/// left a tail of an expert uninitialised would compute with whatever was in the
/// buffer, and produce plausible wrong tokens rather than an error.
pub fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    let mut got = 0usize;
    while got < buf.len() {
        match read_at(file, &mut buf[got..], offset + got as u64) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "the model file ended inside a tensor",
                ));
            }
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// `O_DIRECT` on Linux x86_64: `#define __O_DIRECT 040000` in glibc's
/// `bits/fcntl-linux.h:88`, `00040000` in the kernel's `asm-generic/fcntl.h:48`.
#[cfg(unix)]
const O_DIRECT: i32 = 0o40000;

/// `FILE_FLAG_NO_BUFFERING`, `winnt.h`. The same bargain as `O_DIRECT`: the
/// cache is bypassed, and buffer, offset and length must all be sector-aligned
/// — which [`DIRECT_ALIGN`] already guarantees.
#[cfg(windows)]
const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;

/// What an unbuffered read must be aligned to — buffer, offset and length — in
/// whole 4 KiB pages.
///
/// The disk under the WSL VHDX reports 512-byte logical and 4,096-byte physical
/// blocks, so 4 KiB satisfies both, and it is what stage 0 measured with.
/// Windows wants the volume's physical sector size, which is 4 KiB on every
/// NVMe this targets.
pub const DIRECT_ALIGN: usize = 4096;

/// Whether tier-3 reads are unbuffered unless `INFERRED_FETCH_DIRECT` says
/// otherwise. **Yes on Linux, no on Windows**, and the difference is measured.
///
/// Linux: `O_DIRECT` is faster (below) and keeps 250 MiB a token of fetches out
/// of a page cache that, under WSL's memory cap, hung the machine.
///
/// Windows: the engine holds a mapping of the model file for its whole life,
/// and **while any mapping of a file exists, unbuffered reads of that file run
/// at ~3.0 GB/s instead of ~5.3** — six threads of random 0.88 MiB reads on this
/// Gen5 drive, reproduced outside the engine; untouched, touched, or with only
/// the view unmapped, all the same; closing the mapping object restores it.
/// Buffered reads do not care: 5.9 GB/s with the mapping held. On the 125B,
/// natively: decode 6.15 -> 9.3 tok/s, 292 -> 107 us an expert, text identical.
/// The page-cache concern does not carry over: Windows' standby list is
/// reclaimed under pressure, with no VM cap to fill.
#[cfg(unix)]
pub const DIRECT_BY_DEFAULT: bool = true;
#[cfg(windows)]
pub const DIRECT_BY_DEFAULT: bool = false;

/// Open a model file for tier-3 reads, unbuffered when `direct`.
///
/// **Why unbuffered**: stage 0 on this drive read 4.7–4.9 GB/s with `O_DIRECT`
/// at one thread against 1.18 GB/s buffered and cold, and 10.1 against 6.5 at
/// eight. It also keeps fetches out of the page cache, which filling with model
/// pages is what hung the machine on 16-09.
#[cfg(unix)]
pub fn open_read(path: &Path, direct: bool) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut o = std::fs::OpenOptions::new();
    o.read(true);
    if direct {
        o.custom_flags(O_DIRECT);
    }
    o.open(path)
}

#[cfg(windows)]
pub fn open_read(path: &Path, direct: bool) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    let mut o = std::fs::OpenOptions::new();
    o.read(true);
    if direct {
        o.custom_flags(FILE_FLAG_NO_BUFFERING);
    }
    o.open(path)
}

#[cfg(unix)]
mod fadv {
    /// `POSIX_FADV_DONTNEED`, from `fcntl.h`.
    pub(super) const DONTNEED: std::ffi::c_int = 4;
    unsafe extern "C" {
        pub(super) fn posix_fadvise(
            fd: std::ffi::c_int,
            offset: i64,
            len: i64,
            advice: std::ffi::c_int,
        ) -> std::ffi::c_int;
    }
}

/// Hand a byte range of `path` back to the kernel's page cache.
///
/// `(0, 0)` names the whole file. Advisory: failure is ignored, and the caller
/// must stay correct without it.
///
/// **Why it exists**: placement reads the whole expert pool through the mapping,
/// so the cache climbs to 16 GiB while a 35B is being placed — which is what
/// takes a 32 GB machine to 91%. Evicting each tensor as it is placed holds the
/// cache at roughly one tensor, ~142 MiB.
#[cfg(unix)]
pub fn release_range(path: &Path, offset: i64, len: i64) {
    use std::os::unix::io::AsRawFd;
    let Ok(f) = File::open(path) else { return };
    // SAFETY: `f` owns a valid descriptor for the call, and the advice is a
    // hint the kernel clamps to the file.
    unsafe {
        let _ = fadv::posix_fadvise(f.as_raw_fd(), offset, len, fadv::DONTNEED);
    }
}

/// **A no-op on Windows, deliberately.** There is no per-range equivalent of
/// `POSIX_FADV_DONTNEED` — `SetSystemFileCacheSize` is process-wide and a blunt
/// instrument.
///
/// Acceptable because of *why* the Unix arm exists: the cache pressure it
/// answers is a WSL memory-cap problem. On Windows the file cache is the
/// standby list, which the memory manager hands back under pressure on its own
/// — both the pages the placement mapping touched and the tier-3 fetches, which
/// are buffered here by default (`DIRECT_BY_DEFAULT`).
///
/// If a Windows run is ever seen to thrash during placement, this is the first
/// thing to revisit, and the honest fix is unbuffered placement reads rather
/// than a cache hint.
#[cfg(windows)]
pub fn release_range(_path: &Path, _offset: i64, _len: i64) {}

/// [`release_range`] over a whole file.
pub fn release_file(path: &Path) {
    release_range(path, 0, 0);
}

#[cfg(unix)]
mod madv {
    /// `MADV_DONTNEED`, from `asm-generic/mman-common.h`.
    ///
    /// On a `MAP_PRIVATE` file mapping this discards the resident pages; a later
    /// read faults them back from the file. So it is safe by construction here —
    /// worst case it costs a re-read of data nothing reads again.
    pub(super) const DONTNEED: std::ffi::c_int = 4;
    unsafe extern "C" {
        pub(super) fn madvise(
            addr: *mut std::ffi::c_void,
            len: usize,
            advice: std::ffi::c_int,
        ) -> std::ffi::c_int;
    }
}

/// Release the resident pages of a live mapping that backs `data`, keeping only
/// whole pages inside it.
///
/// Rounds the start up and the end down, because `madvise` needs a page-aligned
/// address and dropping a partial page at either end would discard bytes
/// belonging to a neighbouring tensor.
///
/// Advisory and best-effort: a failure means the pages stay, which is the
/// behaviour before this existed, so the return value is deliberately ignored.
#[cfg(unix)]
pub fn release_mapped(data: &[u8]) {
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
        let _ = madv::madvise(lo as *mut std::ffi::c_void, hi - lo, madv::DONTNEED);
    }
}

/// **A no-op on Windows, deliberately**, for the reason [`release_range`] is.
///
/// The mapping is read-only, so its pages are clean, and Windows moves clean
/// file pages to the standby list and reclaims them under pressure on its own.
/// The Unix arm answers a WSL VM that holds on to memory it has been given;
/// there is no VM here.
///
/// If a Windows run is ever seen to hold the mapping resident during placement,
/// `VirtualUnlock` on the unlocked range trims it from the working set — a
/// documented side effect rather than the call's purpose, which is why it is not
/// used until a run shows it is needed.
#[cfg(windows)]
pub fn release_mapped(_data: &[u8]) {}
