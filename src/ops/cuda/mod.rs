//! CUDA, through the driver API, targeting sm_120.
//!
//! `CLAUDE.md` puts this on our side of the borrowed/ours line: the dense path
//! is llama.cpp's to win, but the MoE expert path and anything the sm_120 stack
//! does not already serve is ours to write. This module is the foundation for
//! that — device discovery, memory, PTX loading, kernel launch — plus a first
//! kernel to prove the toolchain end to end.
//!
//! **The first Q8_0 kernel is bit-exact against [`crate::ops::naive`], and that
//! is deliberate.** One thread per output row, walking the row serially, is the
//! same arithmetic in the same order as the CPU oracle. It is not fast — the
//! reads do not coalesce — but it means the very first GPU result can be
//! checked with `assert_eq!` on raw bits rather than a tolerance we would have
//! to justify. Optimizing it will change the accumulation order and cost that
//! property; doing so knowingly, with a measured number in hand, is the point
//! of starting here.
//!
//! Everything is behind the `cuda` feature so a machine without a toolkit still
//! builds and tests the rest of the crate.

pub mod ffi;
mod ops;

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::{CString, c_void};

use crate::error::{Error, Result};

/// The PTX produced by `build.rs`, embedded so there is no file to ship or find
/// at run time.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn check(code: ffi::CUresult, what: &'static str) -> Result<()> {
    if code == ffi::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::Cuda {
            what,
            detail: ffi::describe(code),
        })
    }
}

/// An owned CUDA context with our kernels loaded.
///
/// Dropping it tears down the module and context in that order, which is what
/// the driver requires.
pub struct Cuda {
    context: ffi::CUcontext,
    module: ffi::CUmodule,
    name: String,
    capability: (i32, i32),
    sm_count: i32,
    total_mem: usize,

    /// Host pointer -> its device copy, for anything that does not change
    /// between calls: quantized weight matrices and the f32 norm vectors.
    ///
    /// Without this the backend would re-upload every weight it touches on
    /// every call. The 0.6B reads 0.59 GiB per forward pass, which at the
    /// measured 28.6 GB/s is ~22 ms of PCIe per token against a 16.4 ms CPU
    /// token — the GPU path would lose to the CPU for reasons that have
    /// nothing to do with any kernel. Keying on the mmap pointer works because
    /// `Weights` borrows the mapping, so the address is stable for the run.
    weights: RefCell<HashMap<usize, DeviceBuffer>>,

    /// Host pointer -> device mirror of one layer's K or V slab.
    ///
    /// The KV cache only ever appends, so each call uploads the positions
    /// added since the last one rather than the whole slab. Re-uploading would
    /// be ~8 MB per layer per token at 4k context; the delta is 2 KB.
    kv: RefCell<HashMap<usize, KvMirror>>,

    /// Reusable device scratch, indexed by role. Growable, never shrunk, so a
    /// steady-state token allocates nothing.
    pool: RefCell<Vec<DeviceBuffer>>,

    /// The first driver error any op hit. See `Cuda::take_error`.
    error: RefCell<Option<Error>>,

    /// Resolved kernel handles, by symbol. `cuModuleGetFunction` is a driver
    /// call and building its argument allocates a `CString`; a decode step
    /// makes ~478 launches, so neither belongs on that path.
    functions: RefCell<HashMap<&'static str, ffi::CUfunction>>,

    /// What this backend actually asked the driver to do. See [`DeviceStats`].
    stats: Cell<DeviceStats>,
}

/// Driver traffic, counted rather than derived.
///
/// `CLAUDE.md`'s profiler rule is "derive bytes, do not count them", because
/// weight and KV traffic are functions of shapes and a counter would recompute
/// a constant. **This is the case that rule does not cover.** How many times a
/// backend crosses the bus is a property of the backend, not of the model, and
/// it is exactly the quantity the `Ops` seam determines — so it has to be
/// observed. Two increments per op, off any inner loop.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeviceStats {
    pub launches: u64,
    pub h2d_calls: u64,
    pub h2d_bytes: u64,
    pub d2h_calls: u64,
    pub d2h_bytes: u64,
}

impl DeviceStats {
    /// Bus crossings per token, given how many tokens produced these counts.
    pub fn crossings_per_token(&self, tokens: u64) -> f64 {
        if tokens == 0 {
            return 0.0;
        }
        (self.h2d_calls + self.d2h_calls) as f64 / tokens as f64
    }
}

/// What one round trip through the `Ops` seam costs on this device, measured
/// rather than assumed. Produced by [`Cuda::benchmark`].
#[derive(Debug, Clone, Copy)]
pub struct DeviceBench {
    pub launch_us: f64,
    pub h2d_us: f64,
    pub d2h_us: f64,
    pub round_trip_us: f64,
}

impl DeviceBench {
    /// Milliseconds per token the seam costs, before any arithmetic.
    ///
    /// Built from the counts actually observed rather than from an assumed two
    /// crossings per op: uploads and downloads are not symmetric, because a
    /// matmul uploads a quantized activation as two buffers and RoPE uploads
    /// three. Multiplying a round-trip average by "ops per token" gets this
    /// wrong in both directions.
    pub fn predicted_ms(&self, stats: &DeviceStats, tokens: u64) -> f64 {
        if tokens == 0 {
            return 0.0;
        }
        let total = stats.h2d_calls as f64 * self.h2d_us
            + stats.d2h_calls as f64 * self.d2h_us
            + stats.launches as f64 * self.launch_us;
        total / tokens as f64 / 1000.0
    }
}

/// A device mirror of a host KV slab, and how much of it is current.
struct KvMirror {
    buf: DeviceBuffer,
    /// Positions already copied. A smaller `n_pos` than this means the cache
    /// was reset, so the mirror is refilled from the start.
    uploaded: usize,
}

// SAFETY: a CUDA context is usable from any thread that has it current, and we
// only ever use it from the thread that owns this value. The raw pointers are
// driver handles, not references into our address space.
unsafe impl Send for Cuda {}

impl Cuda {
    /// Initialize the driver, take device `ordinal`, and load the kernels.
    pub fn new(ordinal: i32) -> Result<Self> {
        // SAFETY: every call below is checked, and each is passed either a
        // valid out-pointer to a local or a handle the driver just produced.
        unsafe {
            check(ffi::cuInit(0), "cuInit")?;

            let mut count = 0;
            check(ffi::cuDeviceGetCount(&mut count), "cuDeviceGetCount")?;
            if ordinal >= count {
                return Err(Error::Cuda {
                    what: "cuDeviceGet",
                    detail: format!("device {ordinal} requested, {count} present"),
                });
            }

            let mut device = 0;
            check(ffi::cuDeviceGet(&mut device, ordinal), "cuDeviceGet")?;

            let mut raw = [0i8; 128];
            check(
                ffi::cuDeviceGetName(raw.as_mut_ptr(), raw.len() as i32, device),
                "cuDeviceGetName",
            )?;
            let name = std::ffi::CStr::from_ptr(raw.as_ptr())
                .to_string_lossy()
                .into_owned();

            let attr = |a| -> Result<i32> {
                let mut v = 0;
                check(
                    ffi::cuDeviceGetAttribute(&mut v, a, device),
                    "cuDeviceGetAttribute",
                )?;
                Ok(v)
            };
            let capability = (attr(ffi::ATTR_CC_MAJOR)?, attr(ffi::ATTR_CC_MINOR)?);
            let sm_count = attr(ffi::ATTR_SM_COUNT)?;

            let mut total_mem = 0usize;
            check(
                ffi::cuDeviceTotalMem_v2(&mut total_mem, device),
                "cuDeviceTotalMem",
            )?;

            let mut context: ffi::CUcontext = std::ptr::null_mut();
            check(ffi::cuCtxCreate_v2(&mut context, 0, device), "cuCtxCreate")?;

            // The PTX is a NUL-terminated image as far as the driver is
            // concerned, so it has to be one.
            let image = CString::new(PTX).map_err(|_| Error::Cuda {
                what: "cuModuleLoadData",
                detail: "PTX contains an interior NUL".to_string(),
            })?;
            let mut module: ffi::CUmodule = std::ptr::null_mut();
            let load = ffi::cuModuleLoadData(&mut module, image.as_ptr() as *const c_void);
            if load != ffi::CUDA_SUCCESS {
                ffi::cuCtxDestroy_v2(context);
                return Err(Error::Cuda {
                    what: "cuModuleLoadData",
                    detail: format!(
                        "{} — a JIT failure here usually means the PTX targets a \
                         newer architecture than the driver understands",
                        ffi::describe(load)
                    ),
                });
            }

            Ok(Self {
                context,
                module,
                name,
                capability,
                sm_count,
                total_mem,
                weights: RefCell::new(HashMap::new()),
                kv: RefCell::new(HashMap::new()),
                pool: RefCell::new(Vec::new()),
                error: RefCell::new(None),
                functions: RefCell::new(HashMap::new()),
                stats: Cell::new(DeviceStats::default()),
            })
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Compute capability as `(major, minor)` — `(12, 0)` is sm_120.
    pub fn capability(&self) -> (i32, i32) {
        self.capability
    }

    pub fn sm_count(&self) -> i32 {
        self.sm_count
    }

    pub fn total_mem(&self) -> usize {
        self.total_mem
    }

    /// Free and total device memory, right now.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: both are valid out-pointers and the context is current.
        unsafe { check(ffi::cuMemGetInfo_v2(&mut free, &mut total), "cuMemGetInfo")? };
        Ok((free, total))
    }

    fn function(&self, name: &str) -> Result<ffi::CUfunction> {
        let cname = CString::new(name).map_err(|_| Error::Cuda {
            what: "cuModuleGetFunction",
            detail: format!("{name:?} is not a valid symbol"),
        })?;
        let mut f: ffi::CUfunction = std::ptr::null_mut();
        // SAFETY: `self.module` is loaded and alive for `self`'s lifetime.
        unsafe {
            check(
                ffi::cuModuleGetFunction(&mut f, self.module, cname.as_ptr()),
                "cuModuleGetFunction",
            )?
        };
        Ok(f)
    }

    /// Block until every launched kernel has finished.
    pub fn sync(&self) -> Result<()> {
        // SAFETY: no arguments; only reports the context's status.
        unsafe { check(ffi::cuCtxSynchronize(), "cuCtxSynchronize") }
    }

    /// Launch a kernel by name over a 1-D grid, then block until it finishes.
    ///
    /// Synchronizing on every launch is the naive shape, and deliberate: the
    /// `Ops` seam hands each method host slices and expects host slices back,
    /// so every call is a self-contained round trip whatever we do here. What
    /// that costs is one of the things this backend exists to measure.
    ///
    /// # Safety
    /// `params` must match the named kernel's signature, and every device
    /// pointer in it must address an allocation large enough for the extents
    /// the kernel will walk.
    unsafe fn launch(
        &self,
        name: &'static str,
        grid: u32,
        block: u32,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        // SAFETY: forwarded to the caller's contract.
        unsafe { self.launch_shared(name, grid, block, 0, params) }
    }

    /// As [`Cuda::launch`], with dynamic shared memory.
    ///
    /// # Safety
    /// As [`Cuda::launch`], and `shared_bytes` must cover what the kernel
    /// indexes through its `extern __shared__` array.
    unsafe fn launch_shared(
        &self,
        name: &'static str,
        grid: u32,
        block: u32,
        shared_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        let f = self.cached_function(name)?;
        // SAFETY: the caller's contract, documented above.
        unsafe {
            check(
                ffi::cuLaunchKernel(
                    f,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    shared_bytes,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?
        };
        self.bump(|s| s.launches += 1);
        // Deliberately no `sync` here. Every `Ops` method ends in a
        // device-to-host copy on the null stream, which is ordered after this
        // kernel and is itself synchronous, so an explicit barrier is a second
        // driver call buying nothing. A launch failure surfaces at that copy.
        Ok(())
    }

    /// Update the counters. `Cell` rather than atomics: this backend is used
    /// from one thread, and the whole point is that it costs nothing.
    pub(super) fn bump(&self, f: impl FnOnce(&mut DeviceStats)) {
        let mut s = self.stats.get();
        f(&mut s);
        self.stats.set(s);
    }

    /// Driver traffic so far.
    pub fn stats(&self) -> DeviceStats {
        self.stats.get()
    }

    /// Time a launch, an upload, a download, and the three together.
    ///
    /// Synthetic on purpose: a 4 KiB payload is about the size of this model's
    /// residual stream, so the result is the *fixed* cost of using the seam,
    /// with as little real work attached as possible.
    pub fn benchmark(&self, reps: u32) -> Result<DeviceBench> {
        use std::time::Instant;

        let n = 1024usize;
        let host = vec![1.0f32; n];
        let x = DeviceBuffer::from_slice(&host)?;
        let y = DeviceBuffer::from_slice(&host)?;
        let mut back = vec![0.0f32; n];

        // Resolve the kernel and let the JIT settle before anything is timed.
        self.saxpy(1.0, &x, &y, n)?;

        let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
            let t = Instant::now();
            for _ in 0..reps {
                f()?;
            }
            Ok(t.elapsed().as_secs_f64() * 1e6 / f64::from(reps))
        };

        let launch_us = time(&mut || self.saxpy(1.0, &x, &y, n))?;
        let h2d_us = time(&mut || x.write(&host))?;
        let d2h_us = time(&mut || y.read(&mut back))?;
        let round_trip_us = time(&mut || {
            x.write(&host)?;
            self.saxpy(1.0, &x, &y, n)?;
            y.read(&mut back)
        })?;

        Ok(DeviceBench {
            launch_us,
            h2d_us,
            d2h_us,
            round_trip_us,
        })
    }

    /// A kernel handle, resolved once per symbol.
    fn cached_function(&self, name: &'static str) -> Result<ffi::CUfunction> {
        let mut map = self.functions.borrow_mut();
        if let Some(f) = map.get(name) {
            return Ok(*f);
        }
        let f = self.function(name)?;
        map.insert(name, f);
        Ok(f)
    }

    /// `y = a * x + y`, elementwise. The toolchain proof.
    pub fn saxpy(&self, a: f32, x: &DeviceBuffer, y: &DeviceBuffer, n: usize) -> Result<()> {
        let f = self.function("saxpy")?;
        let mut n_arg = n as i32;
        let mut a_arg = a;
        let mut x_arg = x.ptr;
        let mut y_arg = y.ptr;
        let mut params = [
            &mut n_arg as *mut _ as *mut c_void,
            &mut a_arg as *mut _ as *mut c_void,
            &mut x_arg as *mut _ as *mut c_void,
            &mut y_arg as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = n.div_ceil(block as usize) as u32;
        // SAFETY: the parameter list matches the kernel's signature in
        // `kernels.cu`, and both buffers hold at least `n` floats — checked by
        // the caller, which is this crate's tests.
        unsafe {
            check(
                ffi::cuLaunchKernel(
                    f,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel(saxpy)",
            )?
        };
        self.sync()
    }

    /// Q8_0 matrix-vector, one thread per output row.
    ///
    /// `w` is the packed weight exactly as it appears in the file. The
    /// activation must already be quantized to Q8_0 — that is what ggml does,
    /// and matching it is what makes the result comparable to the CPU path.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q8_0(
        &self,
        n_in: usize,
        n_out: usize,
        w: &DeviceBuffer,
        x_scales: &DeviceBuffer,
        x_quants: &DeviceBuffer,
        out: &DeviceBuffer,
    ) -> Result<()> {
        let f = self.function("matmul_q8_0")?;
        let mut n_in_arg = n_in as i32;
        let mut n_out_arg = n_out as i32;
        let mut w_arg = w.ptr;
        let mut xs_arg = x_scales.ptr;
        let mut xq_arg = x_quants.ptr;
        let mut out_arg = out.ptr;
        let mut params = [
            &mut n_in_arg as *mut _ as *mut c_void,
            &mut n_out_arg as *mut _ as *mut c_void,
            &mut w_arg as *mut _ as *mut c_void,
            &mut xs_arg as *mut _ as *mut c_void,
            &mut xq_arg as *mut _ as *mut c_void,
            &mut out_arg as *mut _ as *mut c_void,
        ];
        let block = 128u32;
        let grid = n_out.div_ceil(block as usize) as u32;
        // SAFETY: parameters match the kernel signature; buffer sizes are the
        // caller's contract, documented above.
        unsafe {
            check(
                ffi::cuLaunchKernel(
                    f,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel(matmul_q8_0)",
            )?
        };
        self.sync()
    }
}

impl Drop for Cuda {
    fn drop(&mut self) {
        // SAFETY: both handles were produced by the driver and are dropped
        // exactly once, module before context, as the driver requires.
        unsafe {
            ffi::cuModuleUnload(self.module);
            ffi::cuCtxDestroy_v2(self.context);
        }
    }
}

/// An owned allocation in device memory.
pub struct DeviceBuffer {
    ptr: ffi::CUdeviceptr,
    bytes: usize,
}

impl DeviceBuffer {
    pub fn new(bytes: usize) -> Result<Self> {
        // A zero-byte allocation is not an error to ask for, but the driver
        // dislikes it; hand back a null handle we will never dereference.
        if bytes == 0 {
            return Ok(Self { ptr: 0, bytes: 0 });
        }
        let mut ptr: ffi::CUdeviceptr = 0;
        // SAFETY: valid out-pointer, non-zero size.
        unsafe { check(ffi::cuMemAlloc_v2(&mut ptr, bytes), "cuMemAlloc")? };
        Ok(Self { ptr, bytes })
    }

    /// Allocate and fill from a host slice of plain data.
    pub fn from_slice<T: Copy>(data: &[T]) -> Result<Self> {
        let bytes = std::mem::size_of_val(data);
        let buf = Self::new(bytes)?;
        buf.write(data)?;
        Ok(buf)
    }

    pub fn len_bytes(&self) -> usize {
        self.bytes
    }

    /// Copy a host slice in. Fails rather than truncating if it does not fit.
    pub fn write<T: Copy>(&self, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        if bytes > self.bytes {
            return Err(Error::Cuda {
                what: "cuMemcpyHtoD",
                detail: format!("{bytes} bytes into a {} byte buffer", self.bytes),
            });
        }
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: `data` is valid for `bytes` and the device range was checked
        // above. `T: Copy` means there is nothing to drop or relocate.
        unsafe {
            check(
                ffi::cuMemcpyHtoD_v2(self.ptr, data.as_ptr() as *const c_void, bytes),
                "cuMemcpyHtoD",
            )
        }
    }

    /// Copy a host slice in at a byte offset, for appending to a buffer whose
    /// earlier contents are still wanted.
    pub fn write_at<T: Copy>(&self, offset_bytes: usize, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        if offset_bytes + bytes > self.bytes {
            return Err(Error::Cuda {
                what: "cuMemcpyHtoD",
                detail: format!(
                    "{bytes} bytes at offset {offset_bytes} into a {} byte buffer",
                    self.bytes
                ),
            });
        }
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: `data` is valid for `bytes` and the device range was checked
        // above. `T: Copy` means there is nothing to drop or relocate.
        unsafe {
            check(
                ffi::cuMemcpyHtoD_v2(
                    self.ptr + offset_bytes as u64,
                    data.as_ptr() as *const c_void,
                    bytes,
                ),
                "cuMemcpyHtoD",
            )
        }
    }

    /// Copy out into a host slice, which must not ask for more than was
    /// allocated.
    pub fn read<T: Copy>(&self, out: &mut [T]) -> Result<()> {
        let bytes = std::mem::size_of_val(out);
        if bytes > self.bytes {
            return Err(Error::Cuda {
                what: "cuMemcpyDtoH",
                detail: format!("{bytes} bytes from a {} byte buffer", self.bytes),
            });
        }
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: as above, in the other direction.
        unsafe {
            check(
                ffi::cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, self.ptr, bytes),
                "cuMemcpyDtoH",
            )
        }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: allocated by `cuMemAlloc_v2`, freed exactly once.
            unsafe {
                ffi::cuMemFree_v2(self.ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::naive::Naive;
    use crate::ops::{Ops, Weights};
    use crate::gguf::GgmlType;
    use crate::quant::half::{f16_to_f32, f32_to_f16};

    /// Print what we are actually talking to. Not an assertion so much as a
    /// record: if the capability is not 12.0, the PTX we built is wrong for
    /// this device and every other failure here is downstream of that.
    #[test]
    fn reports_the_device() {
        let cuda = Cuda::new(0).expect("no CUDA device");
        let (free, total) = cuda.mem_info().expect("mem info");
        println!(
            "{} | sm_{}{} | {} SMs | {:.2} GiB free of {:.2}",
            cuda.name(),
            cuda.capability().0,
            cuda.capability().1,
            cuda.sm_count(),
            free as f64 / (1024.0 * 1024.0 * 1024.0),
            total as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        assert_eq!(cuda.capability(), (12, 0), "built for sm_120");
    }

    /// Toolchain proof: build, load, allocate, copy up, launch, copy down.
    #[test]
    fn saxpy_round_trips() {
        let cuda = Cuda::new(0).expect("no CUDA device");
        let n = 10_000usize;
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let y: Vec<f32> = (0..n).map(|i| i as f32 * -0.25).collect();

        let dx = DeviceBuffer::from_slice(&x).expect("upload x");
        let dy = DeviceBuffer::from_slice(&y).expect("upload y");
        cuda.saxpy(3.0, &dx, &dy, n).expect("launch");

        let mut got = vec![0.0f32; n];
        dy.read(&mut got).expect("download");
        for i in 0..n {
            let want = 3.0 * x[i] + y[i];
            assert_eq!(got[i].to_bits(), want.to_bits(), "element {i}");
        }
    }

    fn q8_0_row(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for block in values.chunks(32) {
            let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d == 0.0 { 0.0 } else { 1.0 / d };
            bytes.extend_from_slice(&f32_to_f16(d).to_le_bytes());
            for &v in block {
                bytes.push(((v * id).round() as i8) as u8);
            }
        }
        bytes
    }

    /// **The result that matters.** One thread per row, walking the row in
    /// index order, is the same arithmetic in the same order as
    /// `naive::dot_q8_0_q8_0` — so the GPU's answer is compared to the CPU
    /// oracle's raw bits, not to a tolerance.
    ///
    /// This is only possible because the kernel is deliberately unoptimized and
    /// because `build.rs` passes `--fmad=false`. A block-wide reduction, or
    /// letting nvcc contract multiply-add, would break it. When we do optimize,
    /// this test is what tells us exactly what we gave up.
    #[test]
    fn q8_0_matmul_is_bit_identical_to_the_cpu_oracle() {
        let cuda = Cuda::new(0).expect("no CUDA device");

        for (n_in, n_out) in [(128usize, 512usize), (1024, 4096), (256, 3)] {
            let val = |i: usize| ((i as f32) * 0.7391).sin() * 2.0 + ((i as f32) * 0.113).cos();
            let weights: Vec<f32> = (0..n_in * n_out).map(val).collect();
            let mut packed = Vec::new();
            for row in 0..n_out {
                packed.extend(q8_0_row(&weights[row * n_in..(row + 1) * n_in]));
            }
            let x: Vec<f32> = (0..n_in).map(|i| val(i + 9871)).collect();

            // CPU reference, through the ops seam.
            let w = Weights { data: &packed, ty: GgmlType::Q8_0, n_in, n_out };
            let mut expect = vec![0.0f32; n_out];
            Naive.matmul(&w, &x, &mut expect);

            // Quantize the activation on the host, exactly as ggml does, so the
            // kernel receives the same integers the CPU path uses.
            let mut scales = Vec::new();
            let mut quants: Vec<i8> = Vec::new();
            for block in x.chunks(32) {
                let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                scales.push(f16_to_f32(f32_to_f16(d)));
                for &v in block {
                    quants.push((v * id).round() as i8);
                }
            }

            let dw = DeviceBuffer::from_slice(&packed).expect("upload w");
            let ds = DeviceBuffer::from_slice(&scales).expect("upload scales");
            let dq = DeviceBuffer::from_slice(&quants).expect("upload quants");
            let dout = DeviceBuffer::new(n_out * 4).expect("alloc out");
            cuda.matmul_q8_0(n_in, n_out, &dw, &ds, &dq, &dout).expect("launch");

            let mut got = vec![0.0f32; n_out];
            dout.read(&mut got).expect("download");

            let bad: Vec<usize> = (0..n_out)
                .filter(|&j| got[j].to_bits() != expect[j].to_bits())
                .collect();
            assert!(
                bad.is_empty(),
                "{}x{}: {} of {n_out} rows differ; first at {}: gpu {} vs cpu {}",
                n_in, n_out, bad.len(), bad[0], got[bad[0]], expect[bad[0]]
            );
        }
    }

    /// A buffer must refuse a copy that would not fit rather than truncating.
    #[test]
    fn oversized_copies_are_refused() {
        let _cuda = Cuda::new(0).expect("no CUDA device");
        let buf = DeviceBuffer::new(16).expect("alloc");
        assert!(buf.write(&[0u8; 32]).is_err());
        let mut out = [0u8; 32];
        assert!(buf.read(&mut out).is_err());
    }
}
