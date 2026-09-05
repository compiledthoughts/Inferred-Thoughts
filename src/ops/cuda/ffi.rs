//! Raw bindings to the CUDA driver API.
//!
//! Hand-written rather than generated or wrapped, because `CLAUDE.md` puts the
//! CUDA path on our side of the borrowed/ours line, and because the sm_120
//! situation (`HANDOFF.md` §7) may need things a general-purpose wrapper does
//! not expose. This is the whole surface we use — about thirty symbols — so the
//! cost of owning it is small and the control is total.
//!
//! Note the `_v2` suffixes: several driver entry points were revised, and the
//! unsuffixed names are ABI-compatible shims that CUDA's headers `#define`
//! away. Linking wants the real symbol.
//!
//! # Safety
//!
//! Everything here is `unsafe` by nature. The safe wrappers live in
//! [`super`]; nothing outside this module should call these directly.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_uint, c_void};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
/// Device pointers are integers, not host pointers — 64-bit on every platform
/// we care about.
pub type CUdeviceptr = u64;

pub type CUevent = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;
pub type CUgraphNode = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;

/// `CUDA_KERNEL_NODE_PARAMS_v2` from `cuda.h`.
///
/// The v2 layout appeared in CUDA 12 and added `kern` and `ctx` on the end, so
/// the symbol below is the `_v2` one explicitly rather than the unversioned
/// alias — a mismatch between struct and entry point would be read as garbage
/// grid dimensions rather than as an error. Set `func` and leave `kern` and
/// `ctx` null; they are the alternative way of naming the same thing.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KernelNodeParams {
    pub func: CUfunction,
    pub grid_x: c_uint,
    pub grid_y: c_uint,
    pub grid_z: c_uint,
    pub block_x: c_uint,
    pub block_y: c_uint,
    pub block_z: c_uint,
    pub shared_bytes: c_uint,
    pub params: *mut *mut c_void,
    pub extra: *mut *mut c_void,
    pub kern: *mut c_void,
    pub ctx: CUcontext,
}

/// `CU_MEMHOSTALLOC_DEVICEMAP`, from `cuda.h`.
///
/// Makes a pinned host allocation addressable by kernels: the device pointer
/// from [`cuMemHostGetDevicePointer_v2`] can be dereferenced inside a kernel,
/// which reads it across PCIe rather than requiring a copy first. That is what
/// lets an expert live in host RAM without the host having to intervene when it
/// is used -- the precondition for a CUDA graph, which cannot service a miss.
pub const MEMHOSTALLOC_DEVICEMAP: c_uint = 0x02;

/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`, from `cuda.h`.
pub const ATTR_CC_MAJOR: c_int = 75;
/// `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`.
pub const ATTR_CC_MINOR: c_int = 76;
/// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`.
pub const ATTR_SM_COUNT: c_int = 16;

#[link(name = "cuda")]
unsafe extern "C" {
    pub fn cuInit(flags: c_uint) -> CUresult;
    pub fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGetAttribute(value: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;
    pub fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;

    pub fn cuCtxCreate_v2(ctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    pub fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;

    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleUnload(module: CUmodule) -> CUresult;
    pub fn cuModuleGetFunction(
        func: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult;

    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytes: usize) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult;
    pub fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult;
    pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    pub fn cuMemsetD8_v2(dst: CUdeviceptr, value: u8, n: usize) -> CUresult;

    /// Page-locked host memory, optionally mapped into the device's address
    /// space. `flags` takes [`MEMHOSTALLOC_DEVICEMAP`].
    pub fn cuMemHostAlloc(pp: *mut *mut c_void, bytes: usize, flags: c_uint) -> CUresult;
    pub fn cuMemFreeHost(p: *mut c_void) -> CUresult;
    /// The device-side address of a host allocation made with
    /// [`MEMHOSTALLOC_DEVICEMAP`]. `flags` is reserved and must be 0.
    pub fn cuMemHostGetDevicePointer_v2(
        dptr: *mut CUdeviceptr,
        p: *mut c_void,
        flags: c_uint,
    ) -> CUresult;

    pub fn cuLaunchKernel(
        f: CUfunction,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_bytes: c_uint,
        stream: CUstream,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;

    pub fn cuEventCreate(event: *mut CUevent, flags: c_uint) -> CUresult;
    pub fn cuEventDestroy_v2(event: CUevent) -> CUresult;
    pub fn cuEventRecord(event: CUevent, stream: CUstream) -> CUresult;
    pub fn cuEventSynchronize(event: CUevent) -> CUresult;
    pub fn cuEventElapsedTime(ms: *mut f32, start: CUevent, end: CUevent) -> CUresult;

    pub fn cuGraphCreate(graph: *mut CUgraph, flags: c_uint) -> CUresult;
    pub fn cuGraphDestroy(graph: CUgraph) -> CUresult;
    pub fn cuGraphAddKernelNode_v2(
        node: *mut CUgraphNode,
        graph: CUgraph,
        deps: *const CUgraphNode,
        n_deps: usize,
        params: *const KernelNodeParams,
    ) -> CUresult;
    pub fn cuGraphInstantiateWithFlags(
        exec: *mut CUgraphExec,
        graph: CUgraph,
        flags: u64,
    ) -> CUresult;
    pub fn cuGraphExecDestroy(exec: CUgraphExec) -> CUresult;
    pub fn cuGraphLaunch(exec: CUgraphExec, stream: CUstream) -> CUresult;
    pub fn cuGraphExecKernelNodeSetParams_v2(
        exec: CUgraphExec,
        node: CUgraphNode,
        params: *const KernelNodeParams,
    ) -> CUresult;

    pub fn cuGetErrorName(error: CUresult, str_: *mut *const c_char) -> CUresult;
    pub fn cuGetErrorString(error: CUresult, str_: *mut *const c_char) -> CUresult;
}

/// Turn a `CUresult` into the driver's own name and message.
///
/// Errors from a foreign API are worth reporting in that API's vocabulary —
/// `CUDA_ERROR_NO_BINARY_FOR_GPU` says something specific about an sm_120
/// mismatch that "cuda error 209" does not.
pub fn describe(code: CUresult) -> String {
    let mut name: *const c_char = std::ptr::null();
    let mut msg: *const c_char = std::ptr::null();
    // SAFETY: both calls only write a pointer to a static string owned by the
    // driver, and we only read it when the call reports success.
    unsafe {
        let have_name = cuGetErrorName(code, &mut name) == CUDA_SUCCESS && !name.is_null();
        let have_msg = cuGetErrorString(code, &mut msg) == CUDA_SUCCESS && !msg.is_null();
        let text = |p: *const c_char| {
            std::ffi::CStr::from_ptr(p)
                .to_string_lossy()
                .into_owned()
        };
        match (have_name, have_msg) {
            (true, true) => format!("{} ({})", text(name), text(msg)),
            (true, false) => text(name),
            (false, true) => text(msg),
            _ => format!("unknown CUDA error {code}"),
        }
    }
}
