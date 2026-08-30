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

pub const CUDA_SUCCESS: CUresult = 0;

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
