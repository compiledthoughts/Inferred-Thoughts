//! `Ops` on the GPU, one kernel per method.
//!
//! **This is the naive CUDA backend, in the same sense `ops::naive` is the
//! naive CPU one.** Every kernel reproduces the oracle's arithmetic in the
//! oracle's order, and the decomposition is chosen for that rather than for
//! occupancy — attention runs one thread per query head, and RMSNorm's f64
//! reduction runs on a single thread while its block waits. The point is a
//! forward pass that is *known correct on the device*, and a first honest
//! number, not a fast one.
//!
//! # What this backend does not do
//!
//! The [`Ops`] seam takes and returns host slices, so every method here is a
//! round trip: upload the activation, launch, synchronize, download the
//! result. A decode step runs roughly 196 matmuls plus norms and attention, so
//! that is ~250 round trips per token. Removing them means keeping activations
//! resident across a layer, which the seam cannot express — a design change,
//! not a tuning change, and the same one `CLAUDE.md` records as blocking
//! coarser CPU parallelism.
//!
//! Two things are cached anyway, because without them the measurement would be
//! meaningless rather than merely naive: weight matrices (uploaded once, keyed
//! on the mmap pointer) and the KV slabs (appended to, not resent).
//!
//! # Exactness
//!
//! Bit-identical to [`crate::ops::naive`]: `matmul`, `rms_norm`,
//! `rms_norm_heads`, `rope_neox`, `add_assign`. These are integer and f32
//! arithmetic in a fixed order, with `--fmad=false` keeping the compiler from
//! contracting a multiply-add.
//!
//! Not bit-identical: `softmax`, `silu_mul`, `attend`. All three call `expf`,
//! and CUDA's is not obliged to agree with glibc's to the last bit. The
//! accumulation order is still the oracle's; the difference is one library
//! function, and it is why the differential test for these takes a tolerance.

use std::ffi::c_void;

use super::{Cuda, DeviceBuffer, KvMirror, check, ffi};
use crate::error::{Error, Result};
use crate::gguf::GgmlType;
use crate::ops::naive::QuantizedRow;
use crate::ops::{Attn, Ops, Weights};

/// Scratch slots. Distinct within any one method, reused across methods.
mod slot {
    pub const X: usize = 0;
    pub const OUT: usize = 1;
    pub const AUX: usize = 2;
    pub const QSCALES: usize = 3;
    pub const QUANTS: usize = 4;
    pub const SCORES: usize = 5;
    pub const COS: usize = 6;
    pub const SIN: usize = 7;
    pub const Q: usize = 8;
}

fn arg<T>(v: &mut T) -> *mut c_void {
    v as *mut T as *mut c_void
}

impl Cuda {
    /// Host to device, counted.
    fn h2d<T: Copy>(&self, dst: ffi::CUdeviceptr, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        if bytes == 0 {
            return Ok(());
        }
        self.bump(|s| {
            s.h2d_calls += 1;
            s.h2d_bytes += bytes as u64;
        });
        // SAFETY: `data` is valid for `bytes`; `dst` was sized by `pooled` or
        // `resident`.
        unsafe {
            check(
                ffi::cuMemcpyHtoD_v2(dst, data.as_ptr() as *const c_void, bytes),
                "cuMemcpyHtoD",
            )
        }
    }

    /// Device to host, counted.
    fn d2h<T: Copy>(&self, out: &mut [T], src: ffi::CUdeviceptr) -> Result<()> {
        let bytes = std::mem::size_of_val(out);
        if bytes == 0 {
            return Ok(());
        }
        self.bump(|s| {
            s.d2h_calls += 1;
            s.d2h_bytes += bytes as u64;
        });
        // SAFETY: as above, in the other direction.
        unsafe {
            check(
                ffi::cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, src, bytes),
                "cuMemcpyDtoH",
            )
        }
    }

    /// Device copy of a host buffer that does not change, uploaded on first
    /// sight and kept for the life of the backend.
    ///
    /// Keyed on the host address. That is sound here because every caller is
    /// either the mmap (weights) or a `Vec` owned by the model, both of which
    /// outlive the backend, so an address is never recycled underneath us.
    fn resident<T: Copy>(&self, data: &[T]) -> Result<ffi::CUdeviceptr> {
        let key = data.as_ptr() as usize;
        let mut map = self.weights.borrow_mut();
        if let Some(b) = map.get(&key) {
            return Ok(b.ptr);
        }
        let buf = DeviceBuffer::from_slice(data)?;
        let ptr = buf.ptr;
        map.insert(key, buf);
        Ok(ptr)
    }

    /// Device mirror of one layer's K or V slab, brought up to `n_pos`.
    ///
    /// Uploads only the positions added since the last call. A shorter `n_pos`
    /// than last time means the cache was reset, so the mirror is refilled
    /// from position zero.
    fn kv_resident(&self, host: &[u16], n_pos: usize, kv_dim: usize) -> Result<ffi::CUdeviceptr> {
        let key = host.as_ptr() as usize;
        let mut map = self.kv.borrow_mut();
        if !map.contains_key(&key) {
            let buf = DeviceBuffer::new(std::mem::size_of_val(host))?;
            map.insert(key, KvMirror { buf, uploaded: 0 });
        }
        let m = match map.get_mut(&key) {
            Some(m) => m,
            None => {
                return Err(Error::Cuda {
                    what: "kv_resident",
                    detail: "mirror vanished between insert and lookup".to_string(),
                });
            }
        };
        if n_pos < m.uploaded {
            m.uploaded = 0;
        }
        if n_pos > m.uploaded {
            let (from, to) = (m.uploaded * kv_dim, n_pos * kv_dim);
            m.buf.write_at(from * 2, &host[from..to])?;
            m.uploaded = n_pos;
        }
        Ok(m.buf.ptr)
    }

    /// A scratch allocation of at least `bytes`, grown in place if needed.
    fn pooled(&self, slot: usize, bytes: usize) -> Result<ffi::CUdeviceptr> {
        let mut pool = self.pool.borrow_mut();
        while pool.len() <= slot {
            pool.push(DeviceBuffer::new(0)?);
        }
        if pool[slot].len_bytes() < bytes {
            pool[slot] = DeviceBuffer::new(bytes)?;
        }
        Ok(pool[slot].ptr)
    }

    /// Record a failure and keep going. See [`Cuda::take_error`].
    fn note(&self, r: Result<()>) {
        if let Err(e) = r {
            let mut slot = self.error.borrow_mut();
            if slot.is_none() {
                *slot = Some(e);
            }
        }
    }

    /// The first error any op hit, if any, clearing it.
    ///
    /// The `Ops` methods return `()`, so a driver failure has nowhere to go at
    /// the call site. Rather than panic — `CLAUDE.md` forbids `unwrap` outside
    /// tests, and unwinding out of an FFI-heavy path is worse — the first
    /// error is kept and the caller checks it once the run is over. Output
    /// after a failure is meaningless, which is why the CLI treats a present
    /// error as fatal rather than as a warning.
    pub fn take_error(&self) -> Option<Error> {
        self.error.borrow_mut().take()
    }

    // -------------------------------------------------------------- the ops

    fn rms_norm_impl(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) -> Result<()> {
        let w = self.resident(weight)?;
        let xd = self.pooled(slot::X, std::mem::size_of_val(x))?;
        let od = self.pooled(slot::OUT, std::mem::size_of_val(out))?;
        self.h2d(xd, x)?;

        let (mut n, mut eps) = (x.len() as i32, eps);
        let (mut xd, mut w, mut od) = (xd, w, od);
        let mut params = [
            arg(&mut n),
            arg(&mut xd),
            arg(&mut w),
            arg(&mut eps),
            arg(&mut od),
        ];
        // SAFETY: parameters match `rms_norm` in kernels.cu; every buffer was
        // sized from the slice it mirrors.
        unsafe { self.launch("rms_norm", 1, 256, &mut params)? };
        self.d2h(out, od)
    }

    fn rms_norm_heads_impl(
        &self,
        x: &mut [f32],
        weight: &[f32],
        head_dim: usize,
        eps: f32,
    ) -> Result<()> {
        let n_heads = x.len() / head_dim;
        let w = self.resident(weight)?;
        let xd = self.pooled(slot::X, std::mem::size_of_val(x))?;
        self.h2d(xd, x)?;

        let (mut hd, mut eps) = (head_dim as i32, eps);
        let (mut w, mut xd) = (w, xd);
        let mut params = [arg(&mut hd), arg(&mut w), arg(&mut eps), arg(&mut xd)];
        // SAFETY: as above; one block per head, which is the grid below.
        unsafe { self.launch("rms_norm_heads", n_heads as u32, 256, &mut params)? };
        self.d2h(x, xd)
    }

    fn matmul_impl(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) -> Result<()> {
        if w.ty != GgmlType::Q8_0 {
            return Err(Error::Cuda {
                what: "matmul",
                detail: format!(
                    "{:?} has no CUDA kernel; this backend implements Q8_0, which is \
                     every matmul in the models it targets",
                    w.ty
                ),
            });
        }

        // Quantized on the host, deliberately. `quantize_row_q8_0_ref` rounds a
        // scale through f16, and reproducing that rounding in CUDA is a risk
        // taken for no gain: it is one pass over a 1024-element activation, and
        // doing it here means the device sees byte-identical input to the
        // oracle.
        let qx = QuantizedRow::from_f32(x);
        let wd = self.resident(w.data)?;
        let sd = self.pooled(slot::QSCALES, std::mem::size_of_val(qx.scales()))?;
        let qd = self.pooled(slot::QUANTS, std::mem::size_of_val(qx.quants()))?;
        let od = self.pooled(slot::OUT, std::mem::size_of_val(out))?;
        self.h2d(sd, qx.scales())?;
        self.h2d(qd, qx.quants())?;

        let (mut n_in, mut n_out) = (w.n_in as i32, w.n_out as i32);
        let (mut wd, mut sd, mut qd, mut od) = (wd, sd, qd, od);
        let mut params = [
            arg(&mut n_in),
            arg(&mut n_out),
            arg(&mut wd),
            arg(&mut sd),
            arg(&mut qd),
            arg(&mut od),
        ];
        let block = 128u32;
        // SAFETY: parameters match `matmul_q8_0`; the weight buffer is the
        // whole tensor and the grid covers exactly `n_out` rows.
        unsafe {
            self.launch(
                "matmul_q8_0",
                w.n_out.div_ceil(block as usize) as u32,
                block,
                &mut params,
            )?
        };
        self.d2h(out, od)
    }

    fn rope_impl(
        &self,
        x: &mut [f32],
        pos: usize,
        head_dim: usize,
        n_heads: usize,
        theta_base: f32,
    ) -> Result<()> {
        // The table is built here, in f64, exactly as `ops::naive::rope_neox`
        // does. CUDA's double `pow` and `sincos` are not obliged to return
        // glibc's bits, and a one-ulp angle is a real output difference.
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(half);
        let mut sin = Vec::with_capacity(half);
        for i in 0..half {
            let freq = (theta_base as f64).powf(-2.0 * i as f64 / head_dim as f64);
            let (s, c) = (pos as f64 * freq).sin_cos();
            cos.push(c as f32);
            sin.push(s as f32);
        }

        let cd = self.pooled(slot::COS, half * 4)?;
        let sd = self.pooled(slot::SIN, half * 4)?;
        let xd = self.pooled(slot::X, std::mem::size_of_val(x))?;
        self.h2d(cd, &cos)?;
        self.h2d(sd, &sin)?;
        self.h2d(xd, x)?;

        let (mut hd, mut nh) = (head_dim as i32, n_heads as i32);
        let (mut cd, mut sd, mut xd) = (cd, sd, xd);
        let mut params = [
            arg(&mut hd),
            arg(&mut nh),
            arg(&mut cd),
            arg(&mut sd),
            arg(&mut xd),
        ];
        let (block, total) = (256u32, n_heads * half);
        // SAFETY: parameters match `rope_neox`; the grid covers exactly the
        // `n_heads * head_dim/2` rotation pairs.
        unsafe {
            self.launch(
                "rope_neox",
                total.div_ceil(block as usize) as u32,
                block,
                &mut params,
            )?
        };
        self.d2h(x, xd)
    }

    fn softmax_impl(&self, x: &mut [f32]) -> Result<()> {
        let xd = self.pooled(slot::X, std::mem::size_of_val(x))?;
        self.h2d(xd, x)?;

        let (mut n, mut rows) = (x.len() as i32, 1i32);
        let mut xd = xd;
        let mut params = [arg(&mut n), arg(&mut rows), arg(&mut xd)];
        // SAFETY: parameters match `softmax_rows`; one row, so one thread.
        unsafe { self.launch("softmax_rows", 1, 1, &mut params)? };
        self.d2h(x, xd)
    }

    fn attend_impl(&self, a: &Attn<'_>, out: &mut [f32]) -> Result<()> {
        let kd = self.kv_resident(a.k, a.n_pos, a.kv_dim)?;
        let vd = self.kv_resident(a.v, a.n_pos, a.kv_dim)?;
        let qd = self.pooled(slot::Q, std::mem::size_of_val(a.q))?;
        let sd = self.pooled(slot::SCORES, a.n_head * a.n_pos * 4)?;
        let od = self.pooled(slot::OUT, std::mem::size_of_val(out))?;
        self.h2d(qd, a.q)?;

        let mut n_pos = a.n_pos as i32;
        let mut kv_dim = a.kv_dim as i32;
        let mut head_dim = a.head_dim as i32;
        let mut n_head = a.n_head as i32;
        let mut n_head_kv = a.n_head_kv as i32;
        let mut scale = a.scale;
        let (mut qd, mut kd, mut vd, mut sd, mut od) = (qd, kd, vd, sd, od);
        let mut params = [
            arg(&mut n_pos),
            arg(&mut kv_dim),
            arg(&mut head_dim),
            arg(&mut n_head),
            arg(&mut n_head_kv),
            arg(&mut scale),
            arg(&mut qd),
            arg(&mut kd),
            arg(&mut vd),
            arg(&mut sd),
            arg(&mut od),
        ];
        let block = 32u32;
        // SAFETY: parameters match `attend`; the KV mirrors hold at least
        // `n_pos` positions and `scores` is `n_head * n_pos` floats.
        unsafe {
            self.launch(
                "attend",
                a.n_head.div_ceil(block as usize) as u32,
                block,
                &mut params,
            )?
        };
        self.d2h(out, od)
    }

    fn silu_mul_impl(&self, gate: &mut [f32], up: &[f32]) -> Result<()> {
        let gd = self.pooled(slot::X, std::mem::size_of_val(gate))?;
        let ud = self.pooled(slot::AUX, std::mem::size_of_val(up))?;
        self.h2d(gd, gate)?;
        self.h2d(ud, up)?;

        let mut n = gate.len() as i32;
        let (mut gd, mut ud) = (gd, ud);
        let mut params = [arg(&mut n), arg(&mut gd), arg(&mut ud)];
        let block = 256u32;
        // SAFETY: parameters match `silu_mul`; both buffers hold `n` floats.
        unsafe {
            self.launch(
                "silu_mul",
                gate.len().div_ceil(block as usize) as u32,
                block,
                &mut params,
            )?
        };
        self.d2h(gate, gd)
    }

    fn add_assign_impl(&self, a: &mut [f32], b: &[f32]) -> Result<()> {
        let ad = self.pooled(slot::X, std::mem::size_of_val(a))?;
        let bd = self.pooled(slot::AUX, std::mem::size_of_val(b))?;
        self.h2d(ad, a)?;
        self.h2d(bd, b)?;

        let mut n = a.len() as i32;
        let (mut ad, mut bd) = (ad, bd);
        let mut params = [arg(&mut n), arg(&mut ad), arg(&mut bd)];
        let block = 256u32;
        // SAFETY: parameters match `add_assign`; both buffers hold `n` floats.
        unsafe {
            self.launch(
                "add_assign",
                a.len().div_ceil(block as usize) as u32,
                block,
                &mut params,
            )?
        };
        self.d2h(a, ad)
    }
}

impl Ops for Cuda {
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        self.note(self.rms_norm_impl(x, weight, eps, out));
    }

    fn rms_norm_heads(&self, x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
        self.note(self.rms_norm_heads_impl(x, weight, head_dim, eps));
    }

    fn matmul(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) {
        self.note(self.matmul_impl(w, x, out));
    }

    fn rope_neox(&self, x: &mut [f32], pos: usize, head_dim: usize, n_heads: usize, theta: f32) {
        self.note(self.rope_impl(x, pos, head_dim, n_heads, theta));
    }

    fn softmax(&self, x: &mut [f32]) {
        self.note(self.softmax_impl(x));
    }

    fn attend(&self, a: &Attn<'_>, out: &mut [f32]) {
        self.note(self.attend_impl(a, out));
    }

    fn silu_mul(&self, gate: &mut [f32], up: &[f32]) {
        self.note(self.silu_mul_impl(gate, up));
    }

    fn add_assign(&self, a: &mut [f32], b: &[f32]) {
        self.note(self.add_assign_impl(a, b));
    }
}

/// So the caller can keep the device — and its sticky error — while the engine
/// borrows it. `Engine` takes its `Ops` by value, and a `&Cuda` is one.
impl Ops for &Cuda {
    fn rms_norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        (*self).rms_norm(x, weight, eps, out)
    }

    fn rms_norm_heads(&self, x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
        (*self).rms_norm_heads(x, weight, head_dim, eps)
    }

    fn matmul(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) {
        (*self).matmul(w, x, out)
    }

    fn rope_neox(&self, x: &mut [f32], pos: usize, head_dim: usize, n_heads: usize, theta: f32) {
        (*self).rope_neox(x, pos, head_dim, n_heads, theta)
    }

    fn softmax(&self, x: &mut [f32]) {
        (*self).softmax(x)
    }

    fn attend(&self, a: &Attn<'_>, out: &mut [f32]) {
        (*self).attend(a, out)
    }

    fn silu_mul(&self, gate: &mut [f32], up: &[f32]) {
        (*self).silu_mul(gate, up)
    }

    fn add_assign(&self, a: &mut [f32], b: &[f32]) {
        (*self).add_assign(a, b)
    }
}
