//! `Ops` on the GPU, one kernel per method.
//!
//! Activations stay resident on the device across a layer; see `mirror_in`.
//! The seam's `host_wrote` / `host_needs` / `begin_pass` are what make that
//! safe, and they are no-ops for the CPU backends.
//!
//! # Exactness
//!
//! Bit-identical to [`crate::ops::naive`]: `matmul`, `rope_neox`,
//! `add_assign`, and `rms_norm`/`rms_norm_heads` **under `--rms-serial`**.
//! These are integer and f32
//! arithmetic in a fixed order, with `--fmad=false` keeping the compiler from
//! contracting a multiply-add. `matmul` earns this despite being a fast
//! warp-per-row kernel, because Q8_0's 32-element block is an integer sum and
//! therefore order-free; only the accumulation across blocks is f32, and that
//! is kept serial and ascending.
//!
//! Not bit-identical, for two different reasons:
//!
//! * `softmax` and `silu_mul` call `expf`, which CUDA is not obliged to round
//!   as glibc does. Order is unchanged; the gap is one ulp.
//! * `rms_norm` and `rms_norm_heads` reduce the sum of squares as a tree by
//!   default. f64 addition rounds and so is not associative; the tolerance is
//!   `n * 2^-53`, derived rather than fitted. `--rms-serial` restores the
//!   serial walk and with it bit-equality, at ~2.3 ms a token.
//! * `attend` is flash-decoding — it accumulates per chunk of the KV sequence
//!   and combines — so it genuinely reorders. That was a deliberate trade: the
//!   previous order-preserving version parallelized only over `n_head`, so its
//!   cost grew with context while its parallelism did not. Its tolerance is
//!   derived from the decomposition in `tests/cuda_ops.rs`.

use std::ffi::c_void;

use super::{Cuda, DeviceBuffer, KArg, KvMirror, Mirror, check, ffi};
use crate::error::{Error, Result};
use crate::gguf::GgmlType;
/// Tokens one warp of `matmul_q8_0_batch` holds in registers while it loads a
/// weight row once. **Must equal `MM_TOK` in `kernels/kernels.cu`** — it sizes
/// both the grid and the shared-memory request, and a mismatch would read past
/// the partials.
///
/// **Four is measured, and it is a trade-off rather than a maximum.** Prefill
/// tok/s on an 841-token prompt:
///
/// | `MM_TOK` | 1 | 2 | 4 | 8 | 16 |
/// |---|---|---|---|---|---|
/// | Qwen3.5-9B | 52.6 | 83.5 | **105.2** | 103.6 | 85.4 |
/// | Qwen3-0.6B | | | 646 | 649 | 695 |
///
/// 1 -> 4 is the reuse arriving, and it is the whole batched-prefill win on the
/// GPU. Past 4 the 9B *regresses*, because its widest tensor (`ffn_down`,
/// n_in 12288) needs `warps * MM_TOK * 384` floats of shared memory, so 8
/// forces the block down to two warps and 16 to one. Occupancy lost exceeds
/// reuse gained. Four is the largest tile that still keeps four warps there.
///
/// Note the 0.6B keeps improving to 16, because its widest tensor is a quarter
/// as wide and never loses warps. The constant is set for the model that
/// matters, which is the larger one — the same reasoning `CLAUDE.md` applies to
/// judging the GPU on the 0.6B at all.
const MM_TOK: usize = 4;
use crate::ops::{Attn, Delta, Ops, Weights};

/// Scratch slots. Distinct within any one method, reused across methods.
mod slot {
    pub const X: usize = 0;
    pub const OUT: usize = 1;
    pub const AUX: usize = 2;
    pub const QSCALES: usize = 3;
    pub const QUANTS: usize = 4;
    pub const SCORES: usize = 5;
    pub const PART_M: usize = 9;
    pub const PART_L: usize = 10;
    pub const COS: usize = 6;
    pub const SIN: usize = 7;
    pub const Q: usize = 8;
}

impl Cuda {
    /// The device copy of a model-owned activation, brought up to date.
    ///
    /// This is the whole residency mechanism. A mirror is keyed on the host
    /// address and carries one bit: whether the device copy is at least as
    /// fresh as the host one. An op that *writes* a buffer sets that bit and
    /// skips the download; the next op to read the same buffer then finds it
    /// already there and skips the upload. A chain of ops inside a layer
    /// therefore touches the bus once, at the ends, instead of twice per op.
    ///
    /// The bit is cleared by `host_wrote` and by `begin_pass`, which are the
    /// only two ways the host can get ahead of the device.
    fn mirror_in(&self, data: &[f32]) -> Result<ffi::CUdeviceptr> {
        let key = data.as_ptr() as usize;
        let bytes = std::mem::size_of_val(data);
        let (ptr, fresh) = self.slot_for(key, bytes)?;
        if !fresh {
            self.h2d(ptr, data)?;
            if let Some(m) = self.mirrors.borrow_mut().get_mut(&key) {
                m.device_current = true;
            }
        }
        Ok(ptr)
    }

    /// The device buffer an op is about to *write*, without uploading first.
    ///
    /// Marked current on the way out, so the download never happens unless the
    /// model asks for it through `host_needs`.
    fn mirror_out(&self, data: &[f32]) -> Result<ffi::CUdeviceptr> {
        let key = data.as_ptr() as usize;
        let bytes = std::mem::size_of_val(data);
        let (ptr, _) = self.slot_for(key, bytes)?;
        if let Some(m) = self.mirrors.borrow_mut().get_mut(&key) {
            m.device_current = true;
            // The contents are about to change, so a quantization of them is
            // stale — but the buffer holding it is kept.
            m.quant_valid = false;
        }
        Ok(ptr)
    }

    /// The mirror for `key`, allocating only if there is not already one big
    /// enough. Returns its pointer and whether the device copy is current.
    fn slot_for(&self, key: usize, bytes: usize) -> Result<(ffi::CUdeviceptr, bool)> {
        let mut map = self.mirrors.borrow_mut();
        let reusable = match map.get(&key) {
            Some(m) => m.buf.len_bytes() >= bytes,
            None => false,
        };
        if !reusable {
            map.insert(
                key,
                Mirror {
                    buf: DeviceBuffer::new(bytes)?,
                    device_current: false,
                    quant: None,
                    quant_valid: false,
                },
            );
        }
        match map.get(&key) {
            Some(m) => Ok((m.buf.ptr, m.device_current)),
            None => Err(Error::Cuda {
                what: "slot_for",
                detail: "mirror vanished between insert and lookup".to_string(),
            }),
        }
    }

    /// This activation quantized to Q8_0 on the device, computed once.
    ///
    /// Returns `(scales, quants)` device pointers. The result is cached on the
    /// buffer's mirror and dropped the moment anything writes that buffer, so
    /// the five matmuls a layer runs against two distinct activations do two
    /// quantizations rather than five.
    fn quantized(
        &self,
        x: &[f32],
        n_blocks: usize,
    ) -> Result<(ffi::CUdeviceptr, ffi::CUdeviceptr)> {
        let key = x.as_ptr() as usize;
        let xd = self.mirror_in(x)?;

        let existing = match self.mirrors.borrow().get(&key) {
            Some(m) => match &m.quant {
                Some((s, q)) if s.len_bytes() >= n_blocks * 4 => {
                    if m.quant_valid {
                        return Ok((s.ptr, q.ptr));
                    }
                    Some((s.ptr, q.ptr))
                }
                _ => None,
            },
            None => None,
        };

        let (sd, qd, fresh_bufs) = match existing {
            // Buffers already the right size; only the contents are stale.
            Some((s, q)) => (s, q, None),
            None => {
                let scales = DeviceBuffer::new(n_blocks * 4)?;
                let quants = DeviceBuffer::new(n_blocks * 32)?;
                let (s, q) = (scales.ptr, quants.ptr);
                (s, q, Some((scales, quants)))
            }
        };
        {
            let args = [
                KArg::I32(n_blocks as i32),
                KArg::Ptr(xd),
                KArg::Ptr(sd),
                KArg::Ptr(qd),
            ];
            let block = 64u32;
            // SAFETY: parameters match `quantize_q8_0`; the grid covers exactly
            // `n_blocks` blocks and both outputs are sized for them.
            unsafe {
                self.launch(
                    "quantize_q8_0",
                    n_blocks.div_ceil(block as usize) as u32,
                    block,
                    &args,
                )?
            };
        }
        if let Some(m) = self.mirrors.borrow_mut().get_mut(&key) {
            if let Some(bufs) = fresh_bufs {
                m.quant = Some(bufs);
            }
            m.quant_valid = true;
        }
        Ok((sd, qd))
    }

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
        // A device-to-host copy cannot start until the work before it has
        // finished, so this is where the host blocks and it is timed as such.
        let started = std::time::Instant::now();
        // SAFETY: as above, in the other direction.
        let r = unsafe {
            check(
                ffi::cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, src, bytes),
                "cuMemcpyDtoH",
            )
        };
        let waited = started.elapsed().as_nanos() as u64;
        self.bump(|s| {
            s.d2h_calls += 1;
            s.d2h_bytes += bytes as u64;
            s.syncs += 1;
            s.wait_ns += waited;
        });
        r
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
            map.insert(
                key,
                KvMirror {
                    buf,
                    uploaded: 0,
                    device_written: false,
                },
            );
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
        // Once this backend has written the slab, the host copy is stale and
        // uploading it would undo the work.
        if m.device_written {
            return Ok(m.buf.ptr);
        }
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
        // The row length is the weight length, so a batch is self-describing.
        let n = weight.len();
        let rows = x.len() / n;
        let w = self.resident(weight)?;
        let xd = self.mirror_in(x)?;
        let od = self.mirror_out(out)?;

        let args = [
            KArg::I32(n as i32),
            KArg::Ptr(xd),
            KArg::Ptr(w),
            KArg::F32(eps),
            KArg::Ptr(od),
        ];
        // The serial kernel stages one square per element in shared memory so
        // its f64 walk reads shared rather than global — worth 18%, and worth
        // nothing to the tree, which has no serial walk to feed.
        let (kernel, _) = self.rms_kernels();
        let shared = if kernel == "rms_norm" { (n * 4) as u32 } else { 0 };
        // SAFETY: parameters match both `rms_norm` and `rms_norm_tree` in
        // kernels.cu, which share a signature; one block per row of the batch,
        // every buffer was sized from the slice it mirrors, and `shared` is `n`
        // floats or none.
        unsafe { self.launch_shared(kernel, rows as u32, 256, shared, &args)? };
        Ok(())
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
        let xd = self.mirror_in(x)?;

        let args = [
            KArg::I32(head_dim as i32),
            KArg::Ptr(w),
            KArg::F32(eps),
            KArg::Ptr(xd),
        ];
        let (_, kernel) = self.rms_kernels();
        let shared = if kernel == "rms_norm_heads" { (head_dim * 4) as u32 } else { 0 };
        // SAFETY: as above; one block per head, and `shared` is `head_dim`
        // floats or none.
        unsafe { self.launch_shared(kernel, n_heads as u32, 256, shared, &args)? };
        self.mirror_out(x).map(|_| ())
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

        // Quantized on the device. It began on the host, because the scale
        // round-trips through f16 and a differing rounding mode there would be
        // invisible until it moved a quant; `tests/cuda_ops.rs` now checks that
        // against the oracle on real data. Doing it here is what lets a
        // matmul's input stay on the card instead of coming back to be
        // quantized and going out again.
        let n_blocks = w.n_in / 32;
        let n_tok = x.len() / w.n_in;
        let (ws, wq) = self.resident_q8_0(w)?;
        // One thread per 32-element block over the whole batch. The quantize
        // kernel needs no batch awareness: `x` is token-major and contiguous,
        // so its blocks already lay out as `[token][block]`, which is exactly
        // what the matmul indexes.
        let (sd, qd) = self.quantized(x, n_tok * n_blocks)?;
        let od = self.mirror_out(out)?;

        // 128 threads is four warps, so four output rows per block.
        let block = 128u32;
        let rows_per_block = (block / 32) as usize;
        let grid_rows = w.n_out.div_ceil(rows_per_block) as u32;

        // The batched kernel holds `MM_TOK` partial sums per warp per block of
        // the row, so its shared-memory request grows with both `MM_TOK` and
        // `n_in`. Rather than cap `MM_TOK` at whatever the widest tensor
        // allows, narrow the *block* until the request fits: the 9B's
        // `ffn_down` has n_in 12288, so one warp alone wants 24.5 KB at
        // MM_TOK 8, while the 0.6B's widest wants 3 KB and keeps four warps.
        let per_warp = MM_TOK * n_blocks * 4;
        let b_warps = (49152 / per_warp.max(1)).clamp(1, 4);
        let b_block = (b_warps * 32) as u32;
        let b_grid = w.n_out.div_ceil(b_warps) as u32;
        let b_shared = (b_warps * per_warp) as u32;

        if n_tok == 1 {
            let args = [
                KArg::I32(w.n_in as i32),
                KArg::I32(w.n_out as i32),
                KArg::Ptr(ws),
                KArg::Ptr(wq),
                KArg::Ptr(sd),
                KArg::Ptr(qd),
                KArg::Ptr(od),
            ];
            let shared = (rows_per_block * n_blocks * 4) as u32;
            // SAFETY: parameters match `matmul_q8_0_warp`; the grid covers
            // exactly `n_out` rows and `shared` is `warps * n_blocks` floats,
            // which is what the kernel indexes.
            unsafe { self.launch_shared("matmul_q8_0_warp", grid_rows, block, shared, &args)? };
            return Ok(());
        }

        // **Decode keeps the kernel it was measured on.** Every published
        // number here was taken on `matmul_q8_0_warp`, and it is the kernel the
        // decode CUDA graph records; the batched one is strictly for prefill,
        // where a warp amortizes one weight load across `MM_TOK` tokens.
        let args = [
            KArg::I32(w.n_in as i32),
            KArg::I32(w.n_out as i32),
            KArg::I32(n_tok as i32),
            KArg::Ptr(ws),
            KArg::Ptr(wq),
            KArg::Ptr(sd),
            KArg::Ptr(qd),
            KArg::Ptr(od),
        ];
        // SAFETY: parameters match `matmul_q8_0_batch`; the grid covers exactly
        // `n_out` rows by `ceil(n_tok / MM_TOK)` token tiles, and `b_shared` is
        // `warps * MM_TOK * n_blocks` floats, which is what the kernel indexes.
        unsafe {
            self.launch_grid2(
                "matmul_q8_0_batch",
                b_grid,
                n_tok.div_ceil(MM_TOK) as u32,
                b_block,
                b_shared,
                &args,
            )?
        };
        Ok(())
    }

    fn rope_impl(
        &self,
        x: &mut [f32],
        pos: usize,
        head_dim: usize,
        n_rot: usize,
        n_heads: usize,
        theta_base: f32,
    ) -> Result<()> {
        // The table is built here, in f64, exactly as `ops::naive::rope_neox`
        // does. CUDA's double `pow` and `sincos` are not obliged to return
        // glibc's bits, and a one-ulp angle is a real output difference.
        let half = n_rot / 2;
        // `pos` is row 0's position and rows are consecutive, so a batch needs
        // `n_tok` stacked tables. Decode sends one, exactly as before.
        let n_tok = x.len() / (head_dim * n_heads);
        let cd = self.pooled(slot::COS, n_tok * half * 4)?;
        let sd = self.pooled(slot::SIN, n_tok * half * 4)?;

        // Same table for every layer of a token, so build and send it once.
        // Keyed on `n_rot` rather than `head_dim`, since that is what sets the
        // frequencies -- q and k share it, but a model mixing rotation widths
        // would not. `n_tok` joins the key because the table's *length* changes
        // with it, so a decode step after a prefill must not reuse the prefill's.
        let key = (pos, n_rot, theta_base.to_bits(), n_tok);
        if self.rope_pos.get() != Some(key) {
            let mut cos = Vec::with_capacity(n_tok * half);
            let mut sin = Vec::with_capacity(n_tok * half);
            for t in 0..n_tok {
                for i in 0..half {
                    let freq = (theta_base as f64).powf(-2.0 * i as f64 / n_rot as f64);
                    let (s, c) = ((pos + t) as f64 * freq).sin_cos();
                    cos.push(c as f32);
                    sin.push(s as f32);
                }
            }
            self.h2d(cd, &cos)?;
            self.h2d(sd, &sin)?;
            self.rope_pos.set(Some(key));
        }
        let xd = self.mirror_in(x)?;

        let args = [
            KArg::I32(head_dim as i32),
            KArg::I32(n_rot as i32),
            KArg::I32(n_heads as i32),
            KArg::I32(n_tok as i32),
            KArg::Ptr(cd),
            KArg::Ptr(sd),
            KArg::Ptr(xd),
        ];
        let (block, total) = (256u32, n_tok * n_heads * half);
        // SAFETY: parameters match `rope_neox`; the grid covers exactly the
        // `n_heads * head_dim/2` rotation pairs.
        unsafe {
            self.launch(
                "rope_neox",
                total.div_ceil(block as usize) as u32,
                block,
                &args,
            )?
        };
        self.mirror_out(x).map(|_| ())
    }

    fn softmax_impl(&self, x: &mut [f32]) -> Result<()> {
        let xd = self.mirror_in(x)?;

        let args = [KArg::I32(x.len() as i32), KArg::I32(1), KArg::Ptr(xd)];
        // SAFETY: parameters match `softmax_rows`; one row, so one thread.
        unsafe { self.launch("softmax_rows", 1, 1, &args)? };
        self.mirror_out(x).map(|_| ())
    }

    fn attend_impl(&self, a: &Attn<'_>, out: &mut [f32]) -> Result<()> {
        const CHUNK: usize = 128;
        // Widest window in the batch, which is the last row's; it sizes the
        // partial buffers for every row.
        let n_split = a.n_pos.div_ceil(CHUNK);

        let kd = self.kv_resident(a.k, a.n_pos, a.kv_dim)?;
        let vd = self.kv_resident(a.v, a.n_pos, a.kv_dim)?;
        let qd = self.mirror_in(a.q)?;
        let od = self.mirror_out(out)?;

        // Per-chunk partials: an output vector, plus the max and sum that let
        // chunks be combined without ever materializing the scores.
        let pa = self.pooled(slot::SCORES, a.n_head * n_split * a.head_dim * 4)?;
        let pm = self.pooled(slot::PART_M, a.n_head * n_split * 4)?;
        let pl = self.pooled(slot::PART_L, a.n_head * n_split * 4)?;

        // **Query rows are launched one at a time, by device pointer offset.**
        // Every row has a different causal window, so they cannot share a grid
        // without masking most of it away; and the offsets are applied to the
        // device pointers rather than by sub-slicing `a.q` on the host, because
        // this backend keys its mirrors on host addresses -- a sub-slice would
        // look like an unmirrored buffer and be uploaded from a stale host copy,
        // which is the bug `Ops::rope_neox` already carries a note about.
        let (n_q, per_row) = (a.n_q(), a.n_head * a.head_dim);
        for t in 0..n_q {
            let n_pos = a.n_pos_of(t);
            let qd = qd + (t * per_row * 4) as u64;
            let od = od + (t * per_row * 4) as u64;
            self.attend_row(a, n_pos, CHUNK, qd, kd, vd, od, pa, pm, pl)?;
        }
        Ok(())
    }

    /// One query row against `n_pos` cached positions — the flash-decoding pair.
    #[allow(clippy::too_many_arguments)]
    fn attend_row(
        &self,
        a: &Attn<'_>,
        n_pos: usize,
        chunk: usize,
        qd: ffi::CUdeviceptr,
        kd: ffi::CUdeviceptr,
        vd: ffi::CUdeviceptr,
        od: ffi::CUdeviceptr,
        pa: ffi::CUdeviceptr,
        pm: ffi::CUdeviceptr,
        pl: ffi::CUdeviceptr,
    ) -> Result<()> {
        let n_split = n_pos.div_ceil(chunk);
        {
            let args = [
                KArg::I32(n_pos as i32),
                KArg::I32(a.kv_dim as i32),
                KArg::I32(a.head_dim as i32),
                KArg::I32(a.n_head as i32),
                KArg::I32(a.n_head_kv as i32),
                KArg::F32(a.scale),
                KArg::Ptr(qd),
                KArg::Ptr(kd),
                KArg::Ptr(vd),
                KArg::Ptr(pa),
                KArg::Ptr(pm),
                KArg::Ptr(pl),
            ];
            let shared = ((a.head_dim + 2 * chunk) * 4) as u32;
            // SAFETY: parameters match `attn_flash`; the grid is one block per
            // (query head, chunk) so no block sees an empty range, and `shared`
            // is head_dim + 2 * FD_CHUNK floats, which is what it indexes.
            unsafe {
                self.launch_grid2(
                    "attn_flash",
                    a.n_head as u32,
                    n_split as u32,
                    chunk as u32,
                    shared,
                    &args,
                )?
            };
        }

        {
            let args = [
                KArg::I32(n_split as i32),
                KArg::I32(a.head_dim as i32),
                KArg::Ptr(pa),
                KArg::Ptr(pm),
                KArg::Ptr(pl),
                KArg::Ptr(od),
            ];
            let shared = (n_split * 4) as u32;
            // SAFETY: parameters match `attn_flash_combine`; one block per
            // query head, and `shared` is `n_split` floats.
            unsafe {
                self.launch_shared(
                    "attn_flash_combine",
                    a.n_head as u32,
                    chunk as u32,
                    shared,
                    &args,
                )?
            };
        }
        Ok(())
    }

    /// Convert and store K or V without either ever leaving the card.
    /// The repacked device copy of a Q8_0 tensor, built once on first use.
    ///
    /// Returns `(scales, quants)`. The split happens on the host in chunks of
    /// whole rows so the temporary never approaches the tensor size -- the 9B's
    /// LM head alone is over a gigabyte, and holding a second copy of it would
    /// undo the point of mapping the file rather than reading it.
    fn resident_q8_0(&self, w: &Weights<'_>) -> Result<(ffi::CUdeviceptr, ffi::CUdeviceptr)> {
        let key = w.data.as_ptr() as usize;
        if let Some((s, q)) = self.q8.borrow().get(&key) {
            return Ok((s.ptr, q.ptr));
        }

        let n_blocks = w.n_in / 32;
        let scales = DeviceBuffer::new(w.n_out * n_blocks * 2)?;
        let quants = DeviceBuffer::new(w.n_out * w.n_in)?;

        // ~8 MiB of source per chunk, at least one row.
        let row_bytes = n_blocks * 34;
        let rows_per_chunk = (8usize << 20).div_ceil(row_bytes.max(1)).max(1);

        let mut sbuf: Vec<u16> = Vec::with_capacity(rows_per_chunk * n_blocks);
        let mut qbuf: Vec<i8> = Vec::with_capacity(rows_per_chunk * w.n_in);
        let mut row = 0usize;
        while row < w.n_out {
            let rows = rows_per_chunk.min(w.n_out - row);
            sbuf.clear();
            qbuf.clear();
            for r in 0..rows {
                let base = (row + r) * row_bytes;
                for b in 0..n_blocks {
                    let at = base + b * 34;
                    sbuf.push(u16::from_le_bytes([w.data[at], w.data[at + 1]]));
                    // `i8 as u8` is a bit-preserving reinterpretation, which is
                    // what the kernel reads back.
                    qbuf.extend(w.data[at + 2..at + 34].iter().map(|&v| v as i8));
                }
            }
            scales.write_at(row * n_blocks * 2, &sbuf)?;
            quants.write_at(row * w.n_in, &qbuf)?;
            row += rows;
        }

        let ptrs = (scales.ptr, quants.ptr);
        self.q8.borrow_mut().insert(key, (scales, quants));
        Ok(ptrs)
    }

    /// Time a matmul variant at a real weight shape.
    ///
    /// Weights are synthetic -- the kernels are bandwidth and instruction bound
    /// and do not branch on values -- but the *shapes* are the 9B's, because
    /// the serial tail at the end of the kernel grows with `n_in` and the 0.6B
    /// would not show it.
    pub fn bench_matmul(
        &self,
        name: &'static str,
        n_in: usize,
        n_out: usize,
        reps: u32,
    ) -> Result<f64> {
        let n_blocks = n_in / 32;
        let wbytes = n_out * n_blocks * 34;
        let w: Vec<u8> = (0..wbytes).map(|i| (i % 251) as u8).collect();
        let scales: Vec<f32> = (0..n_blocks).map(|i| 0.01 + (i % 7) as f32 * 0.001).collect();
        let quants: Vec<i8> = (0..n_in).map(|i| ((i % 251) as i32 - 125) as i8).collect();

        let wd = DeviceBuffer::from_slice(&w)?;
        let sd = DeviceBuffer::from_slice(&scales)?;
        let qd = DeviceBuffer::from_slice(&quants)?;
        let od = DeviceBuffer::new(n_out * 4)?;

        let args = [
            KArg::I32(n_in as i32),
            KArg::I32(n_out as i32),
            KArg::Ptr(wd.ptr),
            KArg::Ptr(sd.ptr),
            KArg::Ptr(qd.ptr),
            KArg::Ptr(od.ptr),
        ];
        let block = 128u32;
        let warps = (block / 32) as usize;
        // `tree` keeps its partials in registers and asks for none.
        let shared = if name.ends_with("tree") {
            0
        } else {
            (warps * n_blocks * 4) as u32
        };
        let grid = n_out.div_ceil(warps) as u32;

        let was = self.pass_graph.replace(false);
        let run = |reps: u32| -> Result<f64> {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                // SAFETY: the argument list matches every kernel in this
                // diagnostic set, and `shared` is what each indexes.
                unsafe { self.launch_shared(name, grid, block, shared, &args)? };
            }
            self.sync()?;
            Ok(t.elapsed().as_secs_f64() * 1e6 / f64::from(reps))
        };
        run(8)?;
        let us = run(reps);
        self.pass_graph.set(was);
        us
    }

    /// Time a matmul variant that reads a *repacked* weight layout.
    ///
    /// Same shapes and same synthetic data as [`Cuda::bench_matmul`], but the
    /// tensor is split into an aligned f16 scale array and an aligned int8
    /// quant array, which is what makes `int4` loads and `__dp4a` legal.
    pub fn bench_matmul_packed(
        &self,
        name: &'static str,
        n_in: usize,
        n_out: usize,
        reps: u32,
    ) -> Result<f64> {
        let n_blocks = n_in / 32;
        let wscales: Vec<u16> = (0..n_out * n_blocks).map(|i| (0x3800 + (i % 64)) as u16).collect();
        let wquants: Vec<i8> = (0..n_out * n_in).map(|i| ((i % 251) as i32 - 125) as i8).collect();
        let xscales: Vec<f32> = (0..n_blocks).map(|i| 0.01 + (i % 7) as f32 * 0.001).collect();
        let xquants: Vec<i8> = (0..n_in).map(|i| ((i % 251) as i32 - 125) as i8).collect();

        let ws = DeviceBuffer::from_slice(&wscales)?;
        let wq = DeviceBuffer::from_slice(&wquants)?;
        let xs = DeviceBuffer::from_slice(&xscales)?;
        let xq = DeviceBuffer::from_slice(&xquants)?;
        let od = DeviceBuffer::new(n_out * 4)?;

        let args = [
            KArg::I32(n_in as i32),
            KArg::I32(n_out as i32),
            KArg::Ptr(ws.ptr),
            KArg::Ptr(wq.ptr),
            KArg::Ptr(xs.ptr),
            KArg::Ptr(xq.ptr),
            KArg::Ptr(od.ptr),
        ];
        let block = 128u32;
        let warps = (block / 32) as usize;
        let shared = if name.ends_with("tree") { 0 } else { (warps * n_blocks * 4) as u32 };
        let grid = n_out.div_ceil(warps) as u32;

        let was = self.pass_graph.replace(false);
        let run = |reps: u32| -> Result<f64> {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                // SAFETY: the argument list matches both packed variants.
                unsafe { self.launch_shared(name, grid, block, shared, &args)? };
            }
            self.sync()?;
            Ok(t.elapsed().as_secs_f64() * 1e6 / f64::from(reps))
        };
        run(8)?;
        let us = run(reps);
        self.pass_graph.set(was);
        us
    }

    /// Device-resident recurrent state for one layer, uploaded once.
    ///
    /// Unlike `resident`, the device copy is *written* by kernels, so after the
    /// first touch it is the authoritative one and the host slab is stale by
    /// design -- exactly as the KV slabs are. `forget_state` is what makes a
    /// sequence reset visible.
    fn state_resident(&self, host: &[f32]) -> Result<ffi::CUdeviceptr> {
        let key = host.as_ptr() as usize;
        if let Some(b) = self.states.borrow().get(&key) {
            return Ok(b.ptr);
        }
        let buf = DeviceBuffer::from_slice(host)?;
        let ptr = buf.ptr;
        self.states.borrow_mut().insert(key, buf);
        Ok(ptr)
    }

    /// Copy a device-owned recurrent state slab back to the host.
    ///
    /// Nothing on the forward path wants this -- the whole point of the slab
    /// living on the device is that it never comes home. It exists so a test
    /// can check what a kernel *left behind*, which is half of what these ops
    /// do and is invisible from the output alone.
    pub fn read_state_into(&self, host: &mut [f32]) -> Result<()> {
        let key = host.as_ptr() as usize;
        let ptr = match self.states.borrow().get(&key) {
            Some(b) => b.ptr,
            None => return Ok(()),
        };
        self.sync()?;
        self.d2h(host, ptr)
    }

    fn gather_chunks_impl(
        &self,
        src: &[f32],
        chunk: usize,
        stride: usize,
        offset: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let sd = self.mirror_in(src)?;
        let od = self.mirror_out(out)?;
        let args = [
            KArg::I32(out.len() as i32),
            KArg::I32(chunk as i32),
            KArg::I32(stride as i32),
            KArg::I32(offset as i32),
            KArg::Ptr(sd),
            KArg::Ptr(od),
        ];
        let blocks = out.len().div_ceil(256) as u32;
        // SAFETY: parameters match `gather_chunks` in kernels.cu; one thread
        // per output element, guarded against the tail.
        unsafe { self.launch_shared("gather_chunks", blocks, 256, 0, &args)? };
        Ok(())
    }

    fn sigmoid_mul_impl(&self, x: &mut [f32], g: &[f32]) -> Result<()> {
        let gd = self.mirror_in(g)?;
        let xd = self.mirror_in(x)?;
        let args = [KArg::I32(x.len() as i32), KArg::Ptr(xd), KArg::Ptr(gd)];
        let blocks = x.len().div_ceil(256) as u32;
        // SAFETY: parameters match `sigmoid_mul` in kernels.cu.
        unsafe { self.launch_shared("sigmoid_mul", blocks, 256, 0, &args)? };
        self.mirror_out(x).map(|_| ())
    }

    fn l2_norm_heads_impl(&self, x: &mut [f32], head_dim: usize, eps: f32) -> Result<()> {
        let n_heads = x.len() / head_dim;
        let xd = self.mirror_in(x)?;
        let args = [
            KArg::I32(head_dim as i32),
            KArg::F32(eps),
            KArg::Ptr(xd),
        ];
        // SAFETY: parameters match `l2_norm_heads` in kernels.cu; one block per
        // head, and the buffer was sized from the slice it mirrors.
        unsafe { self.launch_shared("l2_norm_heads", n_heads as u32, 128, 0, &args)? };
        self.mirror_out(x).map(|_| ())
    }

    fn ssm_conv_impl(
        &self,
        state: &mut [f32],
        x: &[f32],
        weight: &[f32],
        kernel: usize,
        out: &mut [f32],
    ) -> Result<()> {
        // Channels per token; `n_tok` rows of them.
        let nc = weight.len() / kernel;
        let n_tok = out.len() / nc;
        let sd = self.state_resident(state)?;
        let xd = self.mirror_in(x)?;
        let w = self.resident(weight)?;
        let od = self.mirror_out(out)?;
        let blocks = nc.div_ceil(256) as u32;

        // **Sequential in the batch**: each token convolves over the window the
        // previous one advanced, so the tokens are launched in order against
        // one state. The row offsets go on the device pointers rather than by
        // sub-slicing `x` on the host, because this backend keys its mirrors on
        // host addresses and a sub-slice would be uploaded from a stale copy.
        //
        // This is `n_tok` launches where a chunked algorithm would need one.
        // The seam takes the whole batch precisely so that stays a decision
        // inside this file.
        for t in 0..n_tok {
            let args = [
                KArg::I32(nc as i32),
                KArg::I32(kernel as i32),
                KArg::Ptr(sd),
                KArg::Ptr(xd + (t * nc * 4) as u64),
                KArg::Ptr(w),
                KArg::Ptr(od + (t * nc * 4) as u64),
            ];
            // SAFETY: parameters match `ssm_conv` in kernels.cu; one thread per
            // channel, guarded against the tail, and both offsets stay inside
            // buffers sized for `n_tok * nc` floats.
            unsafe { self.launch_shared("ssm_conv", blocks, 256, 0, &args)? };
        }
        Ok(())
    }

    fn delta_rule_impl(&self, d: &Delta<'_>, state: &mut [f32], out: &mut [f32]) -> Result<()> {
        let sd = self.state_resident(state)?;
        let q = self.mirror_in(d.q)?;
        let k = self.mirror_in(d.k)?;
        let v = self.mirror_in(d.v)?;
        let alpha = self.mirror_in(d.alpha)?;
        let beta = self.mirror_in(d.beta)?;
        // ssm_a and dt_bias are weights, not activations: one upload for the
        // life of the process.
        let ssm_a = self.resident(d.ssm_a)?;
        let dt = self.resident(d.dt_bias)?;
        let od = self.mirror_out(out)?;

        // q and k for the head are staged in shared memory: every thread in the
        // block reads all of both.
        let shared = (2 * d.head_k_dim * 4) as u32;
        let threads = d.head_v_dim.min(256) as u32;

        // **The one op with no batched form.** Token `t`'s rank-1 correction is
        // token `t+1`'s stored state, so these launches are ordered and cannot
        // be collapsed into a grid. Offsets go on the device pointers for the
        // same reason as in `ssm_conv_impl`.
        let kper = d.n_k_heads * d.head_k_dim;
        let vper = d.n_v_heads * d.head_v_dim;
        let heads = d.n_v_heads;
        for t in 0..d.n_tokens() {
            let args = [
                KArg::I32(d.head_k_dim as i32),
                KArg::I32(d.head_v_dim as i32),
                KArg::I32(d.n_k_heads as i32),
                KArg::F32(d.scale()),
                KArg::Ptr(q + (t * kper * 4) as u64),
                KArg::Ptr(k + (t * kper * 4) as u64),
                KArg::Ptr(v + (t * vper * 4) as u64),
                KArg::Ptr(alpha + (t * heads * 4) as u64),
                KArg::Ptr(beta + (t * heads * 4) as u64),
                KArg::Ptr(ssm_a),
                KArg::Ptr(dt),
                KArg::Ptr(sd),
                KArg::Ptr(od + (t * vper * 4) as u64),
            ];
            // SAFETY: parameters match `delta_rule` in kernels.cu; one block
            // per value head, `shared` is the two staged vectors, and every
            // offset stays inside a buffer the model sized for `n_tokens`.
            unsafe {
                self.launch_shared("delta_rule", d.n_v_heads as u32, threads, shared, &args)?
            };
        }
        Ok(())
    }

    fn kv_write_impl(&self, slab: &mut [u16], offset: usize, src: &[f32]) -> Result<()> {
        let key = slab.as_ptr() as usize;
        let bytes = std::mem::size_of_val(slab);
        {
            let mut map = self.kv.borrow_mut();
            if !map.contains_key(&key) {
                map.insert(
                    key,
                    KvMirror {
                        buf: DeviceBuffer::new(bytes)?,
                        uploaded: 0,
                        device_written: false,
                    },
                );
            }
            match map.get_mut(&key) {
                Some(m) => m.device_written = true,
                None => {
                    return Err(Error::Cuda {
                        what: "kv_write",
                        detail: "mirror vanished between insert and lookup".to_string(),
                    });
                }
            }
        }
        let dst = match self.kv.borrow().get(&key) {
            Some(m) => m.buf.ptr + (offset * 2) as u64,
            None => {
                return Err(Error::Cuda {
                    what: "kv_write",
                    detail: "mirror vanished".to_string(),
                });
            }
        };

        // `src` is already on the device -- it is the model's k or v buffer,
        // which rope wrote there.
        let sd = self.mirror_in(src)?;
        let args = [KArg::I32(src.len() as i32), KArg::Ptr(sd), KArg::Ptr(dst)];
        let block = 256u32;
        // SAFETY: parameters match `kv_write_f16`; `dst` is inside the mirror,
        // which the model sized, and the grid covers exactly `src.len()`.
        unsafe {
            self.launch(
                "kv_write_f16",
                src.len().div_ceil(block as usize) as u32,
                block,
                &args,
            )?
        };
        Ok(())
    }

    fn silu_mul_impl(&self, gate: &mut [f32], up: &[f32]) -> Result<()> {
        let gd = self.mirror_in(gate)?;
        let ud = self.mirror_in(up)?;

        let args = [KArg::I32(gate.len() as i32), KArg::Ptr(gd), KArg::Ptr(ud)];
        let block = 256u32;
        // SAFETY: parameters match `silu_mul`; both buffers hold `n` floats.
        unsafe {
            self.launch(
                "silu_mul",
                gate.len().div_ceil(block as usize) as u32,
                block,
                &args,
            )?
        };
        self.mirror_out(gate).map(|_| ())
    }

    fn add_assign_impl(&self, a: &mut [f32], b: &[f32]) -> Result<()> {
        let ad = self.mirror_in(a)?;
        let bd = self.mirror_in(b)?;

        let args = [KArg::I32(a.len() as i32), KArg::Ptr(ad), KArg::Ptr(bd)];
        let block = 256u32;
        // SAFETY: parameters match `add_assign`; both buffers hold `n` floats.
        unsafe {
            self.launch(
                "add_assign",
                a.len().div_ceil(block as usize) as u32,
                block,
                &args,
            )?
        };
        self.mirror_out(a).map(|_| ())
    }
}

/// The error every GatedDeltaNet primitive returns until it has a kernel.
///
/// Named rather than inlined so the three call sites cannot drift, and so that
/// deleting it is the obvious signal that the CUDA path landed.
fn no_gdn_kernel(what: &'static str) -> Error {
    Error::Cuda {
        what,
        detail: "GatedDeltaNet has no CUDA kernel yet; run the qwen35 \
                 architectures on --backend spin until one exists"
            .to_string(),
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

    fn rope_neox(
        &self,
        x: &mut [f32],
        pos: usize,
        head_dim: usize,
        n_rot: usize,
        n_heads: usize,
        theta: f32,
    ) {
        self.note(self.rope_impl(x, pos, head_dim, n_rot, n_heads, theta));
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

    // GatedDeltaNet has no CUDA path yet, and says so rather than quietly
    // falling back to the host.
    //
    // A host fallback would work, and would be a trap: each of these sits
    // inside a GDN layer, so running one on the CPU drags the whole activation
    // home and back, undoing the residency the seam exists for -- and it would
    // read as a mysterious slowdown rather than a missing kernel. It would also
    // break graph capture, which needs an identical launch sequence every pass.
    //
    // The sticky error is the same mechanism `matmul` uses for a quant type it
    // has no kernel for.
    fn l2_norm_heads(&self, x: &mut [f32], head_dim: usize, eps: f32) {
        self.note(self.l2_norm_heads_impl(x, head_dim, eps));
    }

    fn ssm_conv(
        &self,
        state: &mut [f32],
        x: &[f32],
        weight: &[f32],
        kernel: usize,
        out: &mut [f32],
    ) {
        self.note(self.ssm_conv_impl(state, x, weight, kernel, out));
    }

    fn gather_chunks(
        &self,
        src: &[f32],
        chunk: usize,
        stride: usize,
        offset: usize,
        out: &mut [f32],
    ) {
        self.note(self.gather_chunks_impl(src, chunk, stride, offset, out));
    }

    fn sigmoid_mul(&self, x: &mut [f32], g: &[f32]) {
        self.note(self.sigmoid_mul_impl(x, g));
    }

    fn delta_rule(&self, d: &Delta<'_>, state: &mut [f32], out: &mut [f32]) {
        self.note(self.delta_rule_impl(d, state, out));
    }

    fn forget_state(&self) {
        self.states.borrow_mut().clear();
    }

    fn host_wrote(&self, buf: &[f32]) {
        if let Some(m) = self.mirrors.borrow_mut().get_mut(&(buf.as_ptr() as usize)) {
            m.invalidate();
        }
    }

    fn host_needs(&self, buf: &mut [f32]) {
        let key = buf.as_ptr() as usize;
        let ptr = match self.mirrors.borrow().get(&key) {
            Some(m) if m.device_current => m.buf.ptr,
            _ => return,
        };
        self.note(self.d2h(buf, ptr));
    }

    fn kv_write(&self, slab: &mut [u16], offset: usize, src: &[f32]) {
        self.note(self.kv_write_impl(slab, offset, src));
    }

    fn end_pass(&self) {
        self.note(self.graph_end());
        self.note(self.timing_end());
    }

    fn begin_pass(&self, n_tokens: usize) {
        self.note(self.timing_begin());
        self.note(self.graph_begin(n_tokens));
        // Activation buffers are allocated per pass, so an address from the
        // last pass may name a different buffer now. Every mirror is marked
        // stale, which forces a re-upload before anything reads it — that is
        // what makes keying on an address safe. The device allocations are
        // kept: freeing and reallocating ~280 of them per token costs far more
        // than it saves.
        for m in self.mirrors.borrow_mut().values_mut() {
            m.invalidate();
        }
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

    fn rope_neox(
        &self,
        x: &mut [f32],
        pos: usize,
        head_dim: usize,
        n_rot: usize,
        n_heads: usize,
        theta: f32,
    ) {
        (*self).rope_neox(x, pos, head_dim, n_rot, n_heads, theta)
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

    fn l2_norm_heads(&self, x: &mut [f32], head_dim: usize, eps: f32) {
        (*self).l2_norm_heads(x, head_dim, eps)
    }

    fn ssm_conv(
        &self,
        state: &mut [f32],
        x: &[f32],
        weight: &[f32],
        kernel: usize,
        out: &mut [f32],
    ) {
        (*self).ssm_conv(state, x, weight, kernel, out)
    }

    fn gather_chunks(
        &self,
        src: &[f32],
        chunk: usize,
        stride: usize,
        offset: usize,
        out: &mut [f32],
    ) {
        (*self).gather_chunks(src, chunk, stride, offset, out)
    }

    fn sigmoid_mul(&self, x: &mut [f32], g: &[f32]) {
        (*self).sigmoid_mul(x, g)
    }

    fn delta_rule(&self, d: &Delta<'_>, state: &mut [f32], out: &mut [f32]) {
        (*self).delta_rule(d, state, out)
    }

    fn forget_state(&self) {
        (*self).forget_state()
    }

    fn host_wrote(&self, buf: &[f32]) {
        (*self).host_wrote(buf)
    }

    fn host_needs(&self, buf: &mut [f32]) {
        (*self).host_needs(buf)
    }

    fn begin_pass(&self, n_tokens: usize) {
        (*self).begin_pass(n_tokens)
    }

    fn end_pass(&self) {
        (*self).end_pass()
    }

    fn kv_write(&self, slab: &mut [u16], offset: usize, src: &[f32]) {
        (*self).kv_write(slab, offset, src)
    }
}
