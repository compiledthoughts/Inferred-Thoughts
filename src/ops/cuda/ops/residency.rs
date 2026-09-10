//! Where bytes live on the device: activation mirrors and their cached
//! quantizations, the weight, KV and recurrent-state mirrors, the scratch
//! pool, and the counted copies underneath them.

use std::ffi::c_void;

use crate::error::{Error, Result};
use crate::ops::Weights;
use crate::ops::cuda::{Cuda, DeviceBuffer, KArg, KvMirror, Mirror, check, ffi};

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
    pub(super) fn mirror_in(&self, data: &[f32]) -> Result<ffi::CUdeviceptr> {
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
    pub(super) fn mirror_out(&self, data: &[f32]) -> Result<ffi::CUdeviceptr> {
        let key = data.as_ptr() as usize;
        let bytes = std::mem::size_of_val(data);
        let (ptr, _) = self.slot_for(key, bytes)?;
        if let Some(m) = self.mirrors.borrow_mut().get_mut(&key) {
            m.device_current = true;
            // The contents are about to change, so a quantization of them is
            // stale — but the buffer holding it is kept. **Both** forms: a
            // 35B attention layer keeps a Q8_0 and a Q8_K copy of the same
            // activation live at once, and invalidating one would leave the
            // other serving the previous token's values.
            m.quant_valid = false;
            m.quant_k_valid = false;
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
                    quant_k: None,
                    quant_k_valid: false,
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

    /// This activation quantized to **Q8_K** on the device, computed once.
    ///
    /// Returns `(scales, quants, bsums)`. The same caching argument as
    /// [`Cuda::quantized`], on a separate slot: the k-quants pair with Q8_K
    /// where Q8_0 weights pair with Q8_0, so a 35B attention layer needs both
    /// forms of the same normed activation and one slot would thrash.
    ///
    /// `bsums` is carried even though only Q5_K reads it. Sizing the buffer on
    /// demand would make its presence depend on which weight arrived first,
    /// which is exactly the kind of ordering dependence that leaves two formats
    /// right and one wrong.
    pub(super) fn quantized_k(
        &self,
        x: &[f32],
        n_super: usize,
    ) -> Result<(ffi::CUdeviceptr, ffi::CUdeviceptr, ffi::CUdeviceptr)> {
        const QK_K: usize = 256;
        let key = x.as_ptr() as usize;
        let xd = self.mirror_in(x)?;

        let existing = match self.mirrors.borrow().get(&key) {
            Some(m) => match &m.quant_k {
                Some((s, q, b)) if s.len_bytes() >= n_super * 4 => {
                    if m.quant_k_valid {
                        return Ok((s.ptr, q.ptr, b.ptr));
                    }
                    Some((s.ptr, q.ptr, b.ptr))
                }
                _ => None,
            },
            None => None,
        };

        let (sd, qd, bd, fresh_bufs) = match existing {
            Some((s, q, b)) => (s, q, b, None),
            None => {
                let scales = DeviceBuffer::new(n_super * 4)?;
                let quants = DeviceBuffer::new(n_super * QK_K)?;
                let bsums = DeviceBuffer::new(n_super * (QK_K / 16) * 2)?;
                let (s, q, b) = (scales.ptr, quants.ptr, bsums.ptr);
                (s, q, b, Some((scales, quants, bsums)))
            }
        };
        {
            let args = [
                KArg::I32(n_super as i32),
                KArg::Ptr(xd),
                KArg::Ptr(sd),
                KArg::Ptr(qd),
                KArg::Ptr(bd),
            ];
            // One block of 256 threads per super-block: the argmax that sets
            // the scale is a whole-super-block reduction, so the block is the
            // super-block rather than a tunable.
            // SAFETY: parameters match `quantize_q8_k`; the grid covers exactly
            // `n_super` super-blocks and all three outputs are sized for them.
            self.note_shape("quantize_q8_k", n_super * QK_K, 0);
            unsafe { self.launch("quantize_q8_k", n_super as u32, QK_K as u32, &args)? };
        }
        if let Some(m) = self.mirrors.borrow_mut().get_mut(&key) {
            if let Some(bufs) = fresh_bufs {
                m.quant_k = Some(bufs);
            }
            m.quant_k_valid = true;
        }
        Ok((sd, qd, bd))
    }

    /// This activation quantized to Q8_0 on the device, computed once.
    ///
    /// Returns `(scales, quants)` device pointers. The result is cached on the
    /// buffer's mirror and dropped the moment anything writes that buffer, so
    /// the five matmuls a layer runs against two distinct activations do two
    /// quantizations rather than five.
    pub(super) fn quantized(
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
    pub(super) fn h2d<T: Copy>(&self, dst: ffi::CUdeviceptr, data: &[T]) -> Result<()> {
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
    pub(super) fn d2h<T: Copy>(&self, out: &mut [T], src: ffi::CUdeviceptr) -> Result<()> {
        let bytes = std::mem::size_of_val(out);
        if bytes == 0 {
            return Ok(());
        }
        // A device-to-host copy cannot start until the work before it has
        // finished, so this is where the host blocks and it is timed as such.
        let started = std::time::Instant::now();
        // Kept synchronous. Splitting it into an async copy plus
        // `cuCtxSynchronize` was tried, to put the blocking somewhere
        // `CU_CTX_SCHED_BLOCKING_SYNC` governs and so measure host work apart
        // from host spinning. **WSL's driver spins either way** — on-CPU time
        // stayed at 95% of wall — and the split cost 9% of throughput. Reverted;
        // the finding is in `HANDOFF.md`.
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
    pub(super) fn resident<T: Copy>(&self, data: &[T]) -> Result<ffi::CUdeviceptr> {
        let key = data.as_ptr() as usize;
        let mut map = self.weights.borrow_mut();
        if let Some(b) = map.get(&key) {
            return Ok(b.ptr);
        }
        // Split into allocate and copy, and timed around the copy only, so
        // this does not double-count the `cuMemAlloc` that `DeviceBuffer::new`
        // already reports through `alloc_ns`. `from_slice` would do both and
        // there would be no way to add the two totals without overlap.
        //
        // Timed below the lookup, not around it: a hit returns before reaching
        // here, so observing this costs nothing on the path that runs every
        // token.
        let buf = DeviceBuffer::new(std::mem::size_of_val(data))?;
        let started = std::time::Instant::now();
        buf.write(data)?;
        let upload_ns = started.elapsed().as_nanos() as u64;
        // Counted, because on the 35B this is no longer a start-up cost.
        //
        // `DeviceBuffer::from_slice` copies through the driver directly rather
        // than through `Cuda::h2d`, so these uploads were invisible to the
        // crossing counters. That was harmless while `resident` held only norm
        // vectors uploaded once; it is not harmless now, when it is the path
        // every routed expert takes and the traffic it hides is the quantity
        // this project exists to measure. The first 35B run reported "2.0 MiB
        // up" against ~300 MB per token of actual expert streaming.
        self.bump(|s| {
            s.h2d_calls += 1;
            s.h2d_bytes += std::mem::size_of_val(data) as u64;
            s.weight_upload_ns += upload_ns;
        });
        let ptr = buf.ptr;
        map.insert(key, buf);
        Ok(ptr)
    }

    /// Device mirror of one layer's K or V slab, brought up to `n_pos`.
    ///
    /// Uploads only the positions added since the last call. A shorter `n_pos`
    /// than last time means the cache was reset, so the mirror is refilled
    /// from position zero.
    pub(super) fn kv_resident(&self, host: &[u16], n_pos: usize, kv_dim: usize) -> Result<ffi::CUdeviceptr> {
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
    pub(super) fn pooled(&self, slot: usize, bytes: usize) -> Result<ffi::CUdeviceptr> {
        let mut pool = self.pool.borrow_mut();
        while pool.len() <= slot {
            pool.push(DeviceBuffer::new(0)?);
        }
        if pool[slot].len_bytes() < bytes {
            pool[slot] = DeviceBuffer::new(bytes)?;
        }
        Ok(pool[slot].ptr)
    }

    /// The F32 weight transposed into column-major, uploaded once.
    ///
    /// Done on the host at first touch: it is `n_in * n_out` reads of a mapped
    /// file, once per tensor for the life of the run, against a kernel that
    /// then reads it every token.
    pub(super) fn resident_f32_t(&self, w: &Weights<'_>) -> Result<ffi::CUdeviceptr> {
        let key = w.data.as_ptr() as usize;
        if let Some(b) = self.f32t.borrow().get(&key) {
            // **An address alone is not an identity.** Weights from the model's
            // mmap never move, but a caller that frees a buffer and allocates
            // another can be handed the same address for a different shape, and
            // a cached copy that is too small becomes a device read past its
            // end. `the_staged_f32_matmul_is_bit_identical` hit exactly that as
            // a sticky CUDA_ERROR_ILLEGAL_ADDRESS once a heap layout change
            // recycled the address. A size that does not match is re-uploaded.
            if b.len_bytes() == w.n_in * w.n_out * std::mem::size_of::<f32>() {
                return Ok(b.ptr);
            }
        }
        let mut t = vec![0.0f32; w.n_in * w.n_out];
        for j in 0..w.n_out {
            let row = &w.data[j * w.n_in * 4..(j + 1) * w.n_in * 4];
            for (k, c) in row.chunks_exact(4).enumerate() {
                t[k * w.n_out + j] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        let buf = DeviceBuffer::from_slice(&t)?;
        self.bump(|s| {
            s.h2d_calls += 1;
            s.h2d_bytes += std::mem::size_of_val(&t[..]) as u64;
        });
        let ptr = buf.ptr;
        self.f32t.borrow_mut().insert(key, buf);
        Ok(ptr)
    }

    /// Two F32 weights interleaved into one column-major stack, uploaded once.
    ///
    /// **Interleaved, not appended.** `matmul_f32_t` indexes
    /// `wt[k * n_out + j]`, so widening the output means every super-block `k`
    /// gains `b`'s rows after `a`'s. Appending the two buffers would put all of
    /// `a` before all of `b`, which is a different tensor entirely.
    ///
    /// Keyed on both source pointers, so a weight paired with two different
    /// partners gets two stacks rather than silently reusing the first. Built
    /// once and kept for the life of the backend, as `resident_f32_t` is: the
    /// merge costs nothing at run time.
    pub(super) fn resident_f32_t_pair(&self, a: &Weights<'_>, b: &Weights<'_>) -> Result<ffi::CUdeviceptr> {
        let key = (a.data.as_ptr() as usize, b.data.as_ptr() as usize);
        if let Some(buf) = self.f32t_pair.borrow().get(&key) {
            return Ok(buf.ptr);
        }
        let n_out = a.n_out + b.n_out;
        let mut t = vec![0.0f32; a.n_in * n_out];
        {
            let mut fill = |w: &Weights<'_>, base: usize| {
                for j in 0..w.n_out {
                    let row = &w.data[j * w.n_in * 4..(j + 1) * w.n_in * 4];
                    for (k, c) in row.chunks_exact(4).enumerate() {
                        t[k * n_out + base + j] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    }
                }
            };
            fill(a, 0);
            fill(b, a.n_out);
        }
        let buf = DeviceBuffer::from_slice(&t)?;
        self.bump(|st| {
            st.h2d_calls += 1;
            st.h2d_bytes += std::mem::size_of_val(&t[..]) as u64;
        });
        let ptr = buf.ptr;
        self.f32t_pair.borrow_mut().insert(key, buf);
        Ok(ptr)
    }

    /// The repacked device copy of a Q8_0 tensor, built once on first use.
    ///
    /// Returns `(scales, quants)`. The split happens on the host in chunks of
    /// whole rows so the temporary never approaches the tensor size -- the 9B's
    /// LM head alone is over a gigabyte, and holding a second copy of it would
    /// undo the point of mapping the file rather than reading it.
    pub(super) fn resident_q8_0(&self, w: &Weights<'_>) -> Result<(ffi::CUdeviceptr, ffi::CUdeviceptr)> {
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

    /// Device-resident recurrent state for one layer, uploaded once.
    ///
    /// Unlike `resident`, the device copy is *written* by kernels, so after the
    /// first touch it is the authoritative one and the host slab is stale by
    /// design -- exactly as the KV slabs are. `forget_state` is what makes a
    /// sequence reset visible.
    pub(super) fn state_resident(&self, host: &[f32]) -> Result<ffi::CUdeviceptr> {
        let key = host.as_ptr() as usize;
        let epoch = self.state_gen.get();
        // Read the entry out before any copy, so the `RefCell` borrow is not
        // held across `h2d`.
        let found = self
            .states
            .borrow()
            .get(&key)
            .map(|(b, filled)| (b.ptr, b.len_bytes(), *filled));
        if let Some((ptr, bytes, filled)) = found {
            // **Only if it is the same size.** An address alone is not an
            // identity: a slab freed and reallocated at the same address with a
            // different size is a different slab, and re-uploading it into the
            // old allocation would write past its end. Such a slab falls
            // through and gets its own buffer.
            if bytes == std::mem::size_of_val(host) {
                // The allocation is right; only its contents may be stale. This
                // is the checkpoint-restore path, and re-uploading into the
                // buffer that already exists is what makes it cheap.
                if filled == epoch {
                    return Ok(ptr);
                }
                self.h2d(ptr, host)?;
                if let Some(e) = self.states.borrow_mut().get_mut(&key) {
                    e.1 = epoch;
                }
                return Ok(ptr);
            }
        }
        let buf = DeviceBuffer::from_slice(host)?;
        let ptr = buf.ptr;
        self.states.borrow_mut().insert(key, (buf, epoch));
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
            Some((b, _)) => b.ptr,
            None => return Ok(()),
        };
        self.sync()?;
        self.d2h(host, ptr)
    }
}
