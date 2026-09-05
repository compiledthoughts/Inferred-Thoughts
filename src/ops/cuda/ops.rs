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

use super::{Cuda, DeviceBuffer, KArg, KvMirror, Mirror, check, experts, ffi};
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
const MM_TOK: usize = 8;

/// Blocks of a weight row in flight at once. **Must equal `MM_SEG` in
/// `kernels/kernels.cu`.** It fixes the shared-memory request independently of
/// `n_in`, which is what uncapped `MM_TOK`.
const MM_SEG: usize = 64;
use crate::ops::{Attn, Delta, Experts, Ops, Route, Weights};

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
    /// Device-side routing scratch: the ids and weights `moe_topk` writes, and
    /// the expert addresses `moe_gather_ptrs` resolves for gate, up and down.
    ///
    /// One set shared by all forty layers, which is safe because a stream
    /// executes in issue order: a layer's gather completes before its matmuls
    /// start, and its matmuls complete before the next layer's gather.
    pub const ROUTE_IDS: usize = 11;
    pub const ROUTE_W: usize = 12;
    pub const PTR_GATE: usize = 13;
    pub const PTR_UP: usize = 14;
    pub const PTR_DOWN: usize = 15;
}

/// One matmul shape, measured on the live device with the launch queue kept
/// full — the thing `--profile-kernels` cannot report, because synchronizing
/// per launch is exactly what changes the answer.
///
/// `gpu` and `issue` are the same launches timed two ways, so the larger of
/// them is what binds for this shape. `grouped` is the same arithmetic in one
/// launch instead of `group`, which prices fusing the routed FFN.
#[derive(Debug, Clone, Copy)]
pub struct ShapeBench {
    pub kernel: &'static str,
    pub n_in: usize,
    pub n_out: usize,
    /// Launches of this shape observed during the run being profiled.
    pub calls: u64,
    /// Device microseconds per launch.
    pub gpu_us: f64,
    /// Host microseconds per launch, i.e. what it costs to describe the work.
    pub issue_us: f64,
    /// Device microseconds for one launch covering `group` shapes' rows.
    pub grouped_us: f64,
    pub group: usize,
}

/// One kernel launch the run actually made, replayed and timed.
///
/// See [`Cuda::bench_launches`]. Unlike [`ShapeBench`] this cannot omit a
/// kernel: the entry exists because the launch happened.
#[derive(Debug, Clone)]
pub struct LaunchBench {
    pub kernel: &'static str,
    pub grid: (u32, u32),
    pub block: u32,
    /// Times this exact launch was issued during the run.
    pub calls: u64,
    /// Device microseconds for one launch, queue kept full, best of four.
    pub gpu_us: f64,
    /// Host microseconds to issue one.
    pub issue_us: f64,
}

impl LaunchBench {
    pub fn gpu_ms_per_token(&self, tokens: u64) -> f64 {
        self.calls as f64 / tokens.max(1) as f64 * self.gpu_us / 1000.0
    }
    pub fn calls_per_token(&self, tokens: u64) -> f64 {
        self.calls as f64 / tokens.max(1) as f64
    }
    /// Whether the host, not the device, limited this measurement.
    pub fn host_limited(&self) -> bool {
        self.gpu_us > 0.0 && self.issue_us / self.gpu_us > 0.7
    }
}

impl ShapeBench {
    /// Whether the host, not the device, limited this measurement.
    ///
    /// The event pair brackets the launch loop, so if the host cannot keep the
    /// queue full it times the stall and `gpu` collapses onto `issue`. Taking
    /// the best of several runs removes most of it; what survives is reported
    /// rather than trusted, because the alternative is reading a 2 KiB
    /// elementwise kernel as costing 22 us of device time.
    pub fn host_limited(&self) -> bool {
        self.gpu_us > 0.0 && self.issue_us / self.gpu_us > 0.7
    }


    /// Device milliseconds this shape costs per token, at the observed rate.
    pub fn gpu_ms_per_token(&self, tokens: u64) -> f64 {
        self.calls as f64 / tokens.max(1) as f64 * self.gpu_us / 1000.0
    }

    /// Host milliseconds per token, same basis.
    pub fn issue_ms_per_token(&self, tokens: u64) -> f64 {
        self.calls as f64 / tokens.max(1) as f64 * self.issue_us / 1000.0
    }

    /// What fusing `group` launches into one would leave, per token, on the
    /// device. Compare against [`ShapeBench::gpu_ms_per_token`].
    pub fn grouped_ms_per_token(&self, tokens: u64) -> f64 {
        let launches = self.calls as f64 / self.group.max(1) as f64;
        launches / tokens.max(1) as f64 * self.grouped_us / 1000.0
    }
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
    fn quantized_k(
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
    fn resident<T: Copy>(&self, data: &[T]) -> Result<ffi::CUdeviceptr> {
        let key = data.as_ptr() as usize;
        let mut map = self.weights.borrow_mut();
        if let Some(b) = map.get(&key) {
            return Ok(b.ptr);
        }
        let buf = DeviceBuffer::from_slice(data)?;
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
    /// What the expert cache did. `None` if the model has no pooled weights.
    ///
    /// **The measurement the whole design is waiting on.** `HANDOFF.md` §9
    /// item 2 proposed getting the hit rate from an offline replay of a routing
    /// trace; this is better, because it is the policy actually running against
    /// the traffic actually generated.
    pub fn expert_stats(&self) -> Option<experts::ExpertStats> {
        self.absorb_expert_counters();
        self.experts.borrow().as_ref().map(|c| c.stats())
    }

    /// Bring the device-side read counters home and fold them into the stats.
    ///
    /// **Without this the cache is unobservable.** Routing on the device means
    /// the host never learns which experts a token read, so `hits`,
    /// `host_reads` and the coverage distribution would all report zero — a
    /// working cache saying nothing, which is worse than a broken one saying
    /// so. Called from the two accessors rather than per token, because a
    /// counter read costs a synchronize.
    fn absorb_expert_counters(&self) {
        let (counts_ptr, tally_ptr, n) = {
            let b = self.experts.borrow();
            let Some(c) = b.as_ref() else { return };
            let n = c.counted_experts();
            match c.counters_base() {
                Some((counts, tally)) if n > 0 => (counts, tally, n),
                _ => return,
            }
        };
        if self.sync().is_err() {
            return;
        }
        let mut counts = vec![0u32; n];
        let mut tally = [0u64; 2];
        if self.d2h(&mut counts, counts_ptr).is_err() || self.d2h(&mut tally, tally_ptr).is_err() {
            return;
        }
        if let Some(c) = self.experts.borrow_mut().as_mut() {
            c.absorb_counters(&counts, &tally);
        }
    }

    /// Cap the expert slab, in bytes. Zero restores the automatic budget.
    pub fn set_expert_budget(&self, bytes: usize) {
        // Expressed as a reserve because that is what the sizing code has to
        // work with: total free VRAM minus what everything else will need.
        let (free, _) = self.mem_info().unwrap_or((0, 0));
        self.expert_reserve
            .set(if bytes == 0 { experts::DEFAULT_RESERVE } else { free.saturating_sub(bytes) });
    }

    /// Cap the page-locked host tier behind the expert slab, in bytes.
    ///
    /// Zero restores [`experts::DEFAULT_HOST_BUDGET`]. Must be called before
    /// the first pooled tensor, since the cache reads it once at construction.
    pub fn set_expert_host_budget(&self, bytes: usize) {
        self.expert_host_budget
            .set(if bytes == 0 { experts::DEFAULT_HOST_BUDGET } else { bytes });
    }

    /// Hold `bytes` of VRAM back for the KV cache.
    ///
    /// **Fixes a sizing hole rather than tuning one.** The expert slab is sized
    /// from free VRAM at the first pooled tensor, which is inside block 0,
    /// while KV slabs are allocated at the first `kv_write` — block 3 on the
    /// 35B. So without this the slab takes VRAM the KV cache is going to need,
    /// and at a long context the KV allocation fails partway through the first
    /// prefill. At 4k it hides inside `DEFAULT_RESERVE`'s slack; at 256k it is
    /// 5 GiB and does not.
    ///
    /// Additive, because the caller knows the context length and this file
    /// knows the rest.
    pub fn reserve_for_kv(&self, bytes: usize) {
        self.expert_reserve.set(self.expert_reserve.get().saturating_add(bytes));
    }

    /// What fraction of expert reads the busiest slab-many tensors accounted
    /// for. See [`experts::ExpertCache::coverage`].
    pub fn expert_coverage(&self) -> Option<(f64, u64)> {
        self.absorb_expert_counters();
        self.experts.borrow().as_ref().map(|c| c.coverage())
    }

    /// Replace every kernel with a no-op, keeping the launch pattern exactly.
    ///
    /// The output is meaningless; the *time* is the point. See `noop` in
    /// kernels.cu.
    pub fn null_kernels(&self, on: bool) {
        self.null_kernels.set(on);
    }

    /// Whether graphs were turned off because the model read a device result
    /// mid-pass. See [`Cuda::mid_pass_read`]; reported by `--profile-device`
    /// so the throughput loss is visible rather than inferred from a launch
    /// count.
    pub fn graphs_off_for_mid_pass_read(&self) -> bool {
        self.mid_pass_read.get()
    }

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

    /// The k-quant matmuls: Q6_K, Q5_K and IQ4_XS, all against a Q8_K
    /// activation.
    ///
    /// **One kernel serves decode and prefill**, with the batch on `blockIdx.y`
    /// rather than in a second kernel. That is deliberately unlike the Q8_0
    /// pair, where `matmul_q8_0_batch` exists to amortize a weight load across
    /// `MM_TOK` tokens: here a warp still re-reads its weight row per token, so
    /// prefill weight traffic scales with the batch. Correct first, and the
    /// reuse variant arrives as a measured change against this baseline —
    /// which is also what keeps decode and prefill unable to disagree while the
    /// exactness claim is being established.
    fn matmul_kquant(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) -> Result<()> {
        const QK_K: usize = 256;
        let n_tok = x.len() / w.n_in;
        let n_super = w.n_in / QK_K;

        // The weight goes up in the file's own layout, unrepacked. The Q8_0
        // path repacks for 16-byte alignment; that was tried there *and*
        // reverted once for being 20% slower, so the same bet is not made
        // twice untested. Whether these three want a repack is a measurement,
        // not an assumption.
        let wd = self.expert_or_resident(w)?;
        let (sd, qd, bd) = self.quantized_k(x, n_tok * n_super)?;
        let od = self.mirror_out(out)?;

        // 128 threads is four warps, so four output rows per block.
        let block = 128u32;
        let rows_per_block = (block / 32) as usize;
        let grid_rows = w.n_out.div_ceil(rows_per_block) as u32;

        let (name, args): (&'static str, Vec<KArg>) = match w.ty {
            GgmlType::Q6K => (
                "matmul_q6_k_q8_k",
                vec![
                    KArg::I32(w.n_in as i32),
                    KArg::I32(w.n_out as i32),
                    KArg::Ptr(wd),
                    KArg::Ptr(sd),
                    KArg::Ptr(qd),
                    KArg::Ptr(od),
                ],
            ),
            GgmlType::Q5K => (
                "matmul_q5_k_q8_k",
                vec![
                    KArg::I32(w.n_in as i32),
                    KArg::I32(w.n_out as i32),
                    KArg::Ptr(wd),
                    KArg::Ptr(sd),
                    KArg::Ptr(qd),
                    // Only Q5_K reads the per-16 sums, and omitting them here
                    // would leave the other two correct.
                    KArg::Ptr(bd),
                    KArg::Ptr(od),
                ],
            ),
            _ => (
                "matmul_iq4_xs_q8_k",
                vec![
                    KArg::I32(w.n_in as i32),
                    KArg::I32(w.n_out as i32),
                    KArg::Ptr(wd),
                    KArg::Ptr(sd),
                    KArg::Ptr(qd),
                    KArg::Ptr(od),
                ],
            ),
        };

        self.note_shape(name, w.n_in, w.n_out);
        // SAFETY: parameters match the named kernel; the grid covers exactly
        // `n_out` rows by `n_tok` tokens, and the kernels use no dynamic
        // shared memory.
        unsafe { self.launch_grid2(name, grid_rows, n_tok as u32, block, 0, &args) }
    }

    /// The device address of a weight, from whichever residency it belongs to.
    ///
    /// **The one branch that decides whether this engine can serve a model
    /// larger than VRAM.** Anything that fits goes in the permanent mirror, as
    /// it always has. A tensor marked `pooled` — an MoE expert, and only an MoE
    /// expert — goes in the bounded slab, where it may be evicted.
    ///
    /// The distinction is a fact about the tensor, carried on `Weights::pooled`,
    /// not a policy. The policy is [`experts::ExpertCache`]'s.
    ///
    /// Note the slab is two-tier: a pooled tensor that does not fit in VRAM
    /// gets a page-locked host address the kernel dereferences over PCIe, so
    /// this function always returns an address and never stalls on a fill.
    fn expert_or_resident(&self, w: &Weights<'_>) -> Result<ffi::CUdeviceptr> {
        if !w.pooled {
            return self.resident(w.data);
        }
        let mut slot = self.experts.borrow_mut();
        if slot.is_none() {
            // Sized here rather than at construction, because "free VRAM" only
            // means something once the permanent weights are on their way up.
            // Nothing before the first expert of layer 0 is large.
            let (free, _) = self.mem_info()?;
            let reserve = self.expert_reserve.get();
            let budget = free.saturating_sub(reserve);
            let slots = budget / w.data.len().max(1);
            *slot =
                Some(experts::ExpertCache::new(w.data.len(), slots, self.expert_host_budget.get())?);
        }
        let cache = match slot.as_mut() {
            Some(c) => c,
            None => {
                return Err(Error::Cuda {
                    what: "expert cache",
                    detail: "cache vanished between build and use".to_string(),
                });
            }
        };
        let before = cache.stats().filled_bytes;
        let ptr = cache.address_of(w.data.as_ptr() as usize, w.data)?;
        let filled = cache.stats().filled_bytes - before;
        drop(slot);
        // A miss is a bus crossing and is counted as one; a hit moves nothing.
        // Counted rather than derived for the reason `DeviceBuffer::from_slice`
        // taught this session — an upload the counters cannot see reads as
        // "3.9 MiB up" against an actual 3111, and this is the exact traffic
        // the whole design is drawn against.
        if filled > 0 {
            self.bump(|st| {
                st.h2d_calls += 1;
                st.h2d_bytes += filled;
            });
        }
        Ok(ptr)
    }

    /// The F32 matmul, which on the 35B is the MoE router and nothing else.
    ///
    /// Kept as the slow one-thread-per-row shape on purpose; see `matmul_f32`
    /// in kernels.cu for why exactness is worth more than speed here.
    /// The F32 weight transposed into column-major, uploaded once.
    ///
    /// Done on the host at first touch: it is `n_in * n_out` reads of a mapped
    /// file, once per tensor for the life of the run, against a kernel that
    /// then reads it every token.
    fn resident_f32_t(&self, w: &Weights<'_>) -> Result<ffi::CUdeviceptr> {
        let key = w.data.as_ptr() as usize;
        if let Some(b) = self.f32t.borrow().get(&key) {
            return Ok(b.ptr);
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

    fn matmul_f32(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) -> Result<()> {
        let n_tok = x.len() / w.n_in;
        self.note_shape("matmul_f32_t", w.n_in, w.n_out);
        let wd = self.resident_f32_t(w)?;
        let xd = self.mirror_in(x)?;
        let od = self.mirror_out(out)?;
        let args = [
            KArg::I32(w.n_in as i32),
            KArg::I32(w.n_out as i32),
            KArg::Ptr(wd),
            KArg::Ptr(xd),
            KArg::Ptr(od),
        ];
        let block = 128u32;
        // SAFETY: parameters match `matmul_f32_t`; the grid covers exactly
        // `n_out` rows by `n_tok` tokens, and the weight is column-major.
        unsafe {
            self.launch_grid2(
                "matmul_f32_t",
                w.n_out.div_ceil(block as usize) as u32,
                n_tok as u32,
                block,
                0,
                &args,
            )
        }
    }

    /// The routed FFN's matmuls, every expert in one launch.
    ///
    /// IQ4_XS only, which is every `ffn_*_exps` tensor the 35B has. A different
    /// expert format would need its own grouped kernel; failing loudly is
    /// better than falling back to the trait default, whose sub-slicing of
    /// `out` would hand this backend addresses it has never mirrored.
    fn matmul_experts_impl(
        &self,
        w: &Experts<'_>,
        route: &Route,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        const QK_K: usize = 256;
        const MAX: usize = 8;
        if w.ty != GgmlType::Iq4Xs {
            return Err(Error::Cuda {
                what: "matmul_experts",
                detail: format!(
                    "{:?} experts have no grouped CUDA kernel; every ffn_*_exps in the \
                     models this targets is IQ4_XS",
                    w.ty
                ),
            });
        }
        let n_used = route.n_used();
        if n_used == 0 || n_used > MAX {
            return Err(Error::Cuda {
                what: "matmul_experts",
                detail: format!("{n_used} experts per token; the kernel carries at most {MAX}"),
            });
        }

        let n_super = w.n_in / QK_K;
        // One row shared by every expert, or one row each. Derived from the
        // buffer, as the seam's batch count is.
        let rows = x.len() / w.n_in;
        let x_stride_super = if rows == n_used { n_super } else { 0 };
        let (sd, qd, _) = self.quantized_k(x, rows * n_super)?;

        // The picks' addresses, resolved on the device from the slot table.
        // Nothing here reads the router's output, which is what lets this
        // launch live in a graph.
        let table = self.expert_table(w)?;
        let wptrs = self.gather_ptrs(w.data.as_ptr() as usize, table, n_used, slot::PTR_DOWN)?;
        let od = self.mirror_out(out)?;
        self.note_shape("matmul_iq4_xs_q8_k_moe", w.n_in, w.n_out);

        let block = 128u32;
        let rows_per_block = (block / 32) as usize;
        let args = [
            KArg::I32(w.n_in as i32),
            KArg::I32(w.n_out as i32),
            KArg::I32(x_stride_super as i32),
            KArg::I32(n_used as i32),
            KArg::Ptr(wptrs),
            KArg::Ptr(sd),
            KArg::Ptr(qd),
            KArg::Ptr(od),
        ];
        // SAFETY: parameters match `matmul_iq4_xs_q8_k_moe`; the grid covers
        // exactly `n_out` rows by `n_used` experts, unused pointer slots are
        // never dereferenced because the kernel returns on `e >= n_used`, and
        // the kernel uses no dynamic shared memory.
        unsafe {
            self.launch_grid2(
                "matmul_iq4_xs_q8_k_moe",
                w.n_out.div_ceil(rows_per_block) as u32,
                n_used as u32,
                block,
                0,
                &args,
            )
        }
    }

    fn moe_glu_impl(
        &self,
        gate: &Experts<'_>,
        up: &Experts<'_>,
        route: &Route,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        const QK_K: usize = 256;
        const MAX: usize = 8;
        if gate.ty != GgmlType::Iq4Xs || up.ty != GgmlType::Iq4Xs {
            return Err(Error::Cuda {
                what: "moe_glu",
                detail: format!("{:?}/{:?} experts have no fused CUDA kernel", gate.ty, up.ty),
            });
        }
        let n_used = route.n_used();
        if n_used == 0 || n_used > MAX || gate.n_in != up.n_in || gate.n_out != up.n_out {
            return Err(Error::Cuda {
                what: "moe_glu",
                detail: format!("{n_used} experts, gate {:?} vs up {:?}",
                    (gate.n_in, gate.n_out), (up.n_in, up.n_out)),
            });
        }

        let n_super = gate.n_in / QK_K;
        let (sd, qd, _) = self.quantized_k(x, n_super)?;

        // Two slot tables, two gathers, both on the device. `gate` and `up` are
        // separate tensors with separate tables, so an expert's gate and its up
        // need not share a residency tier.
        let gtab = self.expert_table(gate)?;
        let utab = self.expert_table(up)?;
        let gptrs = self.gather_ptrs(gate.data.as_ptr() as usize, gtab, n_used, slot::PTR_GATE)?;
        let uptrs = self.gather_ptrs(up.data.as_ptr() as usize, utab, n_used, slot::PTR_UP)?;
        let od = self.mirror_out(out)?;
        self.note_shape("matmul_iq4_xs_q8_k_moe_glu", gate.n_in, gate.n_out);

        let block = 128u32;
        let rows_per_block = (block / 32) as usize;
        let args = [
            KArg::I32(gate.n_in as i32),
            KArg::I32(gate.n_out as i32),
            KArg::I32(n_used as i32),
            KArg::Ptr(gptrs),
            KArg::Ptr(uptrs),
            KArg::Ptr(sd),
            KArg::Ptr(qd),
            KArg::Ptr(od),
        ];

        // SAFETY: parameters match `matmul_iq4_xs_q8_k_moe_glu`; the grid covers
        // `n_out` rows by `n_used` experts, unused pointer slots are never
        // dereferenced because the kernel returns on `e >= n_used`, and no
        // dynamic shared memory is used.
        unsafe {
            self.launch_grid2(
                "matmul_iq4_xs_q8_k_moe_glu",
                gate.n_out.div_ceil(rows_per_block) as u32,
                n_used as u32,
                block,
                0,
                &args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_finish_impl(
        &self,
        out: &mut [f32],
        at: usize,
        n: usize,
        rows: &[f32],
        route: &Route,
        shared: &[f32],
        logit: &[f32],
        logit_at: usize,
    ) -> Result<()> {
        const MAX: usize = 8;
        let n_used = route.n_used();
        if n_used == 0 || n_used > MAX {
            return Err(Error::Cuda {
                what: "moe_finish",
                detail: format!("{n_used} rows; the kernel carries at most {MAX}"),
            });
        }
        let rd = self.mirror_in(rows)?;
        let shd = self.mirror_in(shared)?;
        let ld = self.mirror_in(logit)?;
        // `mirror_in`: only row `at` is written, so the rest of `out` must
        // already be on the device.
        let od = self.mirror_in(out)?;
        // The weights come from the buffer `moe_topk` wrote, not from eight
        // kernel arguments -- the same move as the expert pointers, and for the
        // same reason: a graph bakes its arguments in at record time.
        let wd = self.pooled(slot::ROUTE_W, n_used * 4)?;
        let args = [
            KArg::I32(n as i32),
            KArg::I32(n_used as i32),
            KArg::I32(at as i32),
            KArg::Ptr(wd),
            KArg::Ptr(rd),
            KArg::Ptr(shd),
            KArg::Ptr(ld),
            KArg::I32(logit_at as i32),
            KArg::Ptr(od),
        ];
        self.note_shape("moe_finish", n, 0);
        // SAFETY: parameters match `moe_finish`; `rows` holds `scales.len() * n`
        // floats, `shared` holds `n`, and one thread covers each element of the
        // output row at `at`.
        unsafe { self.launch_shared("moe_finish", n.div_ceil(256) as u32, 256, 0, &args)? };
        self.mirror_out(out).map(|_| ())
    }

    fn add_scaled_rows_impl(
        &self,
        acc: &mut [f32],
        rows: &[f32],
        scales: &[f32],
    ) -> Result<()> {
        const MAX: usize = 8;
        if scales.is_empty() || scales.len() > MAX {
            return Err(Error::Cuda {
                what: "add_scaled_rows",
                detail: format!("{} rows; the kernel carries at most {MAX}", scales.len()),
            });
        }
        let n = acc.len();
        let rd = self.mirror_in(rows)?;
        // `mirror_out`, not `mirror_in`: every element of `acc` is written, so
        // whatever the host or a previous token left there is irrelevant.
        let ad = self.mirror_out(acc)?;
        let mut s = [0.0f32; MAX];
        s[..scales.len()].copy_from_slice(scales);
        let args = [
            KArg::I32(n as i32),
            KArg::I32(scales.len() as i32),
            KArg::F32(s[0]),
            KArg::F32(s[1]),
            KArg::F32(s[2]),
            KArg::F32(s[3]),
            KArg::F32(s[4]),
            KArg::F32(s[5]),
            KArg::F32(s[6]),
            KArg::F32(s[7]),
            KArg::Ptr(rd),
            KArg::Ptr(ad),
        ];
        self.note_shape("add_scaled_rows", n, 0);
        // SAFETY: parameters match `add_scaled_rows`; `rows` holds
        // `scales.len() * n` floats and one thread covers each output element.
        unsafe { self.launch("add_scaled_rows", n.div_ceil(256) as u32, 256, &args)? };
        Ok(())
    }

    fn matmul_impl(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) -> Result<()> {
        match w.ty {
            GgmlType::Q8_0 => {}
            GgmlType::Q6K | GgmlType::Q5K | GgmlType::Iq4Xs => {
                return self.matmul_kquant(w, x, out);
            }
            GgmlType::F32 => return self.matmul_f32(w, x, out),
            other => {
                return Err(Error::Cuda {
                    what: "matmul",
                    detail: format!(
                        "{other:?} has no CUDA kernel; this backend implements Q8_0 and the \
                         three k-quants the 35B uses (Q5_K, Q6_K, IQ4_XS)"
                    ),
                });
            }
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

        // The batched kernel folds each segment of the row into a running total
        // rather than holding every partial, so its shared-memory request is
        // `warps * MM_TOK * MM_SEG` floats and **does not grow with `n_in`**.
        // It therefore keeps four warps on every tensor either model has, which
        // is what lets `MM_TOK` be chosen for reuse instead of for occupancy.
        let b_shared = (rows_per_block * MM_TOK * MM_SEG * 4) as u32;

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
                grid_rows,
                n_tok.div_ceil(MM_TOK) as u32,
                block,
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

    /// Run `quantize_q8_k` on the device and read all three outputs back.
    ///
    /// **For tests, and it earns its place.** A wrong Q8_K quantization shows
    /// up in every k-quant dot at once, so without this a single defect looks
    /// like three broken matmuls — and a defect in `bsums` alone looks like one
    /// broken matmul (Q5_K) with two healthy ones, which is worse. This makes
    /// the quantizer directly comparable to `quant::kquant::Q8KRow::from_f32`.
    pub fn quantize_q8_k_readback(&self, x: &[f32]) -> Result<(Vec<f32>, Vec<i8>, Vec<i16>)> {
        const QK_K: usize = 256;
        if x.len() % QK_K != 0 {
            return Err(Error::Cuda {
                what: "quantize_q8_k_readback",
                detail: format!("{} values is not a whole number of super-blocks", x.len()),
            });
        }
        let n_super = x.len() / QK_K;
        self.begin_pass(1);
        self.host_wrote(x);
        let (sd, qd, bd) = self.quantized_k(x, n_super)?;
        self.sync()?;

        let mut scales = vec![0.0f32; n_super];
        let mut quants = vec![0i8; n_super * QK_K];
        let mut bsums = vec![0i16; n_super * (QK_K / 16)];
        self.d2h(&mut scales, sd)?;
        self.d2h(&mut quants, qd)?;
        self.d2h(&mut bsums, bd)?;
        Ok((scales, quants, bsums))
    }


    /// Run `moe_topk` on the device and bring its answer home.
    ///
    /// **For tests, and it earns its place the same way
    /// [`Cuda::quantize_q8_k_readback`] does.** Expert selection is the one
    /// decision in the pass that changes *which* weights are read rather than
    /// what is computed from them, so getting it wrong does not degrade the
    /// output — it produces fluent text from the wrong experts, which no
    /// tolerance-based check would catch. This makes the kernel directly
    /// comparable to the host loop in `qwen35::moe_token`.
    ///
    /// `probs` is the router's softmax output, already over all `n_expert`.
    pub fn moe_topk_readback(&self, probs: &[f32], n_used: usize) -> Result<(Vec<i32>, Vec<f32>)> {
        const MAX: usize = 8;
        if n_used == 0 || n_used > MAX || n_used > probs.len() {
            return Err(Error::Cuda {
                what: "moe_topk",
                detail: format!("{n_used} of {} experts; the kernel carries at most {MAX}", probs.len()),
            });
        }
        let pd = DeviceBuffer::from_slice(probs)?;
        let idb = DeviceBuffer::new(n_used * 4)?;
        let wb = DeviceBuffer::new(n_used * 4)?;
        // One block: the reduction is over the whole expert axis, so it cannot
        // be split across blocks without a second pass, and 256 experts is one
        // block's work. `blockDim` need not divide `n_expert` -- the per-thread
        // loop is strided -- but it must be a power of two for the tree.
        let block = 256u32;
        let shared = block * 8;
        let args = [
            KArg::I32(probs.len() as i32),
            KArg::I32(n_used as i32),
            KArg::Ptr(pd.ptr),
            KArg::Ptr(idb.ptr),
            KArg::Ptr(wb.ptr),
        ];
        // SAFETY: parameters match `moe_topk`; the two outputs are sized
        // `n_used` and the kernel writes exactly that many.
        unsafe { self.launch_shared("moe_topk", 1, block, shared, &args)? };
        self.sync()?;
        let mut ids = vec![0i32; n_used];
        let mut weights = vec![0.0f32; n_used];
        self.d2h(&mut ids, idb.ptr)?;
        self.d2h(&mut weights, wb.ptr)?;
        Ok((ids, weights))
    }

    /// The device pointer table for one `Experts` tensor.
    ///
    /// Built on first sight, which places every one of its experts — see
    /// [`experts::ExpertCache::table`] for why eager placement is the price of
    /// a graph, and what it costs.
    fn expert_table(&self, w: &Experts<'_>) -> Result<ffi::CUdeviceptr> {
        let key = w.data.as_ptr() as usize;
        let mut slot = self.experts.borrow_mut();
        if slot.is_none() {
            let (free, _) = self.mem_info()?;
            let budget = free.saturating_sub(self.expert_reserve.get());
            let stride = w.stride();
            *slot = Some(experts::ExpertCache::new(
                stride,
                budget / stride.max(1),
                self.expert_host_budget.get(),
            )?);
        }
        let cache = match slot.as_mut() {
            Some(c) => c,
            None => {
                return Err(Error::Cuda {
                    what: "expert table",
                    detail: "cache vanished between build and use".to_string(),
                });
            }
        };
        let before = cache.stats().filled_bytes;
        let ptr = cache.table(key, w.data, w.n_expert)?;
        let filled = cache.stats().filled_bytes - before;
        drop(slot);
        if filled > 0 {
            self.bump(|st| {
                st.h2d_calls += 1;
                st.h2d_bytes += filled;
            });
        }
        Ok(ptr)
    }

    /// Choose this token's experts on the device.
    ///
    /// The router's probabilities stay on the card, which is the whole point:
    /// the host read they replace is the reason CUDA graphs are off for this
    /// model. `moe_topk` reproduces [`Ops::route`]'s default exactly — see
    /// `device_topk_reproduces_the_host_selection`.
    fn route_impl(&self, probs: &[f32], n_used: usize) -> Result<()> {
        let pd = self.mirror_in(probs)?;
        let idd = self.pooled(slot::ROUTE_IDS, n_used * 4)?;
        let wd = self.pooled(slot::ROUTE_W, n_used * 4)?;
        let block = 256u32;
        let args = [
            KArg::I32(probs.len() as i32),
            KArg::I32(n_used as i32),
            KArg::Ptr(pd),
            KArg::Ptr(idd),
            KArg::Ptr(wd),
        ];
        // SAFETY: parameters match `moe_topk`; both outputs are `n_used` wide
        // and the kernel writes exactly that. Shared memory is one float and
        // one int per thread, which is what the kernel declares.
        unsafe { self.launch_shared("moe_topk", 1, block, block * 8, &args) }
    }

    /// Resolve `n_used` chosen experts to addresses, from `table`.
    ///
    /// Also the only place the expert cache can still be observed. With the
    /// picks living on the device the host never sees a read, so the kernel
    /// counts them; `key` finds this tensor's slice of the global counters.
    fn gather_ptrs(
        &self,
        key: usize,
        table: ffi::CUdeviceptr,
        n_used: usize,
        into: usize,
    ) -> Result<ffi::CUdeviceptr> {
        let idd = self.pooled(slot::ROUTE_IDS, n_used * 4)?;
        let out = self.pooled(into, n_used * 8)?;
        let (vram, counts, tally, base) = match self.experts.borrow().as_ref().and_then(|c| c.counters(key)) {
            Some(x) => x,
            None => {
                return Err(Error::Cuda {
                    what: "moe_gather_ptrs",
                    detail: "no counter slice for this expert tensor; the pool is larger \
                             than the counter arrays and the cache would go unobserved"
                        .to_string(),
                });
            }
        };
        let args = [
            KArg::I32(n_used as i32),
            KArg::I32(base as i32),
            KArg::Ptr(table),
            KArg::Ptr(idd),
            KArg::Ptr(vram),
            KArg::Ptr(counts),
            KArg::Ptr(tally),
            KArg::Ptr(out),
        ];
        // SAFETY: parameters match `moe_gather_ptrs`; one thread per pick, and
        // the kernel returns on `e >= n_used`. `base + id` is inside the
        // counter arrays because `table` refused a base that would not fit.
        unsafe { self.launch("moe_gather_ptrs", 1, 32, &args)? };
        Ok(out)
    }

    /// The host-side picks a `Route` carries, or an error naming the caller.
    ///
    /// **A deliberate error rather than a fallback.** `Route::Device` means the
    /// ids never came home, so an op that needs them on the host cannot proceed
    /// — and silently doing nothing, or routing to expert 0, would produce
    /// fluent text from the wrong weights. Until the device pointer table is
    /// complete this backend's `route` only returns `Host`, so this is a guard
    /// against a future half-wired state, not a live path.
    fn host_picks(&self, route: &Route, what: &'static str) -> Result<Vec<usize>> {
        match route.ids() {
            Some(ids) => Ok(ids.to_vec()),
            None => Err(Error::Cuda {
                what,
                detail: "device routing, but this op still resolves picks on the host"
                    .to_string(),
            }),
        }
    }

    /// As [`Cuda::host_picks`], for the normalized expert weights.
    fn host_weights(&self, route: &Route, what: &'static str) -> Result<Vec<f32>> {
        match route.weights() {
            Some(w) => Ok(w.to_vec()),
            None => Err(Error::Cuda {
                what,
                detail: "device routing, but this op still reads weights on the host"
                    .to_string(),
            }),
        }
    }

    /// Record a matmul's shape, so the microbenchmark can replay it later.
    fn note_shape(&self, kernel: &'static str, n_in: usize, n_out: usize) {
        *self
            .shapes
            .borrow_mut()
            .entry((kernel, n_in, n_out))
            .or_insert(0) += 1;
    }

    /// Turn launch recording on. Costs a hash probe per launch, so it is off
    /// unless `--profile-device` asked for it.
    pub fn record_launches(&self, on: bool) {
        self.record_launches.set(on);
    }

    /// Decompose the IQ4_XS matmul: replay its heaviest recorded launch
    /// against variants that each remove one thing.
    ///
    /// Returns `(label, gpu_us)` with the baseline first. See the `dbg_iq4_*`
    /// kernels for what each removes and what each answer would mean.
    pub fn bench_iq4_variants(&self, reps: u32) -> Result<Vec<(&'static str, f64)>> {
        // The heaviest recorded launch of the real kernel, by calls x geometry.
        let pick = self
            .launches
            .borrow()
            .iter()
            .filter(|(k, _)| k.0 == "matmul_iq4_xs_q8_k")
            .max_by_key(|(k, v)| v.0 * u64::from(k.1))
            .map(|(k, v)| (*k, v.1.clone()));
        let ((_, gx, gy, block, shared), args) = match pick {
            Some(x) => x,
            None => return Ok(Vec::new()),
        };
        let variants = [args];
        let mut out = Vec::new();
        for name in [
            "matmul_iq4_xs_q8_k",
            "dbg_iq4_nounpack",
            "dbg_iq4_nofold",
            "dbg_iq4_noweight",
            "dbg_iq4_notable",
        ] {
            let (us, _) = self.time_launches_2d(name, gx, gy, block, shared, &variants, reps)?;
            out.push((name, us));
        }
        Ok(out)
    }

    /// Replay every launch the run actually made, and time it.
    ///
    /// **Complete by construction.** The previous bench synthesised arguments
    /// from a hand-written table and therefore omitted whatever nobody added to
    /// it — most recently `matmul_iq4_xs_q8_k_moe_glu`, the largest kernel in
    /// the routed FFN, which made its total an undercount of unknown size and
    /// sent a whole line of reasoning the wrong way. Here the entry exists
    /// because the launch happened.
    ///
    /// Safe to run only **after** generation: replaying a kernel re-executes
    /// its writes, so state is corrupted on purpose and nothing may read it
    /// afterwards. Pointers stay valid because weights, mirrors, pool slots and
    /// the expert slab all outlive the pass.
    pub fn bench_launches(&self, reps: u32) -> Result<Vec<LaunchBench>> {
        let recorded: Vec<((&'static str, u32, u32, u32, u32), (u64, Vec<KArg>))> = self
            .launches
            .borrow()
            .iter()
            .map(|(k, v)| (*k, (v.0, v.1.clone())))
            .collect();

        let mut out = Vec::with_capacity(recorded.len());
        for ((kernel, gx, gy, block, shared), (calls, args)) in recorded {
            let variants = [args];
            let (gpu_us, issue_us) =
                self.time_launches_2d(kernel, gx, gy, block, shared, &variants, reps)?;
            out.push(LaunchBench {
                kernel,
                grid: (gx, gy),
                block,
                calls,
                gpu_us,
                issue_us,
            });
        }
        // Most expensive per token first: that is the order in which they matter.
        out.sort_by(|a, b| {
            (b.calls as f64 * b.gpu_us)
                .partial_cmp(&(a.calls as f64 * a.gpu_us))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(out)
    }

    /// Time the shapes this run actually launched — **with no per-launch
    /// synchronize**, which is what `--profile-kernels` cannot do.
    ///
    /// # Why this exists
    ///
    /// `gpu` and `issue` in the pass timings came out at 74.95 and 71.75
    /// ms/token on the 35B, which is ambiguous in the worst way: it is
    /// consistent both with a device saturated by real work and with a device
    /// starved by a host that cannot describe work fast enough. Those call for
    /// opposite fixes — better kernels versus fewer launches — so guessing is
    /// expensive.
    ///
    /// So each shape is measured twice over the same launches:
    ///
    /// * **gpu** — CUDA events either side of the whole batch. Time the device
    ///   spent, per launch, with the queue kept full.
    /// * **issue** — host wall time for the launch loop, which returns as soon
    ///   as the driver accepts each launch. Host cost, per launch.
    ///
    /// Whichever is larger is the one that binds, per shape, and the ratio says
    /// by how much.
    ///
    /// # The grouped column
    ///
    /// Each shape is also run **once** with `group` times the rows, which is
    /// the same arithmetic in one launch instead of `group`. For the routed
    /// FFN that is exactly the change worth costing: 8 experts of `{2048, 512}`
    /// become one `{2048, 4096}`. The difference between `group * gpu` and
    /// `grouped` is what fusing them would return, measured rather than argued.
    ///
    /// Weights are synthetic. These kernels are bandwidth and instruction
    /// bound and do not branch on values — but the f16 scale bytes are set to a
    /// real value anyway, because a NaN scale is the kind of thing that turns
    /// out to matter on some future architecture.
    pub fn bench_shapes(&self, group: usize, tokens: u64, reps: u32) -> Result<Vec<ShapeBench>> {
        let mut want: Vec<((&'static str, usize, usize), u64)> =
            self.shapes.borrow().iter().map(|(k, v)| (*k, *v)).collect();
        // Most-launched first: that is the order in which they matter.
        want.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        let mut out = Vec::with_capacity(want.len());
        for ((kernel, n_in, n_out), calls) in want {
            let one = self.time_shape(kernel, n_in, n_out, reps)?;
            // The grouped form is the same kernel over `group` times the rows.
            // Skipped when it would not fit or would not mean anything.
            // Only meaningful where the shape is launched at least `group`
            // times **per token**, which is what "fuse the experts a token
            // visits" means. Gating on the raw count instead priced fusing
            // eight LM heads — a 2048 x 1,986,560 matmul nobody would build —
            // and that one row put 82 ms into a 7 ms total.
            let per_token = calls as f64 / tokens.max(1) as f64;
            let groupable =
                group > 1 && per_token >= group as f64 && !kernel.ends_with("_moe");
            let grouped = if groupable {
                self.time_shape(kernel, n_in, n_out * group, reps.max(1) / 4 + 1)?.0
            } else {
                one.0
            };
            out.push(ShapeBench {
                kernel,
                n_in,
                n_out,
                calls,
                gpu_us: one.0,
                issue_us: one.1,
                grouped_us: grouped,
                group: if groupable { group } else { 1 },
            });
        }
        Ok(out)
    }

    /// Build the argument list and launch geometry for one shape.
    ///
    /// **Per kernel, by hand, because there is no honest shortcut.** Every
    /// kernel's parameter list is its own, so a generic replay would have to
    /// guess — and a bench that guesses wrong measures a kernel reading
    /// garbage sizes, which is fast and meaningless. The shapes come from the
    /// run; only the plumbing is written down here.
    ///
    /// `n_out == 0` marks an elementwise kernel, for which `n_in` is the
    /// buffer length. Returns `None` for anything not benched, which the
    /// caller reports as `--` rather than as a zero.
    fn shape_args(
        &self,
        kernel: &'static str,
        n_in: usize,
        n_out: usize,
        keep: &mut Vec<DeviceBuffer>,
    ) -> Result<Option<(Vec<KArg>, u32, u32, u32)>> {
        const QK_K: usize = 256;
        // Two f32 buffers of `n` and a couple of quantized outputs cover every
        // elementwise kernel here; they are allocated once and shared.
        let mut fbuf = |n: usize| -> Result<ffi::CUdeviceptr> {
            let v: Vec<f32> = (0..n.max(1)).map(|i| 0.01 + (i % 31) as f32 * 1e-3).collect();
            let b = DeviceBuffer::from_slice(&v)?;
            let p = b.ptr;
            keep.push(b);
            Ok(p)
        };

        let out = match kernel {
            "silu_mul" | "add_assign" | "sigmoid_mul" => {
                let (a, b) = (fbuf(n_in)?, fbuf(n_in)?);
                let args = vec![KArg::I32(n_in as i32), KArg::Ptr(a), KArg::Ptr(b)];
                (args, n_in.div_ceil(256) as u32, 256u32, 0u32)
            }
            "add_scaled" => {
                let (a, b) = (fbuf(n_in)?, fbuf(n_in)?);
                let args = vec![
                    KArg::I32(n_in as i32),
                    KArg::F32(0.125),
                    KArg::Ptr(a),
                    KArg::Ptr(b),
                ];
                (args, n_in.div_ceil(256) as u32, 256, 0)
            }
            "gather_chunks" | "scatter_chunks" => {
                let (src, dst) = (fbuf(n_in * 2)?, fbuf(n_in * 2)?);
                let args = vec![
                    KArg::I32(n_in as i32),
                    KArg::I32(n_in as i32),
                    KArg::I32(n_in as i32),
                    KArg::I32(0),
                    KArg::Ptr(src),
                    KArg::Ptr(dst),
                ];
                (args, n_in.div_ceil(256) as u32, 256, 0)
            }
            "quantize_q8_k" => {
                let n_super = n_in / QK_K;
                let x = fbuf(n_in)?;
                let s = DeviceBuffer::new(n_super * 4)?;
                let q = DeviceBuffer::new(n_super * QK_K)?;
                let b = DeviceBuffer::new(n_super * (QK_K / 16) * 2)?;
                let args = vec![
                    KArg::I32(n_super as i32),
                    KArg::Ptr(x),
                    KArg::Ptr(s.ptr),
                    KArg::Ptr(q.ptr),
                    KArg::Ptr(b.ptr),
                ];
                keep.push(s);
                keep.push(q);
                keep.push(b);
                (args, n_super as u32, QK_K as u32, 0)
            }
            "quantize_q8_0" => {
                let n_blocks = n_in / 32;
                let x = fbuf(n_in)?;
                let s = DeviceBuffer::new(n_blocks * 4)?;
                let q = DeviceBuffer::new(n_blocks * 32)?;
                let args = vec![
                    KArg::I32(n_blocks as i32),
                    KArg::Ptr(x),
                    KArg::Ptr(s.ptr),
                    KArg::Ptr(q.ptr),
                ];
                keep.push(s);
                keep.push(q);
                (args, n_blocks.div_ceil(64) as u32, 64, 0)
            }
            "rms_norm_tree" => {
                let (x, w, o) = (fbuf(n_in)?, fbuf(n_in)?, fbuf(n_in)?);
                let args = vec![
                    KArg::I32(n_in as i32),
                    KArg::Ptr(x),
                    KArg::Ptr(w),
                    KArg::F32(1e-6),
                    KArg::Ptr(o),
                ];
                // One block per row; the recorded shape is a single row.
                (args, 1, 256, 0)
            }
            "matmul_iq4_xs_q8_k_moe" => {
                // Eight experts against one shared activation, which is the
                // gate/up shape. **Eight distinct weights, not one repeated**:
                // the real routed FFN reads eight different experts, and one
                // buffer eight times would be served by L2 rather than VRAM.
                let n_super = n_in / QK_K;
                let row_bytes = n_super * 136;
                let row_set = n_out * row_bytes;
                let mut w = vec![0x11u8; 8 * row_set];
                for c in 0..8 {
                    for r in 0..n_out {
                        for b in 0..n_super {
                            let at = c * row_set + r * row_bytes + b * 136;
                            w[at] = 0x00;
                            w[at + 1] = 0x38;
                        }
                    }
                }
                let wd = DeviceBuffer::from_slice(&w)?;
                let scales: Vec<f32> =
                    (0..n_super).map(|i| 0.01 + (i % 7) as f32 * 1e-3).collect();
                let quants: Vec<i8> = (0..n_in).map(|i| ((i % 251) as i32 - 125) as i8).collect();
                let sd = DeviceBuffer::from_slice(&scales)?;
                let qd = DeviceBuffer::from_slice(&quants)?;
                let od = DeviceBuffer::new(8 * n_out * 4)?;
                let mut args = vec![
                    KArg::I32(n_in as i32),
                    KArg::I32(n_out as i32),
                    KArg::I32(0),
                    KArg::I32(8),
                ];
                for c in 0..8 {
                    args.push(KArg::Ptr(wd.ptr + (c * row_set) as ffi::CUdeviceptr));
                }
                args.push(KArg::Ptr(sd.ptr));
                args.push(KArg::Ptr(qd.ptr));
                args.push(KArg::Ptr(od.ptr));
                let grid = n_out.div_ceil(4) as u32;
                keep.push(wd);
                keep.push(sd);
                keep.push(qd);
                keep.push(od);
                (args, grid, 128, 0)
            }
            _ => return Ok(None),
        };
        let _ = n_out;
        Ok(Some(out))
    }

    /// One shape, `reps` launches, one synchronize. Returns `(gpu, issue)` in
    /// microseconds per launch.
    fn time_shape(
        &self,
        kernel: &'static str,
        n_in: usize,
        n_out: usize,
        reps: u32,
    ) -> Result<(f64, f64)> {
        const QK_K: usize = 256;
        // Everything that is not a plain matmul goes through `shape_args`.
        // The grouped MoE matmul is a matmul by name and not by signature, and
        // leaving it out meant the table silently omitted the single largest
        // kernel in the routed FFN — the same shape of gap that made
        // `resident_bytes` under-report by an order of magnitude.
        let mut keep: Vec<DeviceBuffer> = Vec::new();
        if !kernel.starts_with("matmul_") || kernel.ends_with("_moe") {
            let built = self.shape_args(kernel, n_in, n_out, &mut keep)?;
            let (args, grid, block, shared) = match built {
                Some(b) => b,
                None => return Ok((f64::NAN, f64::NAN)),
            };
            return self.time_launches(kernel, grid, block, shared, &[args], reps);
        }
        // Bytes one row of this format occupies, and the f16 scale's offset
        // within a block. `None` where the format has no f16 to protect.
        let (block_bytes, block_elems, scale_at) = match kernel {
            "matmul_iq4_xs_q8_k" => (136usize, QK_K, Some(0usize)),
            "matmul_q5_k_q8_k" => (176, QK_K, Some(0)),
            "matmul_q6_k_q8_k" => (210, QK_K, Some(208)),
            "matmul_f32" | "matmul_f32_t" => (4, 1, None),
            // Q8_0 has its own benches, and anything else is not a matmul.
            _ => return Ok((f64::NAN, f64::NAN)),
        };
        let n_super = n_in / block_elems;
        let row_bytes = n_super * block_bytes;

        // Enough copies of the weight that consecutive launches cannot all be
        // served by L2. Capped so the LM head, already 398 MB, does not ask for
        // three gigabytes.
        let row_set = n_out * row_bytes;
        let copies = ((64 << 20) / row_set.max(1)).clamp(1, 8);
        let mut w = vec![0x11u8; copies * row_set];
        if let Some(off) = scale_at {
            // 0x3800 is f16 0.5: a real number, so the arithmetic is finite.
            for c in 0..copies {
                for r in 0..n_out {
                    for b in 0..n_super {
                        let at = c * row_set + r * row_bytes + b * block_bytes + off;
                        w[at] = 0x00;
                        w[at + 1] = 0x38;
                    }
                }
            }
        } else {
            for c in w.chunks_exact_mut(4) {
                c.copy_from_slice(&0.01f32.to_le_bytes());
            }
        }

        let wd = DeviceBuffer::from_slice(&w)?;
        let od = DeviceBuffer::new(n_out * 4)?;

        // The activation, in whichever form this kernel dots against.
        let (xs, xq, xb, xf) = if kernel.starts_with("matmul_f32") {
            let x: Vec<f32> = (0..n_in).map(|i| (i % 17) as f32 * 0.01).collect();
            (None, None, None, Some(DeviceBuffer::from_slice(&x)?))
        } else {
            let scales: Vec<f32> = (0..n_super).map(|i| 0.01 + (i % 7) as f32 * 1e-3).collect();
            let quants: Vec<i8> = (0..n_in).map(|i| ((i % 251) as i32 - 125) as i8).collect();
            let bsums: Vec<i16> = (0..n_super * (QK_K / 16)).map(|i| (i % 97) as i16).collect();
            (
                Some(DeviceBuffer::from_slice(&scales)?),
                Some(DeviceBuffer::from_slice(&quants)?),
                Some(DeviceBuffer::from_slice(&bsums)?),
                None,
            )
        };

        let mut variants: Vec<Vec<KArg>> = Vec::with_capacity(copies);
        for c in 0..copies {
            let mut args = vec![
                KArg::I32(n_in as i32),
                KArg::I32(n_out as i32),
                KArg::Ptr(wd.ptr + (c * row_set) as ffi::CUdeviceptr),
            ];
            match (&xf, &xs, &xq, &xb) {
                (Some(x), ..) => args.push(KArg::Ptr(x.ptr)),
                (None, Some(s), Some(q), Some(b)) => {
                    args.push(KArg::Ptr(s.ptr));
                    args.push(KArg::Ptr(q.ptr));
                    if kernel == "matmul_q5_k_q8_k" {
                        args.push(KArg::Ptr(b.ptr));
                    }
                }
                _ => {}
            }
            args.push(KArg::Ptr(od.ptr));
            variants.push(args);
        }

        let block = 128u32;
        let grid = if kernel.starts_with("matmul_f32") {
            n_out.div_ceil(block as usize) as u32
        } else {
            n_out.div_ceil((block / 32) as usize) as u32
        };

        keep.push(wd);
        keep.push(od);
        if let Some(b) = xs { keep.push(b); }
        if let Some(b) = xq { keep.push(b); }
        if let Some(b) = xb { keep.push(b); }
        if let Some(b) = xf { keep.push(b); }
        self.time_launches(kernel, grid, block, 0, &variants, reps)
    }

    /// `reps` launches of one configured kernel, timed two ways: CUDA events
    /// for device microseconds, host wall clock for issue microseconds.
    ///
    /// **One synchronize, at the end.** That is the whole point — a per-launch
    /// sync measures latency where throughput is what matters, and inflates
    /// every kernel in proportion to how often it is called, which is what
    /// makes `--profile-kernels` shares unusable for ranking work.
    /// As above, cycling through `variants` so consecutive launches read
    /// different weights.
    ///
    /// **Replaying one buffer measures the L2, not the card.** This machine has
    /// 32 MB of L2 and a grouped expert matmul reads 4.25 MB, so 300 repeats of
    /// one buffer are 299 cache hits — while the real forward pass streams 11.5
    /// GiB of distinct experts and hits VRAM every time. The tell was that the
    /// LM head, the only shape too large to cache, was also the only one that
    /// measured slow.
    fn time_launches(
        &self,
        kernel: &'static str,
        grid: u32,
        block: u32,
        shared: u32,
        variants: &[Vec<KArg>],
        reps: u32,
    ) -> Result<(f64, f64)> {
        self.time_launches_2d(kernel, grid, 1, block, shared, variants, reps)
    }

    #[allow(clippy::too_many_arguments)]
    fn time_launches_2d(
        &self,
        kernel: &'static str,
        grid_x: u32,
        grid_y: u32,
        block: u32,
        shared: u32,
        variants: &[Vec<KArg>],
        reps: u32,
    ) -> Result<(f64, f64)> {
        // A bench is never part of a graph, and never times its own launches.
        let was_graph = self.pass_graph.replace(false);
        let was_timed = self.time_kernels.replace(false);

        let (mut a, mut b): (ffi::CUevent, ffi::CUevent) =
            (std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: valid out-pointers; flag 0 is the timing-enabled default.
        unsafe {
            check(ffi::cuEventCreate(&mut a, 0), "cuEventCreate")?;
            check(ffi::cuEventCreate(&mut b, 0), "cuEventCreate")?;
        }

        let run = |n: u32| -> Result<(f64, f64)> {
            // SAFETY: `a`/`b` are ours; the null stream is the one everything
            // uses, so the events bracket exactly these launches.
            unsafe { check(ffi::cuEventRecord(a, std::ptr::null_mut()), "cuEventRecord")? };
            let t = std::time::Instant::now();
            for k in 0..n {
                // SAFETY: the caller built `args`, `grid`, `block` and
                // `shared` to match `kernel`, and sized every buffer for the
                // shape it describes.
                let args = &variants[(k as usize) % variants.len()];
                // SAFETY: as above.
                unsafe { self.launch_grid2(kernel, grid_x, grid_y, block, shared, args)? };
            }
            // Host time first: the launch loop returns once the driver has
            // accepted the work, so this is issue cost and not device time.
            let issue = t.elapsed().as_secs_f64() * 1e6 / f64::from(n);
            // SAFETY: as above.
            unsafe {
                check(ffi::cuEventRecord(b, std::ptr::null_mut()), "cuEventRecord")?;
                check(ffi::cuEventSynchronize(b), "cuEventSynchronize")?;
            }
            let mut ms = 0.0f32;
            // SAFETY: both events have completed.
            unsafe {
                check(ffi::cuEventElapsedTime(&mut ms, a, b), "cuEventElapsedTime")?;
            }
            Ok((f64::from(ms) * 1000.0 / f64::from(n), issue))
        };

        run(8)?; // warm the module, the caches and the clocks

        // **Best of several, not the average.** Every error source here only
        // adds time: a stall in the host launch loop, a clock that has not
        // ramped, contention from anything else on the device. WSL's CUDA
        // virtualization makes per-launch host cost spiky in particular, and
        // when the host cannot keep the queue full the event pair measures the
        // stall rather than the kernel — visible as `gpu` collapsing onto
        // `issue`. That was caught by running the same bench twice and finding
        // the affected rows had moved: a 512-element `silu_mul` read 22.4 us
        // once and 13.0 the next time, neither of them credible for 2 KiB.
        //
        // The minimum is the only statistic that is not a function of how busy
        // the machine was, so it is the one reported.
        let mut measured = run(reps.max(1))?;
        for _ in 0..3 {
            let (g, i) = run(reps.max(1))?;
            measured.0 = measured.0.min(g);
            measured.1 = measured.1.min(i);
        }
        let measured = Ok(measured);

        // SAFETY: both events are ours and are no longer in flight.
        unsafe {
            check(ffi::cuEventDestroy_v2(a), "cuEventDestroy")?;
            check(ffi::cuEventDestroy_v2(b), "cuEventDestroy")?;
        }
        self.pass_graph.set(was_graph);
        self.time_kernels.set(was_timed);
        measured
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
        self.note_shape("gather_chunks", out.len(), 0);
        // SAFETY: parameters match `gather_chunks` in kernels.cu; one thread
        // per output element, guarded against the tail.
        unsafe { self.launch_shared("gather_chunks", blocks, 256, 0, &args)? };
        Ok(())
    }

    fn scatter_chunks_impl(
        &self,
        src: &[f32],
        chunk: usize,
        stride: usize,
        offset: usize,
        dst: &mut [f32],
    ) -> Result<()> {
        let sd = self.mirror_in(src)?;
        // `mirror_in`, not `mirror_out`: this writes only the windows it is
        // given, so the rest of `dst` must already be on the device. The MoE
        // fills row `t` of an output whose other rows earlier tokens wrote.
        let dd = self.mirror_in(dst)?;
        let args = [
            KArg::I32(src.len() as i32),
            KArg::I32(chunk as i32),
            KArg::I32(stride as i32),
            KArg::I32(offset as i32),
            KArg::Ptr(sd),
            KArg::Ptr(dd),
        ];
        let blocks = src.len().div_ceil(256) as u32;
        self.note_shape("scatter_chunks", src.len(), 0);
        // SAFETY: parameters match `scatter_chunks` in kernels.cu; one thread
        // per *source* element, guarded against the tail.
        unsafe { self.launch_shared("scatter_chunks", blocks, 256, 0, &args)? };
        self.mirror_out(dst).map(|_| ())
    }

    fn add_scaled_impl(&self, a: &mut [f32], b: &[f32], scale: f32) -> Result<()> {
        let bd = self.mirror_in(b)?;
        let ad = self.mirror_in(a)?;
        let args = [
            KArg::I32(a.len() as i32),
            KArg::F32(scale),
            KArg::Ptr(ad),
            KArg::Ptr(bd),
        ];
        let blocks = a.len().div_ceil(256) as u32;
        self.note_shape("add_scaled", a.len(), 0);
        // SAFETY: parameters match `add_scaled` in kernels.cu.
        unsafe { self.launch_shared("add_scaled", blocks, 256, 0, &args)? };
        self.mirror_out(a).map(|_| ())
    }

    fn add_scaled_sigmoid_impl(
        &self,
        acc: &mut [f32],
        b: &[f32],
        logit: &[f32],
    ) -> Result<()> {
        let ld = self.mirror_in(logit)?;
        let bd = self.mirror_in(b)?;
        let ad = self.mirror_in(acc)?;
        let args = [
            KArg::I32(acc.len() as i32),
            KArg::Ptr(ld),
            KArg::Ptr(ad),
            KArg::Ptr(bd),
        ];
        self.note_shape("add_scaled_sigmoid", acc.len(), 0);
        // SAFETY: parameters match `add_scaled_sigmoid`; `logit` holds at least
        // one float and `acc`/`b` hold `n`.
        unsafe { self.launch_shared("add_scaled_sigmoid", acc.len().div_ceil(256) as u32, 256, 0, &args)? };
        self.mirror_out(acc).map(|_| ())
    }

    fn sigmoid_mul_impl(&self, x: &mut [f32], g: &[f32]) -> Result<()> {
        let gd = self.mirror_in(g)?;
        let xd = self.mirror_in(x)?;
        let args = [KArg::I32(x.len() as i32), KArg::Ptr(xd), KArg::Ptr(gd)];
        let blocks = x.len().div_ceil(256) as u32;
        self.note_shape("sigmoid_mul", x.len(), 0);
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
        self.note_shape("silu_mul", gate.len(), 0);
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
        self.note_shape("add_assign", a.len(), 0);
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

    fn scatter_chunks(
        &self,
        src: &[f32],
        chunk: usize,
        stride: usize,
        offset: usize,
        dst: &mut [f32],
    ) {
        self.note(self.scatter_chunks_impl(src, chunk, stride, offset, dst));
    }

    /// **A device backend must override this**, or the trait default's host
    /// scalar loop reads a buffer the device owns. It is the MoE expert
    /// accumulation, so it runs eight times a layer.
    fn add_scaled(&self, a: &mut [f32], b: &[f32], scale: f32) {
        self.note(self.add_scaled_impl(a, b, scale));
    }

    /// Choose on the device, so the router's probabilities never come home.
    ///
    /// **Always, not conditionally.** Every expert has a permanent device
    /// address from the first sight of its tensor (see
    /// `experts::ExpertCache::table`), so there is no state in which the host
    /// still has to resolve a pick — which is what keeps this backend on one
    /// path rather than two that can disagree.
    fn route(&self, probs: &mut [f32], n_used: usize) -> Route {
        self.note(self.route_impl(probs, n_used));
        Route::Device { n_used }
    }

    fn matmul_experts(&self, w: &Experts<'_>, route: &Route, x: &[f32], out: &mut [f32]) {
        self.note(self.matmul_experts_impl(w, route, x, out));
    }

    fn add_scaled_rows(&self, acc: &mut [f32], rows: &[f32], scales: &[f32]) {
        self.note(self.add_scaled_rows_impl(acc, rows, scales));
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_finish(
        &self,
        out: &mut [f32],
        at: usize,
        n: usize,
        rows: &[f32],
        route: &Route,
        shared: &[f32],
        logit: &[f32],
        logit_at: usize,
    ) {
        self.note(self.moe_finish_impl(out, at, n, rows, route, shared, logit, logit_at));
    }

    fn moe_glu(
        &self,
        gate: &Experts<'_>,
        up: &Experts<'_>,
        route: &Route,
        x: &[f32],
        out: &mut [f32],
        _scratch: &mut [f32],
    ) {
        self.note(self.moe_glu_impl(gate, up, route, x, out));
    }

    fn add_scaled_sigmoid(&self, acc: &mut [f32], b: &[f32], logit: &[f32]) {
        self.note(self.add_scaled_sigmoid_impl(acc, b, logit));
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
        // A read *inside* a pass is the model telling us this pass cannot be a
        // CUDA graph: a graph defers every kernel to `end_pass`, so the
        // download below would return the previous pass's contents. See
        // `Cuda::mid_pass_read`. Latched on the first occurrence, which always
        // happens during the eager warm-up passes, so no graph is ever recorded
        // for such a model rather than one being recorded and silently lying.
        if self.in_pass.get() && !self.mid_pass_read.get() {
            self.mid_pass_read.set(true);
            self.graphs_enabled.set(false);
        }
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
        self.in_pass.set(false);
        self.note(self.graph_end());
        self.note(self.timing_end());
    }

    fn begin_pass(&self, n_tokens: usize) {
        self.in_pass.set(true);
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

    fn scatter_chunks(
        &self,
        src: &[f32],
        chunk: usize,
        stride: usize,
        offset: usize,
        dst: &mut [f32],
    ) {
        (*self).scatter_chunks(src, chunk, stride, offset, dst)
    }

    fn add_scaled(&self, a: &mut [f32], b: &[f32], scale: f32) {
        (*self).add_scaled(a, b, scale)
    }

    fn route(&self, probs: &mut [f32], n_used: usize) -> Route {
        (*self).route(probs, n_used)
    }

    fn matmul_experts(&self, w: &Experts<'_>, route: &Route, x: &[f32], out: &mut [f32]) {
        (*self).matmul_experts(w, route, x, out)
    }

    fn add_scaled_rows(&self, acc: &mut [f32], rows: &[f32], scales: &[f32]) {
        (*self).add_scaled_rows(acc, rows, scales)
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_finish(
        &self,
        out: &mut [f32],
        at: usize,
        n: usize,
        rows: &[f32],
        route: &Route,
        shared: &[f32],
        logit: &[f32],
        logit_at: usize,
    ) {
        (*self).moe_finish(out, at, n, rows, route, shared, logit, logit_at)
    }

    fn moe_glu(
        &self,
        gate: &Experts<'_>,
        up: &Experts<'_>,
        route: &Route,
        x: &[f32],
        out: &mut [f32],
        scratch: &mut [f32],
    ) {
        (*self).moe_glu(gate, up, route, x, out, scratch)
    }

    fn add_scaled_sigmoid(&self, acc: &mut [f32], b: &[f32], logit: &[f32]) {
        (*self).add_scaled_sigmoid(acc, b, logit)
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
