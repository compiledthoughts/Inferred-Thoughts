//! Qwen Sparse Attention past its budget on the device (QSA Q2,
//! `src/model/qwen4exp.md`). Kernels in `kernels/qsa.cuh`; nothing the 35B runs
//! calls these.

use super::slot;
use crate::error::{Error, Result};
use crate::ops::cuda::{Cuda, DeviceBuffer, KArg, ffi};
use crate::ops::{Attn, QsaPool, QsaSelect};

impl Cuda {
    /// A device buffer the device owns, keyed on the host buffer's address and
    /// sized to it: a layer's pooled-key lane, or a pass's selected cells. Never
    /// uploaded and never brought home but by [`Cuda::read_cells`]; a size change
    /// replaces it, zeroed, which is safe because every reader is preceded by a
    /// writer covering what it reads (the watermark for a lane, `qsa_select` for
    /// cells).
    fn qsa_buf<T: Copy>(&self, host: &[T]) -> Result<ffi::CUdeviceptr> {
        let (key, bytes) = (host.as_ptr() as usize, std::mem::size_of_val(host));
        let mut map = self.qsa_bufs.borrow_mut();
        if let Some(b) = map.get(&key)
            && b.len_bytes() == bytes
        {
            return Ok(b.ptr);
        }
        let b = DeviceBuffer::new(bytes)?;
        let ptr = b.ptr;
        map.insert(key, b);
        Ok(ptr)
    }

    /// The RoPE table for pooled keys: row `b` holds `rope_neox`'s cos and sin at
    /// position `b * ratio`, built in f64 on the host as the oracle does. The
    /// positions never change, so it is built once for the largest lane and
    /// never re-uploaded — nothing crosses the bus per pass.
    fn qsa_rope(
        &self,
        n_blocks: usize,
        ratio: usize,
        n_rot: usize,
        theta: f32,
    ) -> Result<(ffi::CUdeviceptr, ffi::CUdeviceptr)> {
        let key = (ratio, n_rot, theta.to_bits());
        let mut t = self.qsa_rope.borrow_mut();
        if let Some((k, n, c, s)) = t.as_ref()
            && *k == key
            && *n >= n_blocks
        {
            return Ok((c.ptr, s.ptr));
        }
        let half = n_rot / 2;
        let n = n_blocks.max(1);
        let (mut cos, mut sin) = (Vec::with_capacity(n * half), Vec::with_capacity(n * half));
        for b in 0..n {
            for i in 0..half {
                let freq = (theta as f64).powf(-2.0 * i as f64 / n_rot as f64);
                let (sn, cs) = ((b * ratio) as f64 * freq).sin_cos();
                cos.push(cs as f32);
                sin.push(sn as f32);
            }
        }
        let (c, s) = (DeviceBuffer::from_slice(&cos)?, DeviceBuffer::from_slice(&sin)?);
        let ptrs = (c.ptr, s.ptr);
        *t = Some((key, n, c, s));
        Ok(ptrs)
    }

    pub(super) fn qsa_pool_impl(&self, raw: &[u16], pooled: &mut [f32], p: &QsaPool<'_>) -> Result<()> {
        let n_blocks = pooled.len() / p.dim.max(1);
        if p.to > n_blocks || p.norm.len() != p.dim || p.n_rot > p.dim || p.to * p.ratio * p.dim > raw.len() {
            return Err(Error::InconsistentArchitecture {
                what: "qsa_pool",
                detail: format!(
                    "blocks {}..{} of {n_blocks}, dim {}, n_rot {}, norm {}, raw {}",
                    p.from,
                    p.to,
                    p.dim,
                    p.n_rot,
                    p.norm.len(),
                    raw.len()
                ),
            });
        }
        let n_new = p.to.saturating_sub(p.from);
        let rd = self.kv_resident(raw, p.to * p.ratio, p.dim)?;
        let pd = self.qsa_buf(pooled)?;
        let wd = self.resident(p.norm)?;
        let (cd, sd) = self.qsa_rope(n_blocks, p.ratio, p.n_rot, p.theta)?;
        let args = [
            KArg::I32(p.from as i32),
            KArg::I32(n_new as i32),
            KArg::I32(p.ratio as i32),
            KArg::I32(p.dim as i32),
            KArg::I32((p.n_rot / 2) as i32),
            KArg::F32(1.0 / p.ratio as f32),
            KArg::F32(p.eps),
            KArg::Ptr(rd),
            KArg::Ptr(pd),
            KArg::Ptr(wd),
            KArg::Ptr(cd),
            KArg::Ptr(sd),
        ];
        let block = 128u32;
        let grid = n_new.max(1).div_ceil(block as usize) as u32;
        // SAFETY: parameters match `qsa_pool`; every block written is inside the
        // lane and every cell read inside the raw lane (checked above), and the
        // table has a row per lane block. A grid of one block when `n_new` is 0,
        // so decode's kernel sequence does not depend on it.
        unsafe { self.launch_grid2("qsa_pool", grid, 1, block, 0, &args) }
    }

    pub(super) fn qsa_select_impl(
        &self,
        q: &[f32],
        pooled: &[f32],
        sel: &QsaSelect,
        scores: &mut [f32],
        cells: &mut [u32],
    ) -> Result<()> {
        let n_q = q.len() / (sel.n_head * sel.dim).max(1);
        if scores.len() != n_q * sel.n_blocks.max(1)
            || cells.len() != n_q * sel.stride()
            || pooled.len() < sel.n_blocks * sel.dim
        {
            return Err(Error::InconsistentArchitecture {
                what: "qsa_select",
                detail: format!(
                    "{n_q} rows, {} blocks, stride {}: scores {}, cells {}, pooled {}",
                    sel.n_blocks,
                    sel.stride(),
                    scores.len(),
                    cells.len(),
                    pooled.len()
                ),
            });
        }
        let qd = self.mirror_in(q)?;
        let pd = self.qsa_buf(pooled)?;
        let sd = self.mirror_out(scores)?;
        let cd = self.qsa_buf(cells)?;
        let block = 128u32;
        let args = [
            KArg::I32(n_q as i32),
            KArg::I32(sel.n_blocks as i32),
            KArg::I32(sel.n_head as i32),
            KArg::I32(sel.dim as i32),
            KArg::Ptr(qd),
            KArg::Ptr(pd),
            KArg::Ptr(sd),
        ];
        let grid = sel.n_blocks.max(1).div_ceil(block as usize) as u32;
        // SAFETY: parameters match `qsa_scores`; the grid covers every (block,
        // row) pair, at least one block, and the kernel guards both axes.
        unsafe { self.launch_grid2("qsa_scores", grid, n_q as u32, block, 0, &args)? };
        let args = [
            KArg::I32(n_q as i32),
            KArg::I32(sel.n_blocks.max(1) as i32),
            KArg::I32(sel.start_pos as i32),
            KArg::I32(sel.ratio as i32),
            KArg::I32(sel.budget as i32),
            KArg::I32(sel.stride() as i32),
            KArg::Ptr(sd),
            KArg::Ptr(cd),
        ];
        let grid = n_q.div_ceil(block as usize) as u32;
        // SAFETY: parameters match `qsa_select_cells`; one thread per row, the
        // scores stride is the buffer's (`n_blocks.max(1)`, as sized above), a row
        // reads only blocks whole at its position (all below `n_blocks`), and it
        // writes at most `stride` cells (`QsaSelect::stride`).
        unsafe { self.launch_grid2("qsa_select_cells", grid, 1, block, 0, &args) }
    }

    /// Sparse attention as a gathered window per query row: the row's kept
    /// cells' K and V copied into a dense window, then the decode attention path
    /// unchanged over it (`src/model/qwen4exp.md`, decided 16-09 by measurement).
    /// Decode is one row: `qsa_gather_kv`, `attn_decode`, `attn_flash_combine`,
    /// whatever the depth. A prefill runs the same three per row.
    pub(super) fn attend_sparse_impl(
        &self,
        a: &Attn<'_>,
        cells: &[u32],
        sel: &QsaSelect,
        out: &mut [f32],
    ) -> Result<()> {
        const CHUNK: usize = 128;
        let (n_q, per_row, stride) = (a.n_q(), a.n_head * a.head_dim, sel.stride());
        if cells.len() != n_q * stride || out.len() != n_q * per_row {
            return Err(Error::InconsistentArchitecture {
                what: "attend_sparse",
                detail: format!("{n_q} rows of stride {stride}: cells {}, out {}", cells.len(), out.len()),
            });
        }
        let kd = self.kv_resident(a.k, a.n_pos, a.kv_dim)?;
        let vd = self.kv_resident(a.v, a.n_pos, a.kv_dim)?;
        let qd = self.mirror_in(a.q)?;
        let od = self.mirror_out(out)?;
        let cd = self.qsa_buf(cells)?;
        let kw = self.pooled(slot::QSA_KW, stride * a.kv_dim * 2)?;
        let vw = self.pooled(slot::QSA_VW, stride * a.kv_dim * 2)?;
        let n_split = stride.div_ceil(CHUNK);
        let pa = self.pooled(slot::SCORES, a.n_head * n_split * a.head_dim * 4)?;
        let pm = self.pooled(slot::PART_M, a.n_head * n_split * 4)?;
        let pl = self.pooled(slot::PART_L, a.n_head * n_split * 4)?;
        let block = 128u32;
        for t in 0..n_q {
            let count = sel.count(a.n_pos_of(t) - 1);
            let args = [
                KArg::I32(count as i32),
                KArg::I32(a.kv_dim as i32),
                KArg::Ptr(cd + (t * stride * 4) as u64),
                KArg::Ptr(kd),
                KArg::Ptr(vd),
                KArg::Ptr(kw),
                KArg::Ptr(vw),
            ];
            let grid = count.div_ceil(block as usize) as u32;
            // SAFETY: parameters match `qsa_gather_kv`; row `t`'s cells are all
            // below its position, hence inside the `n_pos` rows made resident,
            // and both windows hold `stride >= count` rows.
            unsafe { self.launch_grid2("qsa_gather_kv", grid, 1, block, 0, &args)? };
            // One query row over its window: `attend_rows` reads only the shape
            // of `Attn` (the row count from `q`'s length, `n_pos` as the window)
            // and the device pointers it is handed.
            let row = Attn {
                q: &a.q[..per_row],
                k: a.k,
                v: a.v,
                kv_dim: a.kv_dim,
                n_pos: count,
                head_dim: a.head_dim,
                n_head: a.n_head,
                n_head_kv: a.n_head_kv,
                scale: a.scale,
            };
            let (qt, ot) = (qd + (t * per_row * 4) as u64, od + (t * per_row * 4) as u64);
            self.attend_rows(&row, 0, 1, CHUNK, qt, kw, vw, ot, pa, pm, pl)?;
        }
        Ok(())
    }

    /// **Debug readback**: the cells `qsa_select` wrote for `cells`, brought home.
    /// For bisection only — it synchronizes, so a caller runs with graphs off
    /// (`the_0_2b_past_the_budget_is_bit_identical_with_the_expf_ops_on_the_cpu`
    /// uses it to hand the device's selection to the CPU's attention).
    pub fn read_cells(&self, cells: &[u32]) -> Result<Vec<u32>> {
        self.read_qsa(cells)
    }

    /// **Debug readback** of a pooled-key lane, as [`Cuda::read_cells`].
    pub fn read_pooled(&self, lane: &[f32]) -> Result<Vec<f32>> {
        self.read_qsa(lane)
    }

    fn read_qsa<T: Copy + Default>(&self, host: &[T]) -> Result<Vec<T>> {
        self.sync()?;
        let map = self.qsa_bufs.borrow();
        let b = map
            .get(&(host.as_ptr() as usize))
            .filter(|b| b.len_bytes() == std::mem::size_of_val(host))
            .ok_or_else(|| Error::Cuda {
                what: "qsa readback",
                detail: "the device holds no QSA buffer for this host buffer".to_string(),
            })?;
        let mut out = vec![T::default(); host.len()];
        b.read(&mut out)?;
        Ok(out)
    }

    /// Price the gather arm of the Q2 decode fork: device microseconds for one
    /// `qsa_gather_kv` launch copying `cells` rows of K and V (f16, `kv_dim`
    /// each) out of `k` and `v` into a dense window, best of several batches of
    /// `reps` (`Cuda::time_launches_2d`). Returns `(device us, host issue us)`.
    ///
    /// Before timing, one launch is read back and checked against the host
    /// cache, so a bench of a kernel that copies the wrong rows fails instead.
    pub fn bench_qsa_gather(
        &self,
        k: &[u16],
        v: &[u16],
        kv_dim: usize,
        cells: &[u32],
        reps: u32,
    ) -> Result<(f64, f64)> {
        let n_pos = k.len() / kv_dim.max(1);
        if k.len() != v.len() || !k.len().is_multiple_of(kv_dim.max(1)) || cells.iter().any(|&c| c as usize >= n_pos) {
            return Err(Error::InconsistentArchitecture {
                what: "bench_qsa_gather",
                detail: format!("{} cells over a {n_pos}-row cache of {kv_dim}", cells.len()),
            });
        }
        let kd = DeviceBuffer::from_slice(k)?;
        let vd = DeviceBuffer::from_slice(v)?;
        let cd = DeviceBuffer::from_slice(cells)?;
        let kw = DeviceBuffer::new(cells.len() * kv_dim * 2)?;
        let vw = DeviceBuffer::new(cells.len() * kv_dim * 2)?;
        let args = vec![
            KArg::I32(cells.len() as i32),
            KArg::I32(kv_dim as i32),
            KArg::Ptr(cd.ptr),
            KArg::Ptr(kd.ptr),
            KArg::Ptr(vd.ptr),
            KArg::Ptr(kw.ptr),
            KArg::Ptr(vw.ptr),
        ];
        let block = 128u32;
        let grid = cells.len().div_ceil(block as usize) as u32;

        // SAFETY: parameters match `qsa_gather_kv`; the grid covers exactly
        // `cells.len()` window rows, every cell was checked to be inside the
        // cache, and both windows hold `cells.len() * kv_dim` f16 values.
        unsafe { self.launch_grid2("qsa_gather_kv", grid, 1, block, 0, &args)? };
        self.sync()?;
        let (mut gk, mut gv) = (vec![0u16; cells.len() * kv_dim], vec![0u16; cells.len() * kv_dim]);
        kw.read(&mut gk)?;
        vw.read(&mut gv)?;
        for (i, &c) in cells.iter().enumerate() {
            let (s, d) = (c as usize * kv_dim, i * kv_dim);
            if gk[d..d + kv_dim] != k[s..s + kv_dim] || gv[d..d + kv_dim] != v[s..s + kv_dim] {
                return Err(Error::Cuda {
                    what: "qsa_gather_kv",
                    detail: format!("window row {i} is not cache row {c}"),
                });
            }
        }
        self.time_launches_2d("qsa_gather_kv", grid, 1, block, 0, &[args], reps)
    }
}
