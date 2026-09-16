//! Qwen Sparse Attention past its budget on the device (QSA Q2,
//! `src/model/qwen4exp.md`). Kernels in `kernels/qsa.cuh`; nothing the 35B runs
//! calls these.

use crate::error::{Error, Result};
use crate::ops::cuda::{Cuda, DeviceBuffer, KArg};

impl Cuda {
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
        if k.len() != v.len() || k.len() % kv_dim.max(1) != 0 || cells.iter().any(|&c| c as usize >= n_pos) {
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
