//! RMSNorm over rows and over heads, and the per-head L2 norm.

use crate::error::Result;
use crate::ops::cuda::{Cuda, KArg};

impl Cuda {
    pub(super) fn rms_norm_impl(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) -> Result<()> {
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

    pub(super) fn rms_norm_heads_impl(
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

    pub(super) fn l2_norm_heads_impl(&self, x: &mut [f32], head_dim: usize, eps: f32) -> Result<()> {
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
}
