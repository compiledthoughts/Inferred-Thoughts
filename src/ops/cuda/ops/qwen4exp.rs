//! qwen4exp's hyper-connection and PLE ops: the broadcast multiplies, the two
//! elementwise activations, the row dot, PLE's gate and its dilated conv.
//!
//! Kernels in `kernels/qwen4exp.cuh`; the oracle is each op's scalar trait
//! default in `crate::ops`. Nothing the 35B runs calls these.

use crate::error::{Error, Result};
use crate::ops::cuda::{Cuda, KArg};

impl Cuda {
    pub(super) fn mul_rows_impl(&self, x: &mut [f32], w: &[f32]) -> Result<()> {
        if x.is_empty() {
            return Ok(());
        }
        if w.is_empty() || x.len() % w.len() != 0 {
            return Err(Error::InconsistentArchitecture {
                what: "mul_rows",
                detail: format!("{} elements are not whole rows of a {}-wide weight", x.len(), w.len()),
            });
        }
        // The norm weights are model-owned vectors that never change: one upload.
        let wd = self.resident(w)?;
        let xd = self.mirror_in(x)?;
        let args = [KArg::I32(x.len() as i32), KArg::I32(w.len() as i32), KArg::Ptr(xd), KArg::Ptr(wd)];
        self.note_shape("mul_rows", x.len(), 0);
        // SAFETY: parameters match `mul_rows`; one thread per element of `x`,
        // guarded against the tail, and `i % n_w` stays inside `w`.
        unsafe { self.launch_shared("mul_rows", x.len().div_ceil(256) as u32, 256, 0, &args)? };
        self.mirror_out(x).map(|_| ())
    }

    /// `silu_f32` or `sigmoid_f32`, which share a signature.
    pub(super) fn activation_impl(&self, name: &'static str, x: &mut [f32]) -> Result<()> {
        if x.is_empty() {
            return Ok(());
        }
        let xd = self.mirror_in(x)?;
        let args = [KArg::I32(x.len() as i32), KArg::Ptr(xd)];
        self.note_shape(name, x.len(), 0);
        // SAFETY: parameters match `silu_f32` and `sigmoid_f32`; one thread per
        // element, guarded against the tail.
        unsafe { self.launch_shared(name, x.len().div_ceil(256) as u32, 256, 0, &args)? };
        self.mirror_out(x).map(|_| ())
    }

    pub(super) fn mul_streams_impl(&self, out: &mut [f32], h: &[f32], w: &[f32], n_stream: usize) -> Result<()> {
        let n = w.len() / n_stream.max(1);
        let nd = h.len() / n.max(1);
        if n == 0 || nd == 0 || w.len() != n * n_stream || h.len() != n * nd || out.len() != n * n_stream * nd {
            return Err(Error::InconsistentArchitecture {
                what: "mul_streams",
                detail: format!(
                    "out {} from h {} and w {} over {n_stream} streams",
                    out.len(),
                    h.len(),
                    w.len()
                ),
            });
        }
        let hd = self.mirror_in(h)?;
        let wd = self.mirror_in(w)?;
        let od = self.mirror_out(out)?;
        let args = [
            KArg::I32(out.len() as i32),
            KArg::I32(nd as i32),
            KArg::I32(n_stream as i32),
            KArg::Ptr(hd),
            KArg::Ptr(wd),
            KArg::Ptr(od),
        ];
        self.note_shape("mul_streams", out.len(), 0);
        // SAFETY: parameters match `mul_streams`; one thread per output element,
        // guarded against the tail, and the checks above bound every index.
        unsafe { self.launch_shared("mul_streams", out.len().div_ceil(256) as u32, 256, 0, &args)? };
        Ok(())
    }

    pub(super) fn row_dot_impl(&self, a: &[f32], b: &[f32], width: usize, out: &mut [f32]) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        if a.len() != b.len() || a.len() != out.len() * width {
            return Err(Error::InconsistentArchitecture {
                what: "row_dot",
                detail: format!("a {} and b {} are not {} rows of {width}", a.len(), b.len(), out.len()),
            });
        }
        let ad = self.mirror_in(a)?;
        let bd = self.mirror_in(b)?;
        let od = self.mirror_out(out)?;
        let args = [
            KArg::I32(out.len() as i32),
            KArg::I32(width as i32),
            KArg::Ptr(ad),
            KArg::Ptr(bd),
            KArg::Ptr(od),
        ];
        self.note_shape("row_dot", out.len(), width);
        // SAFETY: parameters match `row_dot`; one thread per row, guarded
        // against the tail, each reading `width` elements of a row inside `a`/`b`.
        unsafe { self.launch_shared("row_dot", out.len().div_ceil(256) as u32, 256, 0, &args)? };
        Ok(())
    }

    pub(super) fn signed_sqrt_sigmoid_impl(&self, s: &mut [f32]) -> Result<()> {
        self.activation_impl("signed_sqrt_sigmoid", s)
    }

    pub(super) fn dilated_conv_impl(
        &self,
        state: &mut [f32],
        x: &[f32],
        weight: &[f32],
        kernel: usize,
        dilation: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let nc = weight.len() / kernel.max(1);
        let hist = kernel.saturating_sub(1) * dilation;
        let n_tok = x.len() / nc.max(1);
        let consistent = kernel >= 2
            && dilation > 0
            && nc > 0
            && n_tok > 0
            && weight.len() == nc * kernel
            && x.len() == n_tok * nc
            && out.len() == x.len()
            && state.len() == nc * hist;
        if !consistent {
            return Err(Error::InconsistentArchitecture {
                what: "dilated_conv",
                detail: format!(
                    "state {} x {} weight {} out {} at kernel {kernel}, dilation {dilation}",
                    state.len(),
                    x.len(),
                    weight.len(),
                    out.len()
                ),
            });
        }
        // The history is written by kernels, so the device copy is authoritative
        // after first touch, as for GatedDeltaNet's conv; `forget_state` makes a
        // host reset visible.
        let sd = self.state_resident(state)?;
        let xd = self.mirror_in(x)?;
        let wd = self.resident(weight)?;
        let od = self.mirror_out(out)?;
        let blocks = nc.div_ceil(256) as u32;

        // Two launches at every batch size, decode included, so a pass's kernel
        // sequence does not change with its length.
        let args = [
            KArg::I32(nc as i32),
            KArg::I32(kernel as i32),
            KArg::I32(dilation as i32),
            KArg::I32(n_tok as i32),
            KArg::Ptr(sd),
            KArg::Ptr(xd),
            KArg::Ptr(wd),
            KArg::Ptr(od),
        ];
        self.note_shape("dilated_conv", nc, n_tok);
        // SAFETY: parameters match `dilated_conv`; the grid is one thread per
        // channel by `n_tok` tokens, guarded against the tail, and every tap
        // index is inside the state (`p < 0`, `hist + p >= 0` since the tap
        // reaches back at most `hist`) or inside `x`.
        unsafe { self.launch_grid2("dilated_conv", blocks, n_tok as u32, 256, 0, &args)? };
        let sargs = [
            KArg::I32(nc as i32),
            KArg::I32(kernel as i32),
            KArg::I32(dilation as i32),
            KArg::I32(n_tok as i32),
            KArg::Ptr(sd),
            KArg::Ptr(xd),
        ];
        // SAFETY: parameters match `dilated_conv_state`; one thread per channel,
        // issued after the output launch on the same stream, so every output has
        // read the old history before this rewrites it.
        unsafe { self.launch_shared("dilated_conv_state", blocks, 256, 0, &sargs)? };
        Ok(())
    }
}
