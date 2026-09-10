//! Elementwise and per-row kernels: RoPE, softmax, the gated activations, the
//! residual adds, and the chunk gather and scatter.

use super::slot;
use crate::error::Result;
use crate::ops::cuda::{Cuda, KArg};

impl Cuda {
    pub(super) fn rope_impl(
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

    pub(super) fn softmax_impl(&self, x: &mut [f32], row: usize) -> Result<()> {
        if row == 0 {
            return Ok(());
        }
        let n_rows = x.len() / row;
        let xd = self.mirror_in(x)?;

        let args = [KArg::I32(row as i32), KArg::I32(n_rows as i32), KArg::Ptr(xd)];
        let block = 64u32;
        // SAFETY: parameters match `softmax_rows`; one thread per row, and the
        // kernel returns on `r >= n_rows`.
        unsafe {
            self.launch("softmax_rows", (n_rows as u32).div_ceil(block), block, &args)?
        };
        self.mirror_out(x).map(|_| ())
    }

    pub(super) fn gather_chunks_impl(
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

    pub(super) fn scatter_chunks_impl(
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

    pub(super) fn add_scaled_impl(&self, a: &mut [f32], b: &[f32], scale: f32) -> Result<()> {
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

    pub(super) fn add_scaled_sigmoid_impl(
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

    pub(super) fn sigmoid_mul_impl(&self, x: &mut [f32], g: &[f32]) -> Result<()> {
        let gd = self.mirror_in(g)?;
        let xd = self.mirror_in(x)?;
        let args = [KArg::I32(x.len() as i32), KArg::Ptr(xd), KArg::Ptr(gd)];
        let blocks = x.len().div_ceil(256) as u32;
        self.note_shape("sigmoid_mul", x.len(), 0);
        // SAFETY: parameters match `sigmoid_mul` in kernels.cu.
        unsafe { self.launch_shared("sigmoid_mul", blocks, 256, 0, &args)? };
        self.mirror_out(x).map(|_| ())
    }

    pub(super) fn silu_mul_impl(&self, gate: &mut [f32], up: &[f32]) -> Result<()> {
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

    pub(super) fn add_assign_impl(&self, a: &mut [f32], b: &[f32]) -> Result<()> {
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
