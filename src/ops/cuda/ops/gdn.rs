//! GatedDeltaNet: the causal convolution and the delta rule, per token and
//! batched.

use crate::error::{Error, Result};
use crate::ops::Delta;
use crate::ops::cuda::{Cuda, KArg, SHARED_OPT_IN_BYTES};

impl Cuda {
    pub(super) fn ssm_conv_impl(
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

        // **The per-token loop was never a data dependency.** This is a causal
        // depthwise convolution: token `t` reads a fixed window of the samples
        // before it and feeds nothing to `t+1`. Only the *state shift* the
        // single-token kernel does after each output forced the ordering.
        //
        // Split apart, the outputs are independent and the state is written
        // once, after every output has read the old one — two launches instead
        // of `n_tok`. Measured at 7.8% of a 4,000-token prefill across 120,090
        // launches before this.
        //
        // Gated on `n_tok > 1` so decode runs the identical kernel it always
        // has, and `keep <= 8` because `ssm_conv_state` stages the new window
        // in registers.
        let keep = kernel - 1;
        if n_tok > 1 && keep <= 8 {
            let args = [
                KArg::I32(nc as i32),
                KArg::I32(kernel as i32),
                KArg::I32(n_tok as i32),
                KArg::Ptr(sd),
                KArg::Ptr(xd),
                KArg::Ptr(w),
                KArg::Ptr(od),
            ];
            // SAFETY: parameters match `ssm_conv_batch`; the grid is one
            // thread per channel by `n_tok` tokens, guarded against the tail,
            // and every index stays inside buffers sized `n_tok * nc`.
            unsafe { self.launch_grid2("ssm_conv_batch", blocks, n_tok as u32, 256, 0, &args)? };
            let sargs = [
                KArg::I32(nc as i32),
                KArg::I32(kernel as i32),
                KArg::I32(n_tok as i32),
                KArg::Ptr(sd),
                KArg::Ptr(xd),
            ];
            // SAFETY: parameters match `ssm_conv_state`; one thread per
            // channel, and it runs after the launch above on the same stream,
            // which is what makes reading the old window safe.
            unsafe { self.launch_shared("ssm_conv_state", blocks, 256, 0, &sargs)? };
            return Ok(());
        }

        // Row offsets go on the device pointers rather than by sub-slicing `x`
        // on the host, because this backend keys its mirrors on host addresses
        // and a sub-slice would be uploaded from a stale copy.
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

    pub(super) fn delta_rule_impl(&self, d: &Delta<'_>, state: &mut [f32], out: &mut [f32]) -> Result<()> {
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

        // **The token order is sequential; the heads are not.** Token `t`'s
        // rank-1 correction is token `t+1`'s stored state, which is why this
        // ran one launch per token. But `state` is per value head and no block
        // reads another block's, so the ordering belongs *inside* a block and
        // the heads stay a grid dimension. `delta_rule_batch` moves the loop
        // into the kernel and issues one launch for the whole batch.
        //
        // Measured on the 35B at 11,237 tokens: 337,140 launches, 30 GDN layers
        // times every token, ~14% of prefill in kernel time and ~8% more in
        // launch overhead alone. Gated on `n_tokens > 1` like every other
        // prefill-only path here, so decode runs the identical kernel it always
        // has and a recorded graph never sees this name.
        // **The state, not the launch count, is what this kernel costs.** It
        // is `head_k_dim * head_v_dim` floats per head, read *and written* once
        // per token by the global-memory form: 2.1 GB in 13.1 ms at the 35B's
        // 128x128 and a 512-token batch, which is 164 GB/s and 37% of the bus.
        // Staged in shared it moves twice for the whole batch instead.
        //
        // Padded by one float per row so a warp walking rows hits 32 distinct
        // banks rather than one. Needs more than the 48 KiB default, which
        // `cached_function` opts into; the check is against what the driver
        // actually grants, so a device that refuses falls through rather than
        // failing to launch.
        let padded = d.head_v_dim * (d.head_k_dim + 1) + 2 * d.head_k_dim;
        let shared_state = (padded * 4) as u32;
        if d.n_tokens() > 1 && !self.delta_seq.get() && shared_state <= SHARED_OPT_IN_BYTES as u32 {
            let args = [
                KArg::I32(d.head_k_dim as i32),
                KArg::I32(d.head_v_dim as i32),
                KArg::I32(d.n_k_heads as i32),
                KArg::I32(d.n_tokens() as i32),
                KArg::F32(d.scale()),
                KArg::Ptr(q),
                KArg::Ptr(k),
                KArg::Ptr(v),
                KArg::Ptr(alpha),
                KArg::Ptr(beta),
                KArg::Ptr(ssm_a),
                KArg::Ptr(dt),
                KArg::Ptr(sd),
                KArg::Ptr(od),
            ];
            // SAFETY: parameters match `delta_rule_batch_shared`; one block per
            // value head, and `shared_state` is the two staged vectors plus the
            // padded per-head state, checked against the opt-in cap above.
            unsafe {
                self.launch_shared(
                    "delta_rule_batch_shared",
                    d.n_v_heads as u32,
                    threads,
                    shared_state,
                    &args,
                )?
            };
            return Ok(());
        }

        if d.n_tokens() > 1 && !self.delta_seq.get() {
            let args = [
                KArg::I32(d.head_k_dim as i32),
                KArg::I32(d.head_v_dim as i32),
                KArg::I32(d.n_k_heads as i32),
                KArg::I32(d.n_tokens() as i32),
                KArg::F32(d.scale()),
                KArg::Ptr(q),
                KArg::Ptr(k),
                KArg::Ptr(v),
                KArg::Ptr(alpha),
                KArg::Ptr(beta),
                KArg::Ptr(ssm_a),
                KArg::Ptr(dt),
                KArg::Ptr(sd),
                KArg::Ptr(od),
            ];
            // SAFETY: parameters match `delta_rule_batch` in kernels.cu; one
            // block per value head, `shared` is the two staged vectors, and the
            // kernel derives every per-token offset from buffers the model
            // sized for `n_tokens`.
            unsafe {
                self.launch_shared("delta_rule_batch", d.n_v_heads as u32, threads, shared, &args)?
            };
            return Ok(());
        }

        // Decode, unchanged: one token, one launch. Offsets go on the device
        // pointers for the same reason as in `ssm_conv_impl`.
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
