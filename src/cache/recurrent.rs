//! Recurrent state for GatedDeltaNet layers.
//!
//! The counterpart to [`super::KvCache`], and the reason a 262,144-token
//! context is plausible on this hardware at all: **this does not grow with
//! context.** A GDN layer carries a fixed `head_k_dim x head_v_dim` matrix per
//! value head plus a few past convolution inputs, and that is the whole of its
//! history no matter how long the sequence gets. In the 35B, 30 of 40 layers
//! are GDN, so only 10 layers hold a KV cache at all.
//!
//! Two pieces per layer:
//!
//! * **conv state** — `(kernel - 1) x conv_dim` floats, the sliding window the
//!   depthwise convolution needs. Oldest sample first, which is the order
//!   `build_conv_state` produces by concatenating the stored state and then
//!   this token along the time axis and keeping the last `kernel - 1` entries.
//! * **SSM state** — `n_v_heads x head_k_dim x head_v_dim` floats, the
//!   associative memory the delta rule reads and corrects. Key axis contiguous,
//!   matching what ggml calls `ne[0]`.
//!
//! Stored as **f32, not f16**, which is a deliberate difference from
//! [`super::KvCache`]. That cache is f16 because llama.cpp computes attention
//! over f16-rounded K and V, so matching it *reduces* divergence. Nothing
//! rounds this: `ggml_ssm_conv` and the delta rule read and write f32 state, so
//! rounding here would be divergence we invented. It is also recurrent — every
//! token feeds the next — so an error would compound rather than stay local
//! the way one rounded key does.

use crate::error::{Error, Result};

/// Per-layer conv and SSM state for one sequence.
pub struct RecurrentState {
    conv: Vec<f32>,
    ssm: Vec<f32>,
    conv_len: usize,
    ssm_len: usize,
    n_layer: usize,
}

impl RecurrentState {
    /// `conv_len` and `ssm_len` are per layer.
    ///
    /// Attention layers get slots too, even though they never use them, so
    /// indexing is by absolute layer number. Packing only the recurrent layers
    /// would save a quarter of a few megabytes and introduce a second layer
    /// numbering to get wrong.
    pub fn new(n_layer: usize, conv_len: usize, ssm_len: usize) -> Self {
        Self {
            conv: vec![0.0; n_layer * conv_len],
            ssm: vec![0.0; n_layer * ssm_len],
            conv_len,
            ssm_len,
            n_layer,
        }
    }

    pub fn n_layer(&self) -> usize {
        self.n_layer
    }

    pub fn conv_len(&self) -> usize {
        self.conv_len
    }

    pub fn ssm_len(&self) -> usize {
        self.ssm_len
    }

    /// Bytes held, for the profiler and the placement budget.
    pub fn capacity_bytes(&self) -> u64 {
        ((self.conv.len() + self.ssm.len()) * std::mem::size_of::<f32>()) as u64
    }

    /// Zero everything. A fresh sequence starts from an empty memory, which for
    /// the delta rule means a state predicting zero for every key.
    pub fn reset(&mut self) {
        self.conv.fill(0.0);
        self.ssm.fill(0.0);
    }

    /// The whole conv and SSM slabs, for a checkpoint.
    ///
    /// Whole-buffer rather than per layer because a checkpoint is all-or-
    /// nothing: a state assembled from layers captured at different positions
    /// would be a plausible-looking mixture of two histories, and nothing
    /// downstream could tell.
    pub fn slabs(&self) -> (&[f32], &[f32]) {
        (&self.conv, &self.ssm)
    }

    /// Overwrite both slabs from a checkpoint.
    ///
    /// The caller must tell the backend afterwards — `Ops::forget_state` — or a
    /// device that owns the authoritative copy will carry on from the state it
    /// already has and ignore this entirely.
    pub fn load(&mut self, conv: &[f32], ssm: &[f32]) -> Result<()> {
        if conv.len() != self.conv.len() || ssm.len() != self.ssm.len() {
            return Err(Error::InconsistentArchitecture {
                what: "recurrent checkpoint",
                detail: format!(
                    "checkpoint holds {}+{} floats, this state is {}+{}",
                    conv.len(),
                    ssm.len(),
                    self.conv.len(),
                    self.ssm.len(),
                ),
            });
        }
        self.conv.copy_from_slice(conv);
        self.ssm.copy_from_slice(ssm);
        Ok(())
    }

    pub fn conv(&self, il: usize) -> &[f32] {
        &self.conv[il * self.conv_len..(il + 1) * self.conv_len]
    }

    pub fn conv_mut(&mut self, il: usize) -> &mut [f32] {
        &mut self.conv[il * self.conv_len..(il + 1) * self.conv_len]
    }

    pub fn ssm_mut(&mut self, il: usize) -> &mut [f32] {
        &mut self.ssm[il * self.ssm_len..(il + 1) * self.ssm_len]
    }

    /// Check the state matches the model about to use it.
    ///
    /// A state whose dimensions are wrong does not fault. It produces wrong
    /// numbers that look like a bad checkpoint, so the shapes are compared once
    /// at the top of a pass rather than trusted.
    pub fn check(&self, n_layer: usize, conv_len: usize, ssm_len: usize) -> Result<()> {
        if self.n_layer != n_layer || self.conv_len != conv_len || self.ssm_len != ssm_len {
            return Err(Error::InconsistentArchitecture {
                what: "recurrent state",
                detail: format!(
                    "state holds {} layers of {}+{} floats, model needs {n_layer} of {conv_len}+{ssm_len}",
                    self.n_layer, self.conv_len, self.ssm_len
                ),
            });
        }
        Ok(())
    }
}

/// Shift a channel-major conv window left by one and append `x`.
///
/// The window is `[channel][keep]` with the oldest sample first, so a new token
/// drops index 0 and lands at the end. This lives beside the state rather than
/// in model code because the ordering is the part that is easy to invert, and
/// `Ops::ssm_conv` documents the same convention at the other end.
pub fn push_conv(window: &mut [f32], x: &[f32], keep: usize) {
    debug_assert_eq!(window.len(), x.len() * keep);
    for (c, &v) in x.iter().enumerate() {
        let slot = &mut window[c * keep..(c + 1) * keep];
        slot.rotate_left(1);
        slot[keep - 1] = v;
    }
}

/// Build the `keep + 1`-wide window the convolution reads: each channel's
/// stored samples, then this token.
pub fn conv_window(stored: &[f32], x: &[f32], keep: usize, out: &mut [f32]) {
    let kernel = keep + 1;
    debug_assert_eq!(out.len(), x.len() * kernel);
    for (c, &v) in x.iter().enumerate() {
        let dst = &mut out[c * kernel..(c + 1) * kernel];
        dst[..keep].copy_from_slice(&stored[c * keep..(c + 1) * keep]);
        dst[keep] = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_state_is_zero_and_reset_returns_to_it() {
        let mut s = RecurrentState::new(3, 6, 8);
        s.ssm_mut(1)[0] = 5.0;
        s.conv_mut(2)[3] = 7.0;
        assert_ne!(s.conv(2)[3], 0.0);
        s.reset();
        assert!(s.conv(2).iter().all(|&v| v == 0.0));
        assert!(s.ssm_mut(1).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn layers_do_not_overlap() {
        let mut s = RecurrentState::new(3, 4, 4);
        s.conv_mut(1).fill(1.0);
        assert!(s.conv(0).iter().all(|&v| v == 0.0));
        assert!(s.conv(2).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn the_window_puts_the_newest_sample_last() {
        // Two channels, keep 2. Stored is [a0,a1 | b0,b1] and the new token
        // appends to each channel, so the window is [a0,a1,x0 | b0,b1,x1].
        // Reversing this is the conv-state ordering bug HANDOFF warned about,
        // and it survives every shape check.
        let stored = [1.0f32, 2.0, 10.0, 20.0];
        let x = [3.0f32, 30.0];
        let mut out = [0.0f32; 6];
        conv_window(&stored, &x, 2, &mut out);
        assert_eq!(out, [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
    }

    #[test]
    fn pushing_drops_the_oldest_and_keeps_the_rest_in_order() {
        let mut window = [1.0f32, 2.0, 10.0, 20.0];
        push_conv(&mut window, &[3.0, 30.0], 2);
        assert_eq!(window, [2.0, 3.0, 20.0, 30.0]);
    }

    #[test]
    fn a_full_kernel_of_pushes_replaces_the_window_entirely() {
        let mut window = [1.0f32, 2.0, 3.0];
        for t in 0..3 {
            push_conv(&mut window, &[10.0 + t as f32], 3);
        }
        assert_eq!(window, [10.0, 11.0, 12.0]);
    }

    #[test]
    fn check_rejects_a_state_shaped_for_another_model() {
        let s = RecurrentState::new(2, 4, 8);
        assert!(s.check(2, 4, 8).is_ok());
        assert!(s.check(3, 4, 8).is_err());
        assert!(s.check(2, 5, 8).is_err());
        assert!(s.check(2, 4, 9).is_err());
    }
}
