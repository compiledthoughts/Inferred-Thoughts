//! The routed FFN: expert choice on the device, expert address gathers, and
//! the per-pair and grouped expert matmuls.

use super::slot;
use crate::error::{Error, Result};
use crate::gguf::GgmlType;
use crate::ops::{Experts, Route};
use crate::ops::cuda::{Cuda, DeviceBuffer, KArg, ffi};

/// Pairs per grouped-expert tile. **Must equal `MOE_TOK` in
/// `kernels/kernels.cu`** — it sizes the register arrays there and the grid
/// bound here, and the two disagreeing is silent.
const MOE_TOK: usize = 8;

/// Pairs per grouped-expert tile on the tensor-core path. **Must equal
/// `MOE_MMA_TOK` in `kernels/kernels.cu`.**
///
/// Wider than `MOE_TOK` because the two kernels have opposite constraints: the
/// scalar one holds a per-token register array, the MMA one holds only
/// accumulators per 8-token sub-tile and wants the widest tile routing can
/// fill.
const MOE_MMA_TOK: usize = 16;

impl Cuda {
    /// Tiles the last grouped routed FFN cut its chunk into, or `None` if no
    /// grouped launch has happened.
    ///
    /// **Only so a test can say whether the thing it is testing actually ran.**
    /// `the_grouped_routed_ffn_is_bit_identical` compares grouped against
    /// per-pair on a batch, and would report a clean pass if every tile held a
    /// single token — which is the case the change exists to avoid and the case
    /// where the reuse loop never executes. Comparing this against the pair
    /// count turns "I calculated that tiles should pack" into an observation.
    ///
    /// **Read it after `end_pass`, never during a pass.** It is an ordinary
    /// device read, so mid-pass it returns the previous pass's value and latches
    /// graphs off for the run — exactly the hazard `CLAUDE.md` records.
    pub fn last_moe_tiles(&self) -> Option<u32> {
        let pool = self.pool.borrow();
        let buf = pool.get(slot::N_TILE)?;
        if buf.len_bytes() < 4 {
            return None;
        }
        let ptr = buf.ptr;
        drop(pool);
        let mut n = [0u32; 1];
        self.d2h(&mut n, ptr).ok()?;
        Some(n[0])
    }

    /// Force the per-pair routed FFN even for a batch. The A/B switch for the
    /// two grouped expert kernels; decode is unaffected either way, since it
    /// never reaches the grouped path.
    ///
    /// **This is what makes the bit-exactness claim testable rather than
    /// asserted.** Grouping changes which block computes an output and which
    /// weight loads are shared, never how one output accumulates — so the two
    /// paths must agree to the bit, and
    /// `the_grouped_routed_ffn_is_bit_identical` demands exactly that on a
    /// batch large enough for tiles to actually pack.
    pub fn moe_ungrouped(&self, on: bool) {
        self.moe_ungrouped.set(on);
    }

    /// Group this chunk's (token, pick) pairs by expert, so a weight row is
    /// loaded once per tile instead of once per pair.
    ///
    /// Returns the buffers `moe_group` fills plus the bound the grid must use.
    /// That bound is a function of shape alone —
    /// `sum ceil(count / MOE_TOK) <= n_pair / MOE_TOK + min(n_expert, n_pair)`
    /// — which is what lets it be a launch parameter while the *actual* count
    /// stays on the device: blocks past it read `n_tile` and exit. A grid sized
    /// from a host-visible routing decision would be exactly the mid-pass read
    /// that disables graphs.
    ///
    /// **Run by both expert matmuls rather than once per chunk.** They are
    /// always called in the same order today, so computing it in the first and
    /// reusing it in the second would work — and would make the down matmul
    /// silently wrong the day that order changes. One block over at most 1,024
    /// pairs, against a matmul that reads hundreds of megabytes, is not worth
    /// the coupling.
    fn moe_groups(
        &self,
        n_expert: usize,
        n_used: usize,
        n_tok: usize,
        e_tok: usize,
    ) -> Result<(
        ffi::CUdeviceptr,
        ffi::CUdeviceptr,
        ffi::CUdeviceptr,
        ffi::CUdeviceptr,
        u32,
    )> {
        let n_pair = n_used * n_tok;
        let n_tile_max = n_pair.div_ceil(e_tok) + n_expert.min(n_pair);
        // Exactly the size `route_impl` asked for, so this cannot reallocate
        // the buffer `moe_topk` just wrote its ids into.
        let ids = self.pooled(slot::ROUTE_IDS, n_pair * 4)?;
        let perm = self.pooled(slot::PERM, n_pair * 4)?;
        let first = self.pooled(slot::TILE_FIRST, n_tile_max * 4)?;
        let count = self.pooled(slot::TILE_N, n_tile_max * 4)?;
        let n_tile = self.pooled(slot::N_TILE, 4)?;
        // `n_pair + n_expert + 2 * (n_expert + 1)` ints, as the kernel carves it.
        let shared = ((n_pair + n_expert + 2 * (n_expert + 1)) * 4) as u32;
        let args = [
            KArg::I32(n_pair as i32),
            KArg::I32(n_expert as i32),
            KArg::I32(e_tok as i32),
            KArg::Ptr(ids),
            KArg::Ptr(perm),
            KArg::Ptr(first),
            KArg::Ptr(count),
            KArg::Ptr(n_tile),
        ];
        // SAFETY: parameters match `moe_group`; one block of 256 threads, and
        // `shared` is exactly the four arrays the kernel carves out of it.
        unsafe { self.launch_shared("moe_group", 1, 256, shared, &args)? };
        Ok((perm, first, count, n_tile, n_tile_max as u32))
    }

    /// The routed FFN's matmuls, every expert in one launch.
    ///
    /// IQ4_XS only, which is every `ffn_*_exps` tensor the 35B has. A different
    /// expert format would need its own grouped kernel; failing loudly is
    /// better than falling back to the trait default, whose sub-slicing of
    /// `out` would hand this backend addresses it has never mirrored.
    pub(super) fn matmul_experts_impl(
        &self,
        w: &Experts<'_>,
        route: &Route,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        const QK_K: usize = 256;
        const MAX: usize = 8;
        if w.ty == GgmlType::Nvfp4 {
            return self.matmul_experts_nvfp4(w, route, x, out);
        }
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
        // Every (token, pick) pair gets a block on `y`. Decode is `n_tok == 1`
        // of the same launch, which is why this cannot regress it.
        let n_pair = n_used * route.n_tok();
        // One row shared by every expert, or one row each. Derived from the
        // buffer, as the seam's batch count is — and the comparison is against
        // the *pair* count now, since a batched `down` has one intermediate per
        // pair rather than per pick.
        let rows = x.len() / w.n_in;
        let x_stride_super = if rows == n_pair { n_super } else { 0 };
        let (sd, qd, _) = self.quantized_k(x, rows * n_super)?;

        // The picks' addresses, resolved on the device from the slot table.
        // Nothing here reads the router's output, which is what lets this
        // launch live in a graph.
        let table = self.expert_table(w)?;
        self.stage_route_ids(route)?;
        let wptrs = self.gather_ptrs(w.data.as_ptr() as usize, table, n_used, route.n_tok(), slot::PTR_DOWN)?;
        let od = self.mirror_out(out)?;

        let block = 128u32;
        let rows_per_block = (block / 32) as usize;

        // **Grouped by expert once there is a batch to group.**
        //
        // Gated on `n_tok > 1`, so decode runs the identical kernel and cannot
        // regress by construction rather than by measurement — the same gate
        // the token-tiled dense matmul and the batched `ssm_conv` use. At one
        // token there is nothing to group anyway: eight picks are eight
        // distinct experts, so the per-pair form already loads each row once.
        //
        // `x_stride_super` guards the other precondition. The grouped kernel
        // indexes `x` per *pair*, which is what a batched `down` hands it; the
        // shared-activation form (stride 0) belongs to the gate/up half and
        // goes through `moe_glu` instead.
        if route.n_tok() > 1 && x_stride_super == n_super && !self.moe_ungrouped.get() {
            let mma = self.iq4_mma.get() && w.n_out % 16 == 0;
            let (perm, first, count, n_tile, n_tile_max) = self.moe_groups(
                w.n_expert,
                n_used,
                route.n_tok(),
                if mma { MOE_MMA_TOK } else { MOE_TOK },
            )?;
            // Same argument list either way, so the tensor-core variant is a
            // name and a grid: one warp covers 16 rows there against 4 here.
            let name = if mma {
                "matmul_iq4_xs_q8_k_moe_grouped_mma"
            } else {
                "matmul_iq4_xs_q8_k_moe_grouped"
            };
            let grid_rows = if mma {
                w.n_out.div_ceil(128) as u32
            } else {
                w.n_out.div_ceil(rows_per_block) as u32
            };
            self.note_shape(name, w.n_in, w.n_out);
            let args = [
                KArg::I32(w.n_in as i32),
                KArg::I32(w.n_out as i32),
                KArg::Ptr(n_tile),
                KArg::Ptr(perm),
                KArg::Ptr(first),
                KArg::Ptr(count),
                KArg::Ptr(wptrs),
                KArg::Ptr(sd),
                KArg::Ptr(qd),
                KArg::Ptr(od),
            ];
            // SAFETY: parameters match `matmul_iq4_xs_q8_k_moe_grouped`; the
            // grid covers `n_out` rows by the shape bound on tiles, blocks past
            // the device-side count return before touching a pointer, and the
            // kernel uses no dynamic shared memory.
            return unsafe {
                self.launch_grid2(name, grid_rows, n_tile_max, if mma { 256 } else { block }, 0, &args)
            };
        }

        self.note_shape("matmul_iq4_xs_q8_k_moe", w.n_in, w.n_out);
        let args = [
            KArg::I32(w.n_in as i32),
            KArg::I32(w.n_out as i32),
            KArg::I32(x_stride_super as i32),
            KArg::I32(n_pair as i32),
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
                n_pair as u32,
                block,
                0,
                &args,
            )
        }
    }

    pub(super) fn moe_glu_impl(
        &self,
        gate: &Experts<'_>,
        up: &Experts<'_>,
        route: &Route,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        const QK_K: usize = 256;
        const MAX: usize = 8;
        if gate.ty == GgmlType::Nvfp4 && up.ty == GgmlType::Nvfp4 {
            return self.moe_glu_nvfp4(gate, up, route, x, out);
        }
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
        let n_tok = route.n_tok();
        let n_pair = n_used * n_tok;
        // One activation row per *token*, not per pair: this half of the FFN
        // runs before any per-expert intermediate exists, so all `n_used`
        // experts of a token read the same row.
        let (sd, qd, _) = self.quantized_k(x, n_tok * n_super)?;

        // Two slot tables, two gathers, both on the device. `gate` and `up` are
        // separate tensors with separate tables, so an expert's gate and its up
        // need not share a residency tier.
        let gtab = self.expert_table(gate)?;
        let utab = self.expert_table(up)?;
        // Before either gather, and once for both: they read the same slot.
        self.stage_route_ids(route)?;
        let gptrs = self.gather_ptrs(gate.data.as_ptr() as usize, gtab, n_used, route.n_tok(), slot::PTR_GATE)?;
        let uptrs = self.gather_ptrs(up.data.as_ptr() as usize, utab, n_used, route.n_tok(), slot::PTR_UP)?;
        let od = self.mirror_out(out)?;

        let block = 128u32;
        let rows_per_block = (block / 32) as usize;

        // Grouped by expert once there is a batch to group — see
        // `matmul_experts_impl` for the gate and the reason it is `n_tok > 1`.
        // This is the larger half: measured at 43% of prefill kernel time
        // against the down matmul's 18%.
        if n_tok > 1 && !self.moe_ungrouped.get() {
            let mma = self.iq4_mma.get() && gate.n_out % 16 == 0;
            let (perm, first, count, n_tile, n_tile_max) = self.moe_groups(
                gate.n_expert,
                n_used,
                n_tok,
                if mma { MOE_MMA_TOK } else { MOE_TOK },
            )?;
            let name = if mma {
                "matmul_iq4_xs_q8_k_moe_glu_grouped_mma"
            } else {
                "matmul_iq4_xs_q8_k_moe_glu_grouped"
            };
            let grid_rows = if mma {
                gate.n_out.div_ceil(128) as u32
            } else {
                gate.n_out.div_ceil(rows_per_block) as u32
            };
            self.note_shape(name, gate.n_in, gate.n_out);
            let args = [
                KArg::I32(gate.n_in as i32),
                KArg::I32(gate.n_out as i32),
                KArg::I32(n_used as i32),
                KArg::Ptr(n_tile),
                KArg::Ptr(perm),
                KArg::Ptr(first),
                KArg::Ptr(count),
                KArg::Ptr(gptrs),
                KArg::Ptr(uptrs),
                KArg::Ptr(sd),
                KArg::Ptr(qd),
                KArg::Ptr(od),
            ];
            // SAFETY: parameters match `matmul_iq4_xs_q8_k_moe_glu_grouped`;
            // the grid covers `n_out` rows by the shape bound on tiles, blocks
            // past the device-side count return before touching a pointer, and
            // the kernel uses no dynamic shared memory.
            return unsafe {
                self.launch_grid2(name, grid_rows, n_tile_max, if mma { 256 } else { block }, 0, &args)
            };
        }

        self.note_shape("matmul_iq4_xs_q8_k_moe_glu", gate.n_in, gate.n_out);
        let args = [
            KArg::I32(gate.n_in as i32),
            KArg::I32(gate.n_out as i32),
            KArg::I32(n_pair as i32),
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
                n_pair as u32,
                block,
                0,
                &args,
            )
        }
    }

    /// The routed `down` matmul for NVFP4 experts against a Q8_0 activation,
    /// the exact path (`matmul_nvfp4_q8_0_moe_grouped`).
    ///
    /// **Grouped at every batch size, decode included.** `moe_group` runs on the
    /// device from the device-side ids, so nothing here reads the route on the
    /// host and the launch can live in a graph; at one token each tile is one
    /// pair and the arithmetic is the per-pair product. `moe_ungrouped` is the
    /// IQ4_XS switch and does not apply: there is no per-pair NVFP4 kernel to
    /// fall back to.
    fn matmul_experts_nvfp4(
        &self,
        w: &Experts<'_>,
        route: &Route,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let (n_used, n_tok) = (route.n_used(), route.n_tok());
        let n_pair = n_used * n_tok;
        let rows = x.len() / w.n_in;
        if n_used == 0 || n_used > 8 || rows != n_pair {
            return Err(Error::Cuda {
                what: "matmul_experts",
                detail: format!(
                    "NVFP4 experts read one row per pair: {rows} rows for {n_pair} pairs, \
                     {n_used} experts per token (at most 8)"
                ),
            });
        }
        if self.nvfp4_fp4.get() && cfg!(nvfp4_block_scale) {
            return self.matmul_experts_nvfp4_fp4(w, route, x, out);
        }
        let (sd, qd) = self.quantized(x, rows * (w.n_in / 32))?;
        let table = self.expert_table(w)?;
        self.stage_route_ids(route)?;
        let wptrs = self.gather_ptrs(w.data.as_ptr() as usize, table, n_used, n_tok, slot::PTR_DOWN)?;
        let od = self.mirror_out(out)?;
        let (perm, first, count, n_tile, n_tile_max) =
            self.moe_groups(w.n_expert, n_used, n_tok, MOE_TOK)?;
        let ids = self.pooled(slot::ROUTE_IDS, n_pair * 4)?;
        // The scale table, uploaded once from the mmap. Absent means 1.0; the
        // kernel then never reads the pointer, so any valid one stands in.
        let has_s = !w.scale.is_empty();
        let sc = if has_s { self.resident(w.scale)? } else { ids };
        let name = "matmul_nvfp4_q8_0_moe_grouped";
        self.note_shape(name, w.n_in, w.n_out);
        let args = [
            KArg::I32(w.n_in as i32),
            KArg::I32(w.n_out as i32),
            KArg::I32(has_s as i32),
            KArg::Ptr(n_tile),
            KArg::Ptr(perm),
            KArg::Ptr(first),
            KArg::Ptr(count),
            KArg::Ptr(ids),
            KArg::Ptr(wptrs),
            KArg::Ptr(sc),
            KArg::Ptr(sd),
            KArg::Ptr(qd),
            KArg::Ptr(od),
        ];
        // SAFETY: parameters match `matmul_nvfp4_q8_0_moe_grouped`; the grid
        // covers `n_out` rows by the tile bound, blocks past the device-side tile
        // count return before touching a pointer, and no shared memory is used.
        unsafe { self.launch_grid2(name, w.n_out.div_ceil(32) as u32, n_tile_max, 32, 0, &args) }
    }

    /// Gate, up and the SiLU gating for NVFP4 experts, the exact path
    /// (`matmul_nvfp4_q8_0_moe_glu_grouped`). Grouped at every batch size, for
    /// the reason `matmul_experts_nvfp4` gives.
    fn moe_glu_nvfp4(
        &self,
        gate: &Experts<'_>,
        up: &Experts<'_>,
        route: &Route,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let (n_used, n_tok) = (route.n_used(), route.n_tok());
        let n_pair = n_used * n_tok;
        if n_used == 0
            || n_used > 8
            || gate.n_in != up.n_in
            || gate.n_out != up.n_out
            || x.len() != n_tok * gate.n_in
        {
            return Err(Error::Cuda {
                what: "moe_glu",
                detail: format!(
                    "{n_used} experts, gate {:?} vs up {:?}, {} activation floats for {n_tok} tokens",
                    (gate.n_in, gate.n_out),
                    (up.n_in, up.n_out),
                    x.len()
                ),
            });
        }
        if self.nvfp4_fp4.get() && cfg!(nvfp4_block_scale) {
            return self.moe_glu_nvfp4_fp4(gate, up, route, x, out);
        }
        // One activation row per token: every expert of a token reads it.
        let (sd, qd) = self.quantized(x, n_tok * (gate.n_in / 32))?;
        let gtab = self.expert_table(gate)?;
        let utab = self.expert_table(up)?;
        self.stage_route_ids(route)?;
        let gptrs = self.gather_ptrs(gate.data.as_ptr() as usize, gtab, n_used, n_tok, slot::PTR_GATE)?;
        let uptrs = self.gather_ptrs(up.data.as_ptr() as usize, utab, n_used, n_tok, slot::PTR_UP)?;
        let od = self.mirror_out(out)?;
        let (perm, first, count, n_tile, n_tile_max) =
            self.moe_groups(gate.n_expert, n_used, n_tok, MOE_TOK)?;
        let ids = self.pooled(slot::ROUTE_IDS, n_pair * 4)?;
        let (has_gs, has_us) = (!gate.scale.is_empty(), !up.scale.is_empty());
        let gs = if has_gs { self.resident(gate.scale)? } else { ids };
        let us = if has_us { self.resident(up.scale)? } else { ids };
        let name = "matmul_nvfp4_q8_0_moe_glu_grouped";
        self.note_shape(name, gate.n_in, gate.n_out);
        let args = [
            KArg::I32(gate.n_in as i32),
            KArg::I32(gate.n_out as i32),
            KArg::I32(n_used as i32),
            KArg::I32(has_gs as i32),
            KArg::I32(has_us as i32),
            KArg::Ptr(n_tile),
            KArg::Ptr(perm),
            KArg::Ptr(first),
            KArg::Ptr(count),
            KArg::Ptr(ids),
            KArg::Ptr(gptrs),
            KArg::Ptr(uptrs),
            KArg::Ptr(gs),
            KArg::Ptr(us),
            KArg::Ptr(sd),
            KArg::Ptr(qd),
            KArg::Ptr(od),
        ];
        // SAFETY: parameters match `matmul_nvfp4_q8_0_moe_glu_grouped`; as for
        // `matmul_experts_nvfp4`.
        unsafe { self.launch_grid2(name, gate.n_out.div_ceil(32) as u32, n_tile_max, 32, 0, &args) }
    }

    /// The routed `down` matmul for NVFP4 experts as FP4 x FP4 on the tensor
    /// cores (`matmul_nvfp4_fp4_moe_grouped_mma`): the dense kernel's warp over
    /// an expert tile of `MOE_MMA_TOK` pairs in two 8-pair sub-tiles, sharing
    /// one weight load. The shape checks are the caller's, done before it
    /// chose this path.
    fn matmul_experts_nvfp4_fp4(
        &self,
        w: &Experts<'_>,
        route: &Route,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let (n_used, n_tok) = (route.n_used(), route.n_tok());
        let n_pair = n_used * n_tok;
        let rows = x.len() / w.n_in;
        let (dd, qd) = self.quantized_fp4(x, rows * (w.n_in / 16))?;
        let table = self.expert_table(w)?;
        self.stage_route_ids(route)?;
        let wptrs = self.gather_ptrs(w.data.as_ptr() as usize, table, n_used, n_tok, slot::PTR_DOWN)?;
        let od = self.mirror_out(out)?;
        let (perm, first, count, n_tile, n_tile_max) =
            self.moe_groups(w.n_expert, n_used, n_tok, MOE_MMA_TOK)?;
        let ids = self.pooled(slot::ROUTE_IDS, n_pair * 4)?;
        let has_s = !w.scale.is_empty();
        let sc = if has_s { self.resident(w.scale)? } else { ids };
        let name = "matmul_nvfp4_fp4_moe_grouped_mma";
        self.note_shape(name, w.n_in, w.n_out);
        let args = [
            KArg::I32(w.n_in as i32),
            KArg::I32(w.n_out as i32),
            KArg::I32(has_s as i32),
            KArg::Ptr(n_tile),
            KArg::Ptr(perm),
            KArg::Ptr(first),
            KArg::Ptr(count),
            KArg::Ptr(ids),
            KArg::Ptr(wptrs),
            KArg::Ptr(sc),
            KArg::Ptr(dd),
            KArg::Ptr(qd),
            KArg::Ptr(od),
        ];
        // SAFETY: parameters match `matmul_nvfp4_fp4_moe_grouped_mma`; the grid
        // covers `n_out` rows in 64s by the tile bound, blocks past the
        // device-side tile count or the last row return, and partial row tiles
        // clamp their loads.
        unsafe { self.launch_grid2(name, w.n_out.div_ceil(64) as u32, n_tile_max, 128, 0, &args) }
    }

    /// Gate, up and the SiLU gating for NVFP4 experts as FP4 x FP4 on the
    /// tensor cores (`matmul_nvfp4_fp4_moe_glu_grouped_mma`): one activation
    /// load per sub-tile serves both matrices.
    fn moe_glu_nvfp4_fp4(
        &self,
        gate: &Experts<'_>,
        up: &Experts<'_>,
        route: &Route,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let (n_used, n_tok) = (route.n_used(), route.n_tok());
        let n_pair = n_used * n_tok;
        let (dd, qd) = self.quantized_fp4(x, n_tok * (gate.n_in / 16))?;
        let gtab = self.expert_table(gate)?;
        let utab = self.expert_table(up)?;
        self.stage_route_ids(route)?;
        let gptrs = self.gather_ptrs(gate.data.as_ptr() as usize, gtab, n_used, n_tok, slot::PTR_GATE)?;
        let uptrs = self.gather_ptrs(up.data.as_ptr() as usize, utab, n_used, n_tok, slot::PTR_UP)?;
        let od = self.mirror_out(out)?;
        let (perm, first, count, n_tile, n_tile_max) =
            self.moe_groups(gate.n_expert, n_used, n_tok, MOE_MMA_TOK)?;
        let ids = self.pooled(slot::ROUTE_IDS, n_pair * 4)?;
        let (has_gs, has_us) = (!gate.scale.is_empty(), !up.scale.is_empty());
        let gs = if has_gs { self.resident(gate.scale)? } else { ids };
        let us = if has_us { self.resident(up.scale)? } else { ids };
        let name = "matmul_nvfp4_fp4_moe_glu_grouped_mma";
        self.note_shape(name, gate.n_in, gate.n_out);
        let args = [
            KArg::I32(gate.n_in as i32),
            KArg::I32(gate.n_out as i32),
            KArg::I32(n_used as i32),
            KArg::I32(has_gs as i32),
            KArg::I32(has_us as i32),
            KArg::Ptr(n_tile),
            KArg::Ptr(perm),
            KArg::Ptr(first),
            KArg::Ptr(count),
            KArg::Ptr(ids),
            KArg::Ptr(gptrs),
            KArg::Ptr(uptrs),
            KArg::Ptr(gs),
            KArg::Ptr(us),
            KArg::Ptr(dd),
            KArg::Ptr(qd),
            KArg::Ptr(od),
        ];
        // SAFETY: parameters match `matmul_nvfp4_fp4_moe_glu_grouped_mma`; as for
        // `matmul_experts_nvfp4_fp4`.
        unsafe { self.launch_grid2(name, gate.n_out.div_ceil(64) as u32, n_tile_max, 128, 0, &args) }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn moe_finish_impl(
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
        let n_tok = route.n_tok();
        let wd = self.pooled(slot::ROUTE_W, n_tok * n_used * 4)?;
        let args = [
            KArg::I32(n as i32),
            KArg::I32(n_used as i32),
            KArg::I32(at as i32),
            KArg::I32(n_tok as i32),
            KArg::Ptr(wd),
            KArg::Ptr(rd),
            KArg::Ptr(shd),
            KArg::Ptr(ld),
            KArg::I32(logit_at as i32),
            KArg::Ptr(od),
        ];
        self.note_shape("moe_finish", n, 0);
        // SAFETY: parameters match `moe_finish`; `rows` holds `n_tok * n_used`
        // rows of `n` floats, `shared` holds `n_tok` of them, and one thread
        // covers each element of the `n_tok` output rows starting at `at`.
        unsafe {
            self.launch_shared("moe_finish", (n * n_tok).div_ceil(256) as u32, 256, 0, &args)?
        };
        self.mirror_out(out).map(|_| ())
    }

    pub(super) fn add_scaled_rows_impl(
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
    pub fn moe_topk_readback(
        &self,
        probs: &[f32],
        n_expert: usize,
        n_used: usize,
    ) -> Result<(Vec<i32>, Vec<f32>)> {
        const MAX: usize = 8;
        if n_used == 0 || n_used > MAX || n_used > n_expert || n_expert == 0 {
            return Err(Error::Cuda {
                what: "moe_topk",
                detail: format!("{n_used} of {n_expert} experts; the kernel carries at most {MAX}"),
            });
        }
        let n_tok = probs.len() / n_expert;
        let pd = DeviceBuffer::from_slice(probs)?;
        let idb = DeviceBuffer::new(n_tok * n_used * 4)?;
        let wb = DeviceBuffer::new(n_tok * n_used * 4)?;
        // One block: the reduction is over the whole expert axis, so it cannot
        // be split across blocks without a second pass, and 256 experts is one
        // block's work. `blockDim` need not divide `n_expert` -- the per-thread
        // loop is strided -- but it must be a power of two for the tree.
        let block = 256u32;
        let shared = block * 8;
        let args = [
            KArg::I32(n_expert as i32),
            KArg::I32(n_used as i32),
            KArg::Ptr(pd.ptr),
            KArg::Ptr(idb.ptr),
            KArg::Ptr(wb.ptr),
        ];
        // SAFETY: parameters match `moe_topk`; one block per token, each
        // writing its own `n_used` ids and weights.
        unsafe { self.launch_shared("moe_topk", n_tok as u32, block, shared, &args)? };
        self.sync()?;
        let mut ids = vec![0i32; n_tok * n_used];
        let mut weights = vec![0.0f32; n_tok * n_used];
        self.d2h(&mut ids, idb.ptr)?;
        self.d2h(&mut weights, wb.ptr)?;
        Ok((ids, weights))
    }

    /// Choose this token's experts on the device.
    ///
    /// The router's probabilities stay on the card, which is the whole point:
    /// the host read they replace would turn CUDA graphs off for this model.
    /// `moe_topk` reproduces [`Ops::route`]'s default exactly — see
    /// `device_topk_reproduces_the_host_selection`.
    pub(super) fn route_impl(&self, probs: &[f32], n_expert: usize, n_used: usize) -> Result<()> {
        let n_tok = probs.len() / n_expert.max(1);
        let pd = self.mirror_in(probs)?;
        let idd = self.pooled(slot::ROUTE_IDS, n_tok * n_used * 4)?;
        let wd = self.pooled(slot::ROUTE_W, n_tok * n_used * 4)?;
        let block = 256u32;
        let args = [
            KArg::I32(n_expert as i32),
            KArg::I32(n_used as i32),
            KArg::Ptr(pd),
            KArg::Ptr(idd),
            KArg::Ptr(wd),
        ];
        // SAFETY: parameters match `moe_topk`; one block per token, each
        // writing its own `n_used` ids and weights. Shared memory is one float
        // and one int per thread, which is what the kernel declares.
        unsafe { self.launch_shared("moe_topk", n_tok as u32, block, block * 8, &args) }
    }

    /// Put a host-chosen route's picks where the device gather expects them.
    ///
    /// **`gather_ptrs` reads `slot::ROUTE_IDS` and nothing else.** That slot is
    /// written by exactly one thing, `moe_topk` inside `route_impl`, and
    /// [`Ops::route`] here always returns `Route::Device`, so in the engine the
    /// ids are always fresh and always the size the gather will ask for.
    ///
    /// A caller that builds a `Route::Host` by hand — every MoE bench does —
    /// got neither. `moe_topk` never ran, so the gather read whatever the slot
    /// happened to hold: in-range leftovers from an earlier pass, so the *wrong
    /// experts* and no failure, until a larger batch made `pooled` grow the
    /// slot and hand back an uninitialised block, at which point `table[id]`
    /// went out of bounds. That was the illegal address blocking this branch,
    /// and for as long as it did not fault it was quietly producing numbers
    /// from the wrong weights.
    ///
    /// So: honour the variant, or refuse it. Silently disregarding it is what
    /// this backend must not do.
    ///
    /// Sized exactly as `gather_ptrs` will size it, so its own `pooled` call
    /// cannot reallocate what this just wrote — the same invariant `moe_groups`
    /// depends on.
    fn stage_route_ids(&self, route: &Route) -> Result<()> {
        // `Route::Device` means `moe_topk` has already written the slot.
        let Some(ids) = route.ids() else { return Ok(()) };
        let n = route.n_tok() * route.n_used();
        if ids.len() != n {
            return Err(Error::Cuda {
                what: "stage_route_ids",
                detail: format!(
                    "{} ids for {n} picks ({} tokens x {})",
                    ids.len(),
                    route.n_tok(),
                    route.n_used()
                ),
            });
        }
        let idd = self.pooled(slot::ROUTE_IDS, n * 4)?;
        // The kernel takes `int`; `Route` carries `usize`.
        let as_i32: Vec<i32> = ids.iter().map(|&e| e as i32).collect();
        self.h2d(idd, &as_i32)
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
        n_tok: usize,
        into: usize,
    ) -> Result<ffi::CUdeviceptr> {
        let idd = self.pooled(slot::ROUTE_IDS, n_tok * n_used * 4)?;
        let out = self.pooled(into, n_tok * n_used * 8)?;
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
            KArg::I32(n_tok as i32),
            KArg::Ptr(table),
            KArg::Ptr(idd),
            KArg::Ptr(vram),
            KArg::Ptr(counts),
            KArg::Ptr(tally),
            KArg::Ptr(out),
        ];
        let block = 128u32;
        // SAFETY: parameters match `moe_gather_ptrs`; one thread per
        // (token, pick), and the kernel returns past `n_tok * n_used`.
        // `base + id` is inside the counter arrays because `table` refused a
        // base that would not fit.
        unsafe {
            self.launch(
                "moe_gather_ptrs",
                ((n_tok * n_used) as u32).div_ceil(block),
                block,
                &args,
            )?
        };
        Ok(out)
    }
}
