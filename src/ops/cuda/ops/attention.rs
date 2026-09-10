//! Attention against the KV cache: prefill on the tensor cores or the split
//! flash pair, decode through `attn_decode`, and the f16 KV write.

use super::slot;
use crate::error::{Error, Result};
use crate::ops::Attn;
use crate::ops::cuda::{Cuda, DeviceBuffer, KArg, KvMirror, ffi};

/// Query rows per tensor-core attention tile. **Must equal `ATT_QT` in
/// `kernels/kernels.cu`.**
const ATT_QT: usize = 16;

impl Cuda {
    /// Force the warp-per-position flash-decoding kernel on or off; `None`
    /// restores the depth-based choice. Benchmarks only — see `attend_impl`.
    pub fn attn_warp(&self, force: Option<bool>) {
        self.attn_warp.set(force);
    }

    /// Diagnostic: run the warp score phase for the `n`-th `attend` call of
    /// each pass and the thread path for every other. See `attn_warp_only`.
    pub fn attn_warp_only(&self, n: Option<usize>) {
        self.attn_warp_only.set(n);
    }

    /// Query rows per attention launch.
    ///
    /// **Grouping is a cache effect, not a fusion**: each row keeps its own
    /// blocks and its own arithmetic, but the blocks covering one K/V chunk run
    /// together rather than a whole launch apart, so the chunk is read from
    /// DRAM once instead of once per row. `1` is exactly the old path, which is
    /// what makes this an A/B switch rather than a tuning knob.
    pub fn set_qgroup(&self, n: usize) {
        self.qgroup.set(n.max(1));
    }

    /// Walk the whole KV in one block per (query row, head) for a batch,
    /// rather than splitting the sequence and combining partials through
    /// global memory.
    ///
    /// **Off by default: measured 0.92-1.05x, mean 0.96x.** It removes 537 MiB
    /// of partial traffic per layer at n_q 128, n_pos 8192 and buys nothing,
    /// because attention runs at 4% of this card's bandwidth -- traffic was
    /// never what bound it. Kept as the A/B, and because the same structure is
    /// what a tensor-core score GEMM would be built on.
    pub fn set_attn_fused(&self, on: bool) {
        self.attn_fused.set(on);
    }

    /// Compute the score matrix on the tensor cores.
    ///
    /// **A precision change, not a reordering**: the GEMM operands must be f16
    /// and Q arrives as f32, so it rounds to ten mantissa bits. K and V are
    /// already f16 in the cache, and the running max, running sum, rescaling
    /// and output accumulator all stay f32 -- the split FlashAttention makes,
    /// whose measured RMSE is better than a naive f32 kernel for that reason.
    /// Off by default; `attend_tolerance` does not cover it.
    pub fn set_attn_mma(&self, on: bool) {
        self.attn_mma.set(on);
    }

    /// Put `O += P V` on the tensor cores as well, and select the kernel that
    /// does. Implies [`Cuda::set_attn_mma`], since the two GEMMs live in one
    /// kernel; the score-only path stays available so the second GEMM can be
    /// priced on its own.
    ///
    /// **The same precision statement as `set_attn_mma` plus one term**: the
    /// probabilities also round to f16 on their way into the B operand. The
    /// running max, running sum and output accumulator remain f32.
    pub fn set_attn_vmma(&self, on: bool) {
        self.attn_vmma.set(on);
        if on {
            self.attn_mma.set(true);
        }
    }

    /// Run the split attention path through a decomposed copy of its kernels,
    /// `None` for production.
    ///
    /// The mask names what is left out -- the `DBG_ATT_*` bits in
    /// `kernels.cu` -- and only the combinations instantiated there exist;
    /// anything else fails the launch loudly rather than timing the wrong
    /// kernel. `Some(0)` is the production arithmetic through the copy, which
    /// is what lets a bench prove the copy is the kernel it decomposes.
    ///
    /// **The output is wrong by design for any other mask.** It prices work;
    /// it does not compute attention.
    pub fn attn_dbg(&self, skip: Option<i32>) {
        self.attn_dbg.set(skip);
    }

    /// The attention kernel pair to launch: production, or the decomposed
    /// copies [`Cuda::attn_dbg`] selects.
    ///
    /// Only the masks `kernels.cu` instantiates have kernels, and any other is
    /// an error rather than a fallback: a bench that silently timed production
    /// under a decomposed label is the failure this exists to prevent.
    fn attn_kernel_names(&self) -> Result<(&'static str, &'static str)> {
        let Some(skip) = self.attn_dbg.get() else {
            return Ok(("attn_flash", "attn_flash_combine"));
        };
        let flash = match skip & 127 {
            0 => "dbg_attn_flash_0",
            1 => "dbg_attn_flash_1",
            2 => "dbg_attn_flash_2",
            4 => "dbg_attn_flash_4",
            8 => "dbg_attn_flash_8",
            16 => "dbg_attn_flash_16",
            32 => "dbg_attn_flash_32",
            64 => "dbg_attn_flash_64",
            119 => "dbg_attn_flash_119",
            _ => "",
        };
        let combine = match skip & 384 {
            0 => "dbg_attn_flash_combine_0",
            128 => "dbg_attn_flash_combine_128",
            256 => "dbg_attn_flash_combine_256",
            384 => "dbg_attn_flash_combine_384",
            _ => "",
        };
        if flash.is_empty() || combine.is_empty() || skip & !511 != 0 {
            return Err(Error::Cuda {
                what: "attn_dbg",
                detail: format!("no decomposed attention kernel is instantiated for mask {skip}"),
            });
        }
        Ok((flash, combine))
    }

    /// Run the tensor-core attention path through a decomposed copy of
    /// `attn_flash_mma_v`, `None` for the real kernel. `Some` also selects that
    /// path, so a mask can never be set and then silently not used.
    ///
    /// The mask is the `DBG_VMMA_*` bits in `kernels.cu`; only instantiated
    /// combinations exist and any other fails the launch. `Some(0)` is the
    /// kernel's own arithmetic through the copy.
    pub fn attn_vmma_dbg(&self, skip: Option<i32>) {
        self.attn_vdbg.set(skip);
        if skip.is_some() {
            self.set_attn_vmma(true);
        }
    }

    /// The tensor-core kernel with both GEMMs, or its decomposed copy.
    fn vmma_kernel_name(&self) -> Result<&'static str> {
        let Some(skip) = self.attn_vdbg.get() else {
            return Ok("attn_flash_mma_v");
        };
        Ok(match skip {
            0 => "dbg_attn_mma_v_0",
            1 => "dbg_attn_mma_v_1",
            2 => "dbg_attn_mma_v_2",
            4 => "dbg_attn_mma_v_4",
            8 => "dbg_attn_mma_v_8",
            16 => "dbg_attn_mma_v_16",
            32 => "dbg_attn_mma_v_32",
            18 => "dbg_attn_mma_v_18",
            36 => "dbg_attn_mma_v_36",
            63 => "dbg_attn_mma_v_63",
            192 => "dbg_attn_mma_v_192",
            448 => "dbg_attn_mma_v_448",
            _ => {
                return Err(Error::Cuda {
                    what: "attn_vmma_dbg",
                    detail: format!("no decomposed mma_v kernel is instantiated for mask {skip}"),
                });
            }
        })
    }

    /// Positions from which decode attention takes the tensor cores, unless
    /// forced.
    ///
    /// **Measured 11-09** by `what_attention_costs_as_context_grows` at the
    /// 35B's shape: the tensor-core mode is 0.74-0.83x the warp phase up to
    /// d1024, parity at d1536-2048, 1.78x at d4096 and 1.6-1.9x from there to
    /// d65536. Below the crossover a call is tens of microseconds and the
    /// tile's fixed cost dominates. llama.cpp switches at 8192 for this shape.
    const ATTN_DECODE_MMA_MIN_POS: usize = 2048;

    /// Force decode attention onto the tensor cores (`Some(true)`), off them
    /// (`Some(false)`), or let depth decide (`None`).
    pub fn attn_decode_mma(&self, force: Option<bool>) {
        self.attn_decode_mma.set(force);
    }

    /// Move the depth at which decode attention takes the tensor cores; 0
    /// restores [`Cuda::ATTN_DECODE_MMA_MIN_POS`]. For tests that need the
    /// switch to land in the middle of a generation.
    pub fn set_attn_decode_mma_from(&self, n_pos: usize) {
        self.attn_decode_mma_from.set(n_pos);
    }

    /// Use the 16-slot decode tile even where the 8-slot one fits.
    pub fn attn_decode_tile16(&self, on: bool) {
        self.attn_decode_tile16.set(on);
    }

    /// This decode step's attention mode, as `attn_decode` takes it: 0 the
    /// scalar path, 1 the tensor cores with the 16-slot tile, 2 with the 8-slot
    /// tile.
    ///
    /// The shape has to fit a tile first: `head_dim` a multiple of 16 and at
    /// most 256, and no more query heads per kv head than its slots. **The
    /// 8-slot tile is taken wherever it fits**: at a grouped-query ratio of 8 it
    /// has no empty slots, which is what the 16-slot tile wasted in decode.
    fn decode_mode(&self, a: &Attn<'_>) -> i32 {
        const D8_QT: usize = 8;
        let fits = a.head_dim % 16 == 0
            && a.head_dim <= 256
            && a.n_head_kv > 0
            && a.n_head % a.n_head_kv == 0
            && a.n_head / a.n_head_kv <= ATT_QT;
        if !fits {
            return 0;
        }
        let on = match self.attn_decode_mma.get() {
            Some(force) => force,
            None => {
                let from = match self.attn_decode_mma_from.get() {
                    0 => Self::ATTN_DECODE_MMA_MIN_POS,
                    p => p,
                };
                self.attn_vmma.get() && a.n_pos >= from
            }
        };
        if !on {
            0
        } else if a.n_head / a.n_head_kv <= D8_QT && !self.attn_decode_tile16.get() {
            2
        } else {
            1
        }
    }

    /// Shared memory the 8-slot decode body indexes: the f16 query tile, the
    /// 64-position K/V staging buffer, the f16 probabilities, the score tile,
    /// and three per-slot vectors.
    fn d8_shared_bytes(head_dim: usize) -> u32 {
        const D8_QT: usize = 8;
        const D8_KC: usize = 64;
        (D8_QT * head_dim * 2
            + D8_KC * head_dim * 2
            + D8_QT * D8_KC * 2
            + D8_QT * D8_KC * 4
            + 3 * D8_QT * 4) as u32
    }

    /// Shared memory the `attn_flash_mma_v` body indexes: the f16 query tile,
    /// the K/V staging buffer, the f16 probabilities, the score tile, and four
    /// per-slot vectors.
    fn vmma_shared_bytes(head_dim: usize) -> u32 {
        const ATT_KC: usize = 32;
        (ATT_QT * head_dim * 2
            + ATT_KC * head_dim * 2
            + ATT_QT * ATT_KC * 2
            + ATT_QT * ATT_KC * 4
            + 4 * ATT_QT * 4) as u32
    }

    /// Positions at or above which the warp-per-position score phase is used.
    ///
    /// **`usize::MAX`: it is off, and that is a known-unfinished state.**
    ///
    /// The warp phase fixes a real defect — `attn_flash` reads K transposed,
    /// one thread per position, so adjacent threads are `kv_dim` apart and a
    /// warp's load touches 32 cache lines to use two bytes from each — and it
    /// measures ~3x faster from d2048 up, 43 -> 133 GB/s, worth 6.4 ms/token at
    /// the 19,942 positions a real session reaches:
    ///
    /// | n_pos | thread us | warp us | speedup |
    /// |---|---|---|---|
    /// | 512 | 43.3 | 30.8 | 1.41 |
    /// | 8192 | 387.6 | 124.3 | 3.12 |
    /// | 19942 | 932.1 | 315.9 | 2.95 |
    ///
    /// It also passes `the_warp_attention_agrees_with_the_oracle` at both
    /// models' shapes, twelve depths and both batch shapes — 72 comparisons
    /// against `ops::naive` within the derived tolerance.
    ///
    /// **It was disabled for a day by a test that could not tell drift from a
    /// bug, and the kernel was never wrong.**
    ///
    /// `the_model_agrees_with_the_oracle_to_the_quantization_floor` failed by
    /// "a whole argmax" (17689 against 5429), and the cause was its own method:
    /// it drove decode with each run's `argmax`, so two runs differing *inside*
    /// the floor eventually chose different tokens and then compared different
    /// sequences. `CLAUDE.md` already records that reasoning as the reason
    /// Stage 5's acceptance criterion was replaced.
    ///
    /// Established by a 2x2 rather than by argument
    /// (`what_the_warp_attention_needs_to_fail`): the divergence needs decode
    /// steps and is **identical with graphs on and off**, which eliminates the
    /// suspect the previous session recorded. With the decode sequence fixed,
    /// `the_warp_attention_agrees_when_both_runs_decode_the_same_tokens` gives
    /// 1.69e-2 of magnitude against the same 9e-2 ceiling, and the same argmax.
    ///
    /// 512 is the lowest depth the speedup was measured at (1.41x). Below it the
    /// warp loop still walks all `FD_CHUNK` positions with most guarded off, so
    /// it is plausibly slower and is **unmeasured rather than known good**.
    /// Crossing the threshold mid-run is safe: the two phases are one kernel
    /// with a grid-uniform branch, and `use_warp` is an argument the graph
    /// replay updates like any other.
    ///
    /// `Cuda::attn_warp` forces either path for the benchmarks and the
    /// differential test, which is how the numbers above were taken.
    const ATTN_WARP_MIN_POS: usize = 512;

    /// Whether this call uses the warp-per-position score phase.
    fn warp_scores(&self, n_pos: usize) -> bool {
        if let Some(only) = self.attn_warp_only.get() {
            return self.attn_calls.get() == only + 1;
        }
        match self.attn_warp.get() {
            Some(force) => force,
            None => n_pos >= Self::ATTN_WARP_MIN_POS,
        }
    }

    pub(super) fn attend_impl(&self, a: &Attn<'_>, out: &mut [f32]) -> Result<()> {
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
        // Sized for a whole group of query rows, at the widest window in it.
        let g = self.attend_group(a.n_q()).min(a.n_q());
        let pa = self.pooled(slot::SCORES, g * a.n_head * n_split * a.head_dim * 4)?;
        let pm = self.pooled(slot::PART_M, g * a.n_head * n_split * 4)?;
        let pl = self.pooled(slot::PART_L, g * a.n_head * n_split * 4)?;

        // **Query rows are launched one at a time, by device pointer offset.**
        // Every row has a different causal window, so they cannot share a grid
        // without masking most of it away; and the offsets are applied to the
        // device pointers rather than by sub-slicing `a.q` on the host, because
        // this backend keys its mirrors on host addresses -- a sub-slice would
        // look like an unmirrored buffer and be uploaded from a stale host copy,
        // which is the bug `Ops::rope_neox` already carries a note about.
        let (n_q, per_row) = (a.n_q(), a.n_head * a.head_dim);
        // Counted per `attend`, not per query row: one call is one layer.
        self.attn_calls.set(self.attn_calls.get() + 1);

        // **A batch walks the whole KV in one block per (row, head).** The
        // sequence split exists so decode, which has a single query row, can
        // fill the machine; a batch already has `n_q * n_head` blocks and the
        // split only buys 537 MiB of partials per layer at n_q 128, n_pos 8192.
        // Decode falls through to the two-kernel path unchanged.
        if n_q > 1 && self.attn_fused.get() {
            let args = [
                KArg::I32(a.n_pos_of(0) as i32),
                KArg::I32(a.kv_dim as i32),
                KArg::I32(a.head_dim as i32),
                KArg::I32(a.n_head as i32),
                KArg::I32(a.n_head_kv as i32),
                KArg::I32(i32::from(self.warp_scores(a.n_pos))),
                KArg::F32(a.scale),
                KArg::Ptr(qd),
                KArg::Ptr(kd),
                KArg::Ptr(vd),
                KArg::Ptr(od),
            ];
            // `sq` + `se` + `red` + `acc`, which is what the kernel indexes.
            let shared = ((2 * a.head_dim + 2 * CHUNK) * 4) as u32;
            // SAFETY: parameters match `attn_flash_fused`; one block per
            // (query row, query head), `CHUNK` threads as the reductions
            // assume, and `shared` is the four arrays it carves out.
            unsafe {
                self.launch_shared(
                    "attn_flash_fused",
                    (a.n_head * n_q) as u32,
                    CHUNK as u32,
                    shared,
                    &args,
                )?
            };
            return Ok(());
        }
        // **Query rows go up in groups now.** One launch per row read that
        // row's whole K/V window from DRAM by itself; rows of a group run
        // concurrently, so the blocks sharing a chunk find it in L2. Decode is
        // a group of one and reaches the identical kernels.
        let qg = self.attend_group(n_q);
        for t0 in (0..n_q).step_by(qg) {
            let rows = qg.min(n_q - t0);
            let qd = qd + (t0 * per_row * 4) as u64;
            let od = od + (t0 * per_row * 4) as u64;
            self.attend_rows(a, t0, rows, CHUNK, qd, kd, vd, od, pa, pm, pl)?;
        }
        Ok(())
    }

    /// Query rows per attention launch.
    ///
    /// The tensor-core path needs a whole `ATT_QT` tile in one block, so it
    /// fixes the group; otherwise this is the tunable cache-locality group.
    fn attend_group(&self, n_q: usize) -> usize {
        if n_q > 1 && self.attn_mma.get() {
            ATT_QT
        } else {
            self.qgroup.get().max(1)
        }
    }

    /// A group of query rows against the cache — the flash-decoding pair.
    #[allow(clippy::too_many_arguments)]
    fn attend_rows(
        &self,
        a: &Attn<'_>,
        t0: usize,
        rows: usize,
        chunk: usize,
        qd: ffi::CUdeviceptr,
        kd: ffi::CUdeviceptr,
        vd: ffi::CUdeviceptr,
        od: ffi::CUdeviceptr,
        pa: ffi::CUdeviceptr,
        pm: ffi::CUdeviceptr,
        pl: ffi::CUdeviceptr,
    ) -> Result<()> {
        // Row `t0 + i` attends over `n_pos_first + i` positions, and the grid
        // is sized by the widest window in the group; blocks past a row's own
        // window return before touching anything.
        let n_pos_first = a.n_pos_of(t0);
        let n_split = a.n_pos_of(t0 + rows - 1).div_ceil(chunk);
        {
            let args = [
                KArg::I32(n_pos_first as i32),
                KArg::I32(a.kv_dim as i32),
                KArg::I32(a.head_dim as i32),
                KArg::I32(a.n_head as i32),
                KArg::I32(a.n_head_kv as i32),
                // Which score phase, as an argument rather than a kernel name:
                // it depends on `n_pos`, so a long run crosses the threshold
                // mid-generation and a graph cannot express a changing
                // sequence. See `attn_flash`.
                KArg::I32(i32::from(self.warp_scores(a.n_pos))),
                KArg::I32(n_split as i32),
                KArg::F32(a.scale),
                KArg::Ptr(qd),
                KArg::Ptr(kd),
                KArg::Ptr(vd),
                KArg::Ptr(pa),
                KArg::Ptr(pm),
                KArg::Ptr(pl),
            ];
            let shared = ((a.head_dim + 2 * chunk) * 4) as u32;
            // **`n_q > 1`, as every prefill-only path here is gated.** Without
            // it decode runs a 16-row tile for its single row -- 16x the work,
            // measured at 37.85 -> 23.14 tok/s -- and, worse, `serve` records
            // decode as a CUDA graph, so a kernel the graph has never seen gets
            // its position-dependent arguments baked in and every step after
            // the first attends with stale ones. That produced fluent-looking
            // nonsense in a real session while prefill was correct and faster.
            // **Decode launches one kernel whatever the depth.** Decode is a
            // CUDA graph and a graph cannot change kernels mid-generation, so
            // the tensor-core mode is an argument of `attn_decode`. The modes'
            // grids and shared sizes differ, and replay updates both in place.
            if a.n_q() == 1 && self.attn_dbg.get().is_none() {
                let mode = self.decode_mode(a);
                let (grid_x, dshared) = match mode {
                    2 => (a.n_head_kv as u32, Self::d8_shared_bytes(a.head_dim)),
                    1 => (a.n_head_kv as u32, Self::vmma_shared_bytes(a.head_dim)),
                    _ => ((a.n_head * rows) as u32, shared),
                };
                let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13] = args;
                let dargs = [
                    KArg::I32(mode),
                    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13,
                ];
                // SAFETY: parameters match `attn_decode`: `attn_flash`'s plus the
                // mode. One block per (query head, chunk) in the scalar mode and
                // per (kv head, chunk) in the tensor-core mode, 128 threads either
                // way, and `dshared` is what the chosen mode indexes.
                unsafe {
                    self.launch_grid2(
                        "attn_decode",
                        grid_x,
                        n_split as u32,
                        chunk as u32,
                        dshared,
                        &dargs,
                    )?
                };
            } else if self.attn_mma.get() && a.n_q() > 1 && a.head_dim % 16 == 0 {
                let margs = [
                    KArg::I32(n_pos_first as i32),
                    KArg::I32(rows as i32),
                    KArg::I32(a.kv_dim as i32),
                    KArg::I32(a.head_dim as i32),
                    KArg::I32(a.n_head as i32),
                    KArg::I32(a.n_head_kv as i32),
                    KArg::I32(n_split as i32),
                    KArg::F32(a.scale),
                    KArg::Ptr(qd),
                    KArg::Ptr(kd),
                    KArg::Ptr(vd),
                    KArg::Ptr(pa),
                    KArg::Ptr(pm),
                    KArg::Ptr(pl),
                ];
                const ATT_KC: usize = 32;
                // `attn_flash_mma_v` puts the second GEMM on the tensor cores
                // too. It needs one more shared buffer -- an f16 copy of the
                // probabilities, which is that GEMM's B operand -- and its
                // accumulator is four 16-dim blocks per warp across four warps,
                // so a head_dim past 256 would need registers it does not
                // declare. Both conditions are checked here rather than assumed.
                let vmma = self.attn_vmma.get() && a.head_dim <= 256;
                let mshared = (ATT_QT * a.head_dim * 2
                    + ATT_KC * a.head_dim * 2
                    + if vmma { ATT_QT * ATT_KC * 2 } else { 0 }
                    + ATT_QT * ATT_KC * 4
                    + 4 * ATT_QT * 4) as u32;
                // SAFETY: parameters match `attn_flash_mma`; one block per
                // (head, chunk), 128 threads as its four-warp score tile
                // assumes, and `mshared` is the query tile, the K/V staging
                // buffer, the score tile and the four per-row vectors.
                let mname = if vmma { self.vmma_kernel_name()? } else { "attn_flash_mma" };
                unsafe {
                    self.launch_grid2(
                        mname,
                        a.n_head as u32,
                        n_split as u32,
                        128,
                        mshared,
                        &margs,
                    )?
                };
            } else {
            // SAFETY: parameters match `attn_flash`; the grid is one block per
            // (query head, chunk) so no block sees an empty range, and `shared`
            // is head_dim + 2 * FD_CHUNK floats, which is what it indexes.
            // `attn_flash` reads K transposed -- one thread per position, so
            // adjacent threads are `kv_dim` apart and a warp's load touches 32
            // cache lines. `attn_flash_warp` gives a whole warp to each position
            // so lanes read consecutive keys. Same grid, same shared memory;
            // only the score phase differs. Behind a flag until measured across
            // depth, which is the lesson `f32_staged` cost.
            // The decomposed copies share this kernel's signature exactly, so
            // routing to one is a name and nothing else.
            let (flash, _) = self.attn_kernel_names()?;
            unsafe {
                self.launch_grid2(
                    flash,
                    (a.n_head * rows) as u32,
                    n_split as u32,
                    chunk as u32,
                    shared,
                    &args,
                )?
            };
            }
        }

        {
            let args = [
                KArg::I32(n_pos_first as i32),
                KArg::I32(a.head_dim as i32),
                KArg::I32(a.n_head as i32),
                KArg::I32(n_split as i32),
                KArg::I32(chunk as i32),
                KArg::Ptr(pa),
                KArg::Ptr(pm),
                KArg::Ptr(pl),
                KArg::Ptr(od),
            ];
            let shared = (n_split * 4) as u32;
            let (_, combine) = self.attn_kernel_names()?;
            // SAFETY: parameters match `attn_flash_combine`; one block per
            // query head, and `shared` is `n_split` floats.
            unsafe {
                self.launch_shared(
                    combine,
                    (a.n_head * rows) as u32,
                    chunk as u32,
                    shared,
                    &args,
                )?
            };
        }
        Ok(())
    }

    /// Convert and store K or V without either ever leaving the card.
    pub(super) fn kv_write_impl(&self, slab: &mut [u16], offset: usize, src: &[f32]) -> Result<()> {
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
}
