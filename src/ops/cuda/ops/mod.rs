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
//!
//! # Layout
//!
//! One `impl Cuda` block per family, each in its own file, and the `Ops`
//! impls below delegating to them. `residency` is where bytes live on the
//! device, `placement` is the expert slab and its policy, and `bench`
//! replays what a run launched.

mod attention;
mod bench;
mod elementwise;
mod gdn;
mod matmul;
mod moe;
mod norm;
mod placement;
mod qsa;
mod qwen4exp;
mod residency;

use super::{Cuda, experts};
use crate::error::{Error, Result};
use crate::ops::{Attn, Delta, Experts, Ops, QsaPool, QsaSelect, Route, Weights};

/// Scratch slots. Distinct within any one method, reused across methods.
mod slot {
    pub const SCORES: usize = 5;
    pub const PART_M: usize = 9;
    pub const PART_L: usize = 10;
    pub const COS: usize = 6;
    pub const SIN: usize = 7;
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
    /// What `moe_group` writes: this chunk's (token, pick) pairs permuted into
    /// ascending expert order, the tile table cut over that permutation, and
    /// the tile count.
    ///
    /// Shares the routing scratch's argument for safety — one set for all forty
    /// layers, because a stream executes in issue order, so a layer's grouping
    /// completes before its matmuls start.
    pub const PERM: usize = 16;
    pub const TILE_FIRST: usize = 17;
    pub const TILE_N: usize = 18;
    pub const N_TILE: usize = 19;
    /// QSA's gathered K and V windows (`attend_sparse_impl`).
    pub const QSA_KW: usize = 20;
    pub const QSA_VW: usize = 21;
}

impl Cuda {
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
    pub fn take_error(&self) -> Option<Error> {
        self.error.borrow_mut().take()
    }

    /// The three one-time costs the first forward pass is billed for, in
    /// nanoseconds: `cuMemAlloc`, weight copies, expert placement.
    ///
    /// **Disjoint by construction**, which took a change to `resident` to make
    /// true: it allocated and copied in one `from_slice` call, so the copy
    /// timer contained the allocation timer and the two could not be added.
    /// Adding overlapping measurements is how a breakdown comes to exceed the
    /// thing it decomposes.
    ///
    /// They do not sum to the whole fixed cost and are not meant to. A 0.6B
    /// first pass measures ~842 ms fixed, of which PTX JIT is ~330 —
    /// established by running `CUDA_MODULE_LOADING=EAGER` against `LAZY`, and
    /// not observable from inside the process, since the driver does the work
    /// on first launch and reports nothing.
    pub fn setup_parts(&self) -> (u64, u64, u64) {
        let alloc = super::alloc_ns();
        let upload = self.stats().weight_upload_ns;
        let place = self
            .expert_stats()
            .map_or(0, |e| (e.place_h2d_us + e.place_pin_us + e.place_copy_us) * 1_000);
        (alloc, upload, place)
    }

    /// Print launch and residency counters after each server turn.
    pub fn report_per_turn(&self, on: bool) {
        self.report_per_turn.set(on);
    }

    /// Whether graphs were turned off because the model read a device result
    /// mid-pass. See [`Cuda::mid_pass_read`]; reported by `--profile-device`
    /// so the throughput loss is visible rather than inferred from a launch
    /// count.
    pub fn graphs_off_for_mid_pass_read(&self) -> bool {
        self.mid_pass_read.get()
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

    fn softmax(&self, x: &mut [f32], row: usize) {
        self.note(self.softmax_impl(x, row));
    }

    fn attend(&self, a: &Attn<'_>, out: &mut [f32]) {
        self.note(self.attend_impl(a, out));
    }

    fn silu_mul(&self, gate: &mut [f32], up: &[f32]) {
        self.note(self.silu_mul_impl(gate, up));
    }

    fn scale(&self, buf: &mut [f32], s: f32) {
        self.note(self.scale_impl(buf, s));
    }

    // qwen4exp's hyper-connection and PLE ops (src/model/qwen4exp.md). Every one
    // must be overridden here: the trait defaults would compute on host copies
    // the device may not hold.
    fn mul_rows(&self, x: &mut [f32], w: &[f32]) {
        self.note(self.mul_rows_impl(x, w));
    }
    fn silu(&self, x: &mut [f32]) {
        self.note(self.activation_impl("silu_f32", x));
    }
    fn sigmoid(&self, x: &mut [f32]) {
        self.note(self.activation_impl("sigmoid_f32", x));
    }
    fn mul_streams(&self, out: &mut [f32], h: &[f32], w: &[f32], n_stream: usize) {
        self.note(self.mul_streams_impl(out, h, w, n_stream));
    }
    fn row_dot(&self, a: &[f32], b: &[f32], width: usize, out: &mut [f32]) {
        self.note(self.row_dot_impl(a, b, width, out));
    }
    fn signed_sqrt_sigmoid(&self, s: &mut [f32]) {
        self.note(self.signed_sqrt_sigmoid_impl(s));
    }
    fn dilated_conv(
        &self,
        state: &mut [f32],
        x: &[f32],
        weight: &[f32],
        kernel: usize,
        dilation: usize,
        out: &mut [f32],
    ) {
        self.note(self.dilated_conv_impl(state, x, weight, kernel, dilation, out));
    }

    // QSA past its budget (kernels/qsa.cuh). Overridden, never inherited: the
    // defaults would compute on host copies of device data — the indexer keys,
    // the pooled lanes and K and V live on the device here.
    fn qsa_pool(&self, raw: &[u16], pooled: &mut [f32], p: &QsaPool<'_>) {
        self.note(self.qsa_pool_impl(raw, pooled, p));
    }
    fn qsa_select(&self, q: &[f32], pooled: &[f32], sel: &QsaSelect, scores: &mut [f32], cells: &mut [u32]) {
        self.note(self.qsa_select_impl(q, pooled, sel, scores, cells));
    }
    fn attend_sparse(&self, a: &Attn<'_>, cells: &[u32], sel: &QsaSelect, out: &mut [f32]) {
        self.note(self.attend_sparse_impl(a, cells, sel, out));
    }

    fn add_assign(&self, a: &mut [f32], b: &[f32]) {
        self.note(self.add_assign_impl(a, b));
    }

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
    fn route(&self, probs: &mut [f32], n_expert: usize, n_used: usize) -> Route {
        self.note(self.route_impl(probs, n_expert, n_used));
        Route::Device { n_used, n_tok: probs.len() / n_expert.max(1) }
    }

    fn matmul_pair(
        &self,
        a: &Weights<'_>,
        b: &Weights<'_>,
        x: &[f32],
        out_a: &mut [f32],
        out_b: &mut [f32],
    ) {
        // Declines rather than fails when the merge does not apply: the 9B's
        // `ssm_alpha`/`ssm_beta` are Q8_0 where the 35B's are F32. Stable
        // within a run, so a recorded graph sees one branch or the other and
        // never both.
        if Self::pairable(a, b) {
            self.note(self.matmul_pair_impl(a, b, x, out_a, out_b));
        } else {
            self.matmul(a, x, out_a);
            self.matmul(b, x, out_b);
        }
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
        // Not `clear()`: that frees ~60 device buffers the next pass would
        // immediately allocate again, measured at 69 ms per checkpoint restore.
        // A generation bump marks every slab stale and keeps the memory.
        self.state_gen.set(self.state_gen.get() + 1);
    }

    fn read_state(&self, host: &mut [f32]) {
        let r = self.read_state_into(host);
        self.note(r);
    }

    /// Which paths this process will actually run.
    ///
    /// Resolved state, not flags: what will execute, not what was asked for.
    /// `attn_vmma` implies `attn_mma`, so the order of those arms matters.
    fn config_report(&self) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        out.push((
            "kernels",
            format!(
                "q6k={} q5k={} delta={} iq4={} attn={} rms={} graphs={}",
                if self.q6k_scalar.get() { "per-token" } else { "tiled" },
                // Was missing entirely, though `q5k_scalar` has been a flag
                // since the tiling landed: the banner could not tell a Q5_K
                // A/B's two arms apart. Three states now, and `mma` is the one
                // that is **not bit-exact** — a run's banner has to say so.
                if self.q5k_scalar.get() {
                    "per-token"
                } else if self.q5k_mma.get() {
                    "mma"
                } else {
                    "tiled"
                },
                if self.delta_seq.get() { "per-token" } else { "batched" },
                // `iq4_staged` wins the dispatch, so it is reported first.
                if self.iq4_untiled.get() {
                    "untiled"
                } else if self.iq4_staged.get() {
                    "staged"
                } else if self.iq4_mma.get() {
                    "mma"
                } else {
                    "tiled"
                },
                if self.attn_vmma.get() {
                    "mma+vgemm"
                } else if self.attn_mma.get() {
                    "mma-score"
                } else if self.attn_fused.get() {
                    "fused"
                } else {
                    "split"
                },
                if self.rms_serial.get() { "serial" } else { "tree" },
                if self.graphs_enabled.get() { "on" } else { "off" },
            ),
        ));
        out.push((
            "budgets",
            format!(
                "expert reserve {:.2} GiB | slab cap {} | host tier {:.1} GiB | pool {} | qgroup {}",
                self.expert_reserve.get() as f64 / 1073741824.0,
                self.expert_cap
                    .get()
                    .map_or("explicit".to_string(), |c| format!("{:.1} GiB", c as f64 / 1073741824.0)),
                self.expert_host_budget.get() as f64 / 1073741824.0,
                self.expert_pool.get().map_or("undeclared".to_string(), |n| format!("{n} experts")),
                self.qgroup.get(),
            ),
        ));
        // Named explicitly when set. A resolved path alone does not say whether
        // it came from a default or from the environment, and "why is this run
        // different from the last one" is the question these answer.
        let env: Vec<&str> = [
            ("INFERRED_Q6K_SCALAR", self.q6k_scalar.get()),
            ("INFERRED_Q5K_SCALAR", self.q5k_scalar.get()),
            ("INFERRED_IQ4_STAGED", self.iq4_staged.get()),
            ("INFERRED_DELTA_SEQ", self.delta_seq.get()),
            ("INFERRED_ATTN_MMA", self.attn_mma.get()),
            ("INFERRED_ATTN_VMMA", self.attn_vmma.get()),
            ("INFERRED_ATTN_FUSED", self.attn_fused.get()),
        ]
        .iter()
        .filter(|(_, on)| *on)
        .map(|(k, _)| *k)
        .collect();
        if !env.is_empty() {
            out.push(("env", env.join(" ")));
        }
        out
    }

    /// Launch counts and expert residency for the turn just finished.
    ///
    /// **Counts, not timings.** `bench_launches` also *replays* every recorded
    /// launch, which re-executes its writes and corrupts activations and the KV
    /// cache on purpose — fine once generation is over, ruinous in a server
    /// that has to answer the next turn. For per-kernel timing use
    /// `generate --profile-device` with a long prompt, which runs the same
    /// prefill.
    fn device_report(&self) {
        if !self.report_per_turn.get() {
            return;
        }
        let c = self.stats();
        eprintln!(
            "  device   {} launches, {} crossings, {} syncs this session",
            c.launches,
            c.h2d_calls + c.d2h_calls,
            c.syncs,
        );
        if let Some(e) = self.expert_stats() {
            eprintln!(
                "  experts  {} reads, {:.1}% VRAM / {:.1}% PCIe, {} migrated",
                e.lookups(),
                100.0 * (1.0 - e.host_read_rate()),
                100.0 * e.host_read_rate(),
                e.migrated,
            );
        }
    }

    /// Expert placement, which happens inside the first prefill.
    ///
    /// `ExpertCache::table` is built on first sight of each `Experts` tensor,
    /// and the first sight is the first `moe_glu` — so the whole eager
    /// placement of 30,720 experts is billed to whichever pass touches them
    /// first. That is real time and it is charged in the right place; it is
    /// only the *label* that was missing.
    ///
    /// The three phases are already observed at the point each happens (see
    /// `ExpertStats::place_h2d_us`), because a whole-prefill number cannot say
    /// which part is expensive — an earlier attempt to explain a 46 s prefill
    /// from arithmetic over them was wrong by 40 s.
    ///
    /// **This covers expert placement only**, not the ordinary weight mirrors,
    /// which upload on first touch and are not timed. So the remainder of a
    /// first prefill is still slightly overstated, and saying so here is
    /// cheaper than a reader assuming otherwise.
    fn setup_cost(&self) -> Option<(u64, &'static str)> {
        let (alloc, upload, place) = self.setup_parts();
        let total = alloc + upload + place;
        if total == 0 {
            return None;
        }
        Some((total, "device setup"))
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
        // Placement happens inside the first forward pass, tensor by tensor as
        // each layer is reached, so the second pass is the first moment every
        // expert is certain to be placed and the model file certain to be dead
        // weight. Once: the counter never returns to two.
        self.attn_calls.set(0);
        self.passes_seen.set(self.passes_seen.get() + 1);
        if self.passes_seen.get() == 2 {
            if let Some(p) = self.model_path.borrow().as_ref() {
                experts::drop_file_cache(p);
            }
        }
        // Before the pass, never inside it: a recorded graph cannot have work
        // inserted, but the table it reads can be rewritten between replays.
        //
        // **This used to be gated on `n_tokens == 1`, which made the policy
        // inert in prefill.** The guard was unnecessary — the constraint is
        // *between passes*, which holds at any batch size — and it cost the
        // workload that needs migration most. Measured on a 1501-token prompt:
        // the VRAM read rate sat at 72.9%, which is exactly 22,392/30,720,
        // slots over pool, the signature of a placement using no information at
        // all. In decode the same policy reaches 93.7%.
        //
        // With a host read costing 11.4x a VRAM one (190.8 against 16.7 GB/s,
        // measured by `bench_expert_residency`), that gap was ~72% of prefill.
        self.migrate_experts(n_tokens);
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

    fn softmax(&self, x: &mut [f32], row: usize) {
        (*self).softmax(x, row)
    }

    fn attend(&self, a: &Attn<'_>, out: &mut [f32]) {
        (*self).attend(a, out)
    }

    // **These two were missing, and both defaulted to silence.** `serve` holds
    // `&Cuda`, not `Cuda`, so `engine.ops.device_report()` resolved to the
    // trait's no-op and `--profile-device` printed nothing per turn for as long
    // as it has existed. Twelfth instrument in this repo to fail by producing
    // no number at all. A forwarding impl that forwards *most* methods is the
    // same hazard as a counter that counts *most* allocations.
    fn config_report(&self) -> Vec<(&'static str, String)> {
        (*self).config_report()
    }


    fn device_report(&self) {
        (*self).device_report()
    }

    fn setup_cost(&self) -> Option<(u64, &'static str)> {
        (*self).setup_cost()
    }

    fn silu_mul(&self, gate: &mut [f32], up: &[f32]) {
        (*self).silu_mul(gate, up)
    }

    fn scale(&self, buf: &mut [f32], s: f32) {
        (*self).scale(buf, s)
    }

    fn mul_rows(&self, x: &mut [f32], w: &[f32]) {
        (*self).mul_rows(x, w)
    }
    fn silu(&self, x: &mut [f32]) {
        (*self).silu(x)
    }
    fn sigmoid(&self, x: &mut [f32]) {
        (*self).sigmoid(x)
    }
    fn mul_streams(&self, out: &mut [f32], h: &[f32], w: &[f32], n_stream: usize) {
        (*self).mul_streams(out, h, w, n_stream)
    }
    fn row_dot(&self, a: &[f32], b: &[f32], width: usize, out: &mut [f32]) {
        (*self).row_dot(a, b, width, out)
    }
    fn signed_sqrt_sigmoid(&self, s: &mut [f32]) {
        (*self).signed_sqrt_sigmoid(s)
    }
    fn dilated_conv(
        &self,
        state: &mut [f32],
        x: &[f32],
        weight: &[f32],
        kernel: usize,
        dilation: usize,
        out: &mut [f32],
    ) {
        (*self).dilated_conv(state, x, weight, kernel, dilation, out)
    }

    fn qsa_pool(&self, raw: &[u16], pooled: &mut [f32], p: &QsaPool<'_>) {
        (*self).qsa_pool(raw, pooled, p)
    }
    fn qsa_select(&self, q: &[f32], pooled: &[f32], sel: &QsaSelect, scores: &mut [f32], cells: &mut [u32]) {
        (*self).qsa_select(q, pooled, sel, scores, cells)
    }
    fn attend_sparse(&self, a: &Attn<'_>, cells: &[u32], sel: &QsaSelect, out: &mut [f32]) {
        (*self).attend_sparse(a, cells, sel, out)
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

    fn route(&self, probs: &mut [f32], n_expert: usize, n_used: usize) -> Route {
        (*self).route(probs, n_expert, n_used)
    }

    fn matmul_pair(
        &self,
        a: &Weights<'_>,
        b: &Weights<'_>,
        x: &[f32],
        out_a: &mut [f32],
        out_b: &mut [f32],
    ) {
        (*self).matmul_pair(a, b, x, out_a, out_b)
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

    fn read_state(&self, host: &mut [f32]) {
        (*self).read_state(host)
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
