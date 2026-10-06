//! The dense matmuls: Q8_0, the k-quants against a Q8_K activation (Q6_K,
//! Q5_K, IQ4_XS), and F32 alone or as a merged pair.

use crate::error::{Error, Result};
use crate::gguf::GgmlType;
use crate::ops::{Ops, Weights};
use crate::ops::cuda::{Cuda, KArg};

/// Tokens one warp of `matmul_q8_0_batch` holds in registers while it loads a
/// weight row once. **Must equal `MM_TOK` in `kernels/kernels.cu`** — it sizes
/// both the grid and the shared-memory request, and a mismatch would read past
/// the partials.
///
/// **Four is measured, and it is a trade-off rather than a maximum.** Prefill
/// tok/s on an 841-token prompt:
///
/// | `MM_TOK` | 1 | 2 | 4 | 8 | 16 |
/// |---|---|---|---|---|---|
/// | Qwen3.5-9B | 52.6 | 83.5 | **105.2** | 103.6 | 85.4 |
/// | Qwen3-0.6B | | | 646 | 649 | 695 |
///
/// 1 -> 4 is the reuse arriving, and it is the whole batched-prefill win on the
/// GPU. Past 4 the 9B *regresses*, because its widest tensor (`ffn_down`,
/// n_in 12288) needs `warps * MM_TOK * 384` floats of shared memory, so 8
/// forces the block down to two warps and 16 to one. Occupancy lost exceeds
/// reuse gained. Four is the largest tile that still keeps four warps there.
///
/// Note the 0.6B keeps improving to 16, because its widest tensor is a quarter
/// as wide and never loses warps. The constant is set for the model that
/// matters, which is the larger one — the same reasoning `CLAUDE.md` applies to
/// judging the GPU on the 0.6B at all.
const MM_TOK: usize = 8;

/// Blocks of a weight row in flight at once. **Must equal `MM_SEG` in
/// `kernels/kernels.cu`.** It fixes the shared-memory request independently of
/// `n_in`, which is what uncapped `MM_TOK`.
const MM_SEG: usize = 64;

impl Cuda {
    /// Force the untiled IQ4_XS matmul even for a batch. The A/B switch for
    /// `matmul_iq4_xs_q8_k_batch`; decode is unaffected either way, since it
    /// never reaches the tiled path.
    pub fn iq4_untiled(&self, on: bool) {
        self.iq4_untiled.set(on);
    }

    /// Force the one-token Q6_K matmul, so the token-tiled one can be priced
    /// against it **in the same process on the same request**. Whole-model
    /// prefill numbers taken from separate runs cannot separate the kernel from
    /// the conversation, the depth or the routing.
    pub fn q6k_scalar(&self, on: bool) {
        self.q6k_scalar.set(on);
    }

    /// Force the one-token Q5_K matmul, so the tiled one can be priced against
    /// it in the same process. **This one needs the toggle more than Q6_K did**:
    /// it was reverted on 09-09 on a 2,240-token reading of 399 against 402,
    /// inside a 2% spread, minutes before an 11,237-token reading of 380.9
    /// against 366.2 said the opposite.
    pub fn q5k_scalar(&self, on: bool) {
        self.q5k_scalar.set(on);
    }

    /// Force the `__dp4a` batched Q8_0 matmul, so the tensor-core one can be
    /// priced and checked against it in the same process. `INFERRED_Q8_SCALAR`
    /// sets it from the environment.
    pub fn q8_scalar(&self, on: bool) {
        self.q8_scalar.set(on);
    }

    /// Route batched IQ4_XS matmuls through `mma.m16n8k32.s8`.
    ///
    /// **A probe, not a product.** It answers one question: whether the int8
    /// tensor cores can reproduce the oracle bit for bit on this format. The
    /// reference sums 32 int8 products per sub-block, which is integer and
    /// therefore exact under any decomposition, and one such sub-block is
    /// exactly one `k = 32` MMA tile — so the f32 chain outside it can stay
    /// serial and ascending in a register, which is what the reference does.
    ///
    /// Off by default, and gated on `n_tok > 1` and `n_out % 16 == 0`, so
    /// nothing reaches it unless a caller asks.
    pub fn iq4_mma(&self, on: bool) {
        self.iq4_mma.set(on);
    }

    /// Select the shared-memory staged tile in place of the register-tiled MMA.
    ///
    /// Both kernels compute the same bits in the same order; they differ only
    /// in where the operands come from, so this is an A/B over access shape.
    /// `INFERRED_IQ4_STAGED=1` sets it for a whole process, which is what makes
    /// `serve` measurable without a rebuild.
    pub fn iq4_staged(&self, on: bool) {
        self.iq4_staged.set(on);
    }

    /// Defer the IQ4_XS f32 fold to once per superblock.
    ///
    /// A probe for what sits between the MMAs. **Not bit-exact** — the
    /// integer part is exact and order-free, but folding once per superblock
    /// rounds eight times less often than the reference does, which is a
    /// different f32 answer. Off unless a caller asks.
    pub fn iq4_fold_once(&self, on: bool) {
        self.iq4_fold_once.set(on);
    }

    /// Route batched Q5_K matmuls through the int8 tensor cores.
    ///
    /// On by default. **Not bit-exact** — see `matmul_q5_k_q8_k_mma` — so
    /// `false` here, like `INFERRED_Q5K_SCALAR`, restores the scalar kernel and
    /// with it equal bits. `q5k_scalar_restores_bit_equality` tests that it does.
    pub fn q5k_mma(&self, on: bool) {
        self.q5k_mma.set(on);
    }

    /// Use the staged F32 matmul. Off by default; see `matmul_f32`.
    pub fn f32_staged(&self, on: bool) {
        self.f32_staged.set(on);
    }

    /// The k-quant matmuls: Q6_K, Q5_K and IQ4_XS, all against a Q8_K
    /// activation.
    ///
    /// **One kernel serves decode and prefill**, with the batch on `blockIdx.y`
    /// rather than in a second kernel. That is deliberately unlike the Q8_0
    /// pair, where `matmul_q8_0_batch` exists to amortize a weight load across
    /// `MM_TOK` tokens: here a warp still re-reads its weight row per token, so
    /// prefill weight traffic scales with the batch. Correct first, and the
    /// reuse variant arrives as a measured change against this baseline —
    /// which is also what keeps decode and prefill unable to disagree while the
    /// exactness claim is being established.
    /// NVFP4 matmuls as FP4 x FP4 on the tensor cores (`true`) or against a
    /// Q8_0 activation (`false`, the exact path). On by default in an `sm_120a`
    /// build; `INFERRED_NVFP4_Q8=1` sets `false` for a whole process. Has no
    /// effect in an `sm_120` build, which carries no FP4 kernels.
    pub fn nvfp4_fp4(&self, on: bool) {
        self.nvfp4_fp4.set(on);
    }

    /// NVFP4 as FP4 x FP4 on the tensor cores: `matmul_nvfp4_fp4_mma` against an
    /// activation from `quantize_nvfp4_act`.
    ///
    /// A precision departure from `matmul_nvfp4`: the activation is FP4, and
    /// the core adds sub-block terms in its own order. The reference is
    /// `ops::naive::dot_nvfp4_fp4` on `Fp4Row`, within the f32 chain bound that
    /// `the_fp4_tensor_core_matmul_is_within_the_chain_bound` checks.
    fn matmul_nvfp4_fp4(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) -> Result<()> {
        let n_tok = x.len() / w.n_in;
        let wd = self.expert_or_resident(w)?;
        let (dd, qd) = self.quantized_fp4(x, n_tok * (w.n_in / 16))?;
        let od = self.mirror_out(out)?;
        // Four warps per block, each 16 weight rows; 8 tokens per block row.
        let block = 128u32;
        let args = [
            KArg::I32(w.n_in as i32),
            KArg::I32(w.n_out as i32),
            KArg::I32(n_tok as i32),
            KArg::Ptr(wd),
            KArg::Ptr(dd),
            KArg::Ptr(qd),
            KArg::Ptr(od),
        ];
        self.note_shape("matmul_nvfp4_fp4_mma", w.n_in, w.n_out);
        // SAFETY: parameters match `matmul_nvfp4_fp4_mma`; the grid covers
        // `n_out` rows in 64s by `n_tok` tokens in 8s, warps past either end
        // return, and partial tiles clamp their loads to valid rows and tokens.
        unsafe {
            self.launch_grid2(
                "matmul_nvfp4_fp4_mma",
                w.n_out.div_ceil(64) as u32,
                n_tok.div_ceil(8) as u32,
                block,
                0,
                &args,
            )
        }
    }

    /// NVFP4 against a Q8_0 activation: `ops::naive::dot_nvfp4_q8_0` on the
    /// device, the exact path. The shared expert and the LM head of the NVFP4
    /// checkpoint; the second scale is the model's to apply, as in llama.cpp.
    ///
    /// One thread per output row, serial within it, the batch on `blockIdx.y` —
    /// correct first, as `matmul_q8_0` began. The FP4 x FP4 tensor-core kernel
    /// is the fast path measured against this one.
    fn matmul_nvfp4(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) -> Result<()> {
        let n_tok = x.len() / w.n_in;
        let wd = self.expert_or_resident(w)?;
        let (sd, qd) = self.quantized(x, n_tok * (w.n_in / 32))?;
        let od = self.mirror_out(out)?;
        let block = 32u32;
        let args = [
            KArg::I32(w.n_in as i32),
            KArg::I32(w.n_out as i32),
            KArg::Ptr(wd),
            KArg::Ptr(sd),
            KArg::Ptr(qd),
            KArg::Ptr(od),
        ];
        self.note_shape("matmul_nvfp4_q8_0", w.n_in, w.n_out);
        // SAFETY: parameters match `matmul_nvfp4_q8_0`; the grid covers `n_out`
        // rows by `n_tok` tokens and threads past `n_out` return first.
        unsafe {
            self.launch_grid2(
                "matmul_nvfp4_q8_0",
                w.n_out.div_ceil(block as usize) as u32,
                n_tok as u32,
                block,
                0,
                &args,
            )
        }
    }

    fn matmul_kquant(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) -> Result<()> {
        const QK_K: usize = 256;
        let n_tok = x.len() / w.n_in;
        let n_super = w.n_in / QK_K;

        // The weight goes up in the file's own layout, unrepacked. The Q8_0
        // path repacks for 16-byte alignment; that was tried there *and*
        // reverted once for being 20% slower, so the same bet is not made
        // twice untested. Whether these three want a repack is a measurement,
        // not an assumption.
        let wd = self.expert_or_resident(w)?;
        let (sd, qd, bd) = self.quantized_k(x, n_tok * n_super)?;
        let od = self.mirror_out(out)?;

        // 128 threads is four warps, so four output rows per block.
        let block = 128u32;
        let rows_per_block = (block / 32) as usize;
        let grid_rows = w.n_out.div_ceil(rows_per_block) as u32;

        let (name, args): (&'static str, Vec<KArg>) = match w.ty {
            GgmlType::Q6K => (
                "matmul_q6_k_q8_k",
                vec![
                    KArg::I32(w.n_in as i32),
                    KArg::I32(w.n_out as i32),
                    KArg::Ptr(wd),
                    KArg::Ptr(sd),
                    KArg::Ptr(qd),
                    KArg::Ptr(od),
                ],
            ),
            GgmlType::Q5K => (
                "matmul_q5_k_q8_k",
                vec![
                    KArg::I32(w.n_in as i32),
                    KArg::I32(w.n_out as i32),
                    KArg::Ptr(wd),
                    KArg::Ptr(sd),
                    KArg::Ptr(qd),
                    // Only Q5_K reads the per-16 sums, and omitting them here
                    // would leave the other two correct.
                    KArg::Ptr(bd),
                    KArg::Ptr(od),
                ],
            ),
            _ => (
                "matmul_iq4_xs_q8_k",
                vec![
                    KArg::I32(w.n_in as i32),
                    KArg::I32(w.n_out as i32),
                    KArg::Ptr(wd),
                    KArg::Ptr(sd),
                    KArg::Ptr(qd),
                    KArg::Ptr(od),
                ],
            ),
        };

        // **The token-reuse variant, and only for a batch.** IQ4_XS is the
        // 35B's dense in-block format and `matmul_iq4_xs_q8_k` re-reads its
        // weight row once per token — measured at **23.9% of a 4,000-token
        // prefill**, the largest single kernel, ~1.03 GB of weight per token.
        // `matmul_iq4_xs_q8_k_batch` loads a superblock once for `IQ4_TOK`
        // tokens.
        //
        // Gated on `n_tok > 1`, which is what makes this safe for decode
        // rather than merely tested: **decode runs the identical kernel it ran
        // before**, so its bits, its occupancy and its recorded graph are
        // untouched by construction. A graph only ever sees the `n_tok == 1`
        // name, so the launch-sequence rule the merged attention kernel relies
        // on holds here too.
        const IQ4_TOK: usize = 8;

        // The tensor-core probe. One warp per 16x8 output tile, so `n_out` must
        // be a whole number of 16-row tiles — every IQ4_XS width the 35B uses
        // is (512, 1024, 2048, 4096, 8192), and anything else falls through to
        // the kernels above rather than reading past a row.
        // The staged tile, ahead of the register-tiled MMA because it is the
        // same arithmetic in the same order and differs only in where the
        // operands come from. `ST_TILE_M` is 64, so the row guard is the same
        // 16-row multiple the kernel below needs and a partial tile is clamped
        // on load and dropped on write-back.
        if n_tok > 1
            && self.iq4_staged.get()
            && matches!(w.ty, GgmlType::Iq4Xs)
            && w.n_out % 16 == 0
        {
            const ST_TILE_M: usize = 64;
            const ST_TILE_N: usize = 64;
            let staged = "matmul_iq4_xs_q8_k_staged";
            let sargs = [
                KArg::I32(w.n_in as i32),
                KArg::I32(w.n_out as i32),
                KArg::I32(n_tok as i32),
                KArg::Ptr(wd),
                KArg::Ptr(sd),
                KArg::Ptr(qd),
                KArg::Ptr(od),
            ];
            self.note_shape(staged, w.n_in, w.n_out);
            // SAFETY: parameters match `matmul_iq4_xs_q8_k_staged`; the block is
            // `ST_TILE_M / 16` warps, the kernel declares its shared memory
            // statically, and both tails are clamped inside it.
            return unsafe {
                self.launch_grid2(
                    staged,
                    w.n_out.div_ceil(ST_TILE_M) as u32,
                    n_tok.div_ceil(ST_TILE_N) as u32,
                    (ST_TILE_M / 16 * 32) as u32,
                    0,
                    &sargs,
                )
            };
        }

        if n_tok > 1
            && self.iq4_mma.get()
            && matches!(w.ty, GgmlType::Iq4Xs)
            && w.n_out % 16 == 0
        {
            // Same geometry and same arguments as the real kernel, so the A/B
            // differs in nothing but what happens between the MMAs.
            let mma = if self.iq4_fold_once.get() {
                "dbg_iq4_mma_foldonce"
            } else {
                "matmul_iq4_xs_q8_k_mma"
            };
            let margs = [
                KArg::I32(w.n_in as i32),
                KArg::I32(w.n_out as i32),
                KArg::I32(n_tok as i32),
                KArg::Ptr(wd),
                KArg::Ptr(sd),
                KArg::Ptr(qd),
                KArg::Ptr(od),
            ];
            self.note_shape(mma, w.n_in, w.n_out);
            // SAFETY: parameters match `matmul_iq4_xs_q8_k_mma`; `block` is 128
            // threads, so four warps cover 64 rows per block, the kernel clamps
            // its own token tail, and it uses no dynamic shared memory.
            return unsafe {
                self.launch_grid2(
                    mma,
                    w.n_out.div_ceil(128) as u32,
                    // 8 tokens per MMA times MMA_NTILE tiles per weight load.
                    n_tok.div_ceil(32) as u32,
                    256,
                    0,
                    &margs,
                )
            };
        }

        if n_tok > 1 && !self.iq4_untiled.get() && matches!(w.ty, GgmlType::Iq4Xs) {
            let batched = "matmul_iq4_xs_q8_k_batch";
            let bargs = [
                KArg::I32(w.n_in as i32),
                KArg::I32(w.n_out as i32),
                KArg::I32(n_tok as i32),
                KArg::Ptr(wd),
                KArg::Ptr(sd),
                KArg::Ptr(qd),
                KArg::Ptr(od),
            ];
            self.note_shape(batched, w.n_in, w.n_out);
            // SAFETY: parameters match `matmul_iq4_xs_q8_k_batch`; the grid
            // covers `n_out` rows by ceil(n_tok / IQ4_TOK) token tiles, the
            // kernel clamps its own tail tile, and it uses no dynamic shared
            // memory.
            return unsafe {
                self.launch_grid2(
                    batched,
                    grid_rows,
                    n_tok.div_ceil(IQ4_TOK) as u32,
                    block,
                    0,
                    &bargs,
                )
            };
        }

        // **Q6_K, the same reuse and the same gate.** `matmul_q6_k_q8_k` takes
        // the token as `blockIdx.y`, so it re-reads *and re-unpacks* every
        // weight row once per token — and Q6_K's unpack is a four-way branch on
        // the sub-block over a split nibble/bit-pair, a dozen instructions per
        // weight. Measured at **18.4% of an 11,237-token prefill**, the largest
        // matmul in the profile and second only to attention.
        //
        // `matmul_q6_k_q8_k_tok` holds the unpacked weight across `Q6K_TOK`
        // tokens. It is **bit-identical by construction**: each output keeps
        // the oracle's eight interleaved f32 accumulators, in the same lane and
        // the same ascending order. Only how many outputs one weight load
        // serves changes.
        //
        // Gated on `n_tok > 1` like the IQ4_XS variant above, so decode runs
        // the identical kernel it always has and a recorded graph never sees
        // this name.
        // Q5_K, the same reuse and the same gate. Its twelve-byte scale/min
        // shuffle and its five-bit unpack are both token-independent, and it is
        // **9.7% of prefill at 0.26% of the int8 ceiling** — the worst-utilised
        // kernel in the engine. Both of its f32 chains are preserved per token
        // and in order, so it is bit-identical rather than within a tolerance.
        const Q5K_TOK: usize = 8;

        // **Q5_K on the tensor cores, and the one matmul here outside the
        // bit-exact set.** One `mma.m16n8k32` covers one 32-weight sub-block and
        // one scale, and the reference already folds to f32 once per superblock,
        // so everything reproduces except the eight int32 lanes it keeps — which
        // an MMA cannot hand back. That collapses eight roundings per superblock
        // into one: more accurate, and different. `INFERRED_Q5K_SCALAR` restores
        // the scalar path and with it equal bits.
        //
        // Gated on `n_tok > 1` like every other prefill-only arm, so decode runs
        // the kernel it always has and a recorded graph never sees this name.
        // `n_out % 16 == 0` because a warp owns a 16-row tile; `attn_output` is
        // 4096x2048, so every Q5_K width in the 35B qualifies.
        if n_tok > 1
            && self.q5k_mma.get()
            && !self.q5k_scalar.get()
            && matches!(w.ty, GgmlType::Q5K)
            && w.n_out % 16 == 0
        {
            const Q5K_MMA_NTILE: usize = 4;
            let mma = "matmul_q5_k_q8_k_mma";
            let margs = [
                KArg::I32(w.n_in as i32),
                KArg::I32(w.n_out as i32),
                KArg::I32(n_tok as i32),
                KArg::Ptr(wd),
                KArg::Ptr(sd),
                KArg::Ptr(qd),
                KArg::Ptr(bd),
                KArg::Ptr(od),
            ];
            self.note_shape(mma, w.n_in, w.n_out);
            // SAFETY: parameters match `matmul_q5_k_q8_k_mma`; 256 threads is
            // eight warps covering 128 rows, the kernel clamps both tails, and
            // it uses no dynamic shared memory.
            return unsafe {
                self.launch_grid2(
                    mma,
                    w.n_out.div_ceil(128) as u32,
                    n_tok.div_ceil(8 * Q5K_MMA_NTILE) as u32,
                    256,
                    0,
                    &margs,
                )
            };
        }

        if n_tok > 1 && !self.q5k_scalar.get() && matches!(w.ty, GgmlType::Q5K) {
            let tiled = "matmul_q5_k_q8_k_tok";
            let targs = [
                KArg::I32(w.n_in as i32),
                KArg::I32(w.n_out as i32),
                KArg::I32(n_tok as i32),
                KArg::Ptr(wd),
                KArg::Ptr(sd),
                KArg::Ptr(qd),
                KArg::Ptr(bd),
                KArg::Ptr(od),
            ];
            self.note_shape(tiled, w.n_in, w.n_out);
            // SAFETY: parameters match `matmul_q5_k_q8_k_tok`; the grid covers
            // `n_out` rows by ceil(n_tok / Q5K_TOK) token tiles, the kernel
            // clamps its own tail, and it uses no dynamic shared memory.
            return unsafe {
                self.launch_grid2(tiled, grid_rows, n_tok.div_ceil(Q5K_TOK) as u32, block, 0, &targs)
            };
        }

        const Q6K_TOK: usize = 8;
        if n_tok > 1 && !self.q6k_scalar.get() && matches!(w.ty, GgmlType::Q6K) {
            let tiled = "matmul_q6_k_q8_k_tok";
            let targs = [
                KArg::I32(w.n_in as i32),
                KArg::I32(w.n_out as i32),
                KArg::I32(n_tok as i32),
                KArg::Ptr(wd),
                KArg::Ptr(sd),
                KArg::Ptr(qd),
                KArg::Ptr(od),
            ];
            self.note_shape(tiled, w.n_in, w.n_out);
            // SAFETY: parameters match `matmul_q6_k_q8_k_tok`; the grid covers
            // `n_out` rows by ceil(n_tok / Q6K_TOK) token tiles, the kernel
            // clamps its own tail tile, and it uses no dynamic shared memory.
            return unsafe {
                self.launch_grid2(
                    tiled,
                    grid_rows,
                    n_tok.div_ceil(Q6K_TOK) as u32,
                    block,
                    0,
                    &targs,
                )
            };
        }

        self.note_shape(name, w.n_in, w.n_out);
        // SAFETY: parameters match the named kernel; the grid covers exactly
        // `n_out` rows by `n_tok` tokens, and the kernels use no dynamic
        // shared memory.
        unsafe { self.launch_grid2(name, grid_rows, n_tok as u32, block, 0, &args) }
    }

    /// Whether a staged `matmul_f32_t` can serve this output width.
    ///
    /// One accumulator thread per output row, so the block has to be at least
    /// as wide. Every F32 matmul in `qwen35moe` is 1, 32 or 256 wide; anything
    /// larger falls back to the row-per-thread kernel, which is correct just
    /// slower.
    fn n_out_fits_impl(n_out: usize, block: u32) -> bool {
        n_out > 0 && n_out <= block as usize
    }

    /// Whether the merged kernel can serve this pair.
    ///
    /// **A fast path, not a contract — and getting that wrong broke a model.**
    /// The first version returned an error for anything else, on the assumption
    /// that `ssm_alpha` and `ssm_beta` are F32 because `CLAUDE.md`'s type table
    /// says so. That table describes the *35B file*: on the 9B both are Q8_0.
    /// So an optimisation that does not apply became a reported error, and
    /// `restoring_a_checkpoint_reproduces_the_continuation_on_the_device`
    /// failed on a model the change was never meant to touch.
    ///
    /// An optional merge must decline, not fail. The caller falls back to the
    /// two matmuls, which is exactly what the seam's own default does.
    pub(super) fn pairable(a: &Weights<'_>, b: &Weights<'_>) -> bool {
        a.ty == GgmlType::F32 && b.ty == GgmlType::F32 && a.n_in == b.n_in
    }

    pub(super) fn matmul_pair_impl(
        &self,
        a: &Weights<'_>,
        b: &Weights<'_>,
        x: &[f32],
        out_a: &mut [f32],
        out_b: &mut [f32],
    ) -> Result<()> {
        debug_assert!(Self::pairable(a, b), "matmul_pair_impl given a pair it cannot merge");
        let n_tok = x.len() / a.n_in;
        let n_out = a.n_out + b.n_out;
        self.note_shape("matmul_f32_t_pair", a.n_in, n_out);
        let wd = self.resident_f32_t_pair(a, b)?;
        let xd = self.mirror_in(x)?;
        let oa = self.mirror_out(out_a)?;
        let ob = self.mirror_out(out_b)?;
        let args = [
            KArg::I32(a.n_in as i32),
            KArg::I32(a.n_out as i32),
            KArg::I32(b.n_out as i32),
            KArg::Ptr(wd),
            KArg::Ptr(xd),
            KArg::Ptr(oa),
            KArg::Ptr(ob),
        ];
        let block = 128u32;
        // SAFETY: parameters match `matmul_f32_t_pair`; the grid covers exactly
        // `n_a + n_b` rows by `n_tok` tokens, the weight is the interleaved
        // column-major stack of both, and each thread writes one element of
        // whichever output its row came from.
        unsafe {
            self.launch_grid2(
                "matmul_f32_t_pair",
                n_out.div_ceil(block as usize) as u32,
                n_tok as u32,
                block,
                0,
                &args,
            )
        }
    }

    /// Threads a staged `matmul_f32_t` launches, whatever `n_out` is.
    ///
    /// The point of the staged form is that *loading* is done by the whole
    /// block while only `n_out` threads accumulate, so this is deliberately
    /// unrelated to the output width.
    const F32_STAGE_BLOCK: u32 = 256;

    /// Floats of dynamic shared memory a staged launch may use.
    ///
    /// 48 KiB is the per-block default without an opt-in; staying under it
    /// keeps the launch from needing `cuFuncSetAttribute` and keeps occupancy
    /// at two blocks per SM, which matters not at all here (there is one block)
    /// but would if this kernel ever served a batch.
    const F32_STAGE_FLOATS: usize = 12 * 1024;

    /// The F32 matmul, which on the 35B is the MoE router and nothing else.
    ///
    /// Kept as the slow one-thread-per-row shape on purpose; see `matmul_f32`
    /// in kernels.cu for why exactness is worth more than speed here.
    fn matmul_f32(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) -> Result<()> {
        let n_tok = x.len() / w.n_in;
        self.note_shape("matmul_f32_t", w.n_in, w.n_out);
        let wd = self.resident_f32_t(w)?;
        let xd = self.mirror_in(x)?;
        let od = self.mirror_out(out)?;

        // **Staged when the output fits one block, which on this model is
        // always.** Decomposed at the real geometry: of a 40.0 us call, 14.5 us
        // is the launch and the loop with both loads removed, and the serial
        // accumulation chain costs 1% (`dbg_f32_nochain`, 39.7 us). So the
        // floor was never the exactness — it was one thread per output row
        // issuing 2048 loads with ~8 outstanding, on a block of `n_out` threads,
        // 32 of them for `ssm_alpha`. Staging through shared memory lets the
        // whole block fetch while thread `j` still walks its own row serially in
        // ascending `k`, which is `ops::naive::dot_row`'s order exactly.
        // Measured 40.0 -> 15.1 us, at the no-loads floor, and bit-identical.
        if self.f32_staged.get() && Self::n_out_fits_impl(w.n_out, Self::F32_STAGE_BLOCK) {
            let kt = (Self::F32_STAGE_FLOATS / (w.n_out + 1)).clamp(1, w.n_in);
            let shared = (kt * (w.n_out + 1) * 4) as u32;
            let args = [
                KArg::I32(w.n_in as i32),
                KArg::I32(w.n_out as i32),
                KArg::I32(kt as i32),
                KArg::Ptr(wd),
                KArg::Ptr(xd),
                KArg::Ptr(od),
            ];
            // SAFETY: parameters match `matmul_f32_t_staged`. The shared
            // request is exactly what the kernel indexes — `kt * n_out` floats
            // of weight tile then `kt` of activation — and `n_out` is at most
            // the block, so every output row has an accumulator thread.
            return unsafe {
                self.launch_grid2(
                    "matmul_f32_t_staged",
                    1,
                    n_tok as u32,
                    Self::F32_STAGE_BLOCK,
                    shared,
                    &args,
                )
            };
        }

        let args = [
            KArg::I32(w.n_in as i32),
            KArg::I32(w.n_out as i32),
            KArg::Ptr(wd),
            KArg::Ptr(xd),
            KArg::Ptr(od),
        ];
        let block = 128u32;
        // SAFETY: parameters match `matmul_f32_t`; the grid covers exactly
        // `n_out` rows by `n_tok` tokens, and the weight is column-major.
        unsafe {
            self.launch_grid2(
                "matmul_f32_t",
                w.n_out.div_ceil(block as usize) as u32,
                n_tok as u32,
                block,
                0,
                &args,
            )
        }
    }

    /// F16 weights: `matmul_f32`'s row-per-thread shape over an f16 weight,
    /// bit-identical to the oracle. Only Qwen3.8-Flash-Next's 0.2B test model
    /// reaches it (its QSA indexer), so it has no staged form.
    fn matmul_f16(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) -> Result<()> {
        let n_tok = x.len() / w.n_in;
        self.note_shape("matmul_f16_t", w.n_in, w.n_out);
        let wd = self.resident_f16_t(w)?;
        let xd = self.mirror_in(x)?;
        let od = self.mirror_out(out)?;
        let args = [
            KArg::I32(w.n_in as i32),
            KArg::I32(w.n_out as i32),
            KArg::Ptr(wd),
            KArg::Ptr(xd),
            KArg::Ptr(od),
        ];
        let block = 128u32;
        // SAFETY: parameters match `matmul_f16_t`; the grid covers exactly
        // `n_out` rows by `n_tok` tokens, and the f16 weight is column-major.
        unsafe {
            self.launch_grid2(
                "matmul_f16_t",
                w.n_out.div_ceil(block as usize) as u32,
                n_tok as u32,
                block,
                0,
                &args,
            )
        }
    }

    pub(super) fn matmul_impl(&self, w: &Weights<'_>, x: &[f32], out: &mut [f32]) -> Result<()> {
        match w.ty {
            GgmlType::Q8_0 => {}
            GgmlType::Q6K | GgmlType::Q5K | GgmlType::Iq4Xs => {
                return self.matmul_kquant(w, x, out);
            }
            GgmlType::F32 => return self.matmul_f32(w, x, out),
            GgmlType::F16 => return self.matmul_f16(w, x, out),
            GgmlType::Nvfp4 if self.nvfp4_fp4.get() && cfg!(nvfp4_block_scale) => {
                return self.matmul_nvfp4_fp4(w, x, out);
            }
            GgmlType::Nvfp4 => return self.matmul_nvfp4(w, x, out),
            other => {
                return Err(Error::Cuda {
                    what: "matmul",
                    detail: format!(
                        "{other:?} has no CUDA kernel; this backend implements Q8_0, the \
                         three k-quants the 35B uses (Q5_K, Q6_K, IQ4_XS) and NVFP4"
                    ),
                });
            }
        }

        // Quantized on the device. It began on the host, because the scale
        // round-trips through f16 and a differing rounding mode there would be
        // invisible until it moved a quant; `tests/cuda_ops.rs` now checks that
        // against the oracle on real data. Doing it here is what lets a
        // matmul's input stay on the card instead of coming back to be
        // quantized and going out again.
        let n_blocks = w.n_in / 32;
        let n_tok = x.len() / w.n_in;
        let (ws, wq) = self.resident_q8_0(w)?;
        // One thread per 32-element block over the whole batch. The quantize
        // kernel needs no batch awareness: `x` is token-major and contiguous,
        // so its blocks already lay out as `[token][block]`, which is exactly
        // what the matmul indexes.
        let (sd, qd) = self.quantized(x, n_tok * n_blocks)?;
        let od = self.mirror_out(out)?;

        // 128 threads is four warps, so four output rows per block.
        let block = 128u32;
        let rows_per_block = (block / 32) as usize;
        let grid_rows = w.n_out.div_ceil(rows_per_block) as u32;

        // The batched kernel folds each segment of the row into a running total
        // rather than holding every partial, so its shared-memory request is
        // `warps * MM_TOK * MM_SEG` floats and **does not grow with `n_in`**.
        // It therefore keeps four warps on every tensor either model has, which
        // is what lets `MM_TOK` be chosen for reuse instead of for occupancy.
        let b_shared = (rows_per_block * MM_TOK * MM_SEG * 4) as u32;

        if n_tok == 1 {
            let args = [
                KArg::I32(w.n_in as i32),
                KArg::I32(w.n_out as i32),
                KArg::Ptr(ws),
                KArg::Ptr(wq),
                KArg::Ptr(sd),
                KArg::Ptr(qd),
                KArg::Ptr(od),
            ];
            let shared = (rows_per_block * n_blocks * 4) as u32;
            // SAFETY: parameters match `matmul_q8_0_warp`; the grid covers
            // exactly `n_out` rows and `shared` is `warps * n_blocks` floats,
            // which is what the kernel indexes.
            unsafe { self.launch_shared("matmul_q8_0_warp", grid_rows, block, shared, &args)? };
            return Ok(());
        }

        // **Prefill on the int8 tensor cores**, bit-identical to the batched
        // kernel below, which `INFERRED_Q8_SCALAR` restores. A warp holds 16
        // weight rows for 32 tokens where the batched kernel holds one row for
        // `MM_TOK` 8, so weights are read a quarter as often, and the products
        // run on the tensor cores. Rows come in sixteens, as for IQ4_XS.
        if !self.q8_scalar.get() && w.n_out % 16 == 0 {
            let margs = [
                KArg::I32(w.n_in as i32),
                KArg::I32(w.n_out as i32),
                KArg::I32(n_tok as i32),
                KArg::Ptr(ws),
                KArg::Ptr(wq),
                KArg::Ptr(sd),
                KArg::Ptr(qd),
                KArg::Ptr(od),
            ];
            self.note_shape("matmul_q8_0_mma", w.n_in, w.n_out);
            // SAFETY: parameters match `matmul_q8_0_mma`; `block` is 256 threads,
            // so eight warps cover 128 rows per block, `n_out % 16 == 0` keeps
            // every warp's 16 rows inside the tensor, the kernel clamps its own
            // token tail, and it uses no dynamic shared memory.
            unsafe {
                self.launch_grid2(
                    "matmul_q8_0_mma",
                    w.n_out.div_ceil(128) as u32,
                    // 8 tokens per MMA times MMA_NTILE tiles per weight load.
                    n_tok.div_ceil(32) as u32,
                    256,
                    0,
                    &margs,
                )?
            };
            return Ok(());
        }

        // **Decode keeps the kernel it was measured on.** Every published
        // number here was taken on `matmul_q8_0_warp`, and it is the kernel the
        // decode CUDA graph records; the batched one is strictly for prefill,
        // where a warp amortizes one weight load across `MM_TOK` tokens.
        let args = [
            KArg::I32(w.n_in as i32),
            KArg::I32(w.n_out as i32),
            KArg::I32(n_tok as i32),
            KArg::Ptr(ws),
            KArg::Ptr(wq),
            KArg::Ptr(sd),
            KArg::Ptr(qd),
            KArg::Ptr(od),
        ];
        // SAFETY: parameters match `matmul_q8_0_batch`; the grid covers exactly
        // `n_out` rows by `ceil(n_tok / MM_TOK)` token tiles, and `b_shared` is
        // `warps * MM_TOK * n_blocks` floats, which is what the kernel indexes.
        unsafe {
            self.launch_grid2(
                "matmul_q8_0_batch",
                grid_rows,
                n_tok.div_ceil(MM_TOK) as u32,
                block,
                b_shared,
                &args,
            )?
        };
        Ok(())
    }

    /// Run `quantize_q8_k` on the device and read all three outputs back.
    ///
    /// **For tests, and it earns its place.** A wrong Q8_K quantization shows
    /// up in every k-quant dot at once, so without this a single defect looks
    /// like three broken matmuls — and a defect in `bsums` alone looks like one
    /// broken matmul (Q5_K) with two healthy ones, which is worse. This makes
    /// the quantizer directly comparable to `quant::kquant::Q8KRow::from_f32`.
    pub fn quantize_q8_k_readback(&self, x: &[f32]) -> Result<(Vec<f32>, Vec<i8>, Vec<i16>)> {
        const QK_K: usize = 256;
        if x.len() % QK_K != 0 {
            return Err(Error::Cuda {
                what: "quantize_q8_k_readback",
                detail: format!("{} values is not a whole number of super-blocks", x.len()),
            });
        }
        let n_super = x.len() / QK_K;
        self.begin_pass(1);
        self.host_wrote(x);
        let (sd, qd, bd) = self.quantized_k(x, n_super)?;
        self.sync()?;

        let mut scales = vec![0.0f32; n_super];
        let mut quants = vec![0i8; n_super * QK_K];
        let mut bsums = vec![0i16; n_super * (QK_K / 16)];
        self.d2h(&mut scales, sd)?;
        self.d2h(&mut quants, qd)?;
        self.d2h(&mut bsums, bd)?;
        Ok((scales, quants, bsums))
    }

    /// Run `quantize_nvfp4_act` on the device and read both outputs back:
    /// `(scale codes, packed E2M1 codes)`, for comparison with
    /// `quant::fp4_activation`. For tests, as `quantize_q8_k_readback` is.
    pub fn quantize_nvfp4_act_readback(&self, x: &[f32]) -> Result<(Vec<u8>, Vec<u8>)> {
        if x.len() % 64 != 0 {
            return Err(Error::Cuda {
                what: "quantize_nvfp4_act_readback",
                detail: format!("{} values is not a whole number of NVFP4 blocks", x.len()),
            });
        }
        let n_sub = x.len() / 16;
        self.begin_pass(1);
        self.host_wrote(x);
        let (dd, qd) = self.quantized_fp4(x, n_sub)?;
        self.sync()?;
        let mut scales = vec![0u8; n_sub];
        let mut codes = vec![0u8; n_sub * 8];
        self.d2h(&mut scales, dd)?;
        self.d2h(&mut codes, qd)?;
        Ok((scales, codes))
    }
}
