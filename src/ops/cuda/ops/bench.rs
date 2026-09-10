//! Benches and the launch recorder: the shapes and launches a run made,
//! replayed and timed with the queue kept full, and the decomposition variants.

use crate::error::Result;
use crate::ops::cuda::{Cuda, DeviceBuffer, KArg, LaunchKey, check, ffi};

/// One matmul shape, measured on the live device with the launch queue kept
/// full — the thing `--profile-kernels` cannot report, because synchronizing
/// per launch is exactly what changes the answer.
///
/// `gpu` and `issue` are the same launches timed two ways, so the larger of
/// them is what binds for this shape. `grouped` is the same arithmetic in one
/// launch instead of `group`, which prices fusing the routed FFN.
#[derive(Debug, Clone, Copy)]
pub struct ShapeBench {
    pub kernel: &'static str,
    pub n_in: usize,
    pub n_out: usize,
    /// Launches of this shape observed during the run being profiled.
    pub calls: u64,
    /// Device microseconds per launch.
    pub gpu_us: f64,
    /// Host microseconds per launch, i.e. what it costs to describe the work.
    pub issue_us: f64,
    /// Device microseconds for one launch covering `group` shapes' rows.
    pub grouped_us: f64,
    pub group: usize,
}

/// One kernel launch the run actually made, replayed and timed.
///
/// See [`Cuda::bench_launches`]. Unlike [`ShapeBench`] this cannot omit a
/// kernel: the entry exists because the launch happened.
#[derive(Debug, Clone)]
pub struct LaunchBench {
    pub kernel: &'static str,
    pub grid: (u32, u32),
    pub block: u32,
    /// Times this exact launch was issued during the run.
    pub calls: u64,
    /// Device microseconds for one launch, queue kept full, best of four.
    pub gpu_us: f64,
    /// Host microseconds to issue one.
    pub issue_us: f64,
}

impl LaunchBench {
    pub fn gpu_ms_per_token(&self, tokens: u64) -> f64 {
        self.calls as f64 / tokens.max(1) as f64 * self.gpu_us / 1000.0
    }
    pub fn calls_per_token(&self, tokens: u64) -> f64 {
        self.calls as f64 / tokens.max(1) as f64
    }
    /// Whether the host, not the device, limited this measurement.
    pub fn host_limited(&self) -> bool {
        self.gpu_us > 0.0 && self.issue_us / self.gpu_us > 0.7
    }
}

impl ShapeBench {
    /// Whether the host, not the device, limited this measurement.
    ///
    /// The event pair brackets the launch loop, so if the host cannot keep the
    /// queue full it times the stall and `gpu` collapses onto `issue`. Taking
    /// the best of several runs removes most of it; what survives is reported
    /// rather than trusted, because the alternative is reading a 2 KiB
    /// elementwise kernel as costing 22 us of device time.
    pub fn host_limited(&self) -> bool {
        self.gpu_us > 0.0 && self.issue_us / self.gpu_us > 0.7
    }


    /// Device milliseconds this shape costs per token, at the observed rate.
    pub fn gpu_ms_per_token(&self, tokens: u64) -> f64 {
        self.calls as f64 / tokens.max(1) as f64 * self.gpu_us / 1000.0
    }

    /// Host milliseconds per token, same basis.
    pub fn issue_ms_per_token(&self, tokens: u64) -> f64 {
        self.calls as f64 / tokens.max(1) as f64 * self.issue_us / 1000.0
    }

    /// What fusing `group` launches into one would leave, per token, on the
    /// device. Compare against [`ShapeBench::gpu_ms_per_token`].
    pub fn grouped_ms_per_token(&self, tokens: u64) -> f64 {
        let launches = self.calls as f64 / self.group.max(1) as f64;
        launches / tokens.max(1) as f64 * self.grouped_us / 1000.0
    }
}

impl Cuda {
    /// Replace every kernel with a no-op, keeping the launch pattern exactly.
    ///
    /// The output is meaningless; the *time* is the point. See `noop` in
    /// kernels.cu.
    pub fn null_kernels(&self, on: bool) {
        self.null_kernels.set(on);
    }

    /// Record a matmul's shape, so the microbenchmark can replay it later.
    pub(super) fn note_shape(&self, kernel: &'static str, n_in: usize, n_out: usize) {
        *self
            .shapes
            .borrow_mut()
            .entry((kernel, n_in, n_out))
            .or_insert(0) += 1;
    }

    /// Turn launch recording on. Costs a hash probe per launch, so it is off
    /// unless `--profile-device` asked for it.
    pub fn record_launches(&self, on: bool) {
        self.record_launches.set(on);
    }

    /// Decompose the IQ4_XS matmul: replay its heaviest recorded launch
    /// against variants that each remove one thing.
    ///
    /// Returns `(label, gpu_us)` with the baseline first. See the `dbg_iq4_*`
    /// kernels for what each removes and what each answer would mean.
    pub fn bench_iq4_variants(&self, reps: u32) -> Result<Vec<(&'static str, f64)>> {
        // The heaviest recorded launch of the real kernel, by calls x geometry.
        let pick = self
            .launches
            .borrow()
            .iter()
            .filter(|(k, _)| k.0 == "matmul_iq4_xs_q8_k")
            .max_by_key(|(k, v)| v.0 * u64::from(k.1))
            .map(|(k, v)| (*k, v.1.clone()));
        let ((_, gx, gy, block, shared, _, _), args) = match pick {
            Some(x) => x,
            None => return Ok(Vec::new()),
        };
        let variants = [args];
        let mut out = Vec::new();
        for name in [
            "matmul_iq4_xs_q8_k",
            "dbg_iq4_nounpack",
            "dbg_iq4_nofold",
            "dbg_iq4_noweight",
            "dbg_iq4_notable",
        ] {
            let (us, _) = self.time_launches_2d(name, gx, gy, block, shared, &variants, reps)?;
            out.push((name, us));
        }
        Ok(out)
    }

    /// Decompose `matmul_f32_t`: replay its heaviest recorded launch against
    /// variants that each remove one thing, plus the staged candidate.
    ///
    /// Returns `(label, gpu_us)` with the baseline first. The kernel measures
    /// 6.5 GB/s on a card that reads VRAM at 409.6, and its own comment blames
    /// "memory latency on a single thread" while asserting that going lower
    /// "means splitting the reduction". The first is a hypothesis these
    /// variants test; the second is what `matmul_f32_t_staged` disputes, by
    /// staging through shared memory so the block fetches while thread `j`
    /// still accumulates its own row serially.
    ///
    /// The staged variant is launched with its own geometry — a full block
    /// regardless of `n_out`, and dynamic shared memory — because that is the
    /// whole point of it. The others reuse the recorded launch exactly.
    pub fn bench_f32_variants(&self, reps: u32) -> Result<Vec<(&'static str, f64)>> {
        // Heaviest by calls x rows, which is how it hurts.
        let pick = self
            .launches
            .borrow()
            .iter()
            .filter(|(k, _)| k.0 == "matmul_f32_t")
            .max_by_key(|(_, v)| v.0)
            .map(|(k, v)| (*k, v.1.clone()));
        let ((_, gx, gy, block, shared, _, _), args) = match pick {
            Some(x) => x,
            None => return Ok(Vec::new()),
        };
        let (n_in, n_out) = match (args.first(), args.get(1)) {
            (Some(KArg::I32(a)), Some(KArg::I32(b))) => (*a as usize, *b as usize),
            _ => return Ok(Vec::new()),
        };

        let mut out = Vec::new();
        for name in [
            "matmul_f32_t",
            "dbg_f32_now",
            "dbg_f32_nox",
            "dbg_f32_noloads",
            "dbg_f32_nochain",
        ] {
            let variants = [args.clone()];
            let (us, _) = self.time_launches_2d(name, gx, gy, block, shared, &variants, reps)?;
            out.push((name, us));
        }

        // The candidate. One block, `STAGE_BLOCK` threads whatever `n_out` is,
        // and a tile sized to the shared allocation.
        const STAGE_BLOCK: u32 = 256;
        const STAGE_FLOATS: usize = 12 * 1024;
        if n_out <= STAGE_BLOCK as usize && n_out > 0 {
            let kt = (STAGE_FLOATS / (n_out + 1)).clamp(1, n_in);
            let mut sargs = args.clone();
            sargs.insert(2, KArg::I32(kt as i32));
            let shared_bytes = (kt * (n_out + 1) * 4) as u32;
            let variants = [sargs];
            let (us, _) = self.time_launches_2d(
                "matmul_f32_t_staged",
                1,
                gy,
                STAGE_BLOCK,
                shared_bytes,
                &variants,
                reps,
            )?;
            out.push(("matmul_f32_t_staged", us));
        }
        Ok(out)
    }

    /// How many distinct launches the recorder holds, and how many calls they
    /// stand for.
    ///
    /// **So that [`Cuda::bench_launches`] cannot fail silently.** On the 35B it
    /// produced no output at all — not the table, not its "recorded nothing"
    /// arm, not its error arm — while working on the 0.6B. That is only
    /// possible if the process does not return from it, which no `match` on its
    /// result can report. A caller that prints this first turns "nothing
    /// happened" into "it had N launches to replay and did not come back".
    pub fn recorded_launches(&self) -> (usize, u64) {
        let m = self.launches.borrow();
        (m.len(), m.values().map(|v| v.0).sum())
    }

    /// What an expert read costs from each tier, same kernel, same shape.
    ///
    /// **A controlled A/B, because every previous attempt at this number was a
    /// subtraction.** `CLAUDE.md` carries a withdrawn "~8 GB/s effective
    /// in-kernel" for host-resident experts: it came from a two-point config
    /// comparison that also differed in slab size and fill volume, and a later
    /// long run contradicted its extrapolation. The launch replay hints at the
    /// same figure but cannot establish it either — its recorded pointer table
    /// is one frozen draw from a 73/27 mixture, so its expert rows are a
    /// mixture whose composition is unknown.
    ///
    /// Here nothing differs but the tier. The same 16 expert stacks are built
    /// once, uploaded to VRAM for one variant and copied into a page-locked
    /// `MEMHOSTALLOC_DEVICEMAP` block for the other — which is exactly what
    /// `ExpertCache` does for its host tier, so the second variant is the real
    /// mechanism and not a model of it. The kernel dereferences a pointer table
    /// either way and cannot tell which it got.
    ///
    /// The mixed variant puts 2 of 8 experts on the host, near the 27% the
    /// eager placement actually produces, so the three rows can be checked
    /// against each other: if the cost is linear in the host fraction, mixed
    /// should land a quarter of the way from VRAM to host.
    ///
    /// Shape is the 35B's: `n_in` 2048, `n_out` 512, which is a 557,056-byte
    /// expert — the slab's slot size, not a round number chosen here.
    pub fn bench_expert_residency(&self, reps: u32) -> Result<Vec<(&'static str, f64, f64)>> {
        const QK_K: usize = 256;
        let (n_in, n_out) = (2048usize, 512usize);
        let n_super = n_in / QK_K;
        let row_bytes = n_super * 136;
        let row_set = n_out * row_bytes;
        let sets = 16; // eight gate experts and eight up experts
        let bytes = sets * row_set;

        // Filler with a valid f16 scale of 1.0 at each super-block, as the
        // shape bench builds: the kernel must read plausible data or its
        // arithmetic is not the arithmetic being timed.
        let mut w = vec![0x11u8; bytes];
        for c in 0..sets {
            for r in 0..n_out {
                for b in 0..n_super {
                    let at = c * row_set + r * row_bytes + b * 136;
                    w[at] = 0x00;
                    w[at + 1] = 0x38;
                }
            }
        }

        // One token's activation, shared by every expert — the gate/up shape.
        let scales: Vec<f32> = (0..n_super).map(|i| 0.01 + (i % 7) as f32 * 1e-3).collect();
        let quants: Vec<i8> = (0..n_in).map(|i| ((i % 251) as i32 - 125) as i8).collect();
        let sd = DeviceBuffer::from_slice(&scales)?;
        let qd = DeviceBuffer::from_slice(&quants)?;
        let od = DeviceBuffer::new(8 * n_out * 4)?;
        let vram = DeviceBuffer::from_slice(&w)?;

        // The same bytes again, in a page-locked block mapped into the device
        // address space. This is `ExpertCache::place_on_host`'s allocation.
        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut host_dev: ffi::CUdeviceptr = 0;
        // SAFETY: out-parameters the driver fills, both checked before use.
        unsafe {
            check(
                ffi::cuMemHostAlloc(&mut host_ptr, bytes, ffi::MEMHOSTALLOC_DEVICEMAP),
                "cuMemHostAlloc",
            )?;
            check(
                ffi::cuMemHostGetDevicePointer_v2(&mut host_dev, host_ptr, 0),
                "cuMemHostGetDevicePointer",
            )?;
            // SAFETY: `host_ptr` owns `bytes` and `w` holds exactly that many.
            std::ptr::copy_nonoverlapping(w.as_ptr(), host_ptr as *mut u8, bytes);
        }

        let addr = |base: u64, c: usize| base + (c * row_set) as u64;
        let mut out = Vec::new();
        let mut run = |label: &'static str, on_host: &[bool]| -> Result<(f64, f64)> {
            // Expert `c`'s gate is stack `c` and its up is stack `8 + c`, and
            // both follow that expert's tier — an expert is placed whole, so
            // splitting its two halves across tiers would measure a layout the
            // cache never produces.
            let at = |c: usize, stack: usize| {
                let base = if on_host[c] { host_dev as u64 } else { vram.ptr as u64 };
                addr(base, stack)
            };
            let g: Vec<u64> = (0..8).map(|c| at(c, c)).collect();
            let u: Vec<u64> = (0..8).map(|c| at(c, 8 + c)).collect();
            let gt = DeviceBuffer::from_slice(&g)?;
            let ut = DeviceBuffer::from_slice(&u)?;
            let args = vec![
                KArg::I32(n_in as i32),
                KArg::I32(n_out as i32),
                KArg::I32(8),
                KArg::I32(8),
                KArg::Ptr(gt.ptr),
                KArg::Ptr(ut.ptr),
                KArg::Ptr(sd.ptr),
                KArg::Ptr(qd.ptr),
                KArg::Ptr(od.ptr),
            ];
            let (gpu_us, _) = self.time_launches_2d(
                "matmul_iq4_xs_q8_k_moe_glu",
                n_out.div_ceil(4) as u32,
                8,
                128,
                0,
                &[args],
                reps,
            )?;
            let _ = label;
            // Every launch reads all sixteen stacks: eight gate, eight up.
            let moved = bytes as f64;
            Ok((gpu_us, moved / (gpu_us * 1e-6) / 1e9))
        };

        for (label, mask) in [
            ("all VRAM", [false; 8]),
            ("2 of 8 on host", [true, true, false, false, false, false, false, false]),
            ("all host tier", [true; 8]),
        ] {
            let (us, gbs) = run(label, &mask)?;
            out.push((label, us, gbs));
        }

        // SAFETY: allocated above by `cuMemHostAlloc` and not referenced after
        // the last launch, which `time_launches_2d` has synchronized.
        unsafe { check(ffi::cuMemFreeHost(host_ptr), "cuMemFreeHost")? };
        Ok(out)
    }

    /// Replay every launch the run actually made, and time it.
    ///
    /// **Complete by construction.** The previous bench synthesised arguments
    /// from a hand-written table and therefore omitted whatever nobody added to
    /// it — most recently `matmul_iq4_xs_q8_k_moe_glu`, the largest kernel in
    /// the routed FFN, which made its total an undercount of unknown size and
    /// sent a whole line of reasoning the wrong way. Here the entry exists
    /// because the launch happened.
    ///
    /// Safe to run only **after** generation: replaying a kernel re-executes
    /// its writes, so state is corrupted on purpose and nothing may read it
    /// afterwards. Pointers stay valid because weights, mirrors, pool slots and
    /// the expert slab all outlive the pass.
    ///
    /// # What a replayed expert launch does *not* measure
    ///
    /// The recorded arguments include the expert pointer table as it stood, so
    /// a replayed `matmul_iq4_xs_q8_k_moe*` reads whichever experts that one
    /// launch happened to route to, 200 times over. With 27.1% of the pool in
    /// the pinned host tier those reads are drawn from a mixture of VRAM at
    /// 448 GB/s and PCIe at 26.5, and a single frozen draw is not the mean.
    ///
    /// Observed directly: the decode-shaped launch at grid 128x8 timed 406 us
    /// for eight pairs while the 5-token prefill shape at 128x40 timed 199 us
    /// for forty — 50.8 us per pair against 5.0, a 10x spread between two rows
    /// of the same table. Some of that is real batching benefit and some is
    /// which side of the bus those particular experts were on, and this
    /// instrument cannot separate them.
    ///
    /// So: read the expert rows as an upper bound with a wide error bar, and
    /// use `experts ... % from VRAM` alongside them. The non-expert rows carry
    /// no such caveat — their weights are resident for the life of the backend.
    pub fn bench_launches(&self, reps: u32) -> Result<Vec<LaunchBench>> {
        let recorded: Vec<(LaunchKey, (u64, Vec<KArg>))> = self
            .launches
            .borrow()
            .iter()
            .map(|(k, v)| (*k, (v.0, v.1.clone())))
            .collect();

        let mut out = Vec::with_capacity(recorded.len());
        for ((kernel, gx, gy, block, shared, _, _), (calls, args)) in recorded {
            let variants = [args];
            let (gpu_us, issue_us) =
                self.time_launches_2d(kernel, gx, gy, block, shared, &variants, reps)?;
            out.push(LaunchBench {
                kernel,
                grid: (gx, gy),
                block,
                calls,
                gpu_us,
                issue_us,
            });
        }
        // Most expensive per token first: that is the order in which they matter.
        out.sort_by(|a, b| {
            (b.calls as f64 * b.gpu_us)
                .partial_cmp(&(a.calls as f64 * a.gpu_us))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(out)
    }

    /// Time the shapes this run actually launched — **with no per-launch
    /// synchronize**, which is what `--profile-kernels` cannot do.
    ///
    /// # Why this exists
    ///
    /// `gpu` and `issue` in the pass timings came out at 74.95 and 71.75
    /// ms/token on the 35B, which is ambiguous in the worst way: it is
    /// consistent both with a device saturated by real work and with a device
    /// starved by a host that cannot describe work fast enough. Those call for
    /// opposite fixes — better kernels versus fewer launches — so guessing is
    /// expensive.
    ///
    /// So each shape is measured twice over the same launches:
    ///
    /// * **gpu** — CUDA events either side of the whole batch. Time the device
    ///   spent, per launch, with the queue kept full.
    /// * **issue** — host wall time for the launch loop, which returns as soon
    ///   as the driver accepts each launch. Host cost, per launch.
    ///
    /// Whichever is larger is the one that binds, per shape, and the ratio says
    /// by how much.
    ///
    /// # The grouped column
    ///
    /// Each shape is also run **once** with `group` times the rows, which is
    /// the same arithmetic in one launch instead of `group`. For the routed
    /// FFN that is exactly the change worth costing: 8 experts of `{2048, 512}`
    /// become one `{2048, 4096}`. The difference between `group * gpu` and
    /// `grouped` is what fusing them would return, measured rather than argued.
    ///
    /// Weights are synthetic. These kernels are bandwidth and instruction
    /// bound and do not branch on values — but the f16 scale bytes are set to a
    /// real value anyway, because a NaN scale is the kind of thing that turns
    /// out to matter on some future architecture.
    pub fn bench_shapes(&self, group: usize, tokens: u64, reps: u32) -> Result<Vec<ShapeBench>> {
        let mut want: Vec<((&'static str, usize, usize), u64)> =
            self.shapes.borrow().iter().map(|(k, v)| (*k, *v)).collect();
        // Most-launched first: that is the order in which they matter.
        want.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        let mut out = Vec::with_capacity(want.len());
        for ((kernel, n_in, n_out), calls) in want {
            let one = self.time_shape(kernel, n_in, n_out, reps)?;
            // The grouped form is the same kernel over `group` times the rows.
            // Skipped when it would not fit or would not mean anything.
            // Only meaningful where the shape is launched at least `group`
            // times **per token**, which is what "fuse the experts a token
            // visits" means. Gating on the raw count instead priced fusing
            // eight LM heads — a 2048 x 1,986,560 matmul nobody would build —
            // and that one row put 82 ms into a 7 ms total.
            let per_token = calls as f64 / tokens.max(1) as f64;
            let groupable =
                group > 1 && per_token >= group as f64 && !kernel.ends_with("_moe");
            let grouped = if groupable {
                self.time_shape(kernel, n_in, n_out * group, reps.max(1) / 4 + 1)?.0
            } else {
                one.0
            };
            out.push(ShapeBench {
                kernel,
                n_in,
                n_out,
                calls,
                gpu_us: one.0,
                issue_us: one.1,
                grouped_us: grouped,
                group: if groupable { group } else { 1 },
            });
        }
        Ok(out)
    }

    /// Build the argument list and launch geometry for one shape.
    ///
    /// **Per kernel, by hand, because there is no honest shortcut.** Every
    /// kernel's parameter list is its own, so a generic replay would have to
    /// guess — and a bench that guesses wrong measures a kernel reading
    /// garbage sizes, which is fast and meaningless. The shapes come from the
    /// run; only the plumbing is written down here.
    ///
    /// `n_out == 0` marks an elementwise kernel, for which `n_in` is the
    /// buffer length. Returns `None` for anything not benched, which the
    /// caller reports as `--` rather than as a zero.
    ///
    /// Returns `(args, grid_x, grid_y, block, shared)`. **`grid_y` is not
    /// decoration**: the routed FFN's kernels put the (token, pick) pair on
    /// `blockIdx.y`, so launching them one-dimensionally runs one pair of eight
    /// and reports an eighth of the work as the whole of it.
    fn shape_args(
        &self,
        kernel: &'static str,
        n_in: usize,
        n_out: usize,
        keep: &mut Vec<DeviceBuffer>,
    ) -> Result<Option<(Vec<KArg>, u32, u32, u32, u32)>> {
        const QK_K: usize = 256;
        // Two f32 buffers of `n` and a couple of quantized outputs cover every
        // elementwise kernel here; they are allocated once and shared.
        let mut fbuf = |n: usize| -> Result<ffi::CUdeviceptr> {
            let v: Vec<f32> = (0..n.max(1)).map(|i| 0.01 + (i % 31) as f32 * 1e-3).collect();
            let b = DeviceBuffer::from_slice(&v)?;
            let p = b.ptr;
            keep.push(b);
            Ok(p)
        };

        let out = match kernel {
            "silu_mul" | "add_assign" | "sigmoid_mul" => {
                let (a, b) = (fbuf(n_in)?, fbuf(n_in)?);
                let args = vec![KArg::I32(n_in as i32), KArg::Ptr(a), KArg::Ptr(b)];
                (args, n_in.div_ceil(256) as u32, 1, 256u32, 0u32)
            }
            "add_scaled" => {
                let (a, b) = (fbuf(n_in)?, fbuf(n_in)?);
                let args = vec![
                    KArg::I32(n_in as i32),
                    KArg::F32(0.125),
                    KArg::Ptr(a),
                    KArg::Ptr(b),
                ];
                (args, n_in.div_ceil(256) as u32, 1, 256, 0)
            }
            "gather_chunks" | "scatter_chunks" => {
                let (src, dst) = (fbuf(n_in * 2)?, fbuf(n_in * 2)?);
                let args = vec![
                    KArg::I32(n_in as i32),
                    KArg::I32(n_in as i32),
                    KArg::I32(n_in as i32),
                    KArg::I32(0),
                    KArg::Ptr(src),
                    KArg::Ptr(dst),
                ];
                (args, n_in.div_ceil(256) as u32, 1, 256, 0)
            }
            "quantize_q8_k" => {
                let n_super = n_in / QK_K;
                let x = fbuf(n_in)?;
                let s = DeviceBuffer::new(n_super * 4)?;
                let q = DeviceBuffer::new(n_super * QK_K)?;
                let b = DeviceBuffer::new(n_super * (QK_K / 16) * 2)?;
                let args = vec![
                    KArg::I32(n_super as i32),
                    KArg::Ptr(x),
                    KArg::Ptr(s.ptr),
                    KArg::Ptr(q.ptr),
                    KArg::Ptr(b.ptr),
                ];
                keep.push(s);
                keep.push(q);
                keep.push(b);
                (args, n_super as u32, 1, QK_K as u32, 0)
            }
            "quantize_q8_0" => {
                let n_blocks = n_in / 32;
                let x = fbuf(n_in)?;
                let s = DeviceBuffer::new(n_blocks * 4)?;
                let q = DeviceBuffer::new(n_blocks * 32)?;
                let args = vec![
                    KArg::I32(n_blocks as i32),
                    KArg::Ptr(x),
                    KArg::Ptr(s.ptr),
                    KArg::Ptr(q.ptr),
                ];
                keep.push(s);
                keep.push(q);
                (args, n_blocks.div_ceil(64) as u32, 1, 64, 0)
            }
            "rms_norm_tree" => {
                let (x, w, o) = (fbuf(n_in)?, fbuf(n_in)?, fbuf(n_in)?);
                let args = vec![
                    KArg::I32(n_in as i32),
                    KArg::Ptr(x),
                    KArg::Ptr(w),
                    KArg::F32(1e-6),
                    KArg::Ptr(o),
                ];
                // One block per row; the recorded shape is a single row.
                (args, 1, 1, 256, 0)
            }
            "matmul_iq4_xs_q8_k_moe" | "matmul_iq4_xs_q8_k_moe_glu" => {
                // **This drifted from the kernel and crashed the process.** It
                // used to push eight expert addresses as eight kernel
                // arguments, which was the signature before routing moved onto
                // the device. The kernel now takes one *table* pointer and
                // dereferences `wptrs[e]`, so the old list handed it eight
                // bytes of expert weight to use as an address:
                //
                //     shape bench unavailable: cuEventSynchronize failed:
                //       CUDA_ERROR_ILLEGAL_ADDRESS
                //     error: cuMemAlloc failed: CUDA_ERROR_ILLEGAL_ADDRESS
                //
                // and every later CUDA call failed with the context poisoned,
                // so `--profile-device` exited 1 on the 35B. A hand-written
                // argument list is a second copy of a signature, and this is
                // what the second copy costs when only one of them is updated.
                //
                // Eight *distinct* weights, not one repeated: the real routed
                // FFN reads eight different experts, and one buffer read eight
                // times would be served by L2 rather than by VRAM.
                let glu = kernel.ends_with("_glu");
                let n_super = n_in / QK_K;
                let row_bytes = n_super * 136;
                let row_set = n_out * row_bytes;
                // Two stacks for the fused form: it reads a gate and an up
                // expert per pair, so benching one would halve the traffic.
                let sets = if glu { 16 } else { 8 };
                let mut w = vec![0x11u8; sets * row_set];
                for c in 0..sets {
                    for r in 0..n_out {
                        for b in 0..n_super {
                            let at = c * row_set + r * row_bytes + b * 136;
                            w[at] = 0x00;
                            w[at + 1] = 0x38;
                        }
                    }
                }
                let wd = DeviceBuffer::from_slice(&w)?;

                // The pointer table the kernel actually dereferences, in device
                // memory — which is the whole reason the old argument list was
                // wrong, so building it here is the fix rather than a detail.
                let addrs: Vec<u64> =
                    (0..8).map(|c| wd.ptr as u64 + (c * row_set) as u64).collect();
                let gtab = DeviceBuffer::from_slice(&addrs)?;
                let utab = if glu {
                    let up: Vec<u64> = (0..8)
                        .map(|c| wd.ptr as u64 + ((8 + c) * row_set) as u64)
                        .collect();
                    Some(DeviceBuffer::from_slice(&up)?)
                } else {
                    None
                };

                // `_moe` is the `down` half: one intermediate per pair, so
                // eight activation rows. `_moe_glu` is gate/up: one row shared
                // by every expert of the token, so one.
                let x_rows = if glu { 1 } else { 8 };
                let scales: Vec<f32> = (0..x_rows * n_super)
                    .map(|i| 0.01 + (i % 7) as f32 * 1e-3)
                    .collect();
                let quants: Vec<i8> = (0..x_rows * n_in)
                    .map(|i| ((i % 251) as i32 - 125) as i8)
                    .collect();
                let sd = DeviceBuffer::from_slice(&scales)?;
                let qd = DeviceBuffer::from_slice(&quants)?;
                let od = DeviceBuffer::new(8 * n_out * 4)?;

                let args = if glu {
                    vec![
                        KArg::I32(n_in as i32),
                        KArg::I32(n_out as i32),
                        KArg::I32(8),
                        KArg::I32(8),
                        KArg::Ptr(gtab.ptr),
                        KArg::Ptr(utab.as_ref().map_or(0, |b| b.ptr)),
                        KArg::Ptr(sd.ptr),
                        KArg::Ptr(qd.ptr),
                        KArg::Ptr(od.ptr),
                    ]
                } else {
                    vec![
                        KArg::I32(n_in as i32),
                        KArg::I32(n_out as i32),
                        KArg::I32(n_super as i32),
                        KArg::I32(8),
                        KArg::Ptr(gtab.ptr),
                        KArg::Ptr(sd.ptr),
                        KArg::Ptr(qd.ptr),
                        KArg::Ptr(od.ptr),
                    ]
                };
                let grid = n_out.div_ceil(4) as u32;
                keep.push(wd);
                keep.push(gtab);
                if let Some(b) = utab {
                    keep.push(b);
                }
                keep.push(sd);
                keep.push(qd);
                keep.push(od);
                // `grid.y` is the pair count, which is what these kernels
                // index — eight picks of one token, the decode shape.
                (args, grid, 8, 128, 0)
            }
            _ => return Ok(None),
        };
        let _ = n_out;
        Ok(Some(out))
    }

    /// One shape, `reps` launches, one synchronize. Returns `(gpu, issue)` in
    /// microseconds per launch.
    fn time_shape(
        &self,
        kernel: &'static str,
        n_in: usize,
        n_out: usize,
        reps: u32,
    ) -> Result<(f64, f64)> {
        const QK_K: usize = 256;
        // Everything that is not a plain matmul goes through `shape_args`.
        // The grouped MoE matmul is a matmul by name and not by signature, and
        // leaving it out meant the table silently omitted the single largest
        // kernel in the routed FFN — the same shape of gap that made
        // `resident_bytes` under-report by an order of magnitude.
        let mut keep: Vec<DeviceBuffer> = Vec::new();
        // `contains`, not `ends_with`: `matmul_iq4_xs_q8_k_moe_glu` does not end
        // in `_moe` and so fell through to the plain-matmul path, whose
        // signature it does not have either. The largest kernel in the routed
        // FFN was being benched with the wrong argument list.
        if !kernel.starts_with("matmul_") || kernel.contains("_moe") {
            let built = self.shape_args(kernel, n_in, n_out, &mut keep)?;
            let (args, grid_x, grid_y, block, shared) = match built {
                Some(b) => b,
                None => return Ok((f64::NAN, f64::NAN)),
            };
            return self.time_launches_2d(kernel, grid_x, grid_y, block, shared, &[args], reps);
        }
        // Bytes one row of this format occupies, and the f16 scale's offset
        // within a block. `None` where the format has no f16 to protect.
        let (block_bytes, block_elems, scale_at) = match kernel {
            "matmul_iq4_xs_q8_k" => (136usize, QK_K, Some(0usize)),
            "matmul_q5_k_q8_k" => (176, QK_K, Some(0)),
            "matmul_q6_k_q8_k" => (210, QK_K, Some(208)),
            "matmul_f32" | "matmul_f32_t" => (4, 1, None),
            // Q8_0 has its own benches, and anything else is not a matmul.
            _ => return Ok((f64::NAN, f64::NAN)),
        };
        let n_super = n_in / block_elems;
        let row_bytes = n_super * block_bytes;

        // Enough copies of the weight that consecutive launches cannot all be
        // served by L2. Capped so the LM head, already 398 MB, does not ask for
        // three gigabytes.
        let row_set = n_out * row_bytes;
        let copies = ((64 << 20) / row_set.max(1)).clamp(1, 8);
        let mut w = vec![0x11u8; copies * row_set];
        if let Some(off) = scale_at {
            // 0x3800 is f16 0.5: a real number, so the arithmetic is finite.
            for c in 0..copies {
                for r in 0..n_out {
                    for b in 0..n_super {
                        let at = c * row_set + r * row_bytes + b * block_bytes + off;
                        w[at] = 0x00;
                        w[at + 1] = 0x38;
                    }
                }
            }
        } else {
            for c in w.chunks_exact_mut(4) {
                c.copy_from_slice(&0.01f32.to_le_bytes());
            }
        }

        let wd = DeviceBuffer::from_slice(&w)?;
        let od = DeviceBuffer::new(n_out * 4)?;

        // The activation, in whichever form this kernel dots against.
        let (xs, xq, xb, xf) = if kernel.starts_with("matmul_f32") {
            let x: Vec<f32> = (0..n_in).map(|i| (i % 17) as f32 * 0.01).collect();
            (None, None, None, Some(DeviceBuffer::from_slice(&x)?))
        } else {
            let scales: Vec<f32> = (0..n_super).map(|i| 0.01 + (i % 7) as f32 * 1e-3).collect();
            let quants: Vec<i8> = (0..n_in).map(|i| ((i % 251) as i32 - 125) as i8).collect();
            let bsums: Vec<i16> = (0..n_super * (QK_K / 16)).map(|i| (i % 97) as i16).collect();
            (
                Some(DeviceBuffer::from_slice(&scales)?),
                Some(DeviceBuffer::from_slice(&quants)?),
                Some(DeviceBuffer::from_slice(&bsums)?),
                None,
            )
        };

        let mut variants: Vec<Vec<KArg>> = Vec::with_capacity(copies);
        for c in 0..copies {
            let mut args = vec![
                KArg::I32(n_in as i32),
                KArg::I32(n_out as i32),
                KArg::Ptr(wd.ptr + (c * row_set) as ffi::CUdeviceptr),
            ];
            match (&xf, &xs, &xq, &xb) {
                (Some(x), ..) => args.push(KArg::Ptr(x.ptr)),
                (None, Some(s), Some(q), Some(b)) => {
                    args.push(KArg::Ptr(s.ptr));
                    args.push(KArg::Ptr(q.ptr));
                    if kernel == "matmul_q5_k_q8_k" {
                        args.push(KArg::Ptr(b.ptr));
                    }
                }
                _ => {}
            }
            args.push(KArg::Ptr(od.ptr));
            variants.push(args);
        }

        let block = 128u32;
        let grid = if kernel.starts_with("matmul_f32") {
            n_out.div_ceil(block as usize) as u32
        } else {
            n_out.div_ceil((block / 32) as usize) as u32
        };

        keep.push(wd);
        keep.push(od);
        if let Some(b) = xs { keep.push(b); }
        if let Some(b) = xq { keep.push(b); }
        if let Some(b) = xb { keep.push(b); }
        if let Some(b) = xf { keep.push(b); }
        self.time_launches(kernel, grid, block, 0, &variants, reps)
    }

    /// `reps` launches of one configured kernel, timed two ways: CUDA events
    /// for device microseconds, host wall clock for issue microseconds.
    ///
    /// **One synchronize, at the end.** That is the whole point — a per-launch
    /// sync measures latency where throughput is what matters, and inflates
    /// every kernel in proportion to how often it is called, which is what
    /// makes `--profile-kernels` shares unusable for ranking work.
    /// As above, cycling through `variants` so consecutive launches read
    /// different weights.
    ///
    /// **Replaying one buffer measures the L2, not the card.** This machine has
    /// 32 MB of L2 and a grouped expert matmul reads 4.25 MB, so 300 repeats of
    /// one buffer are 299 cache hits — while the real forward pass streams 11.5
    /// GiB of distinct experts and hits VRAM every time. The tell was that the
    /// LM head, the only shape too large to cache, was also the only one that
    /// measured slow.
    fn time_launches(
        &self,
        kernel: &'static str,
        grid: u32,
        block: u32,
        shared: u32,
        variants: &[Vec<KArg>],
        reps: u32,
    ) -> Result<(f64, f64)> {
        self.time_launches_2d(kernel, grid, 1, block, shared, variants, reps)
    }

    #[allow(clippy::too_many_arguments)]
    fn time_launches_2d(
        &self,
        kernel: &'static str,
        grid_x: u32,
        grid_y: u32,
        block: u32,
        shared: u32,
        variants: &[Vec<KArg>],
        reps: u32,
    ) -> Result<(f64, f64)> {
        // A bench is never part of a graph, and never times its own launches.
        let was_graph = self.pass_graph.replace(false);
        let was_timed = self.time_kernels.replace(false);

        let (mut a, mut b): (ffi::CUevent, ffi::CUevent) =
            (std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: valid out-pointers; flag 0 is the timing-enabled default.
        unsafe {
            check(ffi::cuEventCreate(&mut a, 0), "cuEventCreate")?;
            check(ffi::cuEventCreate(&mut b, 0), "cuEventCreate")?;
        }

        let run = |n: u32| -> Result<(f64, f64)> {
            // SAFETY: `a`/`b` are ours; the null stream is the one everything
            // uses, so the events bracket exactly these launches.
            unsafe { check(ffi::cuEventRecord(a, std::ptr::null_mut()), "cuEventRecord")? };
            let t = std::time::Instant::now();
            for k in 0..n {
                // SAFETY: the caller built `args`, `grid`, `block` and
                // `shared` to match `kernel`, and sized every buffer for the
                // shape it describes.
                let args = &variants[(k as usize) % variants.len()];
                // SAFETY: as above.
                unsafe { self.launch_grid2(kernel, grid_x, grid_y, block, shared, args)? };
            }
            // Host time first: the launch loop returns once the driver has
            // accepted the work, so this is issue cost and not device time.
            let issue = t.elapsed().as_secs_f64() * 1e6 / f64::from(n);
            // SAFETY: as above.
            unsafe {
                check(ffi::cuEventRecord(b, std::ptr::null_mut()), "cuEventRecord")?;
                check(ffi::cuEventSynchronize(b), "cuEventSynchronize")?;
            }
            let mut ms = 0.0f32;
            // SAFETY: both events have completed.
            unsafe {
                check(ffi::cuEventElapsedTime(&mut ms, a, b), "cuEventElapsedTime")?;
            }
            Ok((f64::from(ms) * 1000.0 / f64::from(n), issue))
        };

        run(8)?; // warm the module, the caches and the clocks

        // **Best of several, not the average.** Every error source here only
        // adds time: a stall in the host launch loop, a clock that has not
        // ramped, contention from anything else on the device. WSL's CUDA
        // virtualization makes per-launch host cost spiky in particular, and
        // when the host cannot keep the queue full the event pair measures the
        // stall rather than the kernel — visible as `gpu` collapsing onto
        // `issue`. That was caught by running the same bench twice and finding
        // the affected rows had moved: a 512-element `silu_mul` read 22.4 us
        // once and 13.0 the next time, neither of them credible for 2 KiB.
        //
        // The minimum is the only statistic that is not a function of how busy
        // the machine was, so it is the one reported.
        let mut measured = run(reps.max(1))?;
        for _ in 0..3 {
            let (g, i) = run(reps.max(1))?;
            measured.0 = measured.0.min(g);
            measured.1 = measured.1.min(i);
        }
        let measured = Ok(measured);

        // SAFETY: both events are ours and are no longer in flight.
        unsafe {
            check(ffi::cuEventDestroy_v2(a), "cuEventDestroy")?;
            check(ffi::cuEventDestroy_v2(b), "cuEventDestroy")?;
        }
        self.pass_graph.set(was_graph);
        self.time_kernels.set(was_timed);
        measured
    }

    /// Time a matmul variant at a real weight shape.
    ///
    /// Weights are synthetic -- the kernels are bandwidth and instruction bound
    /// and do not branch on values -- but the *shapes* are the 9B's, because
    /// the serial tail at the end of the kernel grows with `n_in` and the 0.6B
    /// would not show it.
    pub fn bench_matmul(
        &self,
        name: &'static str,
        n_in: usize,
        n_out: usize,
        reps: u32,
    ) -> Result<f64> {
        let n_blocks = n_in / 32;
        let wbytes = n_out * n_blocks * 34;
        let w: Vec<u8> = (0..wbytes).map(|i| (i % 251) as u8).collect();
        let scales: Vec<f32> = (0..n_blocks).map(|i| 0.01 + (i % 7) as f32 * 0.001).collect();
        let quants: Vec<i8> = (0..n_in).map(|i| ((i % 251) as i32 - 125) as i8).collect();

        let wd = DeviceBuffer::from_slice(&w)?;
        let sd = DeviceBuffer::from_slice(&scales)?;
        let qd = DeviceBuffer::from_slice(&quants)?;
        let od = DeviceBuffer::new(n_out * 4)?;

        let args = [
            KArg::I32(n_in as i32),
            KArg::I32(n_out as i32),
            KArg::Ptr(wd.ptr),
            KArg::Ptr(sd.ptr),
            KArg::Ptr(qd.ptr),
            KArg::Ptr(od.ptr),
        ];
        let block = 128u32;
        let warps = (block / 32) as usize;
        // `tree` keeps its partials in registers and asks for none.
        let shared = if name.ends_with("tree") {
            0
        } else {
            (warps * n_blocks * 4) as u32
        };
        let grid = n_out.div_ceil(warps) as u32;

        let was = self.pass_graph.replace(false);
        let run = |reps: u32| -> Result<f64> {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                // SAFETY: the argument list matches every kernel in this
                // diagnostic set, and `shared` is what each indexes.
                unsafe { self.launch_shared(name, grid, block, shared, &args)? };
            }
            self.sync()?;
            Ok(t.elapsed().as_secs_f64() * 1e6 / f64::from(reps))
        };
        run(8)?;
        let us = run(reps);
        self.pass_graph.set(was);
        us
    }

    /// Time a matmul variant that reads a *repacked* weight layout.
    ///
    /// Same shapes and same synthetic data as [`Cuda::bench_matmul`], but the
    /// tensor is split into an aligned f16 scale array and an aligned int8
    /// quant array, which is what makes `int4` loads and `__dp4a` legal.
    pub fn bench_matmul_packed(
        &self,
        name: &'static str,
        n_in: usize,
        n_out: usize,
        reps: u32,
    ) -> Result<f64> {
        let n_blocks = n_in / 32;
        let wscales: Vec<u16> = (0..n_out * n_blocks).map(|i| (0x3800 + (i % 64)) as u16).collect();
        let wquants: Vec<i8> = (0..n_out * n_in).map(|i| ((i % 251) as i32 - 125) as i8).collect();
        let xscales: Vec<f32> = (0..n_blocks).map(|i| 0.01 + (i % 7) as f32 * 0.001).collect();
        let xquants: Vec<i8> = (0..n_in).map(|i| ((i % 251) as i32 - 125) as i8).collect();

        let ws = DeviceBuffer::from_slice(&wscales)?;
        let wq = DeviceBuffer::from_slice(&wquants)?;
        let xs = DeviceBuffer::from_slice(&xscales)?;
        let xq = DeviceBuffer::from_slice(&xquants)?;
        let od = DeviceBuffer::new(n_out * 4)?;

        let args = [
            KArg::I32(n_in as i32),
            KArg::I32(n_out as i32),
            KArg::Ptr(ws.ptr),
            KArg::Ptr(wq.ptr),
            KArg::Ptr(xs.ptr),
            KArg::Ptr(xq.ptr),
            KArg::Ptr(od.ptr),
        ];
        let block = 128u32;
        let warps = (block / 32) as usize;
        let shared = if name.ends_with("tree") { 0 } else { (warps * n_blocks * 4) as u32 };
        let grid = n_out.div_ceil(warps) as u32;

        let was = self.pass_graph.replace(false);
        let run = |reps: u32| -> Result<f64> {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                // SAFETY: the argument list matches both packed variants.
                unsafe { self.launch_shared(name, grid, block, shared, &args)? };
            }
            self.sync()?;
            Ok(t.elapsed().as_secs_f64() * 1e6 / f64::from(reps))
        };
        run(8)?;
        let us = run(reps);
        self.pass_graph.set(was);
        us
    }
}
