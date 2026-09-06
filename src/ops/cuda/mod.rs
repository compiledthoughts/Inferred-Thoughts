//! CUDA, through the driver API, targeting sm_120.
//!
//! `CLAUDE.md` puts this on our side of the borrowed/ours line: the dense path
//! is llama.cpp's to win, but the MoE expert path and anything the sm_120 stack
//! does not already serve is ours to write. This module is the foundation for
//! that — device discovery, memory, PTX loading, kernel launch — plus a first
//! kernel to prove the toolchain end to end.
//!
//! **The first Q8_0 kernel is bit-exact against [`crate::ops::naive`], and that
//! is deliberate.** One thread per output row, walking the row serially, is the
//! same arithmetic in the same order as the CPU oracle. It is not fast — the
//! reads do not coalesce — but it means the very first GPU result can be
//! checked with `assert_eq!` on raw bits rather than a tolerance we would have
//! to justify. Optimizing it will change the accumulation order and cost that
//! property; doing so knowingly, with a measured number in hand, is the point
//! of starting here.
//!
//! Everything is behind the `cuda` feature so a machine without a toolkit still
//! builds and tests the rest of the crate.

pub mod experts;
pub mod ffi;
mod ops;

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::{CString, c_void};

use crate::error::{Error, Result};

/// The PTX produced by `build.rs`, embedded so there is no file to ship or find
/// at run time.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn check(code: ffi::CUresult, what: &'static str) -> Result<()> {
    if code == ffi::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::Cuda {
            what,
            detail: ffi::describe(code),
        })
    }
}

/// An owned CUDA context with our kernels loaded.
///
/// Dropping it tears down the module and context in that order, which is what
/// the driver requires.
pub struct Cuda {
    context: ffi::CUcontext,
    module: ffi::CUmodule,
    name: String,
    capability: (i32, i32),
    sm_count: i32,
    total_mem: usize,

    /// Host pointer -> its device copy, for anything that does not change
    /// between calls: quantized weight matrices and the f32 norm vectors.
    ///
    /// Without this the backend would re-upload every weight it touches on
    /// every call. The 0.6B reads 0.59 GiB per forward pass, which at the
    /// measured 28.6 GB/s is ~22 ms of PCIe per token against a 16.4 ms CPU
    /// token — the GPU path would lose to the CPU for reasons that have
    /// nothing to do with any kernel. Keying on the mmap pointer works because
    /// `Weights` borrows the mapping, so the address is stable for the run.
    weights: RefCell<HashMap<usize, DeviceBuffer>>,

    /// Host pointer -> device mirror of one layer's K or V slab.
    ///
    /// The KV cache only ever appends, so each call uploads the positions
    /// added since the last one rather than the whole slab. Re-uploading would
    /// be ~8 MB per layer per token at 4k context; the delta is 2 KB.
    kv: RefCell<HashMap<usize, KvMirror>>,

    /// Reusable device scratch, indexed by role. Growable, never shrunk, so a
    /// steady-state token allocates nothing.
    pool: RefCell<Vec<DeviceBuffer>>,

    /// The first driver error any op hit. See `Cuda::take_error`.
    error: RefCell<Option<Error>>,

    /// Resolved kernel handles, by symbol. `cuModuleGetFunction` is a driver
    /// call and building its argument allocates a `CString`; a decode step
    /// makes ~478 launches, so neither belongs on that path.
    functions: RefCell<HashMap<&'static str, ffi::CUfunction>>,

    /// What this backend actually asked the driver to do. See [`DeviceStats`].
    stats: Cell<DeviceStats>,

    /// Device copies of the model's activation buffers, keyed on host address.
    /// See `Cuda::mirror_in`.
    mirrors: RefCell<HashMap<usize, Mirror>>,

    /// The decode step as a CUDA graph. See [`GraphState`].
    graph: RefCell<GraphState>,
    /// Whether the pass now running is eligible to use the graph.
    pass_graph: Cell<bool>,
    /// Master switch. Off for callers that drive ops one at a time.
    graphs_enabled: Cell<bool>,
    /// F32 weights, **transposed** at upload into column-major. Keyed on the
    /// mmap address of the tensor.
    ///
    /// Same bytes, and it makes `matmul_f32_t`'s reads coalesce where the
    /// row-major original had every thread walking its own row. ~95 MiB across
    /// the 35B — 40 routers, 60 GatedDeltaNet projections, 40 shared-expert
    /// gates — transposed once each on the host.
    f32t: RefCell<HashMap<usize, DeviceBuffer>>,

    /// Bounded VRAM residency for the MoE expert pool. See [`experts`].
    ///
    /// Separate from `weights`, which is upload-once-keep-forever and right for
    /// anything that fits. The expert pool does not fit — 15.94 GiB against
    /// 14.80 free on the 35B — so it gets a slab with an eviction policy
    /// instead. Built lazily on the first pooled tensor, because its size is
    /// taken from free VRAM at a point when the permanent weights are mostly up.
    experts: RefCell<Option<experts::ExpertCache>>,

    /// VRAM to leave free when sizing the expert slab, in bytes.
    ///
    /// Covers what is not yet allocated when the slab is built: the rest of the
    /// permanent weights, activation mirrors at the configured batch, and the
    /// driver's own working set.
    ///
    /// **The KV cache is not in it by default** — its slabs are allocated
    /// lazily at the first attention layer, which on the 35B is block 3 and so
    /// comes after this sizing. `Cuda::reserve_for_kv` adds it, and only the
    /// caller knows the context length.
    expert_reserve: Cell<usize>,

    /// Page-locked host memory the expert cache's overflow tier may claim.
    ///
    /// The second tier is what makes every expert addressable without host
    /// intervention, which is what a CUDA graph needs; see [`experts`].
    expert_host_budget: Cell<usize>,

    /// Whether `matmul_f32` uses the staged kernel.
    ///
    /// **Off by default, on a measured regression.** See `Cuda::matmul_f32`.
    f32_staged: Cell<bool>,
    /// Two F32 weights interleaved into one column-major stack, keyed on both
    /// source pointers. See `Cuda::resident_f32_t_pair`.
    f32t_pair: RefCell<HashMap<(usize, usize), DeviceBuffer>>,

    /// Force the warp-per-position score phase on or off. `None` picks by
    /// context depth; see `Cuda::attend_impl`.
    attn_warp: Cell<Option<bool>>,
    /// Force the untiled IQ4_XS matmul even for a batch, for the A/B.
    iq4_untiled: Cell<bool>,
    /// Diagnostic: enable the warp score phase for one `attend` call only.
    ///
    /// **A bisector that needs no host reads.** `Ctx::trace` hands out host
    /// slices, which on this backend are stale by design, so comparing traced
    /// intermediates between two CUDA runs compares buffers neither run wrote.
    /// Attention is called once per attending layer per pass, so enabling the
    /// path for a single call index localises a divergence to a layer using
    /// only the logits, which the model does bring home.
    attn_warp_only: Cell<Option<usize>>,
    /// `attend` calls so far this pass, reset by `begin_pass`.
    attn_calls: Cell<usize>,

    /// Print launch and residency counters after each server turn.
    report_per_turn: Cell<bool>,

    /// The model file, so its page cache can be dropped once the experts are
    /// placed. See `experts::drop_file_cache`.
    model_path: RefCell<Option<std::path::PathBuf>>,
    /// Passes seen, to find the moment placement is finished.
    passes_seen: Cell<u64>,
    /// Where the model's mapping starts. See `Cuda::set_map_base`.
    map_base: Cell<Option<usize>>,

    /// Q8_0 weights, repacked at upload into an aligned scale array and an
    /// aligned quant array. Keyed on the mmap address of the tensor.
    ///
    /// Same total bytes as the file layout -- 2 + 32 per block either way --
    /// but split so both operands of `__dp4a` are 16-byte aligned. On disk a
    /// block is 34 bytes, which puts its quants at 34b+2: even, so two-byte
    /// loads are legal, but never a multiple of four, so wide loads and
    /// `__dp4a` are not.
    q8: RefCell<HashMap<usize, (DeviceBuffer, DeviceBuffer)>>,

    /// Device-resident recurrent state, keyed on the host slab address.
    ///
    /// Separate from `weights` because these are *written* by kernels and must
    /// survive a pass, and separate from `mirrors` because `begin_pass`
    /// invalidates every activation mirror by design -- re-uploading ~2 MB per
    /// layer per token would undo the whole reason the seam takes a slab.
    /// Cleared by `forget_state` when the engine resets a sequence.
    /// Device-resident recurrent state, with the generation it was filled at.
    ///
    /// **Invalidated in place, never freed.** `forget_state` used to clear this
    /// map, which dropped ~60 `DeviceBuffer`s and made the next pass allocate
    /// them again — 69 ms per checkpoint restore, measured. That is the same
    /// defect `begin_pass` had against the mirror map, where reallocating ~280
    /// buffers a token cost more than the launches it was meant to save. The
    /// fix there and here is the same: bump a generation and re-upload into the
    /// allocation that already exists.
    states: RefCell<HashMap<usize, (DeviceBuffer, u64)>>,
    /// Bumped by `forget_state`; a slab filled at an older generation is stale.
    state_gen: Cell<u64>,

    /// Replace every kernel with `noop`, keeping the launch pattern. See the
    /// kernel's comment; the output is garbage and the point is the clock.
    null_kernels: Cell<bool>,

    /// Reduce RMSNorm's sum of squares serially rather than as a tree.
    ///
    /// The tree is the default and is ~4x faster, but f64 addition rounds and
    /// is therefore not associative, so it cannot be bit-identical to the
    /// oracle. This restores that at the cost of ~2.3 ms a token. See
    /// `rms_norm_tree` in kernels.cu.
    rms_serial: Cell<bool>,
    /// Decode passes seen. The first few run eagerly so every buffer the graph
    /// will point at has been allocated and settled.
    warmups: Cell<u32>,

    /// Events bracketing a pass, so device time can be separated from wall
    /// time. Created lazily; the pair is reused every pass.
    events: RefCell<Option<(ffi::CUevent, ffi::CUevent)>>,
    /// Whether a pass is open, i.e. whether the event pair holds a live pair of
    /// timestamps still to be read.
    pass_open: Cell<bool>,

    /// True between `begin_pass` and `end_pass`. Distinct from `pass_open`,
    /// which means "the event pair still holds unread timestamps".
    in_pass: Cell<bool>,

    /// The model read a device result **in the middle of a pass**, which
    /// `ARCHITECTURE.md` says it must not do.
    ///
    /// **This is a correctness guard, not a diagnostic.** A graph defers every
    /// kernel to `end_pass`, so a mid-pass `host_needs` downloads a buffer
    /// nothing has written this pass — the *previous* pass's contents. The
    /// graph's own safety check cannot see it: that check compares the kernel
    /// *sequence*, and the sequence is identical. The result is plausible
    /// garbage, which is the failure mode the graph machinery spends code to
    /// avoid everywhere else.
    ///
    /// `qwen35moe` does exactly this: top-k over the router's probabilities is
    /// a host decision, so it reads them per layer, per token. It cost a
    /// session's worth of confusion — the first four tokens were right, because
    /// three warm-up passes run eagerly before the graph records.
    ///
    /// So the read is the declaration: a model that needs one is telling the
    /// backend its pass cannot be a graph, and graphs turn off for the rest of
    /// the run. That is a real throughput loss and it is reported rather than
    /// hidden. It goes away when expert selection moves onto the device.
    mid_pass_read: Cell<bool>,
    /// When the host started issuing the current pass.
    issue_start: Cell<Option<std::time::Instant>>,

    /// Per-kernel wall time, when `time_kernels` is on. See [`Cuda::kernel_times`].
    kernel_ms: RefCell<HashMap<&'static str, (u64, f64)>>,

    /// Whether to synchronize after each launch and attribute the time.
    time_kernels: Cell<bool>,

    /// One real launch of every distinct `(kernel, geometry)` this run issued,
    /// with the arguments it was given, and how many times it happened.
    ///
    /// **Recorded rather than described, because describing kept going wrong.**
    /// The previous bench synthesised arguments per kernel from a hand-written
    /// table, so a kernel nobody added to that table was launched every token
    /// and silently absent from its own accounting — `resident_bytes` missing a
    /// map, the h2d counter missing a path, and this bench twice, most recently
    /// omitting `matmul_iq4_xs_q8_k_moe_glu`, the largest kernel in the routed
    /// FFN. Every one of those instruments defaulted to silence.
    ///
    /// Replaying the launch the backend actually made cannot be incomplete: if
    /// it ran, it is here. The pointers stay valid because weights, mirrors and
    /// pool slots all outlive the pass, and replaying after generation cannot
    /// corrupt anything that is still read.
    ///
    /// Only populated under `--profile-device`; a `HashMap` probe per launch is
    /// not something the forward path should pay for.
    launches: RefCell<HashMap<LaunchKey, (u64, Vec<KArg>)>>,
    /// Whether to populate `launches`.
    record_launches: Cell<bool>,

    /// Every `(kernel, n_in, n_out)` this run actually launched, and how often.
    ///
    /// **So the microbenchmark configures itself.** A hand-written bench picks
    /// shapes someone thought were representative; this one replays the shapes
    /// the model really used, in the proportions it used them, which is the
    /// difference between a number and an answer. It is also the only way to
    /// attribute cost per launch without `--profile-kernels`, whose per-launch
    /// synchronize inflates every share in proportion to call count.
    ///
    /// Two `usize` and a `&'static str` per distinct shape, of which a model
    /// has a few dozen. Off any inner loop: one hash per matmul, against a
    /// kernel that runs for microseconds.
    shapes: RefCell<HashMap<(&'static str, usize, usize), u64>>,

    /// The position the RoPE sin/cos table on the device was built for.
    ///
    /// Every layer rotates at the same position within one token, so the table
    /// is identical across all 28 of them — it was being rebuilt and re-sent 56
    /// times per token for no reason.
    /// `(pos of row 0, n_rot, theta bits, rows)`. `rows` is part of the key
    /// because a batch's table is `rows` stacked tables, so its *length* varies
    /// -- a decode step after a prefill must not reuse the prefill's.
    rope_pos: Cell<Option<(usize, usize, u32, usize)>>,
}

/// A device copy of one host activation buffer.
struct Mirror {
    buf: DeviceBuffer,
    /// Whether the device copy is at least as fresh as the host one.
    device_current: bool,
    /// This buffer quantized to Q8_0, and whether that is still current.
    ///
    /// A layer feeds the *same* normed activation to three matmuls (q, k, v)
    /// and then to two more (gate, up). Quantizing is a function of the buffer
    /// alone, so doing it per matmul repeated identical work five times a
    /// layer.
    quant: Option<(DeviceBuffer, DeviceBuffer)>,
    quant_valid: bool,
    /// The same buffer quantized to **Q8_K**: scales, quants, and the per-16
    /// sums Q5_K needs. Held *beside* the Q8_0 copy rather than replacing it.
    ///
    /// Both are live at once on the 35B: one attention layer feeds its normed
    /// activation to `attn_q` (Q6_K, so Q8_K) and to `attn_k` and `attn_v`
    /// (Q8_0, so Q8_0). ggml's `type_traits_cpu[T].vec_dot_type` is a property
    /// of the *weight* format, so a single activation genuinely needs two
    /// quantizations, and a single slot would thrash between them every layer.
    quant_k: Option<(DeviceBuffer, DeviceBuffer, DeviceBuffer)>,
    quant_k_valid: bool,
}

impl Mirror {
    /// Invalidate without freeing. Allocation is the expensive part — a token
    /// touches ~280 activation slices, and reallocating each one cost more than
    /// the launches saved by caching in the first place.
    fn invalidate(&mut self) {
        self.device_current = false;
        self.quant_valid = false;
        self.quant_k_valid = false;
    }
}

/// Driver traffic, counted rather than derived.
///
/// `CLAUDE.md`'s profiler rule is "derive bytes, do not count them", because
/// weight and KV traffic are functions of shapes and a counter would recompute
/// a constant. **This is the case that rule does not cover.** How many times a
/// backend crosses the bus is a property of the backend, not of the model, and
/// it is exactly the quantity the `Ops` seam determines — so it has to be
/// observed. Two increments per op, off any inner loop.
/// What this backend is holding on the device, by category.
///
/// **Built because a prediction was wrong.** Capping the prefill batch was
/// expected to bring a long 9B session from 14.8 GiB to ~10.5 GiB; it brought
/// it to 12.4. The existing counters cannot see the gap: they count crossings
/// and launches, which are properties of the *traffic*, and this is a property
/// of what was never released. The rule in `ARCHITECTURE.md` is the same one --
/// how a backend uses the device is a fact about the backend, so it is observed
/// rather than derived.
#[derive(Debug, Clone, Copy, Default)]
pub struct Resident {
    /// Repacked Q8_0 weights: scales plus quants. Uploaded once, keyed on the
    /// mmap pointer, genuinely permanent.
    pub weight_bytes: u64,
    pub weight_tensors: u64,
    /// KV slabs, one per attending layer, allocated at full context up front.
    pub kv_bytes: u64,
    pub kv_slabs: u64,
    /// Activation mirrors, keyed on **host address** and never freed. This is
    /// the one that can climb across a session: a fresh `Vec` at an address not
    /// seen before inserts a new entry instead of reusing an old one.
    pub mirror_bytes: u64,
    pub mirrors: u64,
    /// The Q8_0 copies hanging off those mirrors.
    pub quant_bytes: u64,
    /// Scratch slots (RoPE tables, attention partials), indexed rather than
    /// keyed, so bounded by construction.
    pub pool_bytes: u64,
    /// The expert slab: **bounded by policy, not by what was touched.** This is
    /// the one category that does not grow with the length of a run, which is
    /// the whole reason it exists.
    pub expert_slab_bytes: u64,
}

impl Resident {
    pub fn total(&self) -> u64 {
        self.weight_bytes
            + self.kv_bytes
            + self.mirror_bytes
            + self.quant_bytes
            + self.pool_bytes
            + self.expert_slab_bytes
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DeviceStats {
    pub launches: u64,
    pub h2d_calls: u64,
    pub h2d_bytes: u64,
    pub d2h_calls: u64,
    pub d2h_bytes: u64,

    /// Forward passes seen, so the rest can be reported per pass.
    pub passes: u64,
    /// Times the host blocked on the device: an explicit synchronize, or a
    /// device-to-host copy, which cannot start until prior work has finished.
    pub syncs: u64,
    /// Device execution time, from CUDA events bracketing each pass.
    ///
    /// This is time on the *stream*, so it includes any gap where the device
    /// had nothing queued — which is the point. A token where `gpu_ns` is far
    /// below the wall clock is a token the card spent waiting for the CPU.
    pub gpu_ns: u64,
    /// Host time spent issuing a pass: everything between `begin_pass` and
    /// `end_pass` returning. Launches, bookkeeping, the model's own code.
    pub issue_ns: u64,
    /// Host time spent blocked in a device-to-host copy.
    pub wait_ns: u64,
    /// Host time spent uploading weights on first sight, in `Cuda::resident`.
    ///
    /// **One-time, and it lands inside the first forward pass**, which is what
    /// made a 0.6B report `prefill 5 tok 725.2 ms 6.9 tok/s` against a decode
    /// of 189.63 — 0.59 GiB of weights crossing the bus, divided by five prompt
    /// tokens and printed as a throughput. The same defect as expert placement,
    /// two orders of magnitude smaller and on every CUDA model rather than only
    /// the MoE one.
    ///
    /// Only the miss branch is timed, so a resident lookup — the case on every
    /// token after the first — costs nothing to observe.
    pub weight_upload_ns: u64,
}

impl DeviceStats {
    /// Bus crossings per token, given how many tokens produced these counts.
    pub fn crossings_per_token(&self, tokens: u64) -> f64 {
        if tokens == 0 {
            return 0.0;
        }
        (self.h2d_calls + self.d2h_calls) as f64 / tokens as f64
    }
}

/// What one round trip through the `Ops` seam costs on this device, measured
/// rather than assumed. Produced by [`Cuda::benchmark`].
#[derive(Debug, Clone, Copy)]
pub struct DeviceBench {
    pub launch_us: f64,
    pub h2d_us: f64,
    pub d2h_us: f64,
    pub round_trip_us: f64,
}

impl DeviceBench {
    /// Milliseconds per token the seam costs, before any arithmetic.
    ///
    /// Built from the counts actually observed rather than from an assumed two
    /// crossings per op: uploads and downloads are not symmetric, because a
    /// matmul uploads a quantized activation as two buffers and RoPE uploads
    /// three. Multiplying a round-trip average by "ops per token" gets this
    /// wrong in both directions.
    pub fn predicted_ms(&self, stats: &DeviceStats, tokens: u64) -> f64 {
        if tokens == 0 {
            return 0.0;
        }
        let total = stats.h2d_calls as f64 * self.h2d_us
            + stats.d2h_calls as f64 * self.d2h_us
            + stats.launches as f64 * self.launch_us;
        total / tokens as f64 / 1000.0
    }
}

/// A decode step, recorded once and replayed per token.
///
/// # Why this exists
///
/// A decode step issues ~673 kernels and each `cuLaunchKernel` costs ~7 us of
/// *CPU* time — driver bookkeeping, not arithmetic. That is ~4.7 ms of an 11 ms
/// token, a third of it spent describing work rather than doing it, which is
/// why the card drew 50 W of a possible 180 and sat at 55-80% utilization.
///
/// A graph is the same sequence recorded once and replayed with a single call.
///
/// # Why it is built by hand rather than captured
///
/// Stream capture is the usual route and it does not fit here: `n_pos` grows
/// every token, so attention's grid and several arguments change, and a
/// captured graph would be stale on the very next token. Re-capturing each
/// token costs about what the launches did.
///
/// Building the nodes ourselves means we keep every handle, so a replay can
/// update just the parameters that moved — roughly 112 of 673 nodes — with
/// `cuGraphExecKernelNodeSetParams`. The rest are untouched.
///
/// # The safety property that matters
///
/// A graph is only correct if the launch sequence is *identical* every pass.
/// That holds for single-token decode and not for prefill, so only decode is
/// eligible. Rather than trust that, replay checks each launch against the node
/// it is standing in for and **fails loudly** on any divergence in name, count
/// or order. A wrong graph would otherwise produce plausible garbage.
enum GraphState {
    /// Launching eagerly.
    Off,
    /// Building the node list. Kernels are added, not run; the pass that
    /// records is also the pass that first replays, so no token is skipped.
    Recording {
        graph: ffi::CUgraph,
        nodes: Vec<RecordedNode>,
        prev: Option<ffi::CUgraphNode>,
    },
    /// Instantiated. Each pass walks `cursor` through `nodes`, updating any
    /// whose parameters have changed, then launches once.
    Ready {
        graph: ffi::CUgraph,
        exec: ffi::CUgraphExec,
        nodes: Vec<RecordedNode>,
        cursor: usize,
    },
}

/// One node, plus the launch it was recorded from, so a replay can tell what
/// changed.
struct RecordedNode {
    node: ffi::CUgraphNode,
    name: &'static str,
    grid: (u32, u32),
    block: u32,
    shared: u32,
    args: Vec<KArg>,
}

/// One kernel argument, by value.
///
/// The driver wants an array of *pointers* to arguments, which is fine when a
/// launch is a transient thing built from locals. It stops being fine the
/// moment those arguments have to outlive the call — which is what building a
/// CUDA graph needs, since a node keeps its parameters and they are updated
/// later rather than rebuilt.
///
/// So launches carry values and the pointer array is built at the last moment.
/// What makes one recorded launch distinct from another.
///
/// Name, grid, block, shared bytes — **and the first two integer arguments**.
///
/// The scalars are load-bearing and were left out at first. `matmul_f32_t` runs
/// at `{2048, 1}` for the shared-expert gate and `{2048, 32}` for `ssm_alpha`,
/// and both launch a grid of 1x1 with 128 threads: identical geometry,
/// different shapes. Keyed on geometry alone they collide, so the recorder kept
/// whichever arrived first and `bench_f32_variants` could only ever replay one
/// of them.
///
/// That cost a shipped 1.65x regression. A staged rewrite was decomposed
/// against the shape the bench happened to hold, measured 2.65x faster, and was
/// applied to all three — where the widest one needs 44 shared-memory tiles and
/// 88 `__syncthreads` that the bench never saw. **An instrument that cannot
/// distinguish two things will average them and report the average
/// confidently.**
///
/// Two scalars rather than all of them because every kernel here takes its
/// shape first; pointers vary per call and must not be in the key.
pub(super) type LaunchKey = (&'static str, u32, u32, u32, u32, i32, i32);

/// The first two integer arguments of a launch, or zeroes.
pub(super) fn scalar_key(args: &[KArg]) -> (i32, i32) {
    let at = |i: usize| match args.get(i) {
        Some(KArg::I32(v)) => *v,
        _ => 0,
    };
    (at(0), at(1))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum KArg {
    I32(i32),
    F32(f32),
    Ptr(ffi::CUdeviceptr),
}

/// Argument values in a stable place, plus the pointer array the driver reads.
///
/// Every slot is eight bytes and the driver reads four for an `i32` or `f32`.
/// That works because this targets little-endian x86_64 only, which the
/// platform scope in `CLAUDE.md` already fixes.
struct ArgPack {
    slots: Vec<u64>,
}

impl ArgPack {
    fn new(args: &[KArg]) -> Self {
        Self {
            slots: args
                .iter()
                .map(|a| match *a {
                    KArg::I32(v) => v as u32 as u64,
                    KArg::F32(v) => v.to_bits() as u64,
                    KArg::Ptr(p) => p,
                })
                .collect(),
        }
    }

    /// Pointers into `slots`. Borrowed mutably so the array cannot outlive it.
    fn ptrs(&mut self) -> Vec<*mut c_void> {
        self.slots
            .iter_mut()
            .map(|s| s as *mut u64 as *mut c_void)
            .collect()
    }
}

/// A device mirror of a host KV slab, and how much of it is current.
struct KvMirror {
    buf: DeviceBuffer,
    /// Positions already copied from the host. A smaller `n_pos` than this
    /// means the cache was reset, so the mirror is refilled from the start.
    uploaded: usize,
    /// Whether this backend has written the slab itself.
    ///
    /// Once it has, the host copy is stale and must never be uploaded over the
    /// device one. This is what turns the KV cache from something that shuttles
    /// back and forth into something a GPU layer simply owns.
    device_written: bool,
}

// SAFETY: a CUDA context is usable from any thread that has it current, and we
// only ever use it from the thread that owns this value. The raw pointers are
// driver handles, not references into our address space.
unsafe impl Send for Cuda {}

impl Cuda {
    /// Initialize the driver, take device `ordinal`, and load the kernels.
    pub fn new(ordinal: i32) -> Result<Self> {
        Self::with_options(ordinal, false)
    }

    /// As [`Cuda::new`], optionally asking the driver to block rather than spin
    /// on synchronization. See the flag comment below; this exists for
    /// measurement, not for speed.
    pub fn with_options(ordinal: i32, blocking_sync: bool) -> Result<Self> {
        // SAFETY: every call below is checked, and each is passed either a
        // valid out-pointer to a local or a handle the driver just produced.
        unsafe {
            check(ffi::cuInit(0), "cuInit")?;

            let mut count = 0;
            check(ffi::cuDeviceGetCount(&mut count), "cuDeviceGetCount")?;
            if ordinal >= count {
                return Err(Error::Cuda {
                    what: "cuDeviceGet",
                    detail: format!("device {ordinal} requested, {count} present"),
                });
            }

            let mut device = 0;
            check(ffi::cuDeviceGet(&mut device, ordinal), "cuDeviceGet")?;

            let mut raw = [0i8; 128];
            check(
                ffi::cuDeviceGetName(raw.as_mut_ptr(), raw.len() as i32, device),
                "cuDeviceGetName",
            )?;
            let name = std::ffi::CStr::from_ptr(raw.as_ptr())
                .to_string_lossy()
                .into_owned();

            let attr = |a| -> Result<i32> {
                let mut v = 0;
                check(
                    ffi::cuDeviceGetAttribute(&mut v, a, device),
                    "cuDeviceGetAttribute",
                )?;
                Ok(v)
            };
            let capability = (attr(ffi::ATTR_CC_MAJOR)?, attr(ffi::ATTR_CC_MINOR)?);
            let sm_count = attr(ffi::ATTR_SM_COUNT)?;

            let mut total_mem = 0usize;
            check(
                ffi::cuDeviceTotalMem_v2(&mut total_mem, device),
                "cuDeviceTotalMem",
            )?;

            let mut context: ffi::CUcontext = std::ptr::null_mut();
            // `CU_CTX_SCHED_BLOCKING_SYNC` (0x04) makes the driver sleep on a
            // synchronization instead of spinning. The default, `SCHED_AUTO`,
            // busy-waits when there are more cores than contexts — which is
            // correct for latency and ruinous for *measuring*, because a host
            // blocked in a copy then looks exactly like a host doing work.
            // Off by default so nothing changes for a normal run.
            let flags = if blocking_sync { 0x04 } else { 0x00 };
            check(ffi::cuCtxCreate_v2(&mut context, flags, device), "cuCtxCreate")?;

            // The PTX is a NUL-terminated image as far as the driver is
            // concerned, so it has to be one.
            let image = CString::new(PTX).map_err(|_| Error::Cuda {
                what: "cuModuleLoadData",
                detail: "PTX contains an interior NUL".to_string(),
            })?;
            let mut module: ffi::CUmodule = std::ptr::null_mut();
            let load = ffi::cuModuleLoadData(&mut module, image.as_ptr() as *const c_void);
            if load != ffi::CUDA_SUCCESS {
                ffi::cuCtxDestroy_v2(context);
                return Err(Error::Cuda {
                    what: "cuModuleLoadData",
                    detail: format!(
                        "{} — a JIT failure here usually means the PTX targets a \
                         newer architecture than the driver understands",
                        ffi::describe(load)
                    ),
                });
            }

            Ok(Self {
                context,
                module,
                name,
                capability,
                sm_count,
                total_mem,
                weights: RefCell::new(HashMap::new()),
                kv: RefCell::new(HashMap::new()),
                pool: RefCell::new(Vec::new()),
                error: RefCell::new(None),
                functions: RefCell::new(HashMap::new()),
                stats: Cell::new(DeviceStats::default()),
                mirrors: RefCell::new(HashMap::new()),
                rope_pos: Cell::new(None),
                graph: RefCell::new(GraphState::Off),
                pass_graph: Cell::new(false),
                graphs_enabled: Cell::new(true),
                rms_serial: Cell::new(false),
                null_kernels: Cell::new(false),
                states: RefCell::new(HashMap::new()),
                state_gen: Cell::new(0),
                q8: RefCell::new(HashMap::new()),
                f32t: RefCell::new(HashMap::new()),
                experts: RefCell::new(None),
                shapes: RefCell::new(HashMap::new()),
                launches: RefCell::new(HashMap::new()),
                record_launches: Cell::new(false),
                expert_reserve: Cell::new(experts::DEFAULT_RESERVE),
                expert_host_budget: Cell::new(experts::DEFAULT_HOST_BUDGET),
                f32_staged: Cell::new(false),
                f32t_pair: RefCell::new(HashMap::new()),
                attn_warp: Cell::new(None),
                iq4_untiled: Cell::new(false),
                attn_warp_only: Cell::new(None),
                attn_calls: Cell::new(0),
                report_per_turn: Cell::new(false),
                model_path: RefCell::new(None),
                passes_seen: Cell::new(0),
                map_base: Cell::new(None),
                warmups: Cell::new(0),
                events: RefCell::new(None),
                pass_open: Cell::new(false),
                in_pass: Cell::new(false),
                mid_pass_read: Cell::new(false),
                issue_start: Cell::new(None),
                kernel_ms: RefCell::new(HashMap::new()),
                time_kernels: Cell::new(false),
            })
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Compute capability as `(major, minor)` — `(12, 0)` is sm_120.
    pub fn capability(&self) -> (i32, i32) {
        self.capability
    }

    pub fn sm_count(&self) -> i32 {
        self.sm_count
    }

    pub fn total_mem(&self) -> usize {
        self.total_mem
    }

    /// Free and total device memory, right now.
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: both are valid out-pointers and the context is current.
        unsafe { check(ffi::cuMemGetInfo_v2(&mut free, &mut total), "cuMemGetInfo")? };
        Ok((free, total))
    }

    fn function(&self, name: &str) -> Result<ffi::CUfunction> {
        let cname = CString::new(name).map_err(|_| Error::Cuda {
            what: "cuModuleGetFunction",
            detail: format!("{name:?} is not a valid symbol"),
        })?;
        let mut f: ffi::CUfunction = std::ptr::null_mut();
        // SAFETY: `self.module` is loaded and alive for `self`'s lifetime.
        unsafe {
            check(
                ffi::cuModuleGetFunction(&mut f, self.module, cname.as_ptr()),
                "cuModuleGetFunction",
            )?
        };
        Ok(f)
    }

    /// Block until every launched kernel has finished.
    pub fn sync(&self) -> Result<()> {
        // SAFETY: no arguments; only reports the context's status.
        unsafe { check(ffi::cuCtxSynchronize(), "cuCtxSynchronize") }
    }

    /// Launch a kernel by name over a 1-D grid, then block until it finishes.
    ///
    /// Synchronizing on every launch is the naive shape, and deliberate: the
    /// `Ops` seam hands each method host slices and expects host slices back,
    /// so every call is a self-contained round trip whatever we do here. What
    /// that costs is one of the things this backend exists to measure.
    ///
    /// # Safety
    /// `params` must match the named kernel's signature, and every device
    /// pointer in it must address an allocation large enough for the extents
    /// the kernel will walk.
    unsafe fn launch(&self, name: &'static str, grid: u32, block: u32, args: &[KArg]) -> Result<()> {
        // SAFETY: forwarded to the caller's contract.
        unsafe { self.launch_shared(name, grid, block, 0, args) }
    }

    /// As [`Cuda::launch_shared`], with a two-dimensional grid.
    ///
    /// Flash-decoding wants one block per (query head, chunk of positions),
    /// which is what makes its parallelism scale with context instead of with
    /// head count.
    ///
    /// # Safety
    /// As [`Cuda::launch_shared`].
    unsafe fn launch_grid2(
        &self,
        name: &'static str,
        grid_x: u32,
        grid_y: u32,
        block: u32,
        shared_bytes: u32,
        args: &[KArg],
    ) -> Result<()> {
        if self.pass_graph.get() {
            return self.graph_launch(name, grid_x, grid_y, block, shared_bytes, args);
        }
        // Every launch funnels through here, which is what makes the
        // substitution total: same count, same order, same geometry.
        let nulled = self.null_kernels.get();
        let f = self.cached_function(if nulled { "noop" } else { name })?;
        let empty: [KArg; 0] = [];
        let mut pack = ArgPack::new(if nulled { &empty } else { args });
        let mut params = pack.ptrs();
        let started = std::time::Instant::now();
        // SAFETY: the caller's contract, documented above.
        unsafe {
            check(
                ffi::cuLaunchKernel(
                    f,
                    grid_x,
                    grid_y,
                    1,
                    block,
                    1,
                    1,
                    if nulled { 0 } else { shared_bytes },
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?
        };
        self.bump(|s| s.launches += 1);
        if self.record_launches.get() {
            let (s0, s1) = scalar_key(args);
            let key = (name, grid_x, grid_y, block, shared_bytes, s0, s1);
            let mut m = self.launches.borrow_mut();
            match m.get_mut(&key) {
                Some(e) => e.0 += 1,
                None => {
                    m.insert(key, (1, args.to_vec()));
                }
            }
        }
        if self.time_kernels.get() {
            self.sync()?;
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            let mut map = self.kernel_ms.borrow_mut();
            let e = map.entry(name).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ms;
        }
        Ok(())
    }

    /// As [`Cuda::launch`], with dynamic shared memory.
    ///
    /// # Safety
    /// As [`Cuda::launch`], and `shared_bytes` must cover what the kernel
    /// indexes through its `extern __shared__` array.
    unsafe fn launch_shared(
        &self,
        name: &'static str,
        grid: u32,
        block: u32,
        shared_bytes: u32,
        args: &[KArg],
    ) -> Result<()> {
        if self.pass_graph.get() {
            return self.graph_launch(name, grid, 1, block, shared_bytes, args);
        }
        // Every launch funnels through here, which is what makes the
        // substitution total: same count, same order, same geometry.
        let nulled = self.null_kernels.get();
        let f = self.cached_function(if nulled { "noop" } else { name })?;
        let empty: [KArg; 0] = [];
        let mut pack = ArgPack::new(if nulled { &empty } else { args });
        let mut params = pack.ptrs();
        let started = std::time::Instant::now();
        // SAFETY: the caller's contract, documented above.
        unsafe {
            check(
                ffi::cuLaunchKernel(
                    f,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    if nulled { 0 } else { shared_bytes },
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?
        };
        self.bump(|s| s.launches += 1);
        if self.record_launches.get() {
            let (s0, s1) = scalar_key(args);
            let key = (name, grid, 1, block, shared_bytes, s0, s1);
            let mut m = self.launches.borrow_mut();
            match m.get_mut(&key) {
                Some(e) => e.0 += 1,
                None => {
                    m.insert(key, (1, args.to_vec()));
                }
            }
        }
        if self.time_kernels.get() {
            // Synchronizing here is the whole point and also the whole cost:
            // without it the elapsed time measures the CPU-side enqueue, not
            // the kernel. It inflates the total, so read the *shares* rather
            // than the absolute milliseconds.
            self.sync()?;
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            let mut map = self.kernel_ms.borrow_mut();
            let e = map.entry(name).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ms;
        }
        // Deliberately no `sync` here. Every `Ops` method ends in a
        // device-to-host copy on the null stream, which is ordered after this
        // kernel and is itself synchronous, so an explicit barrier is a second
        // driver call buying nothing. A launch failure surfaces at that copy.
        Ok(())
    }

    /// Attribute time to individual kernels, at the cost of a device sync
    /// after every launch.
    ///
    /// The half-level profile says *where* in a layer the time goes; this says
    /// *which kernel*. Off by default because the sync it needs changes the
    /// thing it measures — the absolute total rises, so the useful output is
    /// each kernel's share, not its milliseconds.
    pub fn time_kernels(&self, on: bool) {
        self.time_kernels.set(on);
    }

    /// Per-kernel `(calls, milliseconds)`, busiest first.
    pub fn kernel_times(&self) -> Vec<(&'static str, u64, f64)> {
        let mut v: Vec<_> = self
            .kernel_ms
            .borrow()
            .iter()
            .map(|(k, (n, ms))| (*k, *n, *ms))
            .collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2));
        v
    }

    /// Update the counters. `Cell` rather than atomics: this backend is used
    /// from one thread, and the whole point is that it costs nothing.
    pub(super) fn bump(&self, f: impl FnOnce(&mut DeviceStats)) {
        let mut s = self.stats.get();
        f(&mut s);
        self.stats.set(s);
    }

    /// Driver traffic so far.
    /// Every device allocation this backend is holding, by category.
    ///
    /// Walks the four stores rather than maintaining a running total: it is
    /// called once at the end of a run, and a counter incremented on every
    /// allocation would be one more thing to keep honest.
    pub fn resident_bytes(&self) -> Resident {
        let mut r = Resident::default();
        for (sc, q) in self.q8.borrow().values() {
            r.weight_bytes += (sc.len_bytes() + q.len_bytes()) as u64;
            r.weight_tensors += 1;
        }
        // **This map was missing here until the 35B arrived**, and the omission
        // was invisible for exactly as long as every weight was Q8_0: `q8`
        // above held all of them, and `weights` held only norm vectors. The
        // 35B puts every k-quant tensor — which is every routed expert, 15.94
        // GiB of them — through `resident` instead, so leaving it out
        // under-reported residency by nearly the whole model. The counter
        // exists because a memory prediction was once wrong by 1.9 GiB; it can
        // only do that job if it counts every map that allocates.
        for b in self.weights.borrow().values() {
            r.weight_bytes += b.len_bytes() as u64;
            r.weight_tensors += 1;
        }
        for b in self.f32t.borrow().values() {
            r.weight_bytes += b.len_bytes() as u64;
            r.weight_tensors += 1;
        }
        if let Some(c) = self.experts.borrow().as_ref() {
            r.expert_slab_bytes = c.resident_bytes();
        }
        for m in self.kv.borrow().values() {
            r.kv_bytes += m.buf.len_bytes() as u64;
            r.kv_slabs += 1;
        }
        for m in self.mirrors.borrow().values() {
            r.mirror_bytes += m.buf.len_bytes() as u64;
            r.mirrors += 1;
            if let Some((sc, q)) = &m.quant {
                r.quant_bytes += (sc.len_bytes() + q.len_bytes()) as u64;
            }
            if let Some((sc, q, bs)) = &m.quant_k {
                r.quant_bytes += (sc.len_bytes() + q.len_bytes() + bs.len_bytes()) as u64;
            }
        }
        for b in self.pool.borrow().iter() {
            r.pool_bytes += b.len_bytes() as u64;
        }
        r
    }

    pub fn stats(&self) -> DeviceStats {
        self.stats.get()
    }

    /// Time a launch, an upload, a download, and the three together.
    ///
    /// Synthetic on purpose: a 4 KiB payload is about the size of this model's
    /// residual stream, so the result is the *fixed* cost of using the seam,
    /// with as little real work attached as possible.
    pub fn benchmark(&self, reps: u32) -> Result<DeviceBench> {
        use std::time::Instant;

        let n = 1024usize;
        let host = vec![1.0f32; n];
        let x = DeviceBuffer::from_slice(&host)?;
        let y = DeviceBuffer::from_slice(&host)?;
        let mut back = vec![0.0f32; n];

        // Resolve the kernel and let the JIT settle before anything is timed.
        self.saxpy(1.0, &x, &y, n)?;

        let time = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
            let t = Instant::now();
            for _ in 0..reps {
                f()?;
            }
            Ok(t.elapsed().as_secs_f64() * 1e6 / f64::from(reps))
        };

        // Launches are timed in a batch against a single sync, because that is
        // how the backend issues them. Timing one launch plus one
        // `cuCtxSynchronize` measures a barrier the forward pass never pays.
        let launch_us = {
            let f = self.cached_function("saxpy")?;
            let (mut n_arg, mut a_arg) = (n as i32, 1.0f32);
            let (mut x_arg, mut y_arg) = (x.ptr, y.ptr);
            let mut params = [
                &mut n_arg as *mut _ as *mut c_void,
                &mut a_arg as *mut _ as *mut c_void,
                &mut x_arg as *mut _ as *mut c_void,
                &mut y_arg as *mut _ as *mut c_void,
            ];
            let t = Instant::now();
            for _ in 0..reps {
                // SAFETY: parameters match `saxpy`; both buffers hold `n` floats.
                unsafe {
                    check(
                        ffi::cuLaunchKernel(
                            f,
                            n.div_ceil(256) as u32,
                            1,
                            1,
                            256,
                            1,
                            1,
                            0,
                            std::ptr::null_mut(),
                            params.as_mut_ptr(),
                            std::ptr::null_mut(),
                        ),
                        "cuLaunchKernel",
                    )?
                };
            }
            self.sync()?;
            t.elapsed().as_secs_f64() * 1e6 / f64::from(reps)
        };
        let h2d_us = time(&mut || x.write(&host))?;
        let d2h_us = time(&mut || y.read(&mut back))?;
        let round_trip_us = time(&mut || {
            x.write(&host)?;
            self.saxpy(1.0, &x, &y, n)?;
            y.read(&mut back)
        })?;

        Ok(DeviceBench {
            launch_us,
            h2d_us,
            d2h_us,
            round_trip_us,
        })
    }

    /// Record or replay one launch, instead of issuing it.
    fn graph_launch(
        &self,
        name: &'static str,
        grid_x: u32,
        grid_y: u32,
        block: u32,
        shared: u32,
        args: &[KArg],
    ) -> Result<()> {
        let f = self.cached_function(name)?;
        let mut state = self.graph.borrow_mut();
        match &mut *state {
            GraphState::Recording { graph, nodes, prev } => {
                let mut pack = ArgPack::new(args);
                let mut ptrs = pack.ptrs();
                let p = ffi::KernelNodeParams {
                    func: f,
                    grid_x,
                    grid_y,
                    grid_z: 1,
                    block_x: block,
                    block_y: 1,
                    block_z: 1,
                    shared_bytes: shared,
                    params: ptrs.as_mut_ptr(),
                    extra: std::ptr::null_mut(),
                    kern: std::ptr::null_mut(),
                    ctx: std::ptr::null_mut(),
                };
                let mut node: ffi::CUgraphNode = std::ptr::null_mut();
                let deps = prev.map(|n| [n]);
                let (dep_ptr, n_deps) = match &deps {
                    Some(d) => (d.as_ptr(), 1usize),
                    None => (std::ptr::null(), 0usize),
                };
                // SAFETY: `p` matches CUDA_KERNEL_NODE_PARAMS_v2 and names a
                // function from our own module; the driver copies both it and
                // the parameter array during this call.
                unsafe {
                    check(
                        ffi::cuGraphAddKernelNode_v2(&mut node, *graph, dep_ptr, n_deps, &p),
                        "cuGraphAddKernelNode",
                    )?
                };
                *prev = Some(node);
                nodes.push(RecordedNode {
                    node,
                    name,
                    grid: (grid_x, grid_y),
                    block,
                    shared,
                    args: args.to_vec(),
                });
                Ok(())
            }
            GraphState::Ready {
                exec,
                nodes,
                cursor,
                ..
            } => {
                let i = *cursor;
                *cursor += 1;
                let n = match nodes.get_mut(i) {
                    Some(n) if n.name == name => n,
                    _ => {
                        return Err(Error::Cuda {
                            what: "graph replay",
                            detail: format!(
                                "launch {i} is {name:?}, but the recorded step has {} \
                                 there. The kernel sequence is not identical between \
                                 passes, which a graph cannot express.",
                                nodes.get(i).map(|n| n.name).unwrap_or("nothing")
                            ),
                        });
                    }
                };
                if n.grid == (grid_x, grid_y)
                    && n.block == block
                    && n.shared == shared
                    && n.args == args
                {
                    return Ok(());
                }
                let mut pack = ArgPack::new(args);
                let mut ptrs = pack.ptrs();
                let p = ffi::KernelNodeParams {
                    func: f,
                    grid_x,
                    grid_y,
                    grid_z: 1,
                    block_x: block,
                    block_y: 1,
                    block_z: 1,
                    shared_bytes: shared,
                    params: ptrs.as_mut_ptr(),
                    extra: std::ptr::null_mut(),
                    kern: std::ptr::null_mut(),
                    ctx: std::ptr::null_mut(),
                };
                // SAFETY: as above; `n.node` belongs to the graph `exec` was
                // instantiated from.
                unsafe {
                    check(
                        ffi::cuGraphExecKernelNodeSetParams_v2(*exec, n.node, &p),
                        "cuGraphExecKernelNodeSetParams",
                    )?
                };
                n.grid = (grid_x, grid_y);
                n.block = block;
                n.shared = shared;
                n.args.clear();
                n.args.extend_from_slice(args);
                Ok(())
            }
            GraphState::Off => Err(Error::Cuda {
                what: "graph launch",
                detail: "the pass claimed to use a graph but none is active".to_string(),
            }),
        }
    }

    /// Begin a pass. `n_tokens` decides eligibility: only single-token decode
    /// has a fixed kernel sequence.
    fn graph_begin(&self, n_tokens: usize) -> Result<()> {
        // Per-kernel timing needs a launch it can time, which a graph is not.
        let eligible = n_tokens == 1 && !self.time_kernels.get() && self.graphs_enabled.get();
        self.pass_graph.set(eligible);
        if !eligible {
            return Ok(());
        }

        let warm = self.warmups.get();
        let mut state = self.graph.borrow_mut();
        match &mut *state {
            GraphState::Ready { cursor, .. } => *cursor = 0,
            GraphState::Off => {
                // A few eager passes first, so every buffer the graph will
                // point at has been allocated and stopped moving.
                if warm < 3 {
                    self.warmups.set(warm + 1);
                    self.pass_graph.set(false);
                    return Ok(());
                }
                let mut graph: ffi::CUgraph = std::ptr::null_mut();
                // SAFETY: valid out-pointer; flags must be zero.
                unsafe { check(ffi::cuGraphCreate(&mut graph, 0), "cuGraphCreate")? };
                *state = GraphState::Recording {
                    graph,
                    nodes: Vec::new(),
                    prev: None,
                };
            }
            GraphState::Recording { .. } => {}
        }
        Ok(())
    }

    /// End a pass: instantiate if this was the recording pass, then launch.
    fn graph_end(&self) -> Result<()> {
        if !self.pass_graph.get() {
            return Ok(());
        }
        let mut state = self.graph.borrow_mut();
        let taken = std::mem::replace(&mut *state, GraphState::Off);
        match taken {
            GraphState::Recording { graph, nodes, .. } => {
                let mut exec: ffi::CUgraphExec = std::ptr::null_mut();
                // SAFETY: `graph` is ours and fully built; flags zero.
                unsafe {
                    check(
                        ffi::cuGraphInstantiateWithFlags(&mut exec, graph, 0),
                        "cuGraphInstantiate",
                    )?
                };
                // SAFETY: `exec` was just instantiated; the null stream is the
                // one every copy in this backend uses, so ordering holds.
                unsafe {
                    check(
                        ffi::cuGraphLaunch(exec, std::ptr::null_mut()),
                        "cuGraphLaunch",
                    )?
                };
                self.bump(|s| s.launches += 1);
                *state = GraphState::Ready {
                    graph,
                    exec,
                    nodes,
                    cursor: 0,
                };
                Ok(())
            }
            GraphState::Ready {
                graph,
                exec,
                nodes,
                cursor,
            } => {
                if cursor != nodes.len() {
                    let (a, b) = (cursor, nodes.len());
                    *state = GraphState::Ready {
                        graph,
                        exec,
                        nodes,
                        cursor,
                    };
                    return Err(Error::Cuda {
                        what: "graph replay",
                        detail: format!(
                            "this pass issued {a} launches, the recorded step has {b}. \
                             The kernel sequence is not identical between passes."
                        ),
                    });
                }
                // SAFETY: as above.
                unsafe {
                    check(
                        ffi::cuGraphLaunch(exec, std::ptr::null_mut()),
                        "cuGraphLaunch",
                    )?
                };
                self.bump(|s| s.launches += 1);
                *state = GraphState::Ready {
                    graph,
                    exec,
                    nodes,
                    cursor,
                };
                Ok(())
            }
            GraphState::Off => Ok(()),
        }
    }

    /// Start the clocks for a pass, and bank the previous pass's device time.
    ///
    /// The elapsed time is read at the *start* of the next pass rather than at
    /// the end of this one, because reading it requires the stop event to have
    /// completed and waiting for that here would be the very stall this is
    /// meant to measure.
    fn timing_begin(&self) -> Result<()> {
        let mut slot = self.events.borrow_mut();
        if slot.is_none() {
            let (mut a, mut b): (ffi::CUevent, ffi::CUevent) =
                (std::ptr::null_mut(), std::ptr::null_mut());
            // SAFETY: valid out-pointers; flags zero is CU_EVENT_DEFAULT, which
            // is the timing-enabled one.
            unsafe {
                check(ffi::cuEventCreate(&mut a, 0), "cuEventCreate")?;
                check(ffi::cuEventCreate(&mut b, 0), "cuEventCreate")?;
            }
            *slot = Some((a, b));
        }
        let (start, stop) = match *slot {
            Some(p) => p,
            None => return Ok(()),
        };

        if self.pass_open.get() {
            let mut ms = 0.0f32;
            // SAFETY: both events were recorded during the previous pass, and
            // that pass ended in a blocking copy, so `stop` has completed.
            unsafe {
                check(ffi::cuEventSynchronize(stop), "cuEventSynchronize")?;
                check(
                    ffi::cuEventElapsedTime(&mut ms, start, stop),
                    "cuEventElapsedTime",
                )?;
            }
            let ns = (f64::from(ms) * 1e6) as u64;
            self.bump(|s| s.gpu_ns += ns);
            self.pass_open.set(false);
        }

        // SAFETY: `start` is ours; the null stream is the one everything uses.
        unsafe { check(ffi::cuEventRecord(start, std::ptr::null_mut()), "cuEventRecord")? };
        self.issue_start.set(Some(std::time::Instant::now()));
        self.bump(|s| s.passes += 1);
        Ok(())
    }

    /// Stop the clocks for a pass.
    fn timing_end(&self) -> Result<()> {
        let stop = match *self.events.borrow() {
            Some((_, stop)) => stop,
            None => return Ok(()),
        };
        // SAFETY: as above.
        unsafe { check(ffi::cuEventRecord(stop, std::ptr::null_mut()), "cuEventRecord")? };
        self.pass_open.set(true);
        if let Some(t) = self.issue_start.replace(None) {
            let ns = t.elapsed().as_nanos() as u64;
            self.bump(|s| s.issue_ns += ns);
        }
        Ok(())
    }

    /// Time `reps` launches of a diagnostic kernel taking `(int n, const float
    /// *x, float *out)`, and return microseconds per launch *and* what the
    /// kernel computed.
    ///
    /// Launches are timed as a batch against a single synchronize, because that
    /// is how the backend issues them; timing each against its own barrier
    /// measures a barrier the forward pass never pays. The consequence is a
    /// **floor at the launch issue cost**, ~8-12 us here, so a variant that
    /// reports near that is launch-bound and its true cost is unresolved.
    ///
    /// The result comes back as **f64**, and that is load-bearing rather than
    /// tidy: the question these kernels exist to answer is whether a reduction
    /// is order-free, and reading the answer back through an f32 cast destroys
    /// the ~1e-13 that separates one summation order from another. It is the
    /// same f32 cast that makes a tree reduction pass every exactness test in
    /// the real kernel, which is exactly why a diagnostic must not repeat it.
    ///
    /// `mode` picks the input. 0 is benign. 1 spans ~39 binades, the kind of
    /// dynamic range an outlier feature gives a residual stream. 2 is
    /// adversarial — one enormous term among many equal small ones, the
    /// classic case where *where* a term is added decides whether it survives.
    /// A reduction that only agrees on well-conditioned input has not been
    /// tested.
    pub fn bench_kernel(
        &self,
        name: &'static str,
        n: usize,
        shared_floats: usize,
        threads: u32,
        mode: u8,
        reps: u32,
    ) -> Result<(f64, f64)> {
        let host: Vec<f32> = (0..n)
            .map(|i| {
                let base = (i % 97) as f32 * 0.01 - 0.5;
                match mode {
                    1 => base * 2.0f32.powi((i % 35) as i32 - 17),
                    2 => if i % 512 == 0 { 1.0e8 } else { 1.0 },
                    _ => base,
                }
            })
            .collect();
        let x = DeviceBuffer::from_slice(&host)?;
        let out = DeviceBuffer::new(8)?;
        let args = [KArg::I32(n as i32), KArg::Ptr(x.ptr), KArg::Ptr(out.ptr)];
        let shared = (shared_floats * 4) as u32;

        let was = self.pass_graph.replace(false);
        // SAFETY: the arguments match every kernel in the diagnostic set, and
        // `shared` is what the caller says the kernel indexes.
        let run = |reps: u32| -> Result<f64> {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                unsafe { self.launch_shared(name, 1, threads, shared, &args)? };
            }
            self.sync()?;
            Ok(t.elapsed().as_secs_f64() * 1e6 / f64::from(reps))
        };
        run(64)?; // warm the module and let clocks settle
        let us = run(reps)?;
        self.pass_graph.set(was);
        let mut got = [0.0f64];
        out.read(&mut got)?;
        Ok((us, got[0]))
    }

    /// Turn graph capture off.
    ///
    /// A graph batches a whole pass and runs it at `end_pass`, so a caller that
    /// issues one op and immediately reads the result — the per-op differential
    /// tests, or a backend that mixes CPU and GPU at op granularity — must not
    /// use one. Those callers say so here rather than being silently wrong.
    pub fn use_graphs(&self, on: bool) {
        self.graphs_enabled.set(on);
    }

    /// Reduce RMSNorm serially, trading ~2.3 ms a token for bit-equality with
    /// the `naive` oracle.
    ///
    /// Set it before the first pass. A recorded graph holds whichever kernel
    /// was chosen when it was built, and replay verifies the kernel sequence,
    /// so flipping this mid-run would fail loudly rather than silently.
    pub fn rms_serial(&self, on: bool) {
        self.rms_serial.set(on);
    }

    /// Which RMSNorm kernels are selected. Named here so the dispatch and the
    /// tests cannot drift apart.
    pub(crate) fn rms_kernels(&self) -> (&'static str, &'static str) {
        if self.rms_serial.get() {
            ("rms_norm", "rms_norm_heads")
        } else {
            ("rms_norm_tree", "rms_norm_heads_tree")
        }
    }

    /// Whether a decode step is currently replaying from a graph.
    pub fn graph_active(&self) -> bool {
        matches!(&*self.graph.borrow(), GraphState::Ready { .. })
    }

    /// A kernel handle, resolved once per symbol.
    fn cached_function(&self, name: &'static str) -> Result<ffi::CUfunction> {
        let mut map = self.functions.borrow_mut();
        if let Some(f) = map.get(name) {
            return Ok(*f);
        }
        let f = self.function(name)?;
        map.insert(name, f);
        Ok(f)
    }

    /// `y = a * x + y`, elementwise. The toolchain proof.
    pub fn saxpy(&self, a: f32, x: &DeviceBuffer, y: &DeviceBuffer, n: usize) -> Result<()> {
        let f = self.function("saxpy")?;
        let mut n_arg = n as i32;
        let mut a_arg = a;
        let mut x_arg = x.ptr;
        let mut y_arg = y.ptr;
        let mut params = [
            &mut n_arg as *mut _ as *mut c_void,
            &mut a_arg as *mut _ as *mut c_void,
            &mut x_arg as *mut _ as *mut c_void,
            &mut y_arg as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = n.div_ceil(block as usize) as u32;
        // SAFETY: the parameter list matches the kernel's signature in
        // `kernels.cu`, and both buffers hold at least `n` floats — checked by
        // the caller, which is this crate's tests.
        unsafe {
            check(
                ffi::cuLaunchKernel(
                    f,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel(saxpy)",
            )?
        };
        self.sync()
    }

    /// Q8_0 matrix-vector, one thread per output row.
    ///
    /// `w` is the packed weight exactly as it appears in the file. The
    /// activation must already be quantized to Q8_0 — that is what ggml does,
    /// and matching it is what makes the result comparable to the CPU path.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q8_0(
        &self,
        n_in: usize,
        n_out: usize,
        w: &DeviceBuffer,
        x_scales: &DeviceBuffer,
        x_quants: &DeviceBuffer,
        out: &DeviceBuffer,
    ) -> Result<()> {
        let f = self.function("matmul_q8_0")?;
        let mut n_in_arg = n_in as i32;
        let mut n_out_arg = n_out as i32;
        let mut w_arg = w.ptr;
        let mut xs_arg = x_scales.ptr;
        let mut xq_arg = x_quants.ptr;
        let mut out_arg = out.ptr;
        let mut params = [
            &mut n_in_arg as *mut _ as *mut c_void,
            &mut n_out_arg as *mut _ as *mut c_void,
            &mut w_arg as *mut _ as *mut c_void,
            &mut xs_arg as *mut _ as *mut c_void,
            &mut xq_arg as *mut _ as *mut c_void,
            &mut out_arg as *mut _ as *mut c_void,
        ];
        let block = 128u32;
        let grid = n_out.div_ceil(block as usize) as u32;
        // SAFETY: parameters match the kernel signature; buffer sizes are the
        // caller's contract, documented above.
        unsafe {
            check(
                ffi::cuLaunchKernel(
                    f,
                    grid,
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel(matmul_q8_0)",
            )?
        };
        self.sync()
    }
}

impl GraphState {
    /// Free the driver handles.
    ///
    /// Called from `Cuda`'s `Drop` rather than being a `Drop` of its own:
    /// implementing `Drop` here would forbid moving fields out of the state,
    /// which is exactly what the Recording-to-Ready transition does.
    fn destroy(&mut self) {
        // SAFETY: each handle came from the matching create/instantiate call
        // and is freed once. The exec goes first, as the driver requires.
        unsafe {
            match self {
                GraphState::Recording { graph, .. } => {
                    ffi::cuGraphDestroy(*graph);
                }
                GraphState::Ready { graph, exec, .. } => {
                    ffi::cuGraphExecDestroy(*exec);
                    ffi::cuGraphDestroy(*graph);
                }
                GraphState::Off => {}
            }
        }
        *self = GraphState::Off;
    }
}

impl Drop for Cuda {
    fn drop(&mut self) {
        self.graph.borrow_mut().destroy();
        // SAFETY: both handles were produced by the driver and are dropped
        // exactly once, module before context, as the driver requires.
        unsafe {
            ffi::cuModuleUnload(self.module);
            ffi::cuCtxDestroy_v2(self.context);
        }
    }
}

/// An owned allocation in device memory.
pub struct DeviceBuffer {
    ptr: ffi::CUdeviceptr,
    bytes: usize,
}

/// Cumulative nanoseconds inside `cuMemAlloc`, process-wide.
///
/// **A static because `DeviceBuffer::new` has no backend handle**, and giving
/// it one would thread `&Cuda` through every allocation site to observe
/// something that happens a few hundred times in a process. Process-wide is
/// also the honest scope: a second `Cuda` in the same process shares the
/// driver's allocator, so the cost is not separable per backend anyway.
///
/// Why it exists: a 0.6B CUDA prefill of 64 tokens takes ~890 ms against a
/// marginal cost of 0.76 ms/token, so ~842 ms is fixed. Weight upload accounts
/// for 10 ms of that and PTX JIT for ~330 (measured by `CUDA_MODULE_LOADING`),
/// which left ~500 ms attributed to nothing. Allocation was the leading
/// candidate and a candidate is not a measurement.
static ALLOC_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Nanoseconds spent in `cuMemAlloc` since the process started.
pub fn alloc_ns() -> u64 {
    ALLOC_NS.load(std::sync::atomic::Ordering::Relaxed)
}

impl DeviceBuffer {
    pub fn new(bytes: usize) -> Result<Self> {
        // A zero-byte allocation is not an error to ask for, but the driver
        // dislikes it; hand back a null handle we will never dereference.
        if bytes == 0 {
            return Ok(Self { ptr: 0, bytes: 0 });
        }
        let mut ptr: ffi::CUdeviceptr = 0;
        let started = std::time::Instant::now();
        // SAFETY: valid out-pointer, non-zero size.
        unsafe { check(ffi::cuMemAlloc_v2(&mut ptr, bytes), "cuMemAlloc")? };
        ALLOC_NS.fetch_add(
            started.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(Self { ptr, bytes })
    }

    /// Allocate and fill from a host slice of plain data.
    /// A zero-filled buffer.
    ///
    /// Counters have to start at zero and `cuMemAlloc` does not promise it, so
    /// this is the difference between a read count and a read count plus
    /// whatever the driver last left there.
    pub fn zeroed(bytes: usize) -> Result<Self> {
        let b = Self::new(bytes)?;
        if bytes > 0 {
            // SAFETY: `b.ptr` owns exactly `bytes`, which is what is cleared.
            unsafe { check(ffi::cuMemsetD8_v2(b.ptr, 0, bytes), "cuMemsetD8")? };
        }
        Ok(b)
    }

    pub fn from_slice<T: Copy>(data: &[T]) -> Result<Self> {
        let bytes = std::mem::size_of_val(data);
        let buf = Self::new(bytes)?;
        buf.write(data)?;
        Ok(buf)
    }

    pub fn len_bytes(&self) -> usize {
        self.bytes
    }

    /// Copy a host slice in. Fails rather than truncating if it does not fit.
    pub fn write<T: Copy>(&self, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        if bytes > self.bytes {
            return Err(Error::Cuda {
                what: "cuMemcpyHtoD",
                detail: format!("{bytes} bytes into a {} byte buffer", self.bytes),
            });
        }
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: `data` is valid for `bytes` and the device range was checked
        // above. `T: Copy` means there is nothing to drop or relocate.
        unsafe {
            check(
                ffi::cuMemcpyHtoD_v2(self.ptr, data.as_ptr() as *const c_void, bytes),
                "cuMemcpyHtoD",
            )
        }
    }

    /// Copy a host slice in at a byte offset, for appending to a buffer whose
    /// earlier contents are still wanted.
    pub fn write_at<T: Copy>(&self, offset_bytes: usize, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        if offset_bytes + bytes > self.bytes {
            return Err(Error::Cuda {
                what: "cuMemcpyHtoD",
                detail: format!(
                    "{bytes} bytes at offset {offset_bytes} into a {} byte buffer",
                    self.bytes
                ),
            });
        }
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: `data` is valid for `bytes` and the device range was checked
        // above. `T: Copy` means there is nothing to drop or relocate.
        unsafe {
            check(
                ffi::cuMemcpyHtoD_v2(
                    self.ptr + offset_bytes as u64,
                    data.as_ptr() as *const c_void,
                    bytes,
                ),
                "cuMemcpyHtoD",
            )
        }
    }

    /// Copy out into a host slice, which must not ask for more than was
    /// allocated.
    pub fn read<T: Copy>(&self, out: &mut [T]) -> Result<()> {
        let bytes = std::mem::size_of_val(out);
        if bytes > self.bytes {
            return Err(Error::Cuda {
                what: "cuMemcpyDtoH",
                detail: format!("{bytes} bytes from a {} byte buffer", self.bytes),
            });
        }
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: as above, in the other direction.
        unsafe {
            check(
                ffi::cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, self.ptr, bytes),
                "cuMemcpyDtoH",
            )
        }
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            // SAFETY: allocated by `cuMemAlloc_v2`, freed exactly once.
            unsafe {
                ffi::cuMemFree_v2(self.ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::naive::Naive;
    use crate::ops::{Ops, Weights};
    use crate::gguf::GgmlType;
    use crate::quant::half::{f16_to_f32, f32_to_f16};

    /// Print what we are actually talking to. Not an assertion so much as a
    /// record: if the capability is not 12.0, the PTX we built is wrong for
    /// this device and every other failure here is downstream of that.
    #[test]
    fn reports_the_device() {
        let cuda = Cuda::new(0).expect("no CUDA device");
        let (free, total) = cuda.mem_info().expect("mem info");
        println!(
            "{} | sm_{}{} | {} SMs | {:.2} GiB free of {:.2}",
            cuda.name(),
            cuda.capability().0,
            cuda.capability().1,
            cuda.sm_count(),
            free as f64 / (1024.0 * 1024.0 * 1024.0),
            total as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        assert_eq!(cuda.capability(), (12, 0), "built for sm_120");
    }

    /// Toolchain proof: build, load, allocate, copy up, launch, copy down.
    #[test]
    fn saxpy_round_trips() {
        let cuda = Cuda::new(0).expect("no CUDA device");
        let n = 10_000usize;
        let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let y: Vec<f32> = (0..n).map(|i| i as f32 * -0.25).collect();

        let dx = DeviceBuffer::from_slice(&x).expect("upload x");
        let dy = DeviceBuffer::from_slice(&y).expect("upload y");
        cuda.saxpy(3.0, &dx, &dy, n).expect("launch");

        let mut got = vec![0.0f32; n];
        dy.read(&mut got).expect("download");
        for i in 0..n {
            let want = 3.0 * x[i] + y[i];
            assert_eq!(got[i].to_bits(), want.to_bits(), "element {i}");
        }
    }

    fn q8_0_row(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for block in values.chunks(32) {
            let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d == 0.0 { 0.0 } else { 1.0 / d };
            bytes.extend_from_slice(&f32_to_f16(d).to_le_bytes());
            for &v in block {
                bytes.push(((v * id).round() as i8) as u8);
            }
        }
        bytes
    }

    /// **The result that matters.** One thread per row, walking the row in
    /// index order, is the same arithmetic in the same order as
    /// `naive::dot_q8_0_q8_0` — so the GPU's answer is compared to the CPU
    /// oracle's raw bits, not to a tolerance.
    ///
    /// This is only possible because the kernel is deliberately unoptimized and
    /// because `build.rs` passes `--fmad=false`. A block-wide reduction, or
    /// letting nvcc contract multiply-add, would break it. When we do optimize,
    /// this test is what tells us exactly what we gave up.
    #[test]
    fn q8_0_matmul_is_bit_identical_to_the_cpu_oracle() {
        let cuda = Cuda::new(0).expect("no CUDA device");

        for (n_in, n_out) in [(128usize, 512usize), (1024, 4096), (256, 3)] {
            let val = |i: usize| ((i as f32) * 0.7391).sin() * 2.0 + ((i as f32) * 0.113).cos();
            let weights: Vec<f32> = (0..n_in * n_out).map(val).collect();
            let mut packed = Vec::new();
            for row in 0..n_out {
                packed.extend(q8_0_row(&weights[row * n_in..(row + 1) * n_in]));
            }
            let x: Vec<f32> = (0..n_in).map(|i| val(i + 9871)).collect();

            // CPU reference, through the ops seam.
            let w = Weights { data: &packed, ty: GgmlType::Q8_0, n_in, n_out,
     pooled: false, };
            let mut expect = vec![0.0f32; n_out];
            Naive.matmul(&w, &x, &mut expect);

            // Quantize the activation on the host, exactly as ggml does, so the
            // kernel receives the same integers the CPU path uses.
            let mut scales = Vec::new();
            let mut quants: Vec<i8> = Vec::new();
            for block in x.chunks(32) {
                let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                scales.push(f16_to_f32(f32_to_f16(d)));
                for &v in block {
                    quants.push((v * id).round() as i8);
                }
            }

            let dw = DeviceBuffer::from_slice(&packed).expect("upload w");
            let ds = DeviceBuffer::from_slice(&scales).expect("upload scales");
            let dq = DeviceBuffer::from_slice(&quants).expect("upload quants");
            let dout = DeviceBuffer::new(n_out * 4).expect("alloc out");
            cuda.matmul_q8_0(n_in, n_out, &dw, &ds, &dq, &dout).expect("launch");

            let mut got = vec![0.0f32; n_out];
            dout.read(&mut got).expect("download");

            let bad: Vec<usize> = (0..n_out)
                .filter(|&j| got[j].to_bits() != expect[j].to_bits())
                .collect();
            assert!(
                bad.is_empty(),
                "{}x{}: {} of {n_out} rows differ; first at {}: gpu {} vs cpu {}",
                n_in, n_out, bad.len(), bad[0], got[bad[0]], expect[bad[0]]
            );
        }
    }

    /// A buffer must refuse a copy that would not fit rather than truncating.
    #[test]
    fn oversized_copies_are_refused() {
        let _cuda = Cuda::new(0).expect("no CUDA device");
        let buf = DeviceBuffer::new(16).expect("alloc");
        assert!(buf.write(&[0u8; 32]).is_err());
        let mut out = [0u8; 32];
        assert!(buf.read(&mut out).is_err());
    }
}
