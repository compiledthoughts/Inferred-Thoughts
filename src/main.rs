use std::process::ExitCode;

use clap::{Parser, Subcommand};

use inferred_thoughts::gguf::{Array, GgufFile, TensorInfo, Value};

#[derive(Parser)]
#[command(name = "inferred", version, about = "A from-scratch GGUF inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Parse a GGUF file and print its metadata and tensor index.
    Inspect {
        /// Path to the .gguf file.
        path: String,

        /// Emit machine-readable JSON, for diffing against gguf_dump.py --json.
        #[arg(long)]
        json: bool,

        /// Truncate arrays longer than this in human-readable output.
        #[arg(long, default_value_t = 16)]
        max_array: usize,
    },

    /// Generate text greedily, prefilling the prompt and decoding against a
    /// KV cache.
    Generate {
        /// Path to the .gguf file. Only the `qwen3` architecture loads today.
        #[arg(short, long)]
        model: String,
        /// The prompt, as text. Special tokens in it are parsed, so chat
        /// markers can be written by hand; --chat does it for you.
        #[arg(short, long)]
        prompt: String,
        /// Maximum new tokens to produce.
        #[arg(short = 'n', long, default_value_t = 16)]
        max_tokens: usize,
        /// Keep going past the end-of-sequence token.
        #[arg(long)]
        ignore_eos: bool,
        /// KV cache size in positions. The cache is allocated up front, so
        /// this trades memory for the longest usable context.
        #[arg(short = 'c', long, default_value_t = 4096)]
        ctx: usize,
        /// Prompt tokens per forward pass during prefill.
        ///
        /// A **memory** bound, not a speed knob: a device backend mirrors every
        /// activation buffer and those are sized by the batch — ~405 KiB per
        /// token on the 9B — and are not freed between passes. Raising this
        /// costs VRAM that the model and KV cache would otherwise have.
        #[arg(long, default_value_t = inferred_thoughts::engine::DEFAULT_MAX_BATCH)]
        batch: usize,
        /// Replace every CUDA kernel with a no-op, keeping the launch pattern.
        ///
        /// **A stopwatch, not a mode.** The output is garbage. What it measures
        /// is what a token costs when the work costs nothing — the only direct
        /// way to size the gap between a 53.6 ms token and the 18.05 ms of
        /// kernel time the shape bench accounts for.
        #[arg(long)]
        null_kernels: bool,
        /// Ask the CUDA driver to sleep rather than spin while synchronizing.
        ///
        /// A **measurement** switch, not a speed one. The default context
        /// busy-waits, so a host blocked in a copy burns a core and is
        /// indistinguishable from a host doing work. With this on, the
        /// `host cpu` line below counts only real host work.
        #[arg(long)]
        cuda_blocking: bool,
        /// Cap the MoE expert cache, in GiB. 0 uses the automatic budget.
        ///
        /// The expert pool does not fit in VRAM, so it lives in a bounded slab
        /// with an eviction policy. Larger is not automatically better: the
        /// slab is allocated up front, so an over-large one leaves the driver
        /// short and it begins paging VRAM to host — which costs far more than
        /// the misses it avoids.
        /// Disable CUDA graphs.
        ///
        /// **Also the only way to get kernel attribution back.** `launch_grid2`
        /// returns through `graph_launch` before it reaches the launch
        /// recorder, so with graphs on `--profile-device` replays nothing —
        /// the instrument that is "complete by construction" goes quiet. This
        /// restores it, and gives a direct A/B on what the graph is worth
        /// rather than one inferred across several changes at once.
        #[arg(long, default_value_t = false)]
        no_graphs: bool,
        #[arg(long, default_value_t = 0.0)]
        expert_cache: f64,
        /// Cap the page-locked host tier behind the expert cache, in GiB.
        /// 0 uses the automatic budget.
        ///
        /// Experts that do not fit in VRAM live here and are read across PCIe
        /// by the kernel itself, so the pool stays addressable without the host
        /// in the loop — which is what lets the decode step be a CUDA graph.
        /// Pinned pages cannot be swapped, so this is a real claim on host RAM.
        #[arg(long, default_value_t = 0.0)]
        expert_host: f64,
        /// Print the profile summary: phase timings, byte traffic, and the
        /// top-2 logit margins.
        #[arg(long)]
        profile: bool,
        /// Also time each layer's attention and FFN halves.
        #[arg(long)]
        profile_detail: bool,
        /// Write the profile as JSON, for diffing two runs against each other.
        #[arg(long)]
        profile_json: Option<String>,
        /// With --backend cuda: also time a bare launch, upload and download,
        /// and report what a per-op seam costs before any arithmetic.
        #[arg(long)]
        profile_device: bool,
        /// With --backend cuda: attribute time to individual kernels. Needs a
        /// device sync per launch, so it inflates the total -- read the shares.
        #[arg(long)]
        profile_kernels: bool,
        /// With --backend cuda: reduce RMSNorm's sum of squares serially rather
        /// than as a tree. ~2.3 ms a token slower, and the only way to get
        /// results bit-identical to the `naive` oracle -- f64 addition rounds,
        /// so a tree is a different answer, not just a faster one.
        #[arg(long)]
        rms_serial: bool,
        /// Force the untiled IQ4_XS matmul, for the prefill A/B.
        #[arg(long)]
        iq4_untiled: bool,
        /// Force the one-thread-per-position attention score phase.
        ///
        /// The warp phase is on past `ATTN_WARP_MIN_POS` and is 1.4-3.1x
        /// faster there; this restores the old path so the difference can be
        /// measured end to end rather than only in the kernel bench.
        #[arg(long)]
        attn_thread: bool,
        /// Compute threads. 1 selects the scalar `naive` oracle directly and
        /// ignores --backend; anything more selects --backend, which must
        /// produce identical bits. 0 means physical cores, taken as half the
        /// logical count when the platform reports an even number.
        #[arg(short = 't', long, default_value_t = 0)]
        threads: usize,
        /// Wrap the prompt as a chat turn, so an instruct model answers instead
        /// of continuing the text. Requires a ChatML model; fails loudly if the
        /// file's own template is a shape we do not implement.
        #[arg(long)]
        chat: bool,
        /// Render special tokens such as <think> instead of dropping them.
        #[arg(long)]
        show_special: bool,
        /// Backend: `spin` (persistent spinning pool), `par` (rayon), or
        /// `cuda` if the feature is built. `spin` and `par` must produce
        /// identical bits to `naive`; `spin` exists because a rayon parallel
        /// region costs ~430 us on this machine against 0.4 us for a spin
        /// barrier. `cuda` ignores -t and runs on the GPU.
        #[arg(long, default_value = "spin")]
        backend: String,
    },

    /// Run one forward pass and print a checksum of every intermediate tensor,
    /// for diffing against `llama-eval-callback`.
    Trace {
        /// Path to the .gguf file.
        #[arg(short, long)]
        model: String,
        /// The prompt, as text. One forward pass is run over all of it.
        #[arg(short, long)]
        prompt: String,
        /// Also write every intermediate tensor's full contents here, so a
        /// comparison can check individual elements and not just checksums.
        #[arg(long)]
        dump: Option<String>,
    },

    /// Serve an OpenAI-compatible endpoint, so an existing chat UI can drive
    /// the engine.
    ///
    /// One model, loaded at startup. A chat client re-sends the whole
    /// conversation each turn, and the session only prefills what is new --
    /// which works while the conversation grows by appending, and restarts from
    /// scratch when it does not. See `src/serve` for why a rewind cannot be
    /// cheaper than that on a recurrent architecture.
    Serve {
        /// Path to the .gguf file.
        #[arg(short, long)]
        model: String,
        /// Port to bind on 127.0.0.1.
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Context length. The KV cache is allocated for this up front.
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
        /// Prompt tokens per forward pass during prefill.
        ///
        /// A **memory** bound, not a speed knob: a device backend mirrors every
        /// activation buffer and those are sized by the batch — ~405 KiB per
        /// token on the 9B — and are not freed between passes. Raising this
        /// costs VRAM that the model and KV cache would otherwise have.
        #[arg(long, default_value_t = inferred_thoughts::engine::DEFAULT_MAX_BATCH)]
        batch: usize,
        /// Replace every CUDA kernel with a no-op, keeping the launch pattern.
        ///
        /// **A stopwatch, not a mode.** The output is garbage. What it measures
        /// is what a token costs when the work costs nothing — the only direct
        /// way to size the gap between a 53.6 ms token and the 18.05 ms of
        /// kernel time the shape bench accounts for.
        #[arg(long)]
        null_kernels: bool,
        /// Ask the CUDA driver to sleep rather than spin while synchronizing.
        ///
        /// A **measurement** switch, not a speed one. The default context
        /// busy-waits, so a host blocked in a copy burns a core and is
        /// indistinguishable from a host doing work. With this on, the
        /// `host cpu` line below counts only real host work.
        #[arg(long)]
        cuda_blocking: bool,
        /// Cap the MoE expert cache, in GiB. 0 uses the automatic budget.
        ///
        /// The expert pool does not fit in VRAM, so it lives in a bounded slab
        /// with an eviction policy. Larger is not automatically better: the
        /// slab is allocated up front, so an over-large one leaves the driver
        /// short and it begins paging VRAM to host — which costs far more than
        /// the misses it avoids.
        #[arg(long, default_value_t = 0.0)]
        expert_cache: f64,
        /// Cap the page-locked host tier behind the expert cache, in GiB.
        /// 0 uses the automatic budget.
        ///
        /// Experts that do not fit in VRAM live here and are read across PCIe
        /// by the kernel itself, so the pool stays addressable without the host
        /// in the loop — which is what lets the decode step be a CUDA graph.
        /// Pinned pages cannot be swapped, so this is a real claim on host RAM.
        #[arg(long, default_value_t = 0.0)]
        expert_host: f64,
        /// Default generation budget when the request does not set one.
        #[arg(short = 'n', long, default_value_t = 512)]
        max_tokens: usize,
        /// Count kernel launches, bus crossings and expert residency, and
        /// report them after every turn.
        ///
        /// **Counts, not timings.** `generate --profile-device` also *replays*
        /// every recorded launch to time it, which re-executes their writes and
        /// corrupts activations and the KV cache on purpose — safe once
        /// generation is over, ruinous in a server that has to answer the next
        /// turn. So this reports what a turn issued and where the experts were;
        /// for per-kernel timing use `generate` with a long prompt, which is
        /// the same prefill.
        #[arg(long, default_value_t = false)]
        profile_device: bool,
        /// Compute threads for the CPU backends.
        #[arg(short = 't', long, default_value_t = 0)]
        threads: usize,
        /// Which `ops` implementation runs the model.
        #[arg(long, default_value = "spin")]
        backend: String,
        /// With --backend cuda: reduce RMSNorm serially, for bit-equality with
        /// the oracle at ~2.3 ms a token.
        #[arg(long)]
        rms_serial: bool,
        /// Print each request's body, its turns, and the prompt they render to.
        #[arg(short, long)]
        verbose: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Inspect {
            path,
            json,
            max_array,
        } => inspect(&path, json, max_array),
        Command::Generate {
            model,
            prompt,
            max_tokens,
            ignore_eos,
            ctx,
            batch,
            cuda_blocking,
            null_kernels,
            no_graphs,
            expert_cache,
            expert_host,
            profile,
            profile_detail,
            profile_json,
            profile_device,
            profile_kernels,
            rms_serial,
            attn_thread,
            iq4_untiled,
            threads,
            chat,
            show_special,
            backend,
        } => generate(
            &model,
            &prompt,
            GenOpts {
                max_tokens,
                ignore_eos,
                n_ctx: ctx,
                max_batch: batch,
                expert_cache,
                expert_host,
                cuda_blocking,
                null_kernels,
                no_graphs,
                report: profile || profile_detail,
                detail: profile_detail,
                json: profile_json,
                device: profile_device,
                kernels: profile_kernels,
                rms_serial,
                attn_thread,
                iq4_untiled,
                threads,
                chat,
                show_special,
                backend,
            },
        ),
        Command::Trace { model, prompt, dump } => trace(&model, &prompt, dump.as_deref()),
        Command::Serve {
            model,
            port,
            ctx,
            batch,
            cuda_blocking: _,
            null_kernels: _,
            expert_cache,
            expert_host,
            max_tokens,
            profile_device,
            threads,
            backend,
            rms_serial,
            verbose,
        } => serve(ServeArgs {
            model,
            port,
            ctx,
            max_batch: batch,
            expert_cache,
            expert_host,
            max_tokens,
            profile_device,
            threads,
            backend,
            rms_serial,
            verbose,
        }),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            // thiserror's #[source] chain, so an io error shows the OS message.
            let mut src = std::error::Error::source(&e);
            while let Some(s) = src {
                eprintln!("  caused by: {s}");
                src = s.source();
            }
            ExitCode::FAILURE
        }
    }
}

fn inspect(path: &str, json: bool, max_array: usize) -> inferred_thoughts::Result<()> {
    let f = GgufFile::open(path)?;
    if json {
        print_json(&f);
    } else {
        print_human(&f, max_array);
    }
    Ok(())
}

// -------------------------------------------------------------------- generate

/// Everything `generate` takes beyond the model and prompt. A struct rather
/// than nine positional arguments, which is how the wrong flag ends up in the
/// wrong slot.
struct GenOpts {
    max_tokens: usize,
    ignore_eos: bool,
    n_ctx: usize,
    /// Prompt tokens per forward pass. See the CLI doc on `--batch`: this
    /// bounds activation VRAM, which a batch sizes.
    max_batch: usize,
    /// Cap the MoE expert cache, in GiB. 0 uses the automatic budget.
    expert_cache: f64,
    /// Cap the page-locked host tier behind it, in GiB. 0 is automatic.
    expert_host: f64,
    /// Ask the driver to block rather than spin on sync. Measurement only.
    cuda_blocking: bool,
    /// Disable CUDA graphs, which also restores launch-replay attribution.
    no_graphs: bool,
    /// Replace every kernel with a no-op. Measurement only; output is garbage.
    null_kernels: bool,
    report: bool,
    detail: bool,
    json: Option<String>,
    device: bool,
    kernels: bool,
    rms_serial: bool,
    attn_thread: bool,
    iq4_untiled: bool,
    threads: usize,
    chat: bool,
    show_special: bool,
    backend: String,
}

fn generate(model: &str, prompt: &str, o: GenOpts) -> inferred_thoughts::Result<()> {
    use inferred_thoughts::tok::chat::ChatMl;
    use inferred_thoughts::{Model, Naive, Par, Spin, Tokenizer};

    let f = GgufFile::open(model)?;
    let tk = Tokenizer::from_metadata(&f.metadata)?;
    let m = Model::load(&f)?;

    // Wrapping happens before tokenization so the markers go through
    // `parse_special` and encode as single tokens, not as literal text.
    let text = if o.chat {
        ChatMl::detect(&tk, &f.metadata)?.wrap(prompt)
    } else {
        prompt.to_string()
    };
    let tokens = tk.encode(&text, true, true);

    let n_threads = if o.threads == 0 {
        Par::default_threads()
    } else {
        o.threads
    };

    let label = if o.backend == "cuda" {
        "cuda".to_string()
    } else if n_threads <= 1 {
        "naive (1 thread)".to_string()
    } else {
        format!("{} ({n_threads} threads)", o.backend)
    };
    eprintln!(
        "model {} | {} | {} layers ({} with kv) | {} prompt tokens{} | ctx {} | {label}",
        f.path.file_name().unwrap_or_default().to_string_lossy(),
        m.arch(),
        m.n_layer(),
        m.n_kv_layer(),
        tokens.len(),
        if o.chat { " (chat)" } else { "" },
        o.n_ctx,
    );

    // CUDA is checked before the thread count, which it ignores: the device
    // decides its own parallelism and `-t` describes CPU workers.
    #[cfg(feature = "cuda")]
    if o.backend == "cuda" {
        let cuda = inferred_thoughts::Cuda::with_options(0, o.cuda_blocking)?;
        cuda.time_kernels(o.kernels);
        cuda.rms_serial(o.rms_serial);
        // `Some(false)` pins the thread path; `None` leaves the depth threshold
        // in charge, which is the shipping behaviour.
        cuda.attn_warp(if o.attn_thread { Some(false) } else { None });
        cuda.iq4_untiled(o.iq4_untiled);
        cuda.set_expert_budget((o.expert_cache * 1073741824.0) as usize);
        cuda.set_expert_host_budget((o.expert_host * 1073741824.0) as usize);
        // The KV cache is allocated lazily, at the first attention layer, which
        // is *after* the expert slab has sized itself from free VRAM. Told here
        // because this is the only place that knows the context length.
        cuda.reserve_for_kv(kv_reserve_bytes(&m, o.n_ctx));
        cuda.null_kernels(o.null_kernels);
        cuda.use_graphs(!o.no_graphs);
        cuda.record_launches(o.device);
        cuda.set_model_path(&f.path);
        cuda.set_map_base(f.map_base());
        let (free, total) = cuda.mem_info()?;
        let (major, minor) = cuda.capability();
        eprintln!(
            "device {} | sm_{major}{minor} | {} SMs | {:.2} of {:.2} GiB free",
            cuda.name(),
            cuda.sm_count(),
            free as f64 / 1073741824.0,
            total as f64 / 1073741824.0,
        );
        // Borrowed, not moved, so the sticky error survives the engine. An op
        // that failed has produced meaningless output, so this is fatal.
        let cpu0 = inferred_thoughts::profile::cpu_time_ns();
        let run = run_generation(m, &cuda, &tk, &tokens, &text, &o);
        let host_cpu_ns = match (cpu0, inferred_thoughts::profile::cpu_time_ns()) {
            (Some(a), Some(b)) => Some(b.saturating_sub(a)),
            _ => None,
        };
        if let Some(e) = cuda.take_error() {
            return Err(e);
        }
        if o.report || o.device || o.kernels {
            let r = run.as_ref().ok().copied();
            let n = r.map(|r| r.tokens).unwrap_or((tokens.len() + o.max_tokens) as u64);
            if let (Some(ns), Some(run)) = (host_cpu_ns, r) {
                let per = ns as f64 / n.max(1) as f64 / 1e6;
                eprintln!(
                    "
host     {per:.2} ms/token on-CPU, {:.0}% of the {:.2} ms wall{}",
                    100.0 * per / run.ms_per_token.max(1e-9),
                    run.ms_per_token,
                    if o.cuda_blocking { "" } else { " (driver spins; see --cuda-blocking)" },
                );
            }
            report_device(&cuda, &o, n, tokens.len() as u64, r.map(|r| r.ms_per_pass_token))?;
        }
        return run.map(|_| ());
    }

    // One thread means the oracle itself, not a pool of one -- there is no
    // reason to pay a barrier for a single worker, and it makes `-t 1` the
    // reference run the threaded backends must reproduce bit for bit.
    if n_threads <= 1 {
        return run_generation(m, Naive, &tk, &tokens, &text, &o).map(|_| ());
    }
    match o.backend.as_str() {
        "spin" => run_generation(m, Spin::new(n_threads), &tk, &tokens, &text, &o).map(|_| ()),
        "par" => {
            Par::init(n_threads);
            run_generation(m, Par, &tk, &tokens, &text, &o).map(|_| ())
        }
        other => Err(inferred_thoughts::Error::InconsistentArchitecture {
            what: "--backend",
            detail: format!(
                "{other:?} is not a backend; expected \"spin\", \"par\"{}",
                if cfg!(feature = "cuda") {
                    " or \"cuda\""
                } else {
                    " (\"cuda\" needs --features cuda)"
                }
            ),
        }),
    }
}

/// What the backend asked the driver to do, and what that costs here.
///
/// Separate from the profiler proper because it is backend-specific: `spin` and
/// `naive` have no bus to cross. It answers the question the wall-clock profile
/// cannot — how much of a token is spent moving 4 KiB back and forth rather
/// than computing.
#[cfg(feature = "cuda")]
fn report_device(
    cuda: &inferred_thoughts::Cuda,
    o: &GenOpts,
    tokens: u64,
    prefill_tokens: u64,
    ms_per_pass_token: Option<f64>,
) -> inferred_thoughts::Result<()> {
    let s = cuda.stats();
    // **One-time setup comes out of the device counters, not just the phase
    // timer.** `gpu_ns` brackets every pass with CUDA events, and the first
    // pass carries expert placement -- 23 s of copies that are genuinely on the
    // stream. Left in, it reported `gpu 161.25 ms/token` against a 37.08 ms
    // decode token and concluded "the GPU is busy 435% of it".
    //
    // Subtracting it leaves per-token device time. Approximate in one direction
    // and stated rather than hidden: `cuMemAlloc` is not stream work, so a
    // couple of hundred milliseconds of the subtrahend was never counted in
    // `gpu_ns` to begin with.
    let (alloc, upload, place) = cuda.setup_parts();
    let setup_ns = alloc + upload + place;
    // Plain, for counters setup does not touch: only `d2h` bumps `wait_ns`, and
    // placement copies go the other way.
    let per = |ns: u64| ns as f64 / tokens.max(1) as f64 / 1e6;
    let per_dev = |ns: u64| ns.saturating_sub(setup_ns) as f64 / tokens.max(1) as f64 / 1e6;

    // **Taken before anything replays a launch.** `bench_launches` re-runs every
    // recorded launch 200 times, and `moe_gather_ptrs` carries the atomics that
    // count expert reads -- so reading the counters afterwards reports the
    // benchmark's traffic as the model's. Observed at 4,074,240 reads over 204
    // tokens against a true 195,840: a token makes exactly 960 expert reads, so
    // the figure was 20.8x high and `% from VRAM` and `MiB/token` with it.
    //
    // Fourteenth instrument here to report a wrong basis, and the first where
    // one instrument corrupted another.
    let experts = cuda.expert_stats();
    // Same reason: `coverage` scores the per-expert read counts, which the
    // replay inflates along with the totals.
    let coverage = cuda.expert_coverage();

    if o.kernels {
        let times = cuda.kernel_times();
        let total: f64 = times.iter().map(|(_, _, ms)| ms).sum();
        eprintln!("\nper-kernel (synchronized, so totals are inflated; read the share)");
        eprintln!("  {:<22} {:>8} {:>10} {:>7}", "kernel", "calls", "ms", "share");
        for (name, calls, ms) in &times {
            eprintln!(
                "  {name:<22} {calls:>8} {ms:>10.1} {:>6.1}%",
                100.0 * ms / total.max(1e-9)
            );
        }
        eprintln!("  {:<22} {:>8} {total:>10.1}", "total", "");
    }
    // Where a token's time actually goes. `gpu` is device-stream time from
    // CUDA events, `issue` is host time spent putting work there, and `wait` is
    // host time blocked on a copy. They are measured independently and do not
    // have to sum to the wall clock -- issuing overlaps execution, which is the
    // point of an asynchronous API. What matters is the ratio.
    if s.passes > 0 {
        eprintln!("
time     gpu    {:>6.2} ms/token   device stream, net of one-time setup", per_dev(s.gpu_ns));
        eprintln!("         issue  {:>6.2} ms/token   host, launching and bookkeeping", per_dev(s.issue_ns));
        eprintln!("         wait   {:>6.2} ms/token   host, blocked on a copy", per(s.wait_ns));
        // `gpu`/`issue`/`wait` are divided by prefill + decode, because that is
        // what the device counters accumulate over. `ms_per_token` is one
        // decode token. Those are the same basis only when decode dominates, so
        // the comparison is guarded exactly as the launch replay's is — the
        // alternative is a percentage built from two different denominators,
        // which is what printed "the GPU is busy 97%" of a token that was
        // mostly one-time placement.
        // Same basis on both sides now: pass tokens, net of setup. That is what
        // makes the ratio mean something, and the two earlier versions of this
        // line -- 435% and then 118% -- are what a mismatched one looks like.
        if let Some(total) = ms_per_pass_token {
            eprintln!(
                "         total  {total:>6.2} ms/token   one pass token net of setup, so the GPU is busy {:.0}% of it",
                100.0 * per_dev(s.gpu_ns) / total,
            );
        }
    }

    eprintln!("
device   {} kernel launches", s.launches);
    eprintln!(
        "         {:.1} launches / {:.1} crossings / {:.1} syncs per token",
        s.launches as f64 / tokens.max(1) as f64,
        s.crossings_per_token(tokens),
        s.syncs as f64 / tokens.max(1) as f64,
    );
    eprintln!("         {} up / {} down", s.h2d_calls, s.d2h_calls);
    eprintln!(
        "         {:.1} MiB up, {:.1} MiB down",
        s.h2d_bytes as f64 / 1048576.0,
        s.d2h_bytes as f64 / 1048576.0,
    );
    if cuda.graphs_off_for_mid_pass_read() {
        eprintln!(
            "         CUDA graphs OFF: the model reads a device result mid-pass, so a
         graph would hand it the previous pass's contents. MoE top-k is a host
         decision; this goes away when expert selection moves onto the device."
        );
    }

    // Replay every launch the run actually made. Complete by construction, and
    // therefore the one attribution that cannot silently omit a kernel.
    if o.device {
        // **Printed before the replay, not after it.** `bench_launches` on the
        // 35B returned nothing at all — no table, no "recorded nothing", no
        // error — which a `match` on its result cannot distinguish from never
        // having been called. Announcing the work first means the next failure
        // says how much there was to do.
        let (distinct, calls) = cuda.recorded_launches();
        eprintln!(
            "
launches {distinct} distinct, {calls} calls recorded; replaying each 200x"
        );
        if distinct == 0 {
            eprintln!(
                "         nothing to replay. `--profile-device` arms the recorder, but
         `launch_grid2` returns through `graph_launch` before reaching it, so a
         graphed run records none. Add `--no-graphs`."
            );
        }
        match cuda.bench_launches(200) {
            Ok(b) if !b.is_empty() => {
                eprintln!(
                    "
launches replayed from the run itself (after generation, so writes are moot)"
                );
                eprintln!(
                    "         per token over {} prefill + {tokens} decode",
                    prefill_tokens,
                );
                eprintln!(
                    "         {:<28}{:>10}{:>9}{:>9}{:>10}{:>9}",
                    "kernel", "grid", "calls/t", "gpu us", "ms/token", "share",
                );
                // **Every token the launches were issued for, not just the
                // decoded ones.** The recorder runs from the first pass, so a
                // run with a long prompt has most of its launches in prefill;
                // dividing those by the decode count reported a 19,890 ms
                // "token" and a kernel share of 0%. The shares were still
                // right, which is exactly what makes a wrong denominator hard
                // to notice.
                let norm = tokens + prefill_tokens;
                let total: f64 = b.iter().map(|r| r.gpu_ms_per_token(norm)).sum();
                for r in b.iter().take(28) {
                    let ms = r.gpu_ms_per_token(norm);
                    if ms < 0.005 {
                        continue;
                    }
                    eprintln!(
                        "         {:<28}{:>10}{:>9.1}{:>9.1}{:>10.2}{:>8.1}% {}",
                        r.kernel,
                        format!("{}x{}", r.grid.0, r.grid.1),
                        r.calls_per_token(norm),
                        r.gpu_us,
                        ms,
                        100.0 * ms / total.max(1e-9),
                        if r.host_limited() { "?" } else { "" },
                    );
                }
                eprintln!("         {:<28}{:>38.2} ms/token, {} distinct launches",
                    "TOTAL", total, b.len());
                // Only comparable when the run is mostly decode: `ms` is the
                // decode token's wall clock while `total` is now normalised
                // over prefill too, so on a long prompt these measure different
                // things and subtracting them is meaningless.
                match ms_per_pass_token {
                    Some(ms) if prefill_tokens <= tokens => eprintln!(
                        "         against a {ms:.1} ms token: kernels {:.0}%, everything else {:.1} ms.
         Compare `--null-kernels`, which measures that remainder directly.",
                        100.0 * total / ms, ms - total,
                    ),
                    Some(_) => eprintln!(
                        "         no decode comparison: {prefill_tokens} prefill against {tokens} decode tokens,
         so a decode ms/token and this prefill-weighted total are not the same quantity."
                    ),
                    None => {}
                }
            }
            // **Say so.** An empty replay printed nothing at all, which reads
            // as "this run had no launches" rather than "the recorder was
            // never armed" — and cost a three-minute run to notice. Ninth
            // instrument in this repo to default to silence.
            Ok(_) => eprintln!(
                "         launch replay recorded nothing. `--profile-device` arms the recorder,
         but `launch_grid2` returns through `graph_launch` before reaching it, so a
         graphed run records none. Add `--no-graphs`."
            ),
            Err(e) => eprintln!("         launch replay unavailable: {e}"),
        }
    }

    if o.device {
        match cuda.bench_f32_variants(200) {
            Ok(v) if !v.is_empty() => {
                let base = v[0].1.max(1e-9);
                eprintln!(
                    "\nf32      decomposed, same recorded launch, one thing removed each time"
                );
                for (name, us) in &v {
                    eprintln!(
                        "         {name:<26} {us:7.1} us  {:5.0}% of baseline",
                        100.0 * us / base
                    );
                }
            }
            Ok(_) => {}
            Err(e) => eprintln!("         f32 decomposition unavailable: {e}"),
        }

        match cuda.bench_expert_residency(100) {
            Ok(v) if !v.is_empty() => {
                eprintln!(
                    "
experts  one expert stack read from each tier, same kernel and shape
         (35B gate/up: n_in 2048, n_out 512, 8 experts, 8.50 MiB a launch)"
                );
                for (label, us, gbs) in &v {
                    eprintln!("         {label:<22}{us:>9.1} us{gbs:>9.1} GB/s");
                }
            }
            Ok(_) => {}
            Err(e) => eprintln!("         expert residency bench unavailable: {e}"),
        }

        if let Ok(v) = cuda.bench_iq4_variants(200) {
            if v.len() >= 4 {
                let base = v[0].1;
                eprintln!("
iq4_xs   decomposed, same recorded launch, one thing removed each time");
                for (name, us) in &v {
                    eprintln!(
                        "         {:<22}{:>9.1} us{:>9.0}% of baseline",
                        name, us, 100.0 * us / base.max(1e-9),
                    );
                }
            }
        }
    }

    // The self-configuring microbenchmark. Behind `--profile-device` because it
    // runs real work on the device after the model has finished, which is not
    // something a plain `--profile` should do.
    if o.device {
        let group = 8; // experts per token, so the grouped column prices the routed FFN
        match cuda.bench_shapes(group, tokens, 300) {
            Ok(b) if !b.is_empty() => {
                eprintln!(
                    "
shapes   measured on the live device, queue full, no per-launch sync"
                );
                eprintln!(
                    "         {:<20} {:>12} {:>7} {:>8} {:>9} {:>9} {:>9}",
                    "kernel", "shape", "calls/t", "gpu us", "issue us", "gpu ms/t", "grp ms/t",
                );
                let (mut gpu, mut issue, mut grouped) = (0.0, 0.0, 0.0);
                for r in &b {
                    if r.gpu_us.is_nan() {
                        continue;
                    }
                    gpu += r.gpu_ms_per_token(tokens);
                    issue += r.issue_ms_per_token(tokens);
                    grouped += r.grouped_ms_per_token(tokens);
                    eprintln!(
                        "         {:<20} {:>5}x{:<6} {:>7.1} {:>8.1} {:>9.1} {:>9.2} {:>9.2} {}",
                        r.kernel,
                        r.n_in,
                        r.n_out,
                        r.calls as f64 / tokens.max(1) as f64,
                        r.gpu_us,
                        r.issue_us,
                        r.gpu_ms_per_token(tokens),
                        r.grouped_ms_per_token(tokens),
                        if r.host_limited() { "?" } else { "" },
                    );
                }
                eprintln!(
                    "         {:<20} {:>12} {:>7} {:>8} {:>9} {:>9.2} {:>9.2}",
                    "total", "", "", "", "", gpu, grouped,
                );
                eprintln!(
                    "         matmuls alone: {gpu:.1} ms/token on the device, {issue:.1} ms to issue.
         Whichever is larger is what binds; `grp` is the same arithmetic
         in one launch per {group} instead of {group}.
         `?` marks a row the host could not keep fed, so its gpu figure is an
         upper bound rather than a measurement."
                );
            }
            Ok(_) => {}
            Err(e) => eprintln!("         shape bench unavailable: {e}"),
        }
    }

    // The one-time costs, decomposed. The bare run prints their total on the
    // `setup` line; this says where it went, which is the difference between
    // knowing the first pass was slow and knowing why.
    {
        let (alloc, upload, place) = cuda.setup_parts();
        let ms = |ns: u64| ns as f64 / 1e6;
        eprintln!(
            "
setup    {:.1} ms cuMemAlloc + {:.1} ms weight copies + {:.1} ms expert placement = {:.1} ms",
            ms(alloc),
            ms(upload),
            ms(place),
            ms(alloc + upload + place),
        );
        eprintln!(
            "         PTX JIT is not in this and cannot be: the driver compiles on first
         launch and reports nothing. Measure it with CUDA_MODULE_LOADING=EAGER
         against LAZY — ~330 ms of a 0.6B's ~842 ms fixed first pass."
        );
    }

    if let Some(e) = experts {
        let gib = |b: u64| b as f64 / 1073741824.0;
        eprintln!(
            "
experts  {} slots x {:.2} MiB = {:.2} GiB of bounded cache",
            e.slots,
            e.slot_bytes as f64 / 1048576.0,
            gib(e.capacity_bytes()),
        );
        // **Not a hit rate.** Every expert is placed before it can be routed
        // to, so nothing misses; what varies is which tier a read resolves to.
        // Printing 100% hit would be true and useless.
        eprintln!(
            "         {} reads, {:.1}% from VRAM and {:.1}% across PCIe   {} evictions",
            e.lookups(),
            100.0 * (1.0 - e.host_read_rate()),
            100.0 * e.host_read_rate(),
            e.evictions,
        );
        eprintln!(
            "         {:.2} GiB placed at load ({} tensors): {:.1}s h2d + {:.1}s pin + {:.1}s memcpy",
            gib(e.filled_bytes),
            e.distinct,
            e.place_h2d_us as f64 / 1e6,
            e.place_pin_us as f64 / 1e6,
            e.place_copy_us as f64 / 1e6,
        );
        eprintln!(
            "         {:.2} GiB of mmap released after placement",
            gib(e.released_bytes),
        );
        // The host tier is the point of the two-tier design, so it is reported
        // whether or not it was used: "0 tensors" is a result, not an absence.
        eprintln!(
            "         host tier {} tensors in {:.2} GiB pinned   {:.1}% of reads, {:.1} MiB/token over PCIe in-kernel",
            e.host_slots,
            gib(e.host_bytes),
            100.0 * e.host_read_rate(),
            e.host_reads as f64 * e.slot_bytes as f64 / 1048576.0 / tokens.max(1) as f64,
        );
        if e.migrated > 0 {
            eprintln!(
                "         {} experts migrated between tiers since load",
                e.migrated,
            );
        }
        if e.degraded {
            eprintln!(
                "         DEGRADED: the host tier filled, so placement fell back to
         eviction. The pool is no longer wholly addressable and a CUDA
         graph would be unsound. Raise --expert-host or --expert-cache."
            );
        }
        // What a placement policy that knew the routing distribution in advance
        // could have served from VRAM, against what first-touch arrival order
        // actually served. See `ExpertCache::coverage`.
        if let Some((frac, reads)) = coverage {
            eprintln!(
                "         coverage {:.1}% of {} reads would come from VRAM under an oracle placement,
         against {:.1}% under this one",
                100.0 * frac,
                reads,
                100.0 * (1.0 - e.host_read_rate()),
            );
        }
    }

    // Resident bytes, always printed. What the device *holds* is a different
    // question from what crosses the bus, and only the first one explains why a
    // long session ends up near the card's limit.
    let r = cuda.resident_bytes();
    let mib = |b: u64| b as f64 / 1048576.0;
    eprintln!(
        "
resident {:.0} MiB total on the device",
        mib(r.total())
    );
    eprintln!(
        "         {:.0} MiB weights ({} tensors) | {:.0} MiB kv ({} slabs) | {:.0} MiB pool",
        mib(r.weight_bytes),
        r.weight_tensors,
        mib(r.kv_bytes),
        r.kv_slabs,
        mib(r.pool_bytes),
    );
    // Mirrors are the term that can climb: they are keyed on host address and
    // never freed, so a buffer allocated at a fresh address adds to this rather
    // than reusing. Printed with the count so a climb is attributable.
    eprintln!(
        "         {:.0} MiB activations in {} mirrors (+{:.0} MiB quantized)",
        mib(r.mirror_bytes),
        r.mirrors,
        mib(r.quant_bytes),
    );

    if !o.device {
        eprintln!("         --profile-device times what one crossing costs");
        return Ok(());
    }

    let b = cuda.benchmark(2000)?;
    eprintln!(
        "
         launch {:.1} us | 4 KiB up {:.1} us | down {:.1} us",
        b.launch_us, b.h2d_us, b.d2h_us,
    );
    let seam = b.predicted_ms(&s, tokens);
    eprintln!("         seam costs {seam:.1} ms/token before any arithmetic");
    // `predicted_ms` scales by the per-pass crossing counts, so the measured
    // figure it is compared against has to be per pass token too.
    if let Some(actual) = ms_per_pass_token {
        eprintln!(
            "         {:.0}% of the {actual:.1} ms/token measured is moving 4 KiB about",
            100.0 * seam / actual,
        );
    }
    Ok(())
}

fn run_generation<O: inferred_thoughts::Ops>(
    model: inferred_thoughts::Model<'_>,
    ops: O,
    tk: &inferred_thoughts::Tokenizer,
    tokens: &[u32],
    prompt_text: &str,
    o: &GenOpts,
) -> inferred_thoughts::Result<Run> {
    use inferred_thoughts::Engine;
    use std::io::Write;

    let mut engine = Engine::new(model, ops, o.n_ctx, o.detail);
    engine.set_max_batch(o.max_batch);
    let rs_bytes = engine.recurrent_capacity_bytes();
    eprintln!(
        "kv cache {:.0} MiB resident{}",
        engine.kv_capacity_bytes() as f64 / 1048576.0,
        if rs_bytes > 0 {
            // Reported separately because it does not grow with context, which
            // is the property the hybrid architecture exists for.
            format!(", recurrent state {:.0} MiB", rs_bytes as f64 / 1048576.0)
        } else {
            String::new()
        }
    );

    print!("{prompt_text}");
    let _ = std::io::stdout().flush();

    // Decode incrementally so multi-token characters still render: decode the
    // whole sequence each time and print only what is new.
    let mut shown = tokens.to_vec();
    let mut before = tk.decode(&shown, o.show_special)?;
    let eos = if o.ignore_eos { None } else { tk.eos_token_id };

    let started = std::time::Instant::now();
    let (produced, why) = engine.generate(tokens, o.max_tokens, eos, |id| {
        shown.push(id);
        if let Ok(after) = tk.decode(&shown, o.show_special) {
            if let Some(new_text) = after.strip_prefix(&before) {
                print!("{new_text}");
                let _ = std::io::stdout().flush();
            }
            before = after;
        }
    })?;
    let secs = started.elapsed().as_secs_f64();
    println!();

    // Reported, not inferred: the engine knows which of the three it was.
    eprintln!("[stopped: {}]", why.label());

    // The phase breakdown is free -- the profiler collects it on every run --
    // so it is printed on every run rather than behind `--profile`. A single
    // blended rate over a short prompt and a long generation describes neither
    // phase: prefill batches the whole prompt, decode does one token against
    // the cache, and on this model they differ by an order of magnitude.
    let mut err = std::io::stderr();
    let _ = writeln!(err);
    // Asked *after* the run, because the cost is incurred inside the first
    // forward pass: the CUDA backend places every expert on first sight of its
    // tensor, so `prefill_ns` contains it. `None` on every CPU backend, which
    // is what leaves the 0.6B and 9B output unchanged.
    engine.prof.setup = engine.ops.setup_cost();
    let _ = engine.prof.phases(&mut err);

    // Wall clock over both phases, so its token count is the sum and not
    // `produced` -- dividing prefill-plus-decode time by decode tokens alone
    // would report a rate that is neither phase's and flatter the longer the
    // prompt. The remainder is what the run spends outside the model:
    // detokenizing the whole sequence and printing what is new, once a token.
    let total_tokens = engine.prof.prefill_tokens + produced.len() as u64;
    let outside_ms =
        secs * 1000.0 - (engine.prof.prefill_ns + engine.prof.decode_ns) as f64 / 1e6;
    let _ = writeln!(
        err,
        "wall     {total_tokens:>6} tok  {:>9.1} ms  {:>8.2} tok/s  {outside_ms:>7.1} ms outside the model",
        secs * 1000.0,
        total_tokens as f64 / secs.max(1e-9),
    );

    if o.report {
        let _ = writeln!(err);
        let _ = engine.prof.report(&mut err);
    }
    if let Some(path) = o.json.as_deref() {
        std::fs::write(path, engine.prof.to_json()).map_err(|source| {
            inferred_thoughts::Error::Io {
                path: path.to_string(),
                source,
            }
        })?;
        eprintln!("profile written to {path}");
    }
    // **A decode token, not the run divided by the decoded tokens.** This was
    // `wall / produced.len()`, which folds prefill and one-time setup into a
    // number labelled "ms/token": a 5-token prompt with 25 s of expert
    // placement and 199 decoded tokens reported 170.40 ms/token against a real
    // decode of 38.3, and `report_device` then divided device time by it and
    // printed "the GPU is busy 97% of it".
    //
    // Third wrong denominator of the same shape in this file. The other two are
    // the prefill line (fixed by `Ops::setup_cost`) and the launch replay's
    // normalisation, whose guard is copied below.
    let ms_per_token = if engine.prof.decode_tokens > 0 {
        engine.prof.decode_ns as f64 / engine.prof.decode_tokens as f64 / 1e6
    } else {
        secs * 1000.0 / total_tokens.max(1) as f64
    };
    let setup_ns = engine.prof.setup.map_or(0, |(ns, _)| ns);
    let model_ns = (engine.prof.prefill_ns + engine.prof.decode_ns).saturating_sub(setup_ns);
    Ok(Run {
        ms_per_token,
        ms_per_pass_token: model_ns as f64 / total_tokens.max(1) as f64 / 1e6,
        tokens: total_tokens,
    })
}

/// What a generation actually did, as opposed to what was asked for.
///
/// **`tokens` is the count produced, not `max_tokens`.** The device report
/// divides every counter by it, and a run that stops early — context full, or
/// an EOS — would otherwise have every per-token figure understated by the
/// ratio. That happened: a 1000-token request that stopped at 504 reported
/// 1621 launches and 50% GPU occupancy where the truth was 3195 and ~98%.
#[derive(Clone, Copy)]
struct Run {
    /// Wall clock of one **decode** token, from the profiler's own phase timer.
    ms_per_token: f64,
    /// Model time per **pass token** — prefill plus decode, net of one-time
    /// setup, divided by every token the device did work for.
    ///
    /// **The canonical basis for `report_device`.** Every device counter there
    /// accumulates over passes, and so does the launch replay's total, so a
    /// decode `ms/token` is the wrong thing to divide either by. Comparing them
    /// printed "the GPU is busy 435% of it", then 118% after the setup was
    /// subtracted but the prefill tokens were not — prefill costs ~285 ms a
    /// token against decode's 38, so it dominates a figure normalised over all
    /// of them.
    ms_per_pass_token: f64,
    /// Tokens the device did work for: prefill plus decode. The device counters
    /// accumulate over every pass, so this is the divisor they want — and it is
    /// deliberately not the divisor `ms_per_token` uses, which is why the two
    /// may only be compared when the run is decode-dominated.
    tokens: u64,
}

// ----------------------------------------------------------------------- serve

struct ServeArgs {
    model: String,
    port: u16,
    ctx: usize,
    /// Prompt tokens per forward pass; bounds activation VRAM. See `--batch`.
    max_batch: usize,
    /// Cap the MoE expert cache, in GiB. 0 uses the automatic budget.
    expert_cache: f64,
    /// Cap the page-locked host tier behind it, in GiB. 0 is automatic.
    expert_host: f64,
    max_tokens: usize,
    /// Report launch counts and expert residency after each turn.
    profile_device: bool,
    threads: usize,
    backend: String,
    rms_serial: bool,
    verbose: bool,
}

/// Apply the prefill batch cap to a freshly built engine.
///
/// A free function because `serve` builds an engine in four branches with four
/// different backend types, and the cap must not be something one of them can
/// forget: it bounds activation VRAM, which on the 9B is ~405 KiB per batch
/// token. See `engine::DEFAULT_MAX_BATCH`.
fn with_batch<'a, O: inferred_thoughts::Ops>(
    mut e: inferred_thoughts::Engine<'a, O>,
    n: usize,
) -> inferred_thoughts::Engine<'a, O> {
    e.set_max_batch(n);
    e
}

fn serve(a: ServeArgs) -> inferred_thoughts::Result<()> {
    use inferred_thoughts::serve::{ServeOpts, serve as run_server};
    use inferred_thoughts::tok::chat::ChatMl;
    use inferred_thoughts::{Engine, Model, Naive, Par, Spin, Tokenizer};

    let f = GgufFile::open(&a.model)?;
    let tk = Tokenizer::from_metadata(&f.metadata)?;
    let m = Model::load(&f)?;
    // Refused up front rather than per request: a server that cannot render a
    // chat turn has nothing useful to do.
    let chat = ChatMl::detect(&tk, &f.metadata)?;

    let name = std::path::Path::new(&a.model)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".to_string());
    let opts = ServeOpts {
        port: a.port,
        model_id: format!("{name}-{}", a.backend),
        max_tokens: a.max_tokens,
        verbose: a.verbose,
    };

    eprintln!(
        "model {} | {} | {} layers ({} with kv) | ctx {}",
        name,
        m.arch(),
        m.n_layer(),
        m.n_kv_layer(),
        a.ctx,
    );

    #[cfg(feature = "cuda")]
    if a.backend == "cuda" {
        let cuda = inferred_thoughts::Cuda::new(0)?;
        cuda.rms_serial(a.rms_serial);
        cuda.set_expert_budget((a.expert_cache * 1073741824.0) as usize);
        cuda.set_expert_host_budget((a.expert_host * 1073741824.0) as usize);
        cuda.record_launches(a.profile_device);
        cuda.set_model_path(&f.path);
        cuda.set_map_base(f.map_base());
        cuda.report_per_turn(a.profile_device);
        // See the same call in `generate`: the KV slabs are allocated after the
        // expert slab has already sized itself from free VRAM, so the context
        // length has to be declared here or the slab takes VRAM the cache needs.
        cuda.reserve_for_kv(kv_reserve_bytes(&m, a.ctx));
        let (free, total) = cuda.mem_info()?;
        eprintln!(
            "device {} | {:.2} of {:.2} GiB free",
            cuda.name(),
            free as f64 / 1073741824.0,
            total as f64 / 1073741824.0,
        );
        let engine = with_batch(Engine::new(m, &cuda, a.ctx, false), a.max_batch);
        let r = run_server(engine, tk, chat, opts);
        if let Some(e) = cuda.take_error() {
            return Err(e);
        }
        return r;
    }

    let n_threads = if a.threads == 0 {
        Par::default_threads()
    } else {
        a.threads
    };
    if n_threads <= 1 {
        return run_server(
            with_batch(Engine::new(m, Naive, a.ctx, false), a.max_batch),
            tk,
            chat,
            opts,
        );
    }
    match a.backend.as_str() {
        "spin" => run_server(
            with_batch(Engine::new(m, Spin::new(n_threads), a.ctx, false), a.max_batch),
            tk,
            chat,
            opts,
        ),
        "par" => {
            Par::init(n_threads);
            run_server(
                with_batch(Engine::new(m, Par, a.ctx, false), a.max_batch),
                tk,
                chat,
                opts,
            )
        }
        other => Err(inferred_thoughts::Error::InconsistentArchitecture {
            what: "--backend",
            detail: format!("{other:?} is not a backend; expected \"spin\", \"par\" or \"cuda\""),
        }),
    }
}

// ----------------------------------------------------------------------- trace

/// Emit `name<TAB>n_elements<TAB>sum` per intermediate tensor.
///
/// The sum is accumulated in f64 and covers the whole tensor, which is what
/// makes it comparable to the `sum = ...` line `llama-eval-callback` prints.
fn trace(model: &str, prompt: &str, dump: Option<&str>) -> inferred_thoughts::Result<()> {
    use inferred_thoughts::{Model, Naive, Qwen3, Tokenizer};
    use std::io::Write;

    let f = GgufFile::open(model)?;
    let tk = Tokenizer::from_metadata(&f.metadata)?;
    let m = Model::load(&f)?;

    let tokens = tk.encode(prompt, true, true);
    eprintln!("tokens ({}): {tokens:?}", tokens.len());

    // Optional binary sidecar: [u32 name_len][name][u64 n][n * f32], repeated.
    // Our flat layout matches ggml's, so a comparison can slice this by the
    // reference's own ne0 without us tracking tensor shapes here.
    let mut sink = match dump {
        Some(path) => Some(std::io::BufWriter::new(std::fs::File::create(path).map_err(
            |source| inferred_thoughts::Error::Io {
                path: path.to_string(),
                source,
            },
        )?)),
        None => None,
    };

    let mut emit = |name: &str, il: usize, data: &[f32]| {
        let sum: f64 = data.iter().map(|&v| v as f64).sum();
        // Layer-scoped names carry the index, matching the reference's
        // "attn_norm-0" convention; whole-model ones do not.
        let label = match name {
            "inp_embd" | "result_norm" | "result_output" => name.to_string(),
            _ => format!("{name}-{il}"),
        };
        println!("{label}\t{}\t{sum:.6}", data.len());

        if let Some(w) = sink.as_mut() {
            let nb = label.as_bytes();
            let _ = w.write_all(&(nb.len() as u32).to_le_bytes());
            let _ = w.write_all(nb);
            let _ = w.write_all(&(data.len() as u64).to_le_bytes());
            for &v in data {
                let _ = w.write_all(&v.to_le_bytes());
            }
        }
    };

    // A single pass at position 0, so the cache only needs room for the prompt
    // and the profiler is inert -- `trace` measures numerics, not time.
    //
    // On `qwen35` this walks the prompt one token at a time, which is what the
    // architecture does anyway: the delta rule is a sequential scan. The trace
    // therefore shows the *last* prompt token's tensors, where the reference
    // prints the whole batch. That difference matters when reading the dump and
    // is why the comparison script slices by the reference's own ne0.
    let mut cache = inferred_thoughts::KvCache::new(m.n_kv_layer(), m.kv_dim(), tokens.len());
    let mut recurrent = m
        .recurrent_dims()
        .map(|(n, conv, ssm)| inferred_thoughts::RecurrentState::new(n, conv, ssm));
    let mut prof = inferred_thoughts::Profile::new(false);
    let mut ctx = inferred_thoughts::Ctx::new(&mut emit, &mut prof);
    let logits = m.forward(
        &Naive,
        &tokens,
        0,
        &mut cache,
        recurrent.as_mut(),
        &mut ctx,
    )?;

    let top = Qwen3::argmax(&logits);
    eprintln!(
        "argmax = {top} {:?}",
        tk.token_text(top).unwrap_or("<out of range>")
    );
    Ok(())
}

// ---------------------------------------------------------------- human output

fn print_human(f: &GgufFile, max_array: usize) {
    let arch = f
        .metadata
        .architecture()
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "<missing general.architecture>".to_string());

    println!("file:       {}", f.path.display());
    println!(
        "size:       {} ({} bytes)",
        human_bytes(f.file_size()),
        f.file_size()
    );
    println!("version:    {}", f.version);
    println!("alignment:  {}", f.alignment);
    println!("arch:       {arch}");
    println!("kv pairs:   {}", f.metadata.len());
    println!("tensors:    {}", f.tensors.len());
    println!("data at:    {} (0x{:x})", f.data_offset, f.data_offset);
    println!(
        "weights:    {} across {} params",
        human_bytes(f.total_tensor_bytes()),
        human_count(f.total_parameters())
    );

    println!("\nmetadata ({}):", f.metadata.len());
    let key_width = f.metadata.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
    for (key, value) in f.metadata.iter() {
        println!(
            "  {key:<key_width$}  {:<10}  {}",
            type_label(value),
            format_value(value, max_array)
        );
    }

    println!("\ntensors ({}):", f.tensors.len());
    let name_width = f.tensors.iter().map(|t| t.name.len()).max().unwrap_or(0);
    println!(
        "  {:>4}  {:<name_width$}  {:<8}  {:<26}  {:>14}  {:>14}",
        "idx", "name", "type", "shape", "offset", "bytes"
    );
    for (i, t) in f.tensors.iter().enumerate() {
        println!(
            "  {i:>4}  {:<name_width$}  {:<8}  {:<26}  {:>14}  {:>14}",
            t.name,
            t.ty.name(),
            format_shape(t),
            t.offset,
            t.n_bytes
        );
    }

    // A per-type summary makes it obvious at a glance which quant formats a
    // later stage must implement to load this file.
    println!("\ntype summary:");
    let mut counts: Vec<(&'static str, usize, u64)> = Vec::new();
    for t in &f.tensors {
        match counts.iter_mut().find(|(n, _, _)| *n == t.ty.name()) {
            Some(entry) => {
                entry.1 += 1;
                entry.2 += t.n_bytes;
            }
            None => counts.push((t.ty.name(), 1, t.n_bytes)),
        }
    }
    counts.sort_by(|a, b| b.2.cmp(&a.2));
    for (name, n, bytes) in counts {
        println!("  {name:<8}  {n:>4} tensors  {:>10}", human_bytes(bytes));
    }
}

fn format_shape(t: &TensorInfo) -> String {
    t.dims
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(" x ")
}

fn type_label(v: &Value) -> String {
    match v {
        Value::Array(a) => format!("arr[{}]", a.elem_type_name()),
        other => other.type_name().to_string(),
    }
}

fn format_value(v: &Value, max_array: usize) -> String {
    match v {
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::F32(x) => x.to_string(),
        Value::F64(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::Bool(x) => x.to_string(),
        Value::String(s) => format_string(s),
        Value::Array(a) => {
            let n = a.len();
            let shown = n.min(max_array);
            let mut parts: Vec<String> = (0..shown).map(|i| a.elem_to_string(i)).collect();
            if n > shown {
                parts.push(format!("... {} more", n - shown));
            }
            format!("[{n}] {{{}}}", parts.join(", "))
        }
    }
}

/// Long strings (chat templates run to thousands of characters) are elided so
/// the metadata table stays readable.
fn format_string(s: &str) -> String {
    const LIMIT: usize = 120;
    if s.chars().count() <= LIMIT {
        return format!("{s:?}");
    }
    let head: String = s.chars().take(LIMIT).collect();
    format!("{head:?}... ({} chars total)", s.chars().count())
}

fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", UNITS[u])
    }
}

fn human_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.2}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else {
        n.to_string()
    }
}

// ----------------------------------------------------------------- json output

/// Hand-rolled so the crate keeps no serialization dependency. The shape is
/// chosen to be easy to compare against `gguf_dump.py --json` from a script.
fn print_json(f: &GgufFile) {
    println!("{{");
    println!("  \"filename\": {},", json_string(&f.path.display().to_string()));
    println!("  \"version\": {},", f.version);
    println!("  \"alignment\": {},", f.alignment);
    println!("  \"data_offset\": {},", f.data_offset);
    println!("  \"n_kv\": {},", f.metadata.len());
    println!("  \"n_tensors\": {},", f.tensors.len());

    println!("  \"kv\": {{");
    let n_kv = f.metadata.len();
    for (i, (key, value)) in f.metadata.iter().enumerate() {
        let comma = if i + 1 < n_kv { "," } else { "" };
        println!(
            "    {}: {{\"type\": {}, \"value\": {}}}{comma}",
            json_string(key),
            json_string(&type_label(value)),
            json_value(value)
        );
    }
    println!("  }},");

    println!("  \"tensors\": {{");
    let n_t = f.tensors.len();
    for (i, t) in f.tensors.iter().enumerate() {
        let comma = if i + 1 < n_t { "," } else { "" };
        let shape = t
            .dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "    {}: {{\"index\": {i}, \"shape\": [{shape}], \"type\": {}, \"offset\": {}, \"n_bytes\": {}}}{comma}",
            json_string(&t.name),
            json_string(t.ty.name()),
            t.offset,
            t.n_bytes
        );
    }
    println!("  }}");
    println!("}}");
}

fn json_value(v: &Value) -> String {
    match v {
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::F32(x) => json_float(f64::from(*x)),
        Value::F64(x) => json_float(*x),
        Value::Bool(x) => x.to_string(),
        Value::String(s) => json_string(s),
        Value::Array(a) => json_array(a),
    }
}

fn json_array(a: &Array) -> String {
    fn join<T: ToString>(v: &[T]) -> String {
        v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
    }
    let body = match a {
        Array::U8(v) => join(v),
        Array::I8(v) => join(v),
        Array::U16(v) => join(v),
        Array::I16(v) => join(v),
        Array::U32(v) => join(v),
        Array::I32(v) => join(v),
        Array::U64(v) => join(v),
        Array::I64(v) => join(v),
        Array::Bool(v) => join(v),
        Array::F32(v) => v
            .iter()
            .map(|x| json_float(f64::from(*x)))
            .collect::<Vec<_>>()
            .join(", "),
        Array::F64(v) => v.iter().map(|x| json_float(*x)).collect::<Vec<_>>().join(", "),
        Array::String(v) => v.iter().map(|s| json_string(s)).collect::<Vec<_>>().join(", "),
    };
    format!("[{body}]")
}

/// JSON has no NaN or Infinity literals; emit null so output stays parseable
/// rather than silently producing something a JSON reader rejects.
fn json_float(x: f64) -> String {
    if x.is_finite() { x.to_string() } else { "null".to_string() }
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// VRAM the KV cache will want at this context length, in bytes.
///
/// Mirrors `KvCache::new(n_kv_layer, kv_dim, n_ctx)` and `capacity_bytes`: two
/// f16 tensors of `n_kv_layer * n_ctx * kv_dim`. Computed rather than measured
/// because it has to be known *before* the cache exists — the expert slab sizes
/// itself first, and this is what stops it taking VRAM the cache is going to
/// need.
#[cfg(feature = "cuda")]
fn kv_reserve_bytes(m: &inferred_thoughts::Model, n_ctx: usize) -> usize {
    m.n_kv_layer().saturating_mul(n_ctx).saturating_mul(m.kv_dim()).saturating_mul(2 * 2)
}
