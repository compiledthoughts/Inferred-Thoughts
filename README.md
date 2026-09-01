# inferredThoughts

A from-scratch Rust inference engine for GGUF models. Built toward serving an
MoE model whose weights do not fit in VRAM.

| | |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | the shape — module map, the `Ops` seam, where exactness stops |
| [`BENCHMARKS.md`](BENCHMARKS.md) | every number in sequence, with the method to reproduce it |
| [`HANDOFF.md`](HANDOFF.md) | why the project exists, and what did not survive contact |
| [`CLAUDE.md`](CLAUDE.md) | the rules and the current state |

**Status: v0.1.** Stages 1–6 complete, the `qwen35` architecture decoded, and
the forward pass running on **both CPU and GPU**. Loads a GGUF, tokenizes
exactly like llama.cpp, and generates coherent text on Qwen3-0.6B at **~59
tok/s** decode on the CPU — within **1.09–1.15x** of llama.cpp on the same
CPU, with scalar f32 kernels and results bit-identical to a single-threaded
oracle. KV cache, prefill/decode split, threaded attention, a spin-waiting
thread pool, a chat template, and a profiler. 116 tests by default, 120 with
`--features cuda`, plus 19 that need a model or a device.

The CUDA backend runs the whole model on the GPU at **~215 tok/s** at d384,
flat from short context out to at least d1792, against the CPU's ~54 at the same
depth and llama.cpp's ~308. The decode step is recorded once as a CUDA graph and
replayed: 13 kernel launches a token instead of 673, and the GPU is now busy
~92% of a token. See [`BENCHMARKS.md`](BENCHMARKS.md) for the progression.

## Progress

All figures are Qwen3-0.6B Q8_0 on the machine in `CLAUDE.md` — Ryzen 7 9700X,
8 cores, dual-channel DDR5. Decode is measured at short context unless noted.

| | decode | backend | added | state | key finding |
|---|---|---|---|---|---|
| **v0**<br>`065fc41` | ~2 tok/s<br>*quadratic in context* | `naive` — scalar f32, 1 thread | GGUF parser, dequant, tokenizer, `qwen3` forward pass | 71 tests; verified layer by layer against `llama-eval-callback` | RMSNorm must accumulate in f64, as ggml does — f32 shifts the scale enough to flip Q8_0 quants downstream. And ~1e-3 per tensor is the **floor** for independent Q8_0 implementations, not a defect |
| **v1**<br>`36ebcd8` | **27 tok/s**<br>*linear in context* | `naive` + `par` (rayon, LM head only) | KV cache (f16), engine with prefill/decode split, profiler, threading | 91 tests + 8 model-backed; decode proven **bit-identical** to full recompute; `par` ≡ `naive` bit for bit | Attention scoring is **18.3x** behind llama.cpp while dense matmuls are only **2.45x** behind. Threading at matmul granularity is a *net loss* — cost is per task, not per region |
| **v2** | **19.9 tok/s @ d384**<br>*24.0 @ d64* | + `Ops::attend`, threaded over kv heads | block-wise f16 conversion, `--chat`, `--show-special`, honest stop reasons | 94 tests + 11 model-backed; forward pass **byte-identical** to v1 | Attention growth cut **3.4x** (29.5 → 8.6 ms over 320 positions) with **zero** numeric change — f16→f32 is lossless, so hoisting it out of the dot product cannot move a bit |
| **v3** | **46.9 tok/s @ d384**<br>*58.4 @ d64* | + `spin` — persistent spin-waiting pool, every matmul threaded | `src/ops/pool.rs`, the crate's only `unsafe`; `naive` is now `#![forbid(unsafe_code)]` | 106 tests + 13 model-backed; bit-identical to `naive` at 2/3/5/8 threads | **A rayon parallel region costs ~430 µs here; a spin barrier costs 0.40 µs — 1088x.** Threading was never the problem, dispatch was. Decode bandwidth 12 → **37 GB/s** |
| **v4** | **53.0 tok/s @ d384**<br>*59.5 @ d64* | same, built for the actual CPU | branch-free `f16_to_f32`; `-C target-cpu=native` | 108 tests + 13 model-backed; forward pass still **byte-identical** to v1 | We had been compiling **SSE2-only on a Zen 5**. Enabling AVX-512 doubled attention (2.05x) and did **nothing** for the dense matmuls — the cleanest confirmation yet that one is compute-bound and the other DDR5-bound |
| **v0.1** | **12.5 tok/s**<br>*on the GPU* | + `cuda` — all eight ops as kernels | weights resident in VRAM, KV mirrored on device, `--backend cuda` | 120 tests + 19 device/model-backed; five kernels **bit-identical** to `naive` through the whole model | **Slower than our own CPU.** A round trip through `Ops` costs ~66 us and a decode step makes ~478 of them, so a third of the token goes before any arithmetic. Same shape as v3's rayon finding, one layer down |
| **v0.6** | **215 tok/s**<br>*@ d384* | same | RMSNorm's sum of squares reduced as a tree, by default; the serial walk kept behind `--rms-serial` | 120 tests + 21 device/model-backed; `rms_serial_restores_bit_equality` tests that the flag buys exactness back | **The standing refusal was reversed on evidence.** The serial f64 walk is linear in `n`, so on the 35B it would cost ~7.9 ms against a ~13.3 ms predicted token. An order-free accumulator was built to get the speed *and* keep equal bits; it works, but not without also changing the oracle. The tolerance turns out to be set by the **f32 cast of `mean`**, not by the reorder |
| **v0.5** | 126 tok/s<br>*@ d384, flat to d1792* | same | RMSNorm squares staged through shared memory | 120 tests + 20 device/model-backed | **We are GPU-bound now** — the card is busy ~92% of a token, which retires the megakernel plan. `rms_norm` is the largest remaining kernel, and the cost is **FP64 latency on a dependent chain**, not the loads or the compiler as guessed |
| **v0.4** | 117 tok/s<br>*@ d384, flat to d1792* | same | the decode step recorded once as a CUDA graph and replayed | 120 tests + 19 device/model-backed; the whole-model test now asserts the graph engaged | **673 kernel launches a token became 13.** A `cuLaunchKernel` is ~7 us of CPU bookkeeping, so a third of the token was the CPU describing work. Built by hand rather than captured, because `n_pos` grows every token and a captured graph is stale immediately |
| **v0.3** | 93 tok/s<br>*on the GPU, flat to d1792* | same | flash-decoding attention (split-KV, online softmax); KV cache written as f16 straight into VRAM | 120 tests + 19 device/model-backed | **Attention parallelism was capped by `n_head` = 16**, so its cost grew with context while its parallelism did not — a cliff past d1400, not a slope. And the KV cache round trip was a host barrier in the middle of every layer: 67 bus crossings per token became **5** |
| **v0.2** | **42.4 tok/s**<br>*on the GPU* | same, activations stay on the card | `host_wrote` / `host_needs` / `begin_pass` on the seam (no-ops on CPU); warp-per-row matmul; device-side Q8_0 quantize; `--profile-device` | same counts; still bit-identical, and `naive`/`spin` unchanged | **1329 bus crossings per token became 67**, for 3.4x. And the expected trade never came due: Q8_0's block structure hands you a split that is parallel where the arithmetic is order-free and serial where it is not, so the fast matmul is *still* bit-exact |
| **since v4**<br>`402d262`<br>`29b6f02`<br>`e34ec08` | unchanged | same | Q5_K, Q6_K, IQ4_XS; a CUDA driver-API spike — sm_120 PTX from `build.rs` and one bit-exact kernel; the `qwen35` config decoded and shape-checked | 116 tests + 15 model-backed; the three new quants **bit-exact** against `gguf.quants` on 65,536 real weights each, taken from the 35B itself | Reading the 35B rather than trusting notes about it: **40 blocks not 41, embedding 2048 not 4096**, and it needs **three** k-quant formats — `Q6_K` appeared on no prior list. Structural unit tests hold independently of the fixtures, so a regenerated fixture cannot bless a layout error |

For scale, llama.cpp on the same CPU, same model, same 16-token prompt:
**568 t/s prefill, 65 t/s decode**. The shape of our gap to it matters more than
its size:

| depth | v1 | v2 | v3 | v4 | llama.cpp | v4 gap |
|---|---|---|---|---|---|---|
| d64 | 23.9 | 24.0 | 58.4 | **59.5** | 67.7 | 1.14x |
| d128 | 21.1 | 22.2 | 55.7 | **58.9** | 64.4 | 1.09x |
| d256 | 16.9 | 21.3 | 48.6 | **56.0** | 63.0 | 1.12x |
| d384 | 14.0 | 19.9 | 46.9 | **53.0** | 60.8 | **1.15x** |

v1's gap *widened* with context, 2.8x → 4.4x. v4's is flat at 1.09–1.15x.
Splitting the curve into its constant and per-position terms:

| | v4 | llama.cpp | gap |
|---|---|---|---|
| flat — dense matmuls, per token | 16.39 ms | 14.45 ms | **1.13x** |
| slope — attention, per position | 0.00641 ms | 0.00522 ms | **1.23x** |

Both terms are now within ~1.2x of hand-tuned AVX-512, with every result
bit-identical to the scalar oracle.

llama.cpp is the oracle, not the thing to beat here — the real target is 40
tok/s on Qwen3.6-35B-A3B (60 with MTP), which is where the offload policy
actually matters. For the other end of that range: the same model with no GPU
at all decodes at **5.17 tok/s**. The running log in `HANDOFF.md` records that
measurement, and also why it is not yet trustworthy.

## The CUDA backend

```bash
cargo build --release --features cuda
$B generate -m $MODEL -p "..." -n 64 --backend cuda --profile --profile-device
```

The whole forward pass runs on the GPU: all eight `Ops` methods as kernels,
weights uploaded once and kept in VRAM, the KV cache mirrored on device and
appended to rather than resent, and activations that **stay** on the card
across a layer. Qwen3-0.6B Q8_0, 64 tokens, median of five:

| | tok/s | ms/token | effective |
|---|---|---|---|
| ours, `cuda`, first working version | 12.5 | 80 | 8 GB/s |
| ours, `cuda` | **42.4** | 23.6 | 27 GB/s |
| ours, `spin`, 8 threads | 62.6 | 16.0 | 39.9 GB/s |
| llama.cpp, CUDA, `-ngl 99` | **337.5** | 3.0 | 214 GB/s |

**3.4x over the first version, every logit still identical to the CPU oracle,
and still 0.68x of our own CPU backend.** llama.cpp reaches 48% of the card's
448 GB/s at batch 1.

### The seam was the problem, and it was measured before it was fixed

The first version was *slower than the CPU*, and the reason had nothing to do
with kernels. The `Ops` seam takes host slices and returns host slices, so
every operation was a round trip: upload, launch, download. A decode step runs
~478 of them.

At context 128 — where attention should be nearly free — the attention half was
66% of decode time, and the ratio between the two halves tracked their *op
counts*, not their bytes. That is the signature of fixed per-call cost, so the
profiler grew a way to see it directly rather than by argument:

```
device   40176 kernel launches
         754 up / 3872 down = 67 bus crossings per token
         launch 10.9 us | 4 KiB up 28.0 us | down 61.6 us
         seam costs 10.9 ms/token before any arithmetic
```

It corrected an estimate on its first run: "478 ops, two crossings each" was
wrong — uploads outnumbered downloads nearly two to one, because a matmul sent
its quantized activation as two buffers and RoPE sent three. Real figure at the
time: **1329 crossings per token**.

### Keeping activations on the card

The seam gained three methods with no-op defaults, so `naive`, `par` and `spin`
are untouched and still bit-identical to each other:

| | |
|---|---|
| `host_wrote(buf)` | the model wrote this directly; any device copy is stale |
| `host_needs(buf)` | the model is about to read this; bring it back |
| `begin_pass()` | a pass is starting; addresses may have been recycled |

These are hints *about the host*, not a buffer abstraction — who owns an
activation is a larger question this does not answer. The backend keeps a mirror
per host address carrying one bit: whether the device copy is at least as fresh
as the host one. An op that writes a buffer sets it and skips the download; the
next op to read it finds it there and skips the upload. A layer's chain of ops
therefore touches the bus at its ends instead of twice per op.

**1329 crossings per token became 67** — the two the model actually declares
(`k` and `v` per layer, which the host KV cache reads) plus the logits.

Quantization had to move onto the device for that to work: a matmul quantizing
its input on the host would drag every activation home.

### Three things that were measured rather than assumed

- **Pinned staging buffers.** The copies were 54% of a round trip and pageable
  memory forces the driver to stage through its own pinned buffer, so this
  looked certain. Reverted: ~20% *slower*.
- **Repacking Q8_0 weights for coalescing.** Verified bit-exact, and reverted:
  ~20% *slower*. The premise was wrong — one thread per row already streams its
  row sequentially, which the cache serves well.
- **`begin_pass` clearing the mirror map** freed and reallocated ~280 device
  buffers per token. Invalidating in place, keeping the allocations, is what
  made the residency win show up at all.

### Exactness

| kernels | against `naive` |
|---|---|
| `matmul`, `rope_neox`, `add_assign` | **bit-identical** |
| `rms_norm`, `rms_norm_heads` | derived tolerance; **bit-identical under `--rms-serial`** |
| `softmax`, `silu_mul`, `attend` | ≤ 5.2e-8, about one ulp |

Three of these call `expf`, and CUDA's is not obliged to match glibc's. **The
obstacle was not the expected one.** Parallel reductions were assumed to be what
costs bit-exactness; keeping each reduction serial kept it, and what actually
breaks it is one libm function.

The two RMSNorm kernels are the one place that was traded deliberately. Their
sum of squares is reduced as a tree, which is worth **1.63x end to end** because
the serial f64 walk was dependent-chain FP64 latency and ~40% of device time.
f64 addition rounds, so a tree is a *different answer* — measurably so: it
returns four different results at four block sizes on an adversarial input.
The tolerance is derived not from that reorder, which moves the sum by 1.5e-16,
but from the **f32 cast of `mean`** that follows it, whose grid is a million
times coarser; two orders land on the same f32 unless a rounding boundary falls
between them. `--rms-serial` restores the serial walk and with it bit-equality,
because determinism is hard to recover once it is given up.

That holds even for the fast matmul, which is one warp per row. Q8_0 has a unit
of work that is *already* exact: within a 32-element block the sum of products
is an **integer** sum, so it cannot round and its order does not matter. Only
the accumulation *across* blocks is f32. So lane `b % 32` computes block `b`
into shared memory and lane 0 adds those up in ascending `b` — the oracle's
order exactly. Parallel where the arithmetic is order-free, serial where it is
not. Coalescing falls out: the warp sits in 32 consecutive 34-byte blocks
instead of 32 rows a kilobyte apart.

Whole-model divergence is 1.47e-2 of logit magnitude, which is what `CLAUDE.md`
predicts for a one-ulp perturbation amplified through 28 layers of Q8_0
re-quantization. Greedy decoding does flip a token within a dozen.

Rather than argue about whether that was drift or a defect, there is a test for
it: `only_the_expf_ops_diverge` runs the five exact kernels on the GPU and the
three exp-dependent ones on the CPU and asserts **bit equality on all 151,936
logits**. It passes, which leaves `expf` as the only explanation — and makes any
future GPU divergence bisectable in one run. It is also the first thing here to
straddle both devices, and has to do the same host/device bookkeeping a real
CPU/GPU layer split will need.

```bash
cargo test --release --features cuda --test cuda_ops -- --ignored --nocapture
```

### What is left

Launch overhead. ~580 launches per token at 10.9 us is ~6.3 ms of a 23.6 ms
token, and no amount of kernel tuning removes it — that needs fewer launches,
i.e. fusing ops within a layer.

**This tests none of the offload thesis.** The 0.6B is 0.59 GiB against 14.80
GiB free, and ~580 launches per token is a fixed cost regardless of model size:
at 0.6B each kernel does ~1.3 MB of work, on the 9B roughly 11x more, so the
same overhead is a few percent there instead of a third.

## Build

Everything runs inside WSL (`Ubuntu-24.04`). Build artifacts go on ext4 rather
than DrvFs, which is much faster:

```bash
export CARGO_TARGET_DIR=~/.cargo-target/inferredthoughts
cargo build --release
export B=~/.cargo-target/inferredthoughts/release/inferred
```

`.cargo/config.toml` sets `-C target-cpu=native`. Without it rustc emits
SSE2-only code, which costs about 2x on attention — see the file for the
measurements.

CUDA is behind an off-by-default feature so the crate builds and tests without
a toolkit:

```bash
cargo test --features cuda        # needs CUDA 12.8+ and an sm_120 device
```

`build.rs` finds `nvcc` via `CUDA_PATH`, `CUDA_HOME`, or `/usr/local/cuda`,
compiles `kernels/kernels.cu` to sm_120 PTX, and links the driver API
(`libcuda.so`) — no `libcudart`, no wrapper crate.

## Run

```bash
export MODEL=~/models/Qwen3-0.6B-Q8_0.gguf

# generate text (greedy)
$B generate -m $MODEL -p "The capital of France is" -n 20

# -t threads (0 = physical cores, 1 = the scalar oracle), -c KV context
$B generate -m $MODEL -p "..." -n 200 -t 4 -c 8192

# inspect a model: metadata, tensor table, per-type summary. --json to diff.
$B inspect $MODEL
```

Only the `qwen3` architecture loads today. `Qwen3.5-9B` (`qwen35`) is rejected
with a named error: its config is decoded and shape-checked, but the
GatedDeltaNet forward pass is not written.

A raw prompt runs in **completion** mode, so an instruct model never enters the
assistant turn, never emits `<|im_end|>`, and degenerates into repetition.
`--chat` wraps the prompt as a chat turn instead:

```bash
$B generate -m $MODEL -p "List the capitals of 10 countries" -n 500 --chat
```

ChatML is detected from the model's own vocabulary *and* its
`tokenizer.chat_template`; a model whose template is some other shape is
refused rather than guessed at. Hand-writing the markers still works
(`parse_special` is on) but is fragile: command substitution strips the
trailing newline, which gives a 15-token prompt instead of 16 and a completely
different answer.

`--show-special` renders special tokens in the output, which is how `<think>`
and `</think>` become visible — without it they are silently dropped.

## CLI reference

`inferred --help` and `inferred <command> --help` print this from the source, so
they cannot drift from it. Three commands:

```
inferred inspect  <path>    parse a GGUF, print its metadata and tensor index
inferred generate -m -p     prefill the prompt, then decode against the KV cache
inferred trace    -m -p     one forward pass, checksumming every tensor
```

### `inspect`

| | default | |
|---|---|---|
| `<path>` | *required* | positional — the `.gguf` file |
| `--json` | off | machine-readable, to diff against `gguf_dump.py --json` |
| `--max-array <N>` | `16` | truncate longer arrays in human output. The token list is 151,936 entries, so this is what keeps the dump readable |

### `generate`

| | default | |
|---|---|---|
| `-m`, `--model <PATH>` | *required* | |
| `-p`, `--prompt <TEXT>` | *required* | |
| `-n`, `--max-tokens <N>` | `16` | a ceiling, not a target — EOS stops earlier unless `--ignore-eos` |
| `-c`, `--ctx <N>` | `4096` | KV positions, allocated up front. Trades memory for the longest usable context; the banner prints the resulting resident size |
| `-t`, `--threads <N>` | `0` | `0` means physical cores, taken as half the logical count — SMT siblings do not help a memory-bound kernel. **`1` bypasses the pool entirely and runs `naive`**, the scalar oracle |
| `--backend <NAME>` | `spin` | `spin`, `par`, or `cuda` when built with `--features cuda`. `cuda` ignores `-t`; the other two are ignored at `-t 1`. An unknown name is an error, not a fallback |
| `--chat` | off | wrap the prompt as a chat turn (see above) |
| `--show-special` | off | render special tokens such as `<think>` instead of dropping them |
| `--ignore-eos` | off | keep decoding past the end-of-sequence token |
| `--profile` | off | print the profile summary. Tier 1 is always *collected*; this only controls whether it prints |
| `--profile-detail` | off | also time each layer's attention and FFN halves. Implies `--profile` |
| `--profile-json <PATH>` | — | write one record per token and per layer half, for diffing two runs |
| `--profile-device` | off | with `--backend cuda`, also time a bare launch, upload and download on the live device and report what the seam costs per token. Launch and transfer *counts* are always collected and printed under `--profile` |

### `trace`

| | default | |
|---|---|---|
| `-m`, `--model <PATH>` | *required* | |
| `-p`, `--prompt <TEXT>` | *required* | |
| `--dump <PATH>` | — | also write every intermediate tensor's full contents, so a comparison can check individual elements and not just checksums |

### Streams and exit status

**Generated text and trace checksums go to stdout; everything else goes to
stderr** — the model banner, the stop reason, timings, and the profile. So a
plain redirect captures exactly the payload and nothing else:

```bash
$B generate -m $MODEL -p "..." -n 200 --profile > text.txt   # profile still on screen
$B trace    -m $MODEL -p "..."                  > ours.txt   # checksums only
```

Failure exits non-zero and prints the `thiserror` `#[source]` chain, one
`caused by:` line per level, so an I/O error carries the OS message and a
missing tensor carries its name.

## Test

```bash
cargo test
```

Model-backed tests skip if the model is missing, but the suite fails if *every*
one skips. Override the search path with `INFERRED_MODEL_DIR`.

The Stage 5 acceptance tests load a real model, which is too slow for a debug
build, so they are `#[ignore]`d:

```bash
cargo test --release -- --ignored --test-threads=1
```

They assert that decode-with-cache produces **bit-identical** logits to full
recompute, and that `par` and `spin` reproduce `naive` bit for bit — `spin` at
2, 3, 5 and 8 threads, comparing all 151,936 logits.

## Profiler

Two tiers. Tier 1 is always collected — phase timings and a top-2 logit scan
cost orders of magnitude less than a token — so `--profile` only controls
whether it prints.

```bash
$B generate -m $MODEL -p "$P" -n 500 --profile
$B generate -m $MODEL -p "$P" -n 500 --profile-detail        # + per-layer
$B generate -m $MODEL -p "$P" -n 500 --profile-json p.json   # every record
```

```
prefill      16 tok      639.8 ms      25.0 tok/s
decode      413 tok    23321.3 ms     17.71 tok/s     56.5 ms/tok

weights  0.590 GiB per forward pass
         11.7 GB/s effective during decode
kv       46.9 MiB written, 10088.2 MiB read back

margin   min 0.0015  median 0.1860  (22 of 414 tokens inside the 1% drift band)
```

- **prefill vs decode**, not input vs output. They are compute-bound and
  memory-bound respectively, and most optimizations help only one.
- **bytes**, because the thesis is about bytes and time alone measures the
  symptom. Derived from tensor shapes rather than counted, so nothing contends
  on the hot path.
- **margin** — the top-1/top-2 logit gap. Tokens inside the ~1% drift band could
  differ from `llama-cli` without anything being wrong. A risk measure, not a
  defect count.

`--profile-detail` adds per-layer attention/FFN timing, which is what separates
attention's growth with context from the FFN's flat cost. `--profile-json`
writes one record per token and per layer half, for diffing two runs.

Overhead is verified rather than asserted: 26.3 vs 26.6 tok/s with detail on,
i.e. inside noise.

## Verifying against llama.cpp

llama.cpp is the oracle. A reference build must be CPU-only, or it silently
runs on the GPU and the numbers won't match:

```bash
cmake -S <llama.cpp> -B ~/llama-cpu-ref -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CPU_REPACK=OFF -DGGML_CUDA=OFF -DGGML_LLAMAFILE=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build ~/llama-cpu-ref --target llama-eval-callback -j 16
```

Then diff the forward pass tensor by tensor:

```bash
~/llama-cpu-ref/bin/llama-eval-callback -m $MODEL -p 'The capital of France is' -n 1 > ref.txt 2>&1
$B trace -m $MODEL -p 'The capital of France is' --dump ours.bin > ours.txt
python scripts/compare_eval_callback.py ref.txt ours.txt --dump ours.bin --report
```

`--report` prints worst error per tensor in trace order, which is how you find
where a divergence begins. Expect ~1e-3 per tensor: that is the floor for
independent Q8_0 implementations, not a defect. `CLAUDE.md` explains why.

## Tooling

Python tools need `~/.venvs/inferredthoughts/bin/python` (numpy, pyyaml, tqdm).

```bash
python scripts/compare_gguf_dump.py <gguf> --bin $B     # stage 1 acceptance
python scripts/dump_fixtures.py --model <gguf>          # dequant fixtures
python scripts/dump_tokenizer_fixture.py --model <gguf> # tokenizer fixtures (~6 min)
python scripts/gen_unicode_tables.py                    # after a llama.cpp update
python scripts/check_q8_matmul.py                       # isolate the Q8_0 matmul
```

## Known limits

- `qwen3` only; no MoE, no GatedDeltaNet. Quant support covers what the three
  target models use: F32, F16, BF16, Q8_0, Q5_K, Q6_K, IQ4_XS
- The CUDA backend runs the whole model on the GPU at 0.68x our own CPU
  backend. What remains is launch overhead, not kernels — see above. Only Q8_0
  has a matmul kernel, which is every matmul in the models this targets
- CUDA and `trace` are not usable together: tracing reads intermediate tensors
  from the host, and the device backend leaves them on the card unless the
  model asks for them back. `trace` runs on `naive`
- Greedy sampling only
- Scalar f32 kernels: no SIMD, no GPU, both deliberate. `spin` threads every
  matmul and attention (`-t N`) and scales out to 8 threads
- No hand-written SIMD. What vectorization exists is LLVM's, unlocked by
  `-C target-cpu=native` in `.cargo/config.toml`. Attention is 1.23x behind
  llama.cpp per position, the dense path 1.13x per token
- The `spin` pool busy-waits. It yields after a bounded spin, but it is built
  for a CLI that generates continuously, not a server that idles
- `par` (rayon) is kept only as the control that demonstrates the dispatch
  finding; `spin` supersedes it
