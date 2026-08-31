# inferredThoughts

A from-scratch Rust inference engine for GGUF models. Built toward serving an
MoE model whose weights do not fit in VRAM — see `HANDOFF.md` for why, and
`CLAUDE.md` for how.

**Status: v0.1.** Stages 1–6 complete, the `qwen35` architecture decoded, and
the forward pass running on **both CPU and GPU**. Loads a GGUF, tokenizes
exactly like llama.cpp, and generates coherent text on Qwen3-0.6B at **~59
tok/s** decode on the CPU — within **1.09–1.15x** of llama.cpp on the same
CPU, with scalar f32 kernels and results bit-identical to a single-threaded
oracle. KV cache, prefill/decode split, threaded attention, a spin-waiting
thread pool, a chat template, and a profiler. 116 tests by default, 120 with
`--features cuda`, plus 19 that need a model or a device.

The CUDA backend is **correct and slow on purpose** — 14.2 tok/s against the
CPU's 62.6 and llama.cpp's 337.5 on the same card. See "The CUDA backend"
below for what that measured.

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
| **v0.1** | **14.2 tok/s**<br>*on the GPU* | + `cuda` — all eight ops as kernels | weights resident in VRAM, KV mirrored on device, `--backend cuda` | 120 tests + 19 device/model-backed; five kernels **bit-identical** to `naive` through the whole model | **The seam costs 25.4 ms of a 70.4 ms token.** A round trip through `Ops` is 56.4 us and a decode step makes ~450 of them, so 36% of the token is spent before any arithmetic. Same shape as v3's rayon finding, one layer down |
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
$B generate -m $MODEL -p "..." -n 64 --backend cuda
```

The whole forward pass runs on the GPU: all eight `Ops` methods as kernels,
weights uploaded once and kept in VRAM, the KV cache mirrored on device and
appended to rather than resent. Qwen3-0.6B Q8_0, 64 tokens, median of three:

| | tok/s | ms/token | effective |
|---|---|---|---|
| ours, `cuda` | **14.20** | 70.4 | 9.1 GB/s |
| ours, `spin`, 8 threads | 62.59 | 16.0 | 39.9 GB/s |
| llama.cpp, CUDA, `-ngl 99` | **337.54** | 3.0 | 214 GB/s |

**Slower than our own CPU backend, and that is the finding.** llama.cpp reaches
48% of the card's 448 GB/s at batch 1; we reach 2%.

The cause was measured, not guessed. At context 128 — where attention should be
nearly free — the attention half is 66% of decode time, and the ratio between
the two halves tracks their *op counts*, not their bytes. So the fixed per-call
cost was timed directly:

| | |
|---|---|
| `cuLaunchKernel` + `cuCtxSynchronize` | 18.7 us |
| the same wrapped in 4 KiB H2D + D2H | **56.4 us** |
| ~450 `Ops` calls per decode step, so the floor is | **25.4 ms/token** |

36% of the token is spent before any arithmetic. No kernel work recovers it:
the `Ops` seam takes host slices and returns host slices, so every op is a round
trip by construction. This is v3's rayon result one layer down — a fixed
per-dispatch cost found by measuring the dispatch instead of the payload.

The remaining 64% is kernels written for exactness rather than for the machine:
one thread per output row in the matmul (uncoalesced by construction) and a
single-threaded f64 reduction in RMSNorm.

### Exactness

| kernels | against `naive` |
|---|---|
| `matmul`, `rms_norm`, `rms_norm_heads`, `rope_neox`, `add_assign` | **bit-identical** |
| `softmax`, `silu_mul`, `attend` | ≤ 5.2e-8, about one ulp |

The three that differ all call `expf`, and CUDA's is not obliged to match
glibc's. The obstacle was not the expected one: *parallel reductions* were
assumed to be what costs bit-exactness, but keeping each reduction serial kept
it — what actually breaks exactness is one libm function.

Whole-model divergence is 1.47e-2 of logit magnitude, which is what `CLAUDE.md`
predicts for a one-ulp perturbation amplified through 28 layers of Q8_0
re-quantization. Greedy decoding does flip a token within a dozen.

Rather than argue about whether that was drift or a defect, there is a test for
it: `only_the_expf_ops_diverge` runs the five exact kernels on the GPU and the
three exp-dependent ones on the CPU and asserts **bit equality on all 151,936
logits**. It passes, which leaves `expf` as the only explanation — and makes any
future GPU divergence bisectable in one run.

```bash
cargo test --release --features cuda --test cuda_ops -- --ignored --nocapture
```

**This tests none of the offload thesis.** The 0.6B is 0.59 GiB against 14.80
GiB free. It bought the device path, the kernels, and the number above.

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
- The CUDA backend is a *correctness* vehicle. It runs the whole model on the
  GPU and is 4.4x slower than our own CPU backend, for a measured reason — see
  below. Only Q8_0 has a matmul kernel, which is every matmul in the models
  this targets
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
