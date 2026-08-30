# inferredThoughts

A from-scratch Rust inference engine for GGUF models. Built toward serving an
MoE model whose weights do not fit in VRAM — see `HANDOFF.md` for why, and
`CLAUDE.md` for how.

**Status:** Stages 1–5 complete. Loads a GGUF, tokenizes exactly like
llama.cpp, and generates coherent text on Qwen3-0.6B at **~58 tok/s** decode,
within 1.16x of llama.cpp on the same CPU. KV cache, prefill/decode split,
threaded attention, a spin-waiting thread pool, and a profiler. 106 tests, plus
13 more that need a model on disk. CPU only.

## Progress

All figures are Qwen3-0.6B Q8_0 on the machine in `CLAUDE.md` — Ryzen 7 9700X,
8 cores, dual-channel DDR5. Decode is measured at short context unless noted.

| | decode | backend | added | state | key finding |
|---|---|---|---|---|---|
| **v0**<br>`065fc41` | ~2 tok/s<br>*quadratic in context* | `naive` — scalar f32, 1 thread | GGUF parser, dequant, tokenizer, `qwen3` forward pass | 71 tests; verified layer by layer against `llama-eval-callback` | RMSNorm must accumulate in f64, as ggml does — f32 shifts the scale enough to flip Q8_0 quants downstream. And ~1e-3 per tensor is the **floor** for independent Q8_0 implementations, not a defect |
| **v1**<br>`36ebcd8` | **27 tok/s**<br>*linear in context* | `naive` + `par` (rayon, LM head only) | KV cache (f16), engine with prefill/decode split, profiler, threading | 91 tests + 8 model-backed; decode proven **bit-identical** to full recompute; `par` ≡ `naive` bit for bit | Attention scoring is **18.3x** behind llama.cpp while dense matmuls are only **2.45x** behind. Threading at matmul granularity is a *net loss* — cost is per task, not per region |
| **v2** | **19.9 tok/s @ d384**<br>*24.0 @ d64* | + `Ops::attend`, threaded over kv heads | block-wise f16 conversion, `--chat`, `--show-special`, honest stop reasons | 94 tests + 11 model-backed; forward pass **byte-identical** to v1 | Attention growth cut **3.4x** (29.5 → 8.6 ms over 320 positions) with **zero** numeric change — f16→f32 is lossless, so hoisting it out of the dot product cannot move a bit |
| **v3** | **46.9 tok/s @ d384**<br>*58.4 @ d64* | + `spin` — persistent spin-waiting pool, every matmul threaded | `src/ops/pool.rs`, the crate's only `unsafe`; `naive` is now `#![forbid(unsafe_code)]` | 106 tests + 13 model-backed; bit-identical to `naive` at 2/3/5/8 threads | **A rayon parallel region costs ~430 µs here; a spin barrier costs 0.40 µs — 1088x.** Threading was never the problem, dispatch was. Decode bandwidth 12 → **37 GB/s** |

For scale, llama.cpp on the same CPU, same model, same 16-token prompt:
**568 t/s prefill, 65 t/s decode**. The shape of our gap to it matters more than
its size:

| depth | v1 | v2 | v3 | llama.cpp | v3 gap |
|---|---|---|---|---|---|
| d64 | 23.9 | 24.0 | **58.4** | 67.7 | 1.16x |
| d128 | 21.1 | 22.2 | **55.7** | 64.4 | 1.16x |
| d256 | 16.9 | 21.3 | **48.6** | 63.0 | 1.30x |
| d384 | 14.0 | 19.9 | **46.9** | 60.8 | **1.30x** |

v1's gap *widened* with context, 2.8x → 4.4x. v3's is 1.16–1.30x. Splitting the
curve into its constant and per-position terms: the **dense path is within
1.13x** of hand-tuned AVX-512, because at 37 GB/s both engines are limited by
DDR5 rather than by arithmetic. What remains is attention scoring, still 2.5x
behind.

llama.cpp is the oracle, not the thing to beat here — the real target is 40
tok/s on Qwen3.6-35B-A3B, which is where the offload policy actually matters.

## Build

Everything runs inside WSL (`Ubuntu-24.04`). Build artifacts go on ext4 rather
than DrvFs, which is much faster:

```bash
export CARGO_TARGET_DIR=~/.cargo-target/inferredthoughts
cargo build --release
export B=~/.cargo-target/inferredthoughts/release/inferred
```

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
with a named error — it needs GatedDeltaNet.

**There is no chat template yet.** A raw prompt runs in completion mode, so an
instruct model never enters the assistant turn, never emits `<|im_end|>`, and
degenerates into repetition. Write the markers yourself — `parse_special` is on,
so they tokenize as special tokens:

```bash
P=$(printf '<|im_start|>user\nList the capitals of 10 countries<|im_end|>\n<|im_start|>assistant\n'; echo X); P=${P%X}
$B generate -m $MODEL -p "$P" -n 500
```

The `; echo X` guard matters: plain `$(...)` strips the trailing newline, which
gives a 15-token prompt instead of 16 and a completely different answer.

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
recompute, and that `par` reproduces `naive` bit for bit.

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

- `qwen3` only; no MoE, no GatedDeltaNet
- Greedy sampling only
- Scalar f32 kernels: no SIMD, no GPU, both deliberate. Threading exists
  (`-t N`) but only the LM head is large enough to pay for it — 1.18x. See
  `PARALLEL_THRESHOLD` in `src/ops/par.rs` for the measurements.
- Attention scoring is still **2.5x** behind llama.cpp per position; the dense
  path is within **1.13x**, so that is where the remaining work is
- The `spin` pool busy-waits. It yields after a bounded spin, but it is built
  for a CLI that generates continuously, not a server that idles
- `par` (rayon) is kept only as the control that demonstrates the dispatch
  finding; `spin` supersedes it
