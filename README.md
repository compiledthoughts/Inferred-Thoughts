# inferredThoughts

A from-scratch Rust inference engine for GGUF mixture-of-experts models whose
weights do not fit in VRAM. One binary, CUDA kernels of our own, no PyTorch and
no `libllama`.

The contribution is **tiering and placement policy** — which experts live in
VRAM, which in pinned host memory, which stream from the SSD, and when bytes
move — measured rather than claimed.

| | |
|---|---|
| [`HANDOFF-v3.md`](HANDOFF-v3.md) | why the project exists, where it stands, what is open |
| [`BENCHMARKS-v3.md`](BENCHMARKS-v3.md) | current numbers, each with its method |
| [`SCOREBOARD.md`](SCOREBOARD.md) | one row per landed change |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | module map, the `Ops` seam, where exactness stops |
| [`TIERS.md`](TIERS.md), [`SSD-TIER.md`](SSD-TIER.md) | measured tier costs; the tier-3 decision log |
| [`src/model/*.md`](src/model/) | one living reference per architecture |
| [`docs/v1/`](docs/v1/), [`docs/v2/`](docs/v2/) | the history to v0.3.0 and to v0.8, frozen |

## What runs

On one RTX 5060 Ti (16 GB) with 32 GB of system RAM, native Windows:

| model | file | of which PLE | VRAM | pinned RAM | from SSD | prefill | decode |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next** 176.9B | 119.0 GiB | 50.7 GiB | 13.6 GiB | 6.0 GiB | **76.6%** of experts, 271 MiB/token | 49.2 | **9.06** |
| Qwen3.6-35B-A3B NVFP4 | 19.1 GiB | — | 13.5 GiB | 5.8 GiB | none | 591.2 | 34.3 |
| Qwen3.6-35B-A3B IQ4_XS | 17.5 GiB | — | 13.2 GiB | 5.0 GiB | none | 668.8 | 34.1 |
| Qwen3.5-9B Q8_0 | 9.1 GiB | — | 8.2 GiB | none | none | 281.6 | 40.7 |
| Qwen3-0.6B Q8_0 | 0.6 GiB | — | 1.5 GiB | none | none | 2,659.8 | 292.8 |

**One machine, one run each, 24-09-2026**, native Windows at default budgets:
tok/s from the engine's own profile, tiers from its placement report. Prefill is
a 5,548-token prompt (24 tokens for the 125B's row would measure per-pass
overhead, so it uses the same prompt); decode is a 128-token chat turn, which is
why the 35B reads 34 here and ~41 on the 19,706-token standard run — depth and
turn length both move it. "Pinned RAM" is the page-locked expert tier; Windows
also mirrors VRAM in system memory, so the process peaks higher (19–22 GB on the
three large models). Method and the rest in
[`BENCHMARKS-v3.md`](BENCHMARKS-v3.md).

Under WSL2 the same models run 3–20% slower, except the 125B at 7.5–8.0.
On that WSL footing, llama.cpp CUDA at its best measured fit on this machine
read 5.32–5.36 tok/s on the same model, prompt and day (16-09). The 35B is
behind llama.cpp on both prefill and decode; the gap is in `HANDOFF-v3.md`.

## Prerequisites

- NVIDIA Blackwell GPU (RTX 50-series, `sm_120`)
- NVIDIA driver R570 or newer
- CUDA Toolkit 12.8 or newer
- Rust, current stable
- Windows 10/11: Visual Studio 2022 Build Tools (C++), building from the x64 Native Tools prompt
- Linux or WSL2: Ubuntu 24.04

Models live on a local disk — NTFS on Windows, ext4 on Linux. Never a network
share, and under WSL never `/mnt/*`.

## Build

Windows, from an *x64 Native Tools Command Prompt for VS 2022* (`nvcc` runs
`cl.exe` even to produce PTX):

```bat
cargo build --release --features cuda
```

Linux or WSL:

```bash
cargo build --release --features cuda
```

CUDA is off by default, so the crate builds and tests without a toolkit.
`build.rs` finds `nvcc` through `CUDA_PATH`, `CUDA_HOME` or `/usr/local/cuda`,
compiles `kernels/kernels.cu` to `sm_120a` PTX, embeds it in the binary, and
links the driver API — no `libcudart`, no wrapper crate. **A built binary needs
only the NVIDIA driver.** `INFERRED_SM_ARCH=sm_120` builds forwards-compatible
PTX without the NVFP4 kernels.

`.cargo/config.toml` sets `-C target-cpu=native`, worth 2x on attention and
bit-identical.

## Run

`--chat` is not optional for an instruct model: a raw prompt runs in completion
mode and degenerates into repetition.

```bash
inferred generate -m <gguf> --chat --backend cuda -n 256 -p "Explain MoE routing."
inferred generate -m <gguf> --chat --backend cuda -n 256 --ctx 32768 --profile
inferred serve    -m <gguf> --backend cuda --port 8080 --ctx 8192 -v
inferred inspect  <gguf>
```

`serve` is OpenAI-compatible, so any existing chat client can drive it.

Four commands: `generate`, `serve`, `inspect`, and `trace` (one forward pass,
checksumming every tensor, for diffing against `llama-eval-callback`).
`--help` prints the full flag list from the source, so it cannot drift.

The flags that matter:

| | |
|---|---|
| `--backend cuda` | `spin` (CPU) is the default; `-t 1` runs the scalar oracle |
| `--ctx <N>` | KV positions, **allocated up front** at 28.5 KiB each, taken from the expert slab. Use the smallest the session needs |
| `--expert-cache <GiB>`, `--expert-host <GiB>` | tier caps. The VRAM slab is sized automatically otherwise |
| `--batch <N>` | prompt tokens per pass (512). Uncapped, a 10k prompt holds ~4 GiB |
| `--profile`, `--profile-detail`, `--profile-kernels`, `--profile-device` | timings, per-kernel shares, launch attribution |

**Generated text goes to stdout; everything else to stderr**, so a plain
redirect captures the payload alone. Failure exits non-zero with one
`caused by:` line per level.

Environment switches — quants, attention, fetch behaviour — are listed in
`CLAUDE.md` under *Defaults and their switches*; each restores a slower or
exact path and exists so a default can be A/B'd.

## Test

```bash
cargo test --release --features cuda                 # 174, no model or device needed
cargo test --release --features cuda -- --ignored --test-threads=1   # 96 more
```

The `--ignored` tests need a device and the models; point
`INFERRED_MODEL_DIR` at them. Device tests run serially — concurrent CUDA
contexts fault intermittently.

Correctness is differential, not approximate. `naive` (scalar f32, single
thread, no `unsafe`) is the oracle; every other backend must reproduce it bit
for bit wherever it only redistributes work. A precision departure needs a
tolerance derived from the arithmetic, a switch that restores the exact path,
and a test proving the switch works.

## Verifying against llama.cpp

llama.cpp is the behavioural oracle. Build it CPU-only, then:

```bash
llama-eval-callback -m <model> -p '<prompt>' -n 1 > ref.txt 2>&1
inferred trace -m <model> -p '<prompt>' --dump ours.bin > ours.txt
python scripts/compare_eval_callback.py ref.txt ours.txt --dump ours.bin --report
```

Per-tensor agreement is ~1e-3 at Q8_0, which is the quantization floor: Q8_0 is
a step function, so a one-ulp input difference flips a quant.

## Known limits

- **Blackwell only.** The kernels are built `sm_120a`. On an older card, use llama.cpp.
- **Quants are those the target models use**: F32, F16, BF16, Q8_0, Q5_K, Q6_K, IQ4_XS, NVFP4.
- **The KV cache is fp16**, and `--ctx` reserves it all at start-up.
- **Greedy only** — no sampling, no beam search.
- Architectures: `qwen3`, `qwen35`, `qwen35moe`, `qwen4exp`. Anything else is refused by name rather than guessed at.
