# inferredThoughts

**One card, the whole model.** A 176.9-billion-parameter mixture-of-experts
model, answering at **9–10 tokens a second on a 16 GB consumer GPU** — not a
distilled version of it, that model, 119 GiB of it, with three quarters of its
experts still sitting on the SSD while it talks to you.

It works by treating VRAM, pinned RAM and the NVMe drive as one memory
hierarchy, and deciding token by token which experts belong where. On this
machine 14% of the experts live in VRAM at any moment, and they serve **90.5%
of every expert read** — because the ones the router keeps asking for are the
ones that stay.

```
$ inferred serve -m Qwen3.8-Flash-Next-NVFP4-Q8_0.gguf --backend cuda --port 8080

  ████ █  █ ████ ████ ███  ███  ████ ███
   ██  ██ █ █    █    █  █ █  █ █    █  █
   ██  █ ██ ███  ███  ███  ███  ███  █  █
   ██  █  █ █    █    █ █  █ █  █    █  █
  ████ █  █ █    ████ █  █ █  █ ████ ███

  ████ █  █  ██  █  █  ███ █  █ ████  ███
   ██  █  █ █  █ █  █ █    █  █  ██  █
   ██  ████ █  █ █  █ █ ██ ████  ██   ██
   ██  █  █ █  █ █  █ █  █ █  █  ██     █
   ██  █  █  ██   ██   ███ █  █  ██  ███

                        by compiledthoughts.dev

model Qwen3.8-Flash-Next-NVFP4-Q8_0 | qwen4exp | 48 layers | ctx 32096
device NVIDIA GeForce RTX 5060 Ti | 14.80 of 15.93 GiB free
  chat here  http://127.0.0.1:8080/         <- open it in a browser

  prefill      80 tok     3384.4 ms      23.64 tok/s
  decode      110 tok    10572.1 ms      10.40 tok/s     96.1 ms/tok
  placed at load: 10,247 in VRAM, 6,984 pinned in RAM, 56,497 on disk
```

**Why bother, when you could just buy more RAM?** Because holding that expert
pool in DDR5 costs around $1,600, and holding it on a Gen5 SSD costs about $6 —
and the SSD is only 1.4x more expensive per byte actually moved. A machine built
to serve this class of model the usual way runs to roughly $9,000. This one is
about $1,700, and most of that is not the GPU.

**Where it is going:** 15–20 tok/s on a 120B-class model, then 250B and 500B on
the same 16 GB card.

**What it is not:** a general engine. Four model architectures, one GPU family
(consumer Blackwell), and only the quantization formats those models actually
use. llama.cpp runs everything, everywhere, and is faster on the 35B — it is the
reference every kernel here was checked against, and the honest comparison is
below, including where we lose.

| | |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | how it works, and what is still unfinished |
| [`src/model/*.md`](src/model/) | one living reference per architecture, beside its code |

## What runs

On one RTX 5060 Ti (16 GB) with 32 GB of system RAM, native Windows:

| model | file | of which PLE | VRAM | pinned RAM | from SSD | prefill | decode |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next** 176.9B | 119.0 GiB | 50.7 GiB | 13.6 GiB | 6.0 GiB | **76.6%** of experts, 271 MiB/token | 49.2 | **9.06** |
| Qwen3.6-35B-A3B NVFP4 | 19.1 GiB | — | 13.5 GiB | 5.8 GiB | none | 591.2 | 47.3 |
| Qwen3.6-35B-A3B IQ4_XS | 17.5 GiB | — | 13.2 GiB | 5.0 GiB | none | 668.8 | 41.1 |
| Qwen3-0.6B Q8_0 | 0.6 GiB | — | 1.5 GiB | none | none | 2,659.8 | 292.8 |

The 125B's **best observed turn is 10.40 tok/s** — 110 tokens at shallow depth,
through the chat page. The 9.06 in the table is the measured run below, and the
gap between them is real: decode on this model falls as a conversation grows.

**One machine, one run each, 24-09-2026**, native Windows at default budgets:
tok/s from the engine's own profile, tiers from its placement report. Prefill is
a 5,548-token prompt (24 tokens for the 125B's row would measure per-pass
overhead, so it uses the same prompt); decode is a 128-token chat turn, which is
why the 35B reads 34 here and ~41 on the 19,706-token standard run — depth and
turn length both move it. "Pinned RAM" is the page-locked expert tier; Windows
also mirrors VRAM in system memory, so the process peaks higher (19–22 GB on the
three large models).

### The 176.9B model, against llama.cpp on the same machine

```
 ours, best observed turn  ██████████████████████████████████████ 10.40
 ours, native Windows      █████████████████████████████████     9.06
 ours, WSL2                ███████████████████████████           7.45–7.79
 llama.cpp CUDA, best fit  ███████████████████                   5.32–5.36
 llama.cpp, all experts    ████████████████                      4.36–4.42
   on the CPU
 llama.cpp, CPU only       ██████████                            2.88
                           └─────────┴─────────┴─────────┴─────────┴─ tok/s
                           0         2.5       5.0       7.5      10.0
```

llama.cpp's figures were taken under WSL2 on 16-09 on the same file, prompt and
machine, searching its options for the best fit it could reach (`--n-cpu-moe 42`
won). Ours under WSL2 is the like-for-like row; native is higher. **Decode on
this model falls with conversation length** — 8.02 tok/s at 160 tokens, 6.45 at
1,635 — so a number without its length is not a number.

Where we are behind: the **35B's prefill and decode**, at roughly 0.6x and 0.8x
of llama.cpp on the same file. This engine is narrow by design — four
architectures, one GPU family — and llama.cpp is not.

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

**Every default has a switch that undoes it**, because that is how each one was
measured: `INFERRED_Q5K_SCALAR`, `INFERRED_Q6K_SCALAR` and `--iq4-scalar` leave
the tensor cores; `INFERRED_ATTN_F32` and `--rms-serial` restore the exact
paths; `INFERRED_NVFP4_Q8` restores NVFP4's bit-exact arithmetic;
`INFERRED_FETCH_THREADS`, `INFERRED_FETCH_DIRECT`, `INFERRED_PREFETCH` and
`INFERRED_ASYNC_UPLOAD` control the tier-3 read path;
`INFERRED_RAM_HEADROOM_GIB` sets how much memory is left for the rest of the
machine. `inferred --help` lists the flags.

## Test

```bash
cargo test --release --features cuda                 # 178, no model or device needed
cargo test --release --features cuda -- --ignored --test-threads=1   # 98 more
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
