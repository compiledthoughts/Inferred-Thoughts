# inferredThoughts

A Rust + CUDA inference engine for mixture-of-experts models, streaming their
experts from the SSD.

It runs **Qwen3.8-Flash-Next — 176.9B parameters, a 119 GiB file — at ~9 tokens
a second on a 16 GB RTX 5060 Ti with 32 GB of RAM.** About 99 GiB of that file,
83%, stays on the NVMe drive. VRAM, pinned RAM and the SSD work as one memory
hierarchy: the experts the router keeps asking for stay in VRAM, and the rest
are read from the SSD as they are picked.

- One binary: Rust host, CUDA kernels written here, compiled to PTX and embedded.
  No Python, no PyTorch, no `libllama`. A built binary needs only the NVIDIA driver.
- An OpenAI-compatible server with a chat page built in.
- Checked against llama.cpp tensor by tensor; 178 tests run without a GPU, 98
  more with one.

How the SSD streaming works: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Numbers

RTX 5060 Ti 16 GB, Ryzen 7 9700X, 32 GB DDR5, Gen5 NVMe, native Windows 11,
default settings.

| model | file | in VRAM | pinned RAM | on SSD | prefill tok/s | decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| **Qwen3.8-Flash-Next** NVFP4, 176.9B | 119.0 GiB | 13.6 GiB | 6.0 GiB | 76.6% of experts | 49.2 | **9.06** |
| **Qwen3.6-35B-A3B** NVFP4 | 19.1 GiB | 13.5 GiB | 5.8 GiB | none | 591.2 | **47.3** |
| Qwen3.6-35B-A3B IQ4_XS | 17.5 GiB | 13.2 GiB | 5.0 GiB | none | 668.8 | 41.1 |
| Qwen3-0.6B Q8_0 | 0.6 GiB | 1.5 GiB | none | none | 2,659.8 | 292.8 |

- Prefill is a 5,548-token prompt. Decode is a chat turn: a 128-token turn for
  the 176.9B and the 0.6B, a turn through `serve` for the 35B.
- **The 176.9B's best turn so far is 10.40 tok/s.** Its decode falls as a
  conversation grows: 6.45 tok/s twenty turns into a session.
- **The 35B holds up in long sessions**: 41.0 tok/s at 36k tokens of context,
  through `serve`.
- Although 76.6% of the 176.9B's experts live on the SSD, about 90% of expert
  reads are served from VRAM. It reads ~270 MiB a token from the drive.

**Against llama.cpp on the same machine** (both under WSL2, same file and prompt):
on the 176.9B, llama.cpp's best configuration (`--n-cpu-moe 42`) decodes at
5.32–5.36 tok/s against our 7.45–7.79 there, and 9.06 natively. On the 35B,
llama.cpp is ahead: ~965 prefill and ~52 decode on the IQ4_XS file.

## Tested on, and what you need

| | tested | required |
|---|---|---|
| GPU | RTX 5060 Ti 16 GB | NVIDIA Blackwell (RTX 50-series, `sm_120`) |
| CPU | Ryzen 7 9700X (Zen 5) | any x86-64; the build targets the CPU it runs on. Intel untested |
| Driver / CUDA | 591.86 / CUDA 12.8 | R570+ / CUDA Toolkit 12.8+ |
| OS | Windows 11 native; WSL2 Ubuntu 24.04 | native Linux should work, but is untested |
| Build tools | Rust 1.98, VS 2022 Build Tools (MSVC 14.44) | Rust stable; on Windows, VS 2022 Build Tools (C++) |
| RAM | 32 GB DDR5 | 32 GB for the 176.9B, which peaks at ~22 GB |
| Disk | Gen5 NVMe | a fast local NVMe for the 176.9B — it reads from the file every token |

Keep models on a local disk. Under WSL, that means ext4 (`~/models`), never
`/mnt/c` and friends: the Windows-drive bridge is ~45x slower.

## Models

| model | GGUF |
|---|---|
| Qwen3.8-Flash-Next, NVFP4 experts, 176.9B | [CompiledThoughts/Qwen3.8-Flash-Next-NVFP4-Q8_0](https://huggingface.co/CompiledThoughts/Qwen3.8-Flash-Next-NVFP4-Q8_0) — 128 GB, one file |
| Qwen3.6-35B-A3B, NVFP4 experts | [CompiledThoughts/Qwen3.6-35B-A3B-NVFP4-Q8_0-it](https://huggingface.co/CompiledThoughts/Qwen3.6-35B-A3B-NVFP4-Q8_0-it) — 20.5 GB |

Also supported: Qwen3 and Qwen3.5 dense GGUFs (tested at 0.6B and 9B, Q8_0) and
the Qwen3.6-35B-A3B IQ4_XS GGUF. Other architectures are refused by name.

## Build

Windows, from an **x64 Native Tools Command Prompt for VS 2022** (`nvcc` needs
`cl.exe`, and a plain shell cannot find it):

```bat
cargo build --release --features cuda
```

Linux or WSL2: the same command. `build.rs` finds `nvcc` through `CUDA_PATH`,
`CUDA_HOME` or `/usr/local/cuda`. The build targets the CPU it is built on
(`.cargo/config.toml`).

## Run

The build puts the binary at `target/release/inferred` (`target\release\inferred.exe`
on Windows).

```bash
./target/release/inferred serve -m Qwen3.8-Flash-Next-NVFP4-Q8_0.gguf --backend cuda --port 8080 --ctx 8192
```

Open `http://127.0.0.1:8080/` for the chat page. Any OpenAI-compatible client
can use `http://127.0.0.1:8080/v1`.

One-shot, from the command line:

```bash
./target/release/inferred generate -m <gguf> --backend cuda --chat -n 256 -p "Explain MoE routing."
```

| flag | |
|---|---|
| `--backend cuda` | required for the GPU; the default is the CPU |
| `--ctx <N>` | context length. The KV cache is reserved up front and comes out of VRAM the experts would use, so use the smallest that fits the session |
| `--expert-host <GiB>` | pinned-RAM expert tier, 6 GiB by default |
| `--expert-cache <GiB>` | cap on the VRAM expert tier; sized automatically otherwise |

**A large `--ctx` can push experts onto the SSD without a warning**, and decode
drops sharply. If the 35B slows down at a long context, raise `--expert-host`:
at `--ctx 64096` on Windows it needed `--expert-host 10`. `inferred --help`
lists every flag.

Tests:

```bash
cargo test --release --features cuda                                  # 178, no GPU needed
cargo test --release --features cuda -- --ignored --test-threads=1    # 98, GPU + models
```

The `--ignored` tests look for models in `INFERRED_MODEL_DIR`.

## Known limits

- NVIDIA Blackwell only. On an older card, use llama.cpp.
- Greedy decoding only, no sampling.
- No tool calling yet: the server ignores `tools`.
- The KV cache is fp16.

## Coming

- **More models**: larger MoE models on the same 16 GB card — 250B, then 500B —
  and more architectures as they are needed.
- **More hardware**: smaller cards, down to 6 GB of VRAM; older NVIDIA
  generations; AMD and Apple. A direction, not a promise.
- Tool calling, and a live view of where each token's experts came from.

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). llama.cpp is the
behavioural reference every kernel was checked against.
