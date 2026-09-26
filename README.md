# inferredThoughts

A Rust + CUDA inference engine for mixture-of-experts models, streaming their
experts from the SSD.

## Why

RAM prices shot up. Running a large model the usual way means holding all of
it in VRAM and RAM, and that memory is the expensive part. In a
mixture-of-experts model each token uses only a few experts, so most of the
model can stay on an NVMe SSD and be read when it is needed. The goal: good
large models, at acceptable speed, on reasonably priced hardware.

Today that means **Qwen3.8-Flash-Next — 176.9B parameters, a 119 GiB file — at
~9 tokens a second on a 16 GB RTX 5060 Ti with 32 GB of RAM**, with about 83%
of the file left on the SSD. How it works:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Tested so far

Two models, on one machine: RTX 5060 Ti 16 GB, Ryzen 7 9700X, 32 GB DDR5,
Gen5 NVMe.

| model | GGUF | prefill tok/s | decode tok/s |
|---|---|---:|---:|
| Qwen3.8-Flash-Next, NVFP4, 176.9B | [119 GiB](https://huggingface.co/CompiledThoughts/Qwen3.8-Flash-Next-NVFP4-Q8_0) | 49.2 | **9.06** (best turn 10.40) |
| Qwen3.6-35B-A3B, NVFP4 | [19.1 GiB](https://huggingface.co/CompiledThoughts/Qwen3.6-35B-A3B-NVFP4-Q8_0-it) | 591.2 | **47.3** |

Native Windows 11, default settings. Prefill is a 5,548-token prompt; decode is
a chat turn, and on the 176.9B it falls as a conversation grows. On the same
machine under WSL2, llama.cpp's best configuration decodes the 176.9B at 5.3
tok/s, against 7.5–7.8 for this engine there.

**Only tested on:**
- **Windows 11 (native) and WSL2 (Ubuntu 24.04).** Native Linux is not tested.
- **NVIDIA RTX 50-series / RTX PRO Blackwell (`sm_120`).** The kernels are
  built for `sm_120a`; older GPUs are not supported yet.

## What you need

- An RTX 50-series or RTX PRO Blackwell GPU; NVIDIA driver R570+; CUDA 12.8+
- Rust (stable). On Windows: VS 2022 Build Tools, building from the
  *x64 Native Tools Command Prompt*
- For the 176.9B: 32 GB of RAM and a fast local NVMe. Under WSL, keep models
  on ext4 (`~/models`), not `/mnt/c`.

## Build and run

```bash
cargo build --release --features cuda
./target/release/inferred serve -m Qwen3.8-Flash-Next-NVFP4-Q8_0.gguf --backend cuda --port 8080 --ctx 8192
```

On Windows the binary is `target\release\inferred.exe`. Open
`http://127.0.0.1:8080/` for the chat page; any OpenAI-compatible client can
use `http://127.0.0.1:8080/v1`.

`--ctx` reserves the KV cache up front, out of the VRAM the experts would use:
keep it as small as the session needs. If decode slows sharply at a large
`--ctx`, raise `--expert-host` (pinned-RAM expert tier, 6 GiB by default).

Tests: `cargo test --release --features cuda` (178, no GPU needed).

## Limits and next

- Tested on two models, one GPU family, Windows and WSL2 only.
- Greedy decoding only; no tool calling yet.
- Next: older NVIDIA GPUs, native Linux, more models.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
