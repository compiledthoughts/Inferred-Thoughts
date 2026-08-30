# inferredThoughts

A from-scratch Rust inference engine for GGUF models. Built toward serving an
MoE model whose weights do not fit in VRAM — see `HANDOFF.md` for why, and
`CLAUDE.md` for how.

**Status:** Stages 1–5 complete. Loads a GGUF, tokenizes exactly like
llama.cpp, and generates coherent text at ~27 tok/s on Qwen3-0.6B. KV cache,
prefill/decode split, and a profiler. 91 tests, plus 8 more that need a model
on disk. CPU only.

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
# generate text (greedy)
$B generate -m ~/models/Qwen3-0.6B-Q8_0.gguf -p "The capital of France is" -n 20

# inspect a model: metadata, tensor table, per-type summary
$B inspect ~/models/Qwen3-0.6B-Q8_0.gguf

# machine-readable, for diffing
$B inspect ~/models/Qwen3-0.6B-Q8_0.gguf --json
```

Only the `qwen3` architecture loads today. `Qwen3.5-9B` (`qwen35`) is rejected
with a named error — it needs GatedDeltaNet.

## Test

```bash
cargo test
```

Model-backed tests skip if the model is missing, but the suite fails if *every*
one skips. Override the search path with `INFERRED_MODEL_DIR`.

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
- Attention scoring is the bottleneck past a few hundred tokens: 64% of decode
  time at 384 tokens, and ~10x less efficient per byte than the matmul path
