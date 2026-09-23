# How it works

A Rust engine that runs GGUF mixture-of-experts models whose weights do not fit
in VRAM. One binary, CUDA kernels written here, no PyTorch and no `libllama`.

The interesting part is not the arithmetic — it is **where each expert lives and
when its bytes move**. Everything below serves that.

---

## The shape

```
  GGUF file  ─►  model  ─►  Engine  ─►  Ops (the seam)  ─►  naive | spin | cuda
   mmap          arch       KV cache,       one trait          scalar  threads  GPU
   metadata      decode     batching,       per kernel
   tensors       + weights  sampling
```

- **`gguf`** reads the file: metadata, the tensor index, and a memory map. Every
  architectural constant comes from the file, never from a table in the code.
- **`model`** decodes one architecture per file (`src/model/<arch>.rs`), with a
  living reference beside it (`src/model/<arch>.md`) describing the forward pass
  as the reference implementation performs it.
- **`engine`** owns the pass: prefill in batches, decode a token at a time, the
  KV cache, the recurrent state, and the profile counters.
- **`ops`** is the seam. One trait, one method per kernel. Everything above it
  is backend-agnostic; everything below is a backend.

## The backends, and why there are four

| | what it is | why it exists |
|---|---|---|
| `naive` | scalar f32, single thread, no `unsafe` | **the oracle.** Never optimised. Everything else is checked against it |
| `par` | rayon | superseded, kept as the control for a measured dispatch finding |
| `spin` | `naive`'s kernels on a spin-waiting pool | the CPU engine |
| `cuda` | driver API, PTX compiled by `build.rs` and embedded | the GPU engine |

**A backend that only redistributes work must reproduce the oracle bit for
bit.** Parallel kernels are parallel over independent outputs, so each
accumulation keeps its serial order. Where that is impossible — tensor cores
fold differently from a scalar loop — the departure needs three things: a
tolerance derived from the arithmetic, a switch that restores the exact path,
and a test proving the switch works.

## The expert tiers

An MoE model's routed experts are most of its bytes and only a fraction are
touched per token. They live in three places:

```
  VRAM slab  ─ one allocation, cut into slots, one expert each, evicted by GCLOCK
  host tier  ─ page-locked RAM, read across PCIe by the kernel itself
  the file   ─ everything else, fetched on demand when a layer picks it
```

- **Every expert is addressable at all times.** A CUDA graph replays a fixed
  sequence with no host involvement, so a miss cannot be serviced mid-token; an
  expert that is not resident still has a valid device address, in the host tier
  or after a fetch.
- **The slab is a cache, not storage.** Slots are reused; the victim's table
  entry is repointed before its slot is overwritten, and the experts a layer is
  currently resolving are leased so they cannot be evicted underneath it.
- **Tier 3 fetches in parallel** — several threads, each with its own file
  handle, into page-locked staging, unbuffered on Linux and buffered on Windows
  (where an open memory mapping of a file makes unbuffered reads of it slower).
- **Lookahead prefetch** runs the next layer's router on this layer's output, so
  the experts it will want are read while the GPU is still busy.

The policy is measured rather than asserted: on a 176.9B model with 14% of its
experts in VRAM, **90.5% of expert reads are served from VRAM** — ahead of the
best fixed placement of the busiest experts, which serves 75.6%.

## What the engine refuses to do

- **Never materialise an f32 copy of a weight.** Matmuls walk quantized blocks.
- **Never hand the seam a sub-slice of an activation buffer.** A device backend
  keys its mirrors on the host address, so `&x[k..]` is an address it has never
  seen; there are explicit gather and scatter ops for that.
- **Never change a decode pass's kernel sequence with depth.** Decode replays a
  recorded graph; anything depth-dependent is a kernel argument.
- **Never invent a constant.** Read it from the file or from the reference
  implementation's source, cite it, and fail loudly when a key is missing.

## How correctness is established

1. **Fixtures from the reference.** Every numeric kernel has a unit test against
   values dumped from llama.cpp before it is used anywhere.
2. **Differential tests between backends.** Bit equality where the arithmetic
   allows it, and a named, switchable tolerance where it does not.
3. **Whole-model comparison.** `inferred trace` checksums every intermediate
   tensor for a prompt, and a script diffs that against
   `llama-eval-callback`'s. Agreement of ~1e-3 per tensor at Q8_0 is the
   quantization floor, not slack.
4. **Acceptance tests.** The same prompt generates identical tokens at full
   budgets and at tier caps that force most experts to stream from disk.

## The server

`inferred serve` speaks the OpenAI chat API, and serves a chat page from the
binary itself at `/`. Two things beyond the protocol:

- **Conversation reuse.** The session records the exact text the engine has
  consumed. A turn that extends it prefills only the difference; a turn that
  diverges returns to the nearest checkpoint rather than to zero.
- **Reasoning is separated.** A reasoning model's `<think>…</think>` is streamed
  as `reasoning_content`, the answer as `content`, and the markers never reach
  the client.

---

## Work in progress

Honest about what is not done, in the order it is likely to be tackled.

- **Where the token goes, natively.** The decomposition that drives the work —
  roughly half GPU kernels, half host path, barely overlapping — was measured
  under WSL before two changes that moved both halves. The largest single host
  item is the router's picks coming back to the host, ~41 ms of a ~118 ms token.
- **Fewer bytes per expert.** Capacity on a 16 GB card is exhausted; two
  independent lines of measurement end at "only VRAM residency helps", and the
  way to more residency is a smaller expert quant.
- **A fused expert slot.** Gate, up and down of one expert in one slot: one
  table and one gather instead of three, the expert placed as a unit.
- **Tool calling.** The server accepts `tools` and ignores it, so a client that
  expects function calls gets prose. The models' own template is Hermes-style
  (`<tools>` in the system block, `<tool_call>` in the reply), which is what
  should be implemented.
- **A UI beyond chat.** Live tier and memory figures — how much came from VRAM,
  from RAM, from the SSD — and sliders for the context and tier caps. The engine
  reports VRAM today; host memory is not exposed yet.
- **Wider hardware.** Kernels are built `sm_120a`: consumer Blackwell only,
  because the FP4 tensor cores are the point. Older cards would run scalar
  fallbacks and be a slower llama.cpp, which already serves those users well.
- **Native Linux** should work unchanged — every WSL mention in the source is a
  comment rather than a branch — but that is untested.
- **Prefill on the 125B** is ~50 tok/s and has had no attention at all.
