# How the SSD streaming works

A mixture-of-experts model is mostly experts, and each token uses only a few of
them. Qwen3.8-Flash-Next has 512 experts in each of its 48 layers and picks 10
per layer per token. Its experts are 63.3 GiB; the card has 16 GB. So the
experts live in three tiers, and the engine decides token by token which ones
belong where.

```
  VRAM slab     ~400 GB/s   fixed slots, one expert each, a cache with GCLOCK eviction
     ▲
  pinned RAM     ~26 GB/s   page-locked host memory, read by the GPU across PCIe
     ▲
  NVMe (GGUF)     ~5 GB/s   everything else, read from the model file when picked
```

## At load

The dense weights and the KV cache go into VRAM first. The expert slab takes
most of what is left, then the pinned tier fills up to its budget (6 GiB by
default, never more than available RAM less 1 GiB). Every other expert stays in
the file. For the 176.9B at default settings:

| VRAM | pinned RAM | on SSD |
|---:|---:|---:|
| 10,247 | 6,984 | 56,497 |

expert tensors, of 73,728. The file is read in place: GGUF already aligns its
tensors, so there is no repacked copy on disk. The model's 50.7 GiB n-gram table
also stays in the file; each token reads 16 of its rows.

## Each token, each layer

1. **The router picks** 10 experts.
2. **In VRAM:** used directly.
3. **In pinned RAM:** the kernel reads the weights across PCIe itself. No copy,
   no host involvement.
4. **On the SSD:** the layer's cold experts are read as one batch by 8 threads,
   each with its own file handle, into page-locked staging. Each upload into a
   slab slot is queued rather than waited on. The slot's previous expert is
   evicted by GCLOCK; its table entry is repointed first, and experts the
   current layer is using cannot be evicted.
5. **Lookahead:** the next layer's router is run early on this layer's output,
   so the experts it will want are already being read while the GPU works.

Reads are unbuffered (`O_DIRECT`) on Linux. On Windows they are buffered,
because while a file is memory-mapped, Windows serves unbuffered reads of it at
~3.0 GB/s against ~5.3 buffered.

## What it achieves

On the 176.9B, with 14% of the experts in VRAM, **about 90% of expert reads are
served from VRAM**, and the drive supplies ~270 MiB a token. The cache adapts
as a conversation moves: the best *fixed* choice of the busiest experts would
serve 81%.

## Correctness

Where an expert comes from never changes the answer. The acceptance test
generates the same prompt at full budgets and at tier caps that force most
experts to stream from disk, and requires identical tokens. Every kernel is
checked against llama.cpp and against a scalar reference implementation kept
in the engine.

## What is next

- **Fewer bytes per expert.** VRAM is the only tier that makes decode faster,
  and it is full; a smaller expert format fits more of them.
- **Less host time per token.** Bringing the router's picks back to the host
  each layer is the largest single host cost, ~41 ms of a ~118 ms token.
- **One slot per expert.** Gate, up and down of an expert placed together: one
  lookup and one read instead of three.

Source comments cite internal design notes (`SSD-TIER.md`, `qwen4exp.md` and
others) that are not published; the code and its tests are the record here.
