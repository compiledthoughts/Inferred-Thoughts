#!/usr/bin/env python3
"""Write a copy of a GGUF with some tensors removed from its index.

**Why this exists.** llama.cpp's `qwen35moe` loader at `ac9338c34` creates no
output-head scale, so it refuses the NVFP4 GGUF our converter writes:
`done_getting_tensors: wrong number of tensors; expected 1215, got 1213`. The two
it does not want are `output.scale` and `output.input_scale`, four bytes each.
A copy without them loads, which is what the pre-merge `llama-eval-callback`
comparison needs — see HANDOFF-v2 12-09 (plan for next session) item 4. The
logits then differ by a constant; **every intermediate tensor is unaffected**,
and intermediates are what the comparison reads.

**ggml requires the data section to be contiguous.** Dropping only the *index*
entries and leaving the bytes in place does not work: `gguf_init_from_file_ptr`
walks the tensors and rejects the first offset that is not where it expects it
("tensor 'blk.37.ffn_down_exps.weight' has offset 17112208384, expected
17112208320"). So the dropped tensors' bytes are excised too, and every tensor
after them has its offset shifted down by what was removed.

Each tensor's extent is taken from **the next tensor's offset** rather than
computed from its type and dimensions -- contiguity, which ggml has just been
shown to enforce, makes that exact and needs no type-size table. Every kept
tensor's *bytes* are still copied through unchanged; only its recorded offset
moves.

Usage:
  python scripts/gguf_drop_tensors.py <in.gguf> <out.gguf> <tensor> [<tensor>...]
"""

from __future__ import annotations

import os
import struct
import sys

MAGIC = b"GGUF"
# From `enum gguf_metadata_value_type` in ggml's gguf.h. Fixed-width sizes; 8 is
# a string and 9 an array, both handled separately.
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
STRING, ARRAY = 8, 9
# magic + version + tensor count + metadata count.
HEADER = 4 + 4 + 8 + 8


class Cursor:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.at = 0

    def take(self, n: int) -> bytes:
        b = self.buf[self.at : self.at + n]
        if len(b) != n:
            raise SystemExit(f"truncated GGUF at byte {self.at}")
        self.at += n
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> bytes:
        return self.take(self.u64())

    def skip_value(self, ty: int) -> None:
        if ty == STRING:
            self.string()
        elif ty == ARRAY:
            inner = self.u32()
            count = self.u64()
            if inner == STRING:
                for _ in range(count):
                    self.string()
            elif inner == ARRAY:
                raise SystemExit("nested arrays are not in any GGUF we write")
            elif inner in SCALAR:
                self.take(SCALAR[inner] * count)
            else:
                raise SystemExit(f"unknown array element type {inner}")
        elif ty in SCALAR:
            self.take(SCALAR[ty])
        else:
            raise SystemExit(f"unknown metadata value type {ty}")


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__.strip())
        return 2
    src, dst, drop = sys.argv[1], sys.argv[2], set(sys.argv[3:])

    with open(src, "rb") as fh:
        # The header is small; read a generous prefix rather than the whole file.
        head = fh.read(64 << 20)

        c = Cursor(head)
        if c.take(4) != MAGIC:
            raise SystemExit(f"{src} is not a GGUF file")
        version = c.u32()
        if version != 3:
            raise SystemExit(f"GGUF version {version}, expected 3")
        n_tensors = c.u64()
        n_kv = c.u64()

        alignment = 32
        for _ in range(n_kv):
            key = c.string()
            ty = c.u32()
            if key == b"general.alignment":
                if ty != 4:
                    raise SystemExit("general.alignment is not a UINT32")
                alignment = struct.unpack("<I", c.take(4))[0]
            else:
                c.skip_value(ty)
        kv_end = c.at

        # Each tensor info, as its exact byte range, so kept ones are copied
        # through untouched rather than re-encoded.
        infos = []
        for _ in range(n_tensors):
            start = c.at
            name = c.string().decode("utf-8", "replace")
            n_dims = c.u32()
            c.take(8 * n_dims)
            c.take(4)  # ggml_type
            offset = c.u64()  # relative to the data section
            infos.append({"name": name, "a": start, "b": c.at, "off": offset})
        info_end = c.at

        missing = drop - {t["name"] for t in infos}
        if missing:
            raise SystemExit(f"not in {src}: {', '.join(sorted(missing))}")

        data_start = (info_end + alignment - 1) // alignment * alignment
        total = os.path.getsize(src)
        data_size = total - data_start

        # Extents from the next tensor's offset. ggml enforces contiguity, so
        # this is exact and needs no per-type size table.
        order = sorted(infos, key=lambda t: t["off"])
        for i, t in enumerate(order):
            t["size"] = (order[i + 1]["off"] if i + 1 < len(order) else data_size) - t["off"]
            if t["size"] < 0:
                raise SystemExit(f"{t['name']}: tensors are not in offset order")

        print(f"{src}")
        print(f"  {n_tensors} tensors, {n_kv} metadata keys, alignment {alignment}")
        print(f"  data section starts at {data_start}, {data_size} bytes")

        # One pass in offset order: kept tensors get a new offset and a source
        # range to copy; dropped ones get neither, and shift everything after.
        removed, new_off, copy = 0, {}, []
        for t in order:
            if t["name"] in drop:
                print(f"  dropping {t['name']}  ({t['b'] - t['a']} B index, {t['size']} B data)")
                removed += t["size"]
            else:
                new_off[t["name"]] = t["off"] - removed
                if copy and copy[-1][0] + copy[-1][1] == t["off"]:
                    copy[-1] = (copy[-1][0], copy[-1][1] + t["size"])  # coalesce
                else:
                    copy.append((t["off"], t["size"]))

        keep = [t for t in infos if t["name"] not in drop]

        # Rebuild: same magic, version and metadata, fewer tensor infos, and
        # each kept entry's trailing 8-byte offset rewritten.
        out = bytearray()
        out += MAGIC
        out += struct.pack("<I", version)
        out += struct.pack("<Q", len(keep))
        out += struct.pack("<Q", n_kv)
        out += head[HEADER:kv_end]  # every metadata key, verbatim
        for t in keep:
            out += head[t["a"] : t["b"] - 8] + struct.pack("<Q", new_off[t["name"]])
        out += b"\x00" * ((-len(out)) % alignment)

        with open(dst, "wb") as w:
            w.write(out)
            copied = 0
            for off, size in copy:
                fh.seek(data_start + off)
                left = size
                while left:
                    chunk = fh.read(min(left, 64 << 20))
                    if not chunk:
                        raise SystemExit("unexpected end of file in the data section")
                    w.write(chunk)
                    left -= len(chunk)
                copied += size

    print(f"{dst}")
    print(f"  {len(keep)} tensors, data section at {len(out)}, {copied} bytes copied")
    if copied != data_size - removed:
        raise SystemExit(f"copied {copied}, expected {data_size - removed}")
    print(f"  {removed} bytes excised in {len(copy)} run(s); offsets after them shifted down")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
