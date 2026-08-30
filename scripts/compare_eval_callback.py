#!/usr/bin/env python3
"""Stage 4 acceptance: diff our forward pass against llama-eval-callback and
report the FIRST diverging tensor.

Two checks:

1. **Elements.** llama-eval-callback prints sampled values from each tensor.
   We resolve each printed value's exact coordinates and compare it against the
   same element of our own dump. This is the discriminating check.
2. **Checksums.** It also prints `sum = ...` over the whole tensor, which
   catches anything the sampled elements miss.

Two subtleties the printer forces on us, both learned the hard way:

* Rows are truncated. Within a group it prints the first 3 rows, a bare `...,`
  and the last 3. Numbering printed rows sequentially compares unrelated
  elements. Same for values within a row: first 3, `...`, last 3.
* Values are printed to 4 decimals, so a reference value is only known to
  +/-5e-5. That print quantum has to be allowed for on top of any tolerance.

Error is judged against each tensor's own magnitude, not each element's:
numerical error in a matmul scales with the tensor, so a near-zero element
would otherwise fail on noise that is negligible in context.

Usage:
  python scripts/compare_eval_callback.py <ref.txt> <ours.txt> [--dump ours.bin]
"""

from __future__ import annotations

import argparse
import re
import struct
import sys

NAME_RE = re.compile(r"\s*common_debug_cb_eval:\s*(\S.*?)\s*=\s*\(")
SUM_RE = re.compile(r"\s*sum\s*=\s*(-?[\d.]+(?:[eE][-+]?\d+)?)\s*$")
DIMS_RE = re.compile(r"=\s*\{([\d,\s]+)\}\s*$")
ROW_RE = re.compile(r"^\s*\[\s*(-?\d.*?)\s*\],?\s*$")
NUM_RE = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")

# Our trace name -> the reference's name for the same tensor.
ALIASES = {"inp_embd": "embd"}


class Tensor:
    """One reference dump: dims, sampled values with resolved coordinates."""

    def __init__(self, dims: list[int]):
        self.dims = dims
        # (i2, i1, [(offset_in_row, value), ...])
        self.rows: list[tuple[int, int, list[tuple[int, float]]]] = []
        self.total: float | None = None

    def scale(self) -> float:
        vals = [abs(v) for _, _, r in self.rows for _, v in r]
        return max(vals) if vals else 1.0


def parse_ref(path: str) -> dict[str, Tensor]:
    """Parse every tensor dump, keeping the last occurrence of each name.

    A name is re-emitted after each op that rewrites it (Qcur after mul_mat,
    after reshape, after rope); the final one is what flows onward.
    """
    out: dict[str, Tensor] = {}
    name: str | None = None
    cur: Tensor | None = None

    # Group state: which i2 block we are in, rows seen before/after the `...`.
    # The dump nests as `[ outer [ group [ row ] ... ] ... ]`, so a group is a
    # bracket at depth 2 -- counting every bare `[` would include the outermost
    # one and shift every block index by one.
    group = -1
    depth = 0
    before: list[list[float]] = []
    after: list[list[float]] = []
    truncated = False

    def close_group():
        nonlocal before, after, truncated
        if cur is None or (not before and not after):
            before, after, truncated = [], [], False
            return
        ne1 = cur.dims[1] if len(cur.dims) > 1 else 1
        ne0 = cur.dims[0]
        for i, vals in enumerate(before):
            cur.rows.append((group, i, resolve_row(vals, ne0)))
        for j, vals in enumerate(reversed(after)):
            cur.rows.append((group, ne1 - 1 - j, resolve_row(vals, ne0)))
        before, after, truncated = [], [], False

    with open(path, errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            m = NAME_RE.match(line)
            if m:
                close_group()
                if name is not None and cur is not None:
                    out[name] = cur
                name = m.group(1).strip()
                d = DIMS_RE.search(line)
                dims = [int(x) for x in d.group(1).split(",")] if d else [1]
                cur = Tensor(dims)
                group = -1
                depth = 0
                continue

            if cur is None:
                continue

            stripped = line.strip()

            if stripped == "[":
                depth += 1
                if depth == 2:
                    close_group()
                    group += 1
                continue
            if stripped in ("]", "],"):
                if depth == 2:
                    close_group()
                depth -= 1
                continue
            if stripped.startswith("...."):
                truncated = True
                continue
            if stripped == "...," or stripped == "...":
                truncated = True
                continue

            m = ROW_RE.match(line)
            if m:
                vals = [float(v) for v in NUM_RE.findall(m.group(1))]
                if vals:
                    (after if truncated else before).append(vals)
                continue

            m = SUM_RE.match(line)
            if m:
                close_group()
                cur.total = float(m.group(1))
                continue

    close_group()
    if name is not None and cur is not None:
        out[name] = cur
    return out


def resolve_row(vals: list[float], ne0: int) -> list[tuple[int, float]]:
    """Map printed values to their offsets within a row."""
    if len(vals) == ne0:
        return list(enumerate(vals))
    if len(vals) == 6 and ne0 >= 6:
        # first 3, then last 3
        return list(zip([0, 1, 2, ne0 - 3, ne0 - 2, ne0 - 1], vals))
    if len(vals) < ne0:
        # Only the leading values are unambiguous.
        return list(enumerate(vals[: len(vals) // 2]))
    return []


def read_dump(path: str) -> dict[str, list[float]]:
    """Read the sidecar written by `inferred trace --dump`:
    [u32 name_len][name][u64 n][n * f32], repeated."""
    out: dict[str, list[float]] = {}
    with open(path, "rb") as f:
        while True:
            head = f.read(4)
            if len(head) < 4:
                break
            (name_len,) = struct.unpack("<I", head)
            name = f.read(name_len).decode()
            (n,) = struct.unpack("<Q", f.read(8))
            out[name] = list(struct.unpack(f"<{n}f", f.read(4 * n)))
    return out


def parse_ours(path: str) -> list[tuple[str, int, float]]:
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                rows.append((parts[0], int(parts[1]), float(parts[2])))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("ours")
    ap.add_argument("--dump", help="binary sidecar from `inferred trace --dump`")
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="allowed error as a fraction of each tensor's magnitude")
    ap.add_argument("--report", action="store_true",
                    help="print worst error per tensor in trace order instead of "
                         "stopping at the first failure -- shows where drift begins")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ref = parse_ref(args.ref)
    ours = parse_ours(args.ours)
    if not ref:
        print(f"FAIL: no tensors parsed from {args.ref}", file=sys.stderr)
        return 2
    if not ours:
        print(f"FAIL: no tensors parsed from {args.ours}", file=sys.stderr)
        return 2

    # ---- element check --------------------------------------------------
    worst_elem = ("", 0.0)
    checked = 0
    per_tensor: list[tuple[str, float, float, float, int]] = []
    if args.dump:
        dump = read_dump(args.dump)
        for name, _, _ in ours:
            tensor_worst = 0.0
            tensor_detail = (0.0, 0.0, 0)
            key = ALIASES.get(name, name)
            if key not in ref or name not in dump:
                continue
            t = ref[key]
            data = dump[name]
            ne0 = t.dims[0]
            ne1 = t.dims[1] if len(t.dims) > 1 else 1
            scale = t.scale()
            allowed = 1e-4 + args.tol * scale

            # At the final layer llama.cpp applies ggml_get_rows(cur, inp_out_ids),
            # keeping only the position(s) it will produce logits for -- so its
            # tensor holds 1 token where ours holds all of them. Align from the
            # end, which is where the kept token sits.
            ref_total = ne0 * ne1 * (t.dims[2] if len(t.dims) > 2 else 1)
            skip = max(0, len(data) - ref_total)

            for i2, i1, cells in t.rows:
                base = skip + max(i2, 0) * ne1 * ne0 + i1 * ne0
                for off, want in cells:
                    idx = base + off
                    if idx >= len(data):
                        continue
                    got = data[idx]
                    diff = abs(got - want)
                    d = diff / max(scale, 1e-9)
                    checked += 1
                    if d > worst_elem[1]:
                        worst_elem = (f"{name}[t{i2} r{i1} +{off}]", d)
                    if d > tensor_worst:
                        tensor_worst = d
                        tensor_detail = (want, got, idx)
                    if args.report:
                        continue
                    if diff > allowed:
                        print(f"FAIL - first element divergence in {name}")
                        print(f"  dims {t.dims}, block {i2}, row {i1}, offset {off}")
                        print(f"  flat index   {idx}")
                        print(f"  reference    {want:.6f}")
                        print(f"  ours         {got:.6f}")
                        print(f"  tensor scale {scale:.6f}")
                        print(f"  error/scale  {d:.3e}   (allowed {args.tol:g})")
                        print("")
                        print("Fix this tensor before looking at anything downstream.")
                        return 1
            if tensor_worst > 0.0 or name in dump:
                w, g, ix = tensor_detail
                per_tensor.append((name, tensor_worst, w, g, ix))

        if args.report:
            print(f"{'tensor':<20} {'worst err/scale':>15}  {'reference':>12} {'ours':>12}  idx")
            for name, d, w, g, ix in per_tensor:
                mark = " <-- first over 1%" if d > 0.01 else ""
                print(f"{name:<20} {d:>15.3e}  {w:>12.6f} {g:>12.6f}  {ix}{mark}")
            return 0

        print(f"element check: {checked} values OK; "
              f"worst {worst_elem[0]} at {worst_elem[1]:.2e} of tensor scale")

    # ---- checksum check -------------------------------------------------
    compared = 0
    worst_sum = ("", 0.0)
    first_bad = None
    for name, n, got in ours:
        key = ALIASES.get(name, name)
        if key not in ref or ref[key].total is None:
            continue
        want = ref[key].total
        # f32 accumulation error grows with element count, and the printed sum
        # itself carries rounding, so scale the allowance by n.
        allowed = max(1e-3, 3e-6 * n)
        diff = abs(got - want)
        compared += 1
        rel = diff / max(1.0, abs(want))
        if rel > worst_sum[1]:
            worst_sum = (name, rel)
        if diff > allowed and first_bad is None:
            first_bad = (name, n, want, got, diff, allowed)
        if args.verbose:
            flag = "  " if diff <= allowed else "!!"
            print(f"{flag} {name:<20} n={n:<8} ref={want:>14.4f} ours={got:>14.4f} diff={diff:.4f}")

    if first_bad:
        name, n, want, got, diff, allowed = first_bad
        print(f"FAIL - checksum divergence at {name}")
        print(f"  elements   {n}")
        print(f"  reference  {want:.6f}")
        print(f"  ours       {got:.6f}")
        print(f"  difference {diff:.6f}  (allowed {allowed:.6f})")
        return 1

    print(f"checksum check: {compared} tensors OK; worst relative {worst_sum[1]:.2e} ({worst_sum[0]})")
    print("")
    print("PASS - the forward pass matches llama.cpp at every traced tensor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
