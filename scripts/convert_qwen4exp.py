"""Qwen3.8-Flash-Next (`qwen4exp`) NVFP4 checkpoint -> GGUF, with the fork's converter
and one correction applied from outside it.

**The correction: the PLE table's FP8 scale.** The checkpoint stores the n-gram
table as 128 FP8 E4M3 shards, `ngram_embedding.shard_N.weight`, sharing **one**
per-tensor scale, `ngram_embedding.weight_scale` (BF16, one value, 1.993e-4 in
`nvidia/Qwen3.8-Flash-Next-NVFP4`). The fork's converter drops it:

- `conversion/base.py:559-563`, under `modelopt`, pairs every `X.weight_scale` with
  `X.weight` and deletes a scale whose weight does not exist. There is no
  `ngram_embedding.weight`, only the shards, so the scale is removed silently.
- `conversion/qwen4exp.py:180`, `_load_ple_shard`, casts FP8 to float32 and applies
  no scale.
- `src/models/qwen4exp.cpp:1185` reads the table with a plain `get_rows`.

So every PLE row would be ~5,017x too large, in llama.cpp's reference and in any
engine tested against it. colibri applies the scale
(`c/qwen38_core.h:1354`, `e4m3_decode(raw[d]) * m->ple_weight_scale`). Of the
checkpoint's 73,729 `weight_scale` tensors this is the only one without a matching
weight (checked 15-09-2026). SSD-TIER.md, "Toward the 125B".

The fix is applied here rather than in the fork, as `convert_nvfp4.sh` fixed the
35B's NVFP4 detection: the fork and the checkpoint stay untouched. The scale is
multiplied in **before** quantization, so Q8_0 is computed from the true values;
rescaling a quantized table afterwards would round through each block's f16 scale.

Usage: python convert_qwen4exp.py <fork llama.cpp dir> <convert_hf_to_gguf args...>
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger("convert_qwen4exp")

SCALE_SUFFIX = "ple_embedding.ngram_embedding.weight_scale"


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: convert_qwen4exp.py <fork llama.cpp dir> <convert_hf_to_gguf args...>")
    fork = Path(sys.argv[1]).resolve()
    if not (fork / "convert_hf_to_gguf.py").is_file():
        sys.exit(f"{fork} has no convert_hf_to_gguf.py")

    # The fork's entry point puts its own gguf-py on the path when imported.
    sys.path.insert(0, str(fork))
    import torch
    import convert_hf_to_gguf
    from conversion.base import LazyTorchTensor
    from conversion.qwen4exp import Qwen4ExpTextModel

    base_dequant = Qwen4ExpTextModel.dequant_model

    def dequant_model(self) -> None:
        # Capture the scale through the converter's own loader, before
        # `dequant_model` deletes it as an orphan.
        # A checkpoint with a float PLE table (the BF16 base, or the 0.2B test model)
        # has no scale; whether one is required is decided per shard, by its dtype.
        names = [k for k in self.model_tensors if k.endswith(SCALE_SUFFIX)]
        self._ple_scale = None
        if len(names) > 1:
            raise ValueError(f"{len(names)} tensors end in {SCALE_SUFFIX}: {names}")
        if names:
            t = LazyTorchTensor.to_eager(self.model_tensors[names[0]]())
            if t.numel() != 1:
                raise ValueError(f"{names[0]} has {t.numel()} values; one per-tensor scale was expected")
            self._ple_scale = float(t.float().reshape(()).item())
            if not (self._ple_scale > 0.0 and self._ple_scale == self._ple_scale):
                raise ValueError(f"{names[0]} = {self._ple_scale}, not a positive scale")
            logger.info(f"PLE table scale {names[0]} = {self._ple_scale!r}, applied to every shard")
        base_dequant(self)

    def _load_ple_shard(self, name: str):
        if not hasattr(self, "_ple_scale"):
            raise ValueError(f"PLE shard {name} reached before dequant_model looked for its scale")
        scale = self._ple_scale

        def load():
            # A fresh lazy tensor every call, or to_eager() memoizes every shard
            # (the fork's own comment, qwen4exp.py:178).
            eager = LazyTorchTensor.to_eager(self.model_tensors[name]())
            fp8 = eager.dtype == torch.float8_e4m3fn
            # FP8 without a scale is exactly the defect this script exists for; a
            # scale beside float shards would mean the checkpoint is not what
            # this was written against. Either is refused rather than guessed.
            if fp8 and scale is None:
                raise ValueError(f"PLE shard {name} is FP8 but the checkpoint has no {SCALE_SUFFIX}")
            if not fp8 and scale is not None:
                raise ValueError(f"PLE shard {name} is {eager.dtype}, yet a per-tensor FP8 scale exists")
            out = eager.to(torch.float32)
            if fp8:
                out = out * scale
            out = out.contiguous().numpy()
            if name.endswith(".shard_0.weight"):
                logger.info(
                    f"PLE shard 0 is {eager.dtype}, scale {scale!r}; row 0, first 4 as written: "
                    f"{out[0, :4].tolist()}; shard |max| {float(abs(out).max())!r}"
                )
            return out

        return load

    Qwen4ExpTextModel.dequant_model = dequant_model
    Qwen4ExpTextModel._load_ple_shard = _load_ple_shard

    sys.argv = [str(fork / "convert_hf_to_gguf.py"), *sys.argv[2:]]
    convert_hf_to_gguf.main()


if __name__ == "__main__":
    main()
