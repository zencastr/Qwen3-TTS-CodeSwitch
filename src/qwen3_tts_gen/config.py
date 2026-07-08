"""Runtime + cache configuration, driven by environment variables.

Defaults are tuned for the B300 cluster:
  * weights cached on /scratch (big files don't belong in $HOME),
  * bf16 compute (well supported on Blackwell),
  * PyTorch SDPA attention. flash-attn is opt-in: it has no prebuilt Blackwell
    wheels and the login node has no CUDA toolkit to build it, so SDPA is the
    safe default. Set QWEN3_TTS_ATTN=flash_attention_2 if you installed it.

Environment variables:
  QWEN3_TTS_CACHE   cache dir for model weights   (default /scratch/ick/qwen3_tts/cache)
  QWEN3_TTS_MODEL   model alias or repo id        (default 1.7b-base)
  QWEN3_TTS_DEVICE  torch device                  (default cuda:0)
  QWEN3_TTS_DTYPE   bfloat16 | float16 | float32  (default bfloat16)
  QWEN3_TTS_ATTN    sdpa | flash_attention_2 | eager (default sdpa)
  QWEN3_TTS_CUDNN_SDP  use the cuDNN attention backend (default 0 = off)

On the B300 (sm_103) nodes the cuDNN 9.x SDPA backend has no valid execution
plan and raises "cuDNN Frontend error: No valid execution plans built" from
inside the Mimi audio codec. We therefore disable *only* the cuDNN SDPA backend
by default (mem-efficient/math backends still work, cuDNN convolutions stay on).
Set QWEN3_TTS_CUDNN_SDP=1 to re-enable it on hardware where it works.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .models import DEFAULT_MODEL, resolve_model

# Big files (model weights) belong on scratch, per the cluster storage layout.
DEFAULT_CACHE = "/scratch/ick/qwen3_tts/cache"

_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float": "float32",
}
_VALID_DTYPES = {"bfloat16", "float16", "float32"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def cache_dir() -> Path:
    return Path(os.environ.get("QWEN3_TTS_CACHE", DEFAULT_CACHE)).expanduser()


def configure_hf_cache(path: str | os.PathLike | None = None) -> Path:
    """Point HuggingFace at the shared scratch cache.

    Call this *before* importing/loading any model so the download script and
    the generation jobs share one cache and nothing re-downloads. An already
    exported HF_HOME is respected.
    """
    cache = Path(path or cache_dir()).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return cache


@dataclass(frozen=True)
class RuntimeConfig:
    model_id: str             # resolved HF repo id or local path
    device: str               # e.g. "cuda:0"
    dtype: str                # canonical torch dtype name, e.g. "bfloat16"
    attn_implementation: str  # "sdpa" | "flash_attention_2" | "eager"
    cudnn_sdp: bool = False    # cuDNN SDPA backend (off by default; broken on sm_103)

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        attn: str | None = None,
        cudnn_sdp: bool | None = None,
    ) -> "RuntimeConfig":
        model = model or os.environ.get("QWEN3_TTS_MODEL", DEFAULT_MODEL)
        dtype = (dtype or os.environ.get("QWEN3_TTS_DTYPE", "bfloat16")).lower()
        dtype = _DTYPE_ALIASES.get(dtype, dtype)
        if dtype not in _VALID_DTYPES:
            raise ValueError(
                f"Unsupported dtype {dtype!r}; choose one of {sorted(_VALID_DTYPES)}."
            )
        return cls(
            model_id=resolve_model(model),
            device=device or os.environ.get("QWEN3_TTS_DEVICE", "cuda:0"),
            dtype=dtype,
            attn_implementation=attn or os.environ.get("QWEN3_TTS_ATTN", "sdpa"),
            cudnn_sdp=_env_bool("QWEN3_TTS_CUDNN_SDP", False) if cudnn_sdp is None else cudnn_sdp,
        )

    @property
    def torch_dtype(self):
        import torch  # local import: keep torch out of the --help / dry-run path

        return getattr(torch, self.dtype)
