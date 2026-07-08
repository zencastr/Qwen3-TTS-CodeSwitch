"""Batch speech generation with Qwen3-TTS.

Voice-cloning (and custom-voice / voice-design) speech synthesis driven by
manifests, designed to run at scale under SLURM. Heavy imports (torch,
qwen_tts) are deferred into :mod:`qwen3_tts_gen.backend` so that ``--help`` and
manifest validation work without a GPU.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .config import RuntimeConfig, configure_hf_cache
from .models import DEFAULT_MODEL, MODELS, TOKENIZER, resolve_model

__all__ = [
    "RuntimeConfig",
    "configure_hf_cache",
    "MODELS",
    "TOKENIZER",
    "DEFAULT_MODEL",
    "resolve_model",
    "__version__",
]
