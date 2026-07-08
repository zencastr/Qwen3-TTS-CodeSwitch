"""Thin wrapper around the ``qwen_tts`` inference API.

All heavy imports (torch, qwen_tts, soundfile, numpy) live here and are pulled
in lazily so the rest of the package stays importable without a GPU.

The voice-clone path (``generate_voice_clone``) is the documented, verified API
and the primary workflow. Custom-voice and voice-design are wired through the
same dispatcher; their exact method signatures can vary between qwen-tts
releases, so :func:`synthesize` resolves the method by name and raises a clear
error if it is absent in the installed version.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import RuntimeConfig
from .manifest import Item

# Method name on Qwen3TTSModel for each manifest mode.
_MODE_METHODS = {
    "clone": "generate_voice_clone",
    "custom": "generate_custom_voice",
    "design": "generate_voice_design",
}


def _configure_sdpa_backends(torch, cfg: RuntimeConfig) -> None:
    """Toggle the cuDNN SDPA backend (off by default).

    On Blackwell sm_103 the cuDNN 9.x attention backend has no valid execution
    plan and raises "No valid execution plans built" from inside the Mimi audio
    codec's scaled_dot_product_attention. Disabling just that backend lets SDPA
    fall back to the mem-efficient/math backends; cuDNN convolutions stay on.
    """
    fn = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
    if fn is not None:
        try:
            fn(cfg.cudnn_sdp)
        except Exception:  # noqa: BLE001 - never let a backend toggle abort a run
            pass


def load_model(cfg: RuntimeConfig):
    """Load a Qwen3TTSModel onto the configured device/dtype/attention."""
    import torch
    from qwen_tts import Qwen3TTSModel

    _configure_sdpa_backends(torch, cfg)

    return Qwen3TTSModel.from_pretrained(
        cfg.model_id,
        device_map=cfg.device,
        dtype=cfg.torch_dtype,
        attn_implementation=cfg.attn_implementation,
    )


def _kwargs_for(item: Item) -> dict[str, Any]:
    """Build the keyword arguments for the mode's generate_* method.

    Only non-empty optional arguments are included, so a release that doesn't
    accept e.g. ``instruct=None`` still works.
    """
    if item.mode == "clone":
        return {
            "text": item.text,
            "language": item.language,
            "ref_audio": item.ref_audio,
            "ref_text": item.ref_text,
        }
    if item.mode == "custom":
        kw = {"text": item.text, "language": item.language, "speaker": item.speaker}
        if item.instruct:
            kw["instruct"] = item.instruct
        return kw
    if item.mode == "design":
        kw = {"text": item.text, "language": item.language}
        if item.voice_prompt:
            kw["voice_prompt"] = item.voice_prompt
        if item.instruct:
            kw["instruct"] = item.instruct
        return kw
    raise ValueError(f"unknown mode {item.mode!r}")


def synthesize(model, item: Item, gen_kwargs: dict[str, Any] | None = None):
    """Synthesize one item. Returns ``(wav, sample_rate)`` with wav as 1-D float."""
    method_name = _MODE_METHODS[item.mode]
    method = getattr(model, method_name, None)
    if method is None:
        raise NotImplementedError(
            f"The installed qwen_tts model has no '{method_name}' method for "
            f"mode={item.mode!r}. Check the API with `python -m qwen3_tts_gen.backend "
            "--inspect` and update qwen3_tts_gen/backend.py:_MODE_METHODS."
        )
    wavs, sr = method(**_kwargs_for(item), **(gen_kwargs or {}))
    wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    return _to_mono_float(wav), int(sr)


def _to_mono_float(wav):
    import numpy as np

    arr = np.asarray(wav)
    arr = np.squeeze(arr)
    if arr.ndim > 1:  # (channels, T) or (T, channels) -> mix to mono
        ch_axis = 0 if arr.shape[0] < arr.shape[1] else 1
        arr = arr.mean(axis=ch_axis)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def write_wav(path: str | Path, wav, sample_rate: int) -> float:
    """Write a mono float waveform to ``path``; return its duration in seconds."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, int(sample_rate))
    return float(len(wav)) / float(sample_rate) if sample_rate else 0.0


def inspect_api() -> str:
    """Return a short report of the installed qwen_tts generate_* methods."""
    import qwen_tts
    from qwen_tts import Qwen3TTSModel

    methods = sorted(
        m for m in dir(Qwen3TTSModel)
        if m.startswith("generate") and callable(getattr(Qwen3TTSModel, m, None))
    )
    version = getattr(qwen_tts, "__version__", "unknown")
    lines = [f"qwen_tts version: {version}", "generate_* methods on Qwen3TTSModel:"]
    lines += [f"  - {m}" for m in methods] or ["  (none found)"]
    lines.append("mode -> method mapping used by this package:")
    for mode, name in _MODE_METHODS.items():
        ok = "ok" if name in methods else "MISSING"
        lines.append(f"  - {mode:<7} -> {name}  [{ok}]")
    return "\n".join(lines)


def _main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Inspect the installed qwen_tts API.")
    ap.add_argument("--inspect", action="store_true", help="list generate_* methods")
    ap.parse_args(argv)
    print(inspect_api())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
