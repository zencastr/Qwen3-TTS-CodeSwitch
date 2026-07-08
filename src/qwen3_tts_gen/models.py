"""Registry of Qwen3-TTS checkpoints on the HuggingFace Hub.

All weights are public (Apache-2.0). The 12 Hz tokenizer is the shared neural
audio codec required by every generation model, so it is always fetched
alongside them.

Three model families:
  * ``base``        - reference-audio voice cloning (in-context learning). This
                      is the family used by the manifest workflow.
  * ``customvoice`` - preset named speakers + instruction control, no reference
                      audio needed.
  * ``voicedesign`` - build a voice from a free-form natural-language prompt.
"""
from __future__ import annotations

from dataclasses import dataclass

# Shared neural audio tokenizer/codec (required by every generation model).
TOKENIZER = "Qwen/Qwen3-TTS-Tokenizer-12Hz"


@dataclass(frozen=True)
class ModelSpec:
    alias: str        # short, CLI-friendly handle
    repo_id: str      # full HuggingFace repo id
    family: str       # "base" | "customvoice" | "voicedesign"
    params: str       # "1.7B" | "0.6B"
    summary: str      # one-line description


_SPECS = [
    ModelSpec(
        "1.7b-base", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "base", "1.7B",
        "Reference-audio voice cloning (ICL). Best quality; default for manifests.",
    ),
    ModelSpec(
        "0.6b-base", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "base", "0.6B",
        "Smaller/faster voice cloning. Good for high-throughput batch runs.",
    ),
    ModelSpec(
        "1.7b-customvoice", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "customvoice", "1.7B",
        "Preset named speakers + instruction control (no reference audio).",
    ),
    ModelSpec(
        "0.6b-customvoice", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "customvoice", "0.6B",
        "Smaller preset-speaker model.",
    ),
    ModelSpec(
        "1.7b-voicedesign", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "voicedesign", "1.7B",
        "Free-form voice design from a natural-language voice description.",
    ),
]

MODELS: dict[str, ModelSpec] = {s.alias: s for s in _SPECS}

DEFAULT_MODEL = "1.7b-base"

# Named groupings for the download script. The tokenizer is added implicitly.
DOWNLOAD_SETS: dict[str, list[str]] = {
    "all": [s.alias for s in _SPECS],
    "base": ["1.7b-base", "0.6b-base"],
    "1.7b": ["1.7b-base", "1.7b-customvoice", "1.7b-voicedesign"],
    "0.6b": ["0.6b-base", "0.6b-customvoice"],
    "minimal": [DEFAULT_MODEL],
}


def resolve_model(name: str) -> str:
    """Map an alias (``1.7b-base``) or a full repo id / local path to a repo id."""
    if "/" in name:  # already a full HF repo id or a local directory path
        return name
    key = name.strip().lower()
    if key in MODELS:
        return MODELS[key].repo_id
    raise KeyError(
        f"Unknown model {name!r}. Known aliases: {', '.join(MODELS)} "
        "(or pass a full HF repo id, e.g. 'Qwen/Qwen3-TTS-12Hz-1.7B-Base')."
    )
