"""Manifest parsing for batch generation.

A manifest is a JSONL file (one JSON object per line) or a CSV/TSV file. Each
row describes one utterance to synthesize. For the default voice-clone mode a
row looks like::

    {"id": "u001",
     "ref_audio": "refs/alice.wav",
     "ref_text": "transcript of the reference clip",
     "text": "the new text to speak in Alice's voice",
     "language": "English",
     "out": "alice/u001.wav"}

Recognised fields (synonyms in parentheses):
  id            optional unique id (utt_id, name); auto-assigned item_000001 if
                missing. Also the default output name (<id>.wav).
  text          (script, target_text) the text to synthesize. REQUIRED.
  language      (lang) target language; defaults to the job's --language.
  mode          "clone" (default) | "custom" | "design".
  ref_audio     (reference_audio, prompt_audio, audio) reference/example audio
                path or URL.                                   [clone]
  ref_text      (reference_text, prompt_text) transcript of ref_audio. [clone]
  speaker       (voice) preset speaker name.                   [custom]
  instruct      (instruction, style) optional style instruction. [custom/design]
  voice_prompt  (voice_description, description) voice to design. [design]
  out           (output, out_path, wav) output path, relative to --out-dir
                unless absolute.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Languages supported by Qwen3-TTS.
SUPPORTED_LANGUAGES = (
    "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
)

VALID_MODES = ("clone", "custom", "design")


@dataclass
class Item:
    id: str
    text: str
    language: str
    mode: str = "clone"
    ref_audio: str | None = None
    ref_text: str | None = None
    speaker: str | None = None
    instruct: str | None = None
    voice_prompt: str | None = None
    out: str | None = None
    # Populated by resolve_paths(); excluded from equality.
    out_path: Path | None = field(default=None, compare=False)

    def validate(self) -> None:
        if not self.text or not str(self.text).strip():
            raise ValueError(f"item {self.id!r}: empty 'text'")
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"item {self.id!r}: invalid mode {self.mode!r} (use {VALID_MODES})"
            )
        if self.mode == "clone":
            if not self.ref_audio:
                raise ValueError(f"item {self.id!r}: clone mode requires 'ref_audio'")
            if not self.ref_text:
                raise ValueError(f"item {self.id!r}: clone mode requires 'ref_text'")
        elif self.mode == "custom":
            if not self.speaker:
                raise ValueError(f"item {self.id!r}: custom mode requires 'speaker'")
        elif self.mode == "design":
            if not (self.voice_prompt or self.instruct):
                raise ValueError(
                    f"item {self.id!r}: design mode requires 'voice_prompt'"
                )


def _is_url(s: str) -> bool:
    return s.startswith(("http://", "https://", "data:"))


def _coalesce(row: dict, *names):
    """First present, non-empty value among ``names`` (else None)."""
    for n in names:
        if n in row and row[n] not in (None, ""):
            v = row[n]
            return v.strip() if isinstance(v, str) else v
    return None


def _row_to_item(row: dict, index: int, default_language: str, default_mode: str) -> Item:
    item_id = _coalesce(row, "id", "utt_id", "name") or f"item_{index:06d}"
    return Item(
        id=str(item_id),
        text=_coalesce(row, "text", "script", "target_text") or "",
        language=_coalesce(row, "language", "lang") or default_language,
        mode=str(_coalesce(row, "mode") or default_mode).lower(),
        ref_audio=_coalesce(row, "ref_audio", "reference_audio", "prompt_audio", "audio"),
        ref_text=_coalesce(row, "ref_text", "reference_text", "prompt_text"),
        speaker=_coalesce(row, "speaker", "voice"),
        instruct=_coalesce(row, "instruct", "instruction", "style"),
        voice_prompt=_coalesce(row, "voice_prompt", "voice_description", "description"),
        out=_coalesce(row, "out", "output", "out_path", "wav"),
    )


def _iter_records(path: Path) -> Iterator[dict]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter=delimiter):
                yield {k: v for k, v in row.items() if k is not None}
        return
    if suffix not in {".jsonl", ".ndjson", ".json"}:
        raise ValueError(
            f"Unsupported manifest type {suffix!r} for {path} "
            "(use .jsonl, .ndjson, .json, .csv or .tsv)."
        )
    # JSON family: support both a single JSON array and line-delimited JSON.
    with path.open(encoding="utf-8") as fh:
        first = fh.read(1)
        while first and first.isspace():
            first = fh.read(1)
        fh.seek(0)
        if first == "[":
            data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError(f"{path}: top-level JSON must be a list of objects.")
            yield from data
        else:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc


def load_manifest(
    path: str | Path,
    *,
    default_language: str = "English",
    default_mode: str = "clone",
    validate: bool = True,
) -> list[Item]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    items: list[Item] = []
    for record in _iter_records(path):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: every record must be a JSON object, got {type(record)}")
        items.append(_row_to_item(record, len(items), default_language, default_mode))
    if not items:
        raise ValueError(f"No items found in manifest {path}")
    if validate:
        for it in items:
            it.validate()
    return items


def resolve_paths(
    items: list[Item],
    out_dir: str | Path,
    audio_root: str | Path | None = None,
) -> list[Item]:
    """Fill in ``out_path`` and make ref_audio paths absolute.

    Output: ``out`` if given (absolute respected, else under ``out_dir``), else
    ``<id>.wav`` under ``out_dir``. Relative local ``ref_audio`` paths are
    resolved against ``audio_root`` when provided (URLs are left untouched).
    """
    out_dir = Path(out_dir)
    audio_root = Path(audio_root) if audio_root else None
    for it in items:
        rel = Path(it.out or f"{it.id}.wav")
        it.out_path = rel if rel.is_absolute() else out_dir / rel
        if it.ref_audio and audio_root and not _is_url(it.ref_audio):
            p = Path(it.ref_audio)
            if not p.is_absolute():
                it.ref_audio = str(audio_root / p)
    return items


def shard(items: list[Item], num_shards: int, shard_index: int) -> list[Item]:
    """Round-robin shard selection (balances variable-length items across tasks)."""
    if num_shards <= 1:
        return items
    if not (0 <= shard_index < num_shards):
        raise ValueError(f"shard_index {shard_index} out of range [0, {num_shards})")
    return items[shard_index::num_shards]


def filter_existing(items: list[Item]) -> list[Item]:
    """Drop items whose output already exists (for --skip-existing / resume)."""
    return [it for it in items if not (it.out_path and it.out_path.exists())]


def missing_local_refs(items: list[Item]) -> list[Item]:
    """Clone items whose local ref_audio file is missing (URLs are assumed OK)."""
    out = []
    for it in items:
        if it.mode != "clone" or not it.ref_audio or _is_url(it.ref_audio):
            continue
        if not Path(it.ref_audio).exists():
            out.append(it)
    return out
