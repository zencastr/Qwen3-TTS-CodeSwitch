#!/usr/bin/env python3
"""Build a qwen3_tts_gen manifest from the CodemixQA dataset for voice cloning.

CodemixQA (gentaiscool/codemixqa) is a synthetically code-mixed English trivia-QA
benchmark: each row's `problem` mixes English with one of 16 other languages via
an algorithmic method (grammarforce_source/target, selective, random). Only rows
whose non-English language is one Qwen3-TTS actually supports are usable here —
see manifests/codemixqa_full.jsonl's provenance / memory `codemixqa-dataset`.

Each row is voice-cloned onto an English FLEURS reference speaker (round-robin
over the pool, for speaker diversity) with `language: "auto"` so Qwen3-TTS's
language autodetection handles the embedded non-English spans, matching the
precedent in manifests/codeswitch_demo.jsonl.

Usage:
  python scripts/build_codemixqa_manifest.py \
      --codemixqa-parquet /scratch/ick/qwen3_tts/codemixqa/codemixqa_24k.parquet \
      --fleurs-root /scratch/ick/LID/fleurs --out manifests/codemixqa_full.jsonl
  python scripts/build_codemixqa_manifest.py --per-lang 2 --out manifests/codemixqa_smoke.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

# CodemixQA `language` value -> still code-mixed with English, so every row
# gets Qwen3-TTS's autodetect language rather than the named target language.
CODEMIXQA_SUPPORTED = {
    "Simplified Chinese", "French", "Italian", "Japanese", "Korean", "Spanish",
}


def _load_fleurs_refs(fleurs_root: Path, audio_root: Path) -> list[tuple[str, str]]:
    """Return [(ref_audio_path, ref_text), ...] for FLEURS en_us clips."""
    tsv = fleurs_root / "manifest.tsv"
    if not tsv.exists():
        raise SystemExit(f"FLEURS manifest.tsv not found: {tsv}")
    refs: list[tuple[str, str]] = []
    with tsv.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if (row.get("config") or "").strip() != "en_us":
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            audio_path = (row.get("audio_path") or "").strip()
            ref = audio_root / audio_path
            if not ref.exists():
                alt = fleurs_root / "en_us" / Path(audio_path).name
                if not alt.exists():
                    continue
                ref = alt
            refs.append((str(ref), text))
    return refs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codemixqa-parquet",
                    default="/scratch/ick/qwen3_tts/codemixqa/codemixqa_24k.parquet",
                    help="pre-filtered CodemixQA parquet (Qwen-supported languages only)")
    ap.add_argument("--fleurs-root", default="/scratch/ick/LID/fleurs",
                    help="dir containing manifest.tsv and <lang>/ clip dirs")
    ap.add_argument("--audio-root", default=None,
                    help="root the tsv 'audio_path' is relative to "
                         "(default: parent of --fleurs-root)")
    ap.add_argument("--out", required=True, help="output JSONL manifest path")
    ap.add_argument("--per-lang", type=int, default=None,
                    help="take at most N rows per language (for smoke tests)")
    ap.add_argument("--languages", nargs="+", default=None,
                    help="subset of CodemixQA language values (default: all 6 supported)")
    args = ap.parse_args(argv)

    fleurs_root = Path(args.fleurs_root)
    audio_root = Path(args.audio_root) if args.audio_root else fleurs_root.parent
    refs = _load_fleurs_refs(fleurs_root, audio_root)
    if not refs:
        ap.error(f"no FLEURS en_us reference clips found under {fleurs_root}")

    wanted = set(args.languages) if args.languages else set(CODEMIXQA_SUPPORTED)
    unknown = wanted - CODEMIXQA_SUPPORTED
    if unknown:
        ap.error(f"not supported by Qwen3-TTS: {sorted(unknown)}; "
                 f"supported CodemixQA languages: {sorted(CODEMIXQA_SUPPORTED)}")

    table = pq.read_table(
        args.codemixqa_parquet, columns=["id", "problem", "language", "method", "answer", "topic"]
    ).to_pydict()
    n = len(table["id"])

    per_lang: dict[str, int] = {}
    empty_text = 0
    dup_ids = 0
    rows_out = 0
    seen_ids: set[str] = set()
    ref_idx = 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for i in range(n):
            lang = (table["language"][i] or "").strip()
            if lang not in CODEMIXQA_SUPPORTED or lang not in wanted:
                continue
            if args.per_lang is not None and per_lang.get(lang, 0) >= args.per_lang:
                continue
            text = (table["problem"][i] or "").strip()
            if not text:
                empty_text += 1
                continue
            row_id = str(table["id"][i]).replace(" ", "_")   # e.g. "...-Simplified Chinese" -> safe filename
            if row_id in seen_ids:
                dup_ids += 1
                continue
            seen_ids.add(row_id)

            ref_audio, ref_text = refs[ref_idx % len(refs)]
            ref_idx += 1

            out.write(json.dumps({
                "id": row_id,
                "mode": "clone",
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "text": text,
                "language": "auto",          # code-mixed English + embedded language
                "cmqa_language": lang,        # provenance, ignored by qwen3_tts_gen
                "cmqa_method": table["method"][i],
            }, ensure_ascii=False) + "\n")
            per_lang[lang] = per_lang.get(lang, 0) + 1
            rows_out += 1

    print(f"[done] wrote {rows_out} rows to {out_path} (from {len(refs)} FLEURS ref speakers)")
    for lang in sorted(per_lang):
        print(f"    {lang:<20} {per_lang[lang]}")
    if empty_text:
        print(f"[warn] {empty_text} rows skipped (empty problem text)")
    if dup_ids:
        print(f"[warn] {dup_ids} rows skipped (duplicate id)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
