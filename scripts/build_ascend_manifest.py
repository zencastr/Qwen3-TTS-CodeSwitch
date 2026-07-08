#!/usr/bin/env python3
"""Build a qwen3_tts_gen manifest from the ASCEND dataset for voice cloning.

ASCEND (CAiRE/ASCEND) is a spontaneous Chinese-English *code-switching* corpus.
For each clip we emit a voice-clone row whose reference IS the clip and whose
target text is the clip's own transcript -> a voice-cloned copy of the utterance
(output written as <out-dir>/<id>.wav).

Language handling (ASCEND's `language` field is one of zh / en / mixed):
  zh    -> "Chinese"
  en    -> "English"
  mixed -> "auto"     # code-switch; let Qwen3-TTS language-autodetect handle it
                      # (matches the repo's committed manifests/codeswitch_demo.jsonl)

The output `id` is the reference WAV stem, NOT the parquet `id` field: ASCEND's
`id` is only unique per-split (9869 unique / 12314 rows) and would collide across
train/val/test, silently overwriting outputs. The WAV stem is globally unique.

Metadata comes from the parquet files under <ascend-root>/main/; audio comes from
the flat <ascend-root>/wav/ directory (filename == basename of the parquet `path`).

Usage:
  python scripts/build_ascend_manifest.py --out manifests/ascend_full.jsonl
  python scripts/build_ascend_manifest.py --per-lang 1 --out manifests/ascend_smoke.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

# ASCEND language code -> Qwen3-TTS language name ("auto" = autodetect for
# code-switched clips). Chinese/English are the two Qwen-supported single
# languages present; "mixed" is Chinese-English code-switching.
ASCEND_TO_QWEN = {
    "zh": "Chinese",
    "en": "English",
    "mixed": "auto",
}

COLUMNS = ["path", "transcription", "language"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ascend-root", default="/scratch/ick/ASCEND",
                    help="dir containing main/*.parquet and wav/ (default: /scratch/ick/ASCEND)")
    ap.add_argument("--out", required=True, help="output JSONL manifest path")
    ap.add_argument("--per-lang", type=int, default=None,
                    help="take at most N clips per language (for smoke tests)")
    ap.add_argument("--languages", nargs="+", default=None,
                    help="subset of ASCEND codes to include (default: zh en mixed)")
    ap.add_argument("--splits", nargs="+", default=None,
                    help="subset of splits to include, e.g. train test validation "
                         "(default: all)")
    args = ap.parse_args(argv)

    ascend_root = Path(args.ascend_root)
    wav_dir = ascend_root / "wav"
    parquets = sorted(glob.glob(str(ascend_root / "main" / "*.parquet")))
    if not parquets:
        ap.error(f"no parquet files under {ascend_root / 'main'}")
    if not wav_dir.is_dir():
        ap.error(f"wav dir not found: {wav_dir}")

    wanted = set(args.languages) if args.languages else set(ASCEND_TO_QWEN)
    unknown = wanted - set(ASCEND_TO_QWEN)
    if unknown:
        ap.error(f"unknown ASCEND language code(s): {sorted(unknown)}; "
                 f"known: {sorted(ASCEND_TO_QWEN)}")

    per_lang: dict[str, int] = {}
    missing_audio = 0
    empty_text = 0
    dup_ids = 0
    rows_out = 0
    seen_ids: set[str] = set()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for pf in parquets:
            split = os.path.basename(pf).split("-")[0]
            if args.splits and split not in args.splits:
                continue
            table = pq.read_table(pf, columns=COLUMNS).to_pydict()
            n = len(table["path"])
            for i in range(n):
                lang = (table["language"][i] or "").strip()
                if lang not in ASCEND_TO_QWEN or lang not in wanted:
                    continue
                if args.per_lang is not None and per_lang.get(lang, 0) >= args.per_lang:
                    continue
                text = (table["transcription"][i] or "").strip()
                if not text:
                    empty_text += 1
                    continue
                stem = Path(table["path"][i]).stem            # globally-unique id
                ref = wav_dir / f"{stem}.wav"
                if not ref.exists():
                    missing_audio += 1
                    continue
                if stem in seen_ids:                          # paranoia: skip dups
                    dup_ids += 1
                    continue
                seen_ids.add(stem)
                out.write(json.dumps({
                    "id": stem,
                    "mode": "clone",
                    "ref_audio": str(ref),
                    "ref_text": text,
                    "text": text,                             # re-speak the transcript
                    "language": ASCEND_TO_QWEN[lang],
                }, ensure_ascii=False) + "\n")
                per_lang[lang] = per_lang.get(lang, 0) + 1
                rows_out += 1

    print(f"[done] wrote {rows_out} rows to {out_path}")
    for lang in sorted(per_lang):
        print(f"    {lang:<6} {ASCEND_TO_QWEN[lang]:<8} {per_lang[lang]}")
    if empty_text:
        print(f"[warn] {empty_text} rows skipped (empty transcription)")
    if missing_audio:
        print(f"[warn] {missing_audio} rows skipped (wav not found in {wav_dir})")
    if dup_ids:
        print(f"[warn] {dup_ids} rows skipped (duplicate wav stem)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
