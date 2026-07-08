#!/usr/bin/env python3
"""Build a qwen3_tts_gen manifest from the FLEURS dataset for voice cloning.

For each FLEURS clip in a language Qwen3-TTS supports, emit a voice-clone row
whose reference IS the clip and whose target text is the clip's own transcript
-> a voice-cloned copy of the utterance (output organized as <lang>/<id>.wav).

Only the 10 Qwen3-TTS-supported languages are included; every other FLEURS
language is skipped and reported (Qwen3-TTS would reject them).

Usage:
  python scripts/build_fleurs_manifest.py \
      --fleurs-root /scratch/ick/LID/fleurs --out manifests/fleurs_full.jsonl
  python scripts/build_fleurs_manifest.py --per-lang 1 --out manifests/fleurs_smoke.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# FLEURS language code -> Qwen3-TTS language name (the supported 10, verified
# against the model config's talker_config.codec_language_id).
FLEURS_TO_QWEN = {
    "cmn_hans_cn": "Chinese",
    "en_us": "English",
    "fr_fr": "French",
    "de_de": "German",
    "it_it": "Italian",
    "ja_jp": "Japanese",
    "ko_kr": "Korean",
    "pt_br": "Portuguese",
    "ru_ru": "Russian",
    "es_419": "Spanish",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fleurs-root", default="/scratch/ick/LID/fleurs",
                    help="dir containing manifest.tsv and <lang>/ clip dirs")
    ap.add_argument("--manifest-tsv", default=None,
                    help="FLEURS manifest.tsv (default: <fleurs-root>/manifest.tsv)")
    ap.add_argument("--audio-root", default=None,
                    help="root the tsv 'audio_path' is relative to "
                         "(default: parent of --fleurs-root)")
    ap.add_argument("--out", required=True, help="output JSONL manifest path")
    ap.add_argument("--per-lang", type=int, default=None,
                    help="take at most N clips per language (default: all)")
    ap.add_argument("--languages", nargs="+", default=None,
                    help="subset of FLEURS codes to include (default: all 10 supported)")
    args = ap.parse_args(argv)

    fleurs_root = Path(args.fleurs_root)
    tsv = Path(args.manifest_tsv) if args.manifest_tsv else fleurs_root / "manifest.tsv"
    audio_root = Path(args.audio_root) if args.audio_root else fleurs_root.parent
    if not tsv.exists():
        ap.error(f"manifest.tsv not found: {tsv}")

    wanted = set(args.languages) if args.languages else set(FLEURS_TO_QWEN)
    unknown = wanted - set(FLEURS_TO_QWEN)
    if unknown:
        ap.error(f"not supported by Qwen3-TTS: {sorted(unknown)}; "
                 f"supported FLEURS codes: {sorted(FLEURS_TO_QWEN)}")

    per_lang: dict[str, int] = {}
    skipped: dict[str, int] = {}
    missing_audio = 0
    rows_out = 0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv.open(encoding="utf-8", newline="") as fh, out_path.open("w", encoding="utf-8") as out:
        for row in csv.DictReader(fh, delimiter="\t"):
            lang = (row.get("config") or "").strip()
            if lang not in FLEURS_TO_QWEN:
                skipped[lang] = skipped.get(lang, 0) + 1
                continue
            if lang not in wanted:
                continue
            if args.per_lang is not None and per_lang.get(lang, 0) >= args.per_lang:
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            audio_path = (row.get("audio_path") or "").strip()
            ref = audio_root / audio_path
            if not ref.exists():
                alt = fleurs_root / lang / Path(audio_path).name
                if alt.exists():
                    ref = alt
                else:
                    missing_audio += 1
                    continue
            utt_id = (row.get("utt_id") or f"{lang}/{per_lang.get(lang, 0):04d}").strip()
            out.write(json.dumps({
                "id": utt_id,
                "mode": "clone",
                "ref_audio": str(ref),
                "ref_text": text,
                "text": text,                       # re-speak the clip's own transcript
                "language": FLEURS_TO_QWEN[lang],
            }, ensure_ascii=False) + "\n")
            per_lang[lang] = per_lang.get(lang, 0) + 1
            rows_out += 1

    print(f"[done] wrote {rows_out} rows to {out_path}")
    for lang in sorted(per_lang):
        print(f"    {lang:<12} {FLEURS_TO_QWEN[lang]:<10} {per_lang[lang]}")
    if missing_audio:
        print(f"[warn] {missing_audio} rows skipped (audio file not found)")
    if skipped:
        print(f"[info] skipped {sum(skipped.values())} rows across "
              f"{len(skipped)} unsupported FLEURS languages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
