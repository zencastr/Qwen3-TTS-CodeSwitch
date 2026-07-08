"""Build a JSONL manifest from a folder of reference clips + target texts.

This is glue for generating at scale: point it at a directory of reference
WAVs and a file of target lines, and it emits one manifest row per
(reference, text) pair ready for `qwen3_tts_gen.generate`.

Reference transcripts (ref_text) are read from a sidecar text file next to each
clip (same stem, ``--ref-text-suffix`` extension, default ``.txt``). Clips
without a transcript are skipped with a warning (clone mode needs ref_text).

Pairing:
  --pair zip      i-th reference with i-th text (counts should match)
  --pair product  every reference x every text   (default; good for trying a
                  set of lines across many voices)

Examples:
  python -m qwen3_tts_gen.build_manifest \\
      --refs-dir refs/ --texts-file lines.txt --language English \\
      --out manifests/run.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def _read_ref_text(audio_path: Path, suffix: str) -> str | None:
    sidecar = audio_path.with_suffix(suffix)
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip()
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="qwen3-tts-build-manifest",
        description="Build a JSONL manifest from reference clips + target texts.",
    )
    ap.add_argument("--refs-dir", required=True, help="directory of reference audio clips")
    ap.add_argument("--texts-file", required=True,
                    help="file with one target text per line (blank lines ignored)")
    ap.add_argument("--out", required=True, help="output manifest path (.jsonl)")
    ap.add_argument("--language", default="English", help="language for every row")
    ap.add_argument("--ref-text-suffix", default=".txt",
                    help="extension of the sidecar transcript file (default: .txt)")
    ap.add_argument("--pair", choices=("product", "zip"), default="product",
                    help="how to combine references and texts (default: product)")
    ap.add_argument("--abs-paths", action="store_true",
                    help="write absolute ref_audio paths (default: as given)")
    args = ap.parse_args(argv)

    refs_dir = Path(args.refs_dir)
    refs = sorted(p for p in refs_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not refs:
        ap.error(f"no audio clips found in {refs_dir} (looked for {AUDIO_EXTS})")

    texts = [ln.strip() for ln in Path(args.texts_file).read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    if not texts:
        ap.error(f"no non-empty lines in {args.texts_file}")

    # Pair up references with transcripts, skipping any without a sidecar.
    usable = []
    for ref in refs:
        ref_text = _read_ref_text(ref, args.ref_text_suffix)
        if ref_text is None:
            print(f"[skip] {ref.name}: no transcript ({ref.stem}{args.ref_text_suffix})")
            continue
        usable.append((ref, ref_text))
    if not usable:
        ap.error("no reference clips had a sidecar transcript; clone mode needs ref_text")

    if args.pair == "zip":
        n = min(len(usable), len(texts))
        if len(usable) != len(texts):
            print(f"[warn] zip pairing uses {n} of {len(usable)} refs / {len(texts)} texts")
        pairs = [(usable[i][0], usable[i][1], texts[i]) for i in range(n)]
    else:
        pairs = [(ref, rt, txt) for (ref, rt) in usable for txt in texts]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for ref, ref_text, text in pairs:
            ref_audio = str(ref.resolve()) if args.abs_paths else str(ref)
            text_idx = texts.index(text)
            fh.write(json.dumps({
                "id": f"{ref.stem}__{text_idx:04d}",
                "mode": "clone",
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "text": text,
                "language": args.language,
            }, ensure_ascii=False) + "\n")

    print(f"[done] wrote {len(pairs)} rows to {out_path} "
          f"({len(usable)} refs x {len(texts)} texts, pair={args.pair})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
