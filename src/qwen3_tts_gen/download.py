"""Initialize the project: download Qwen3-TTS weights into the scratch cache.

The 12 Hz tokenizer is always fetched (every generation model needs it).
Re-running is safe: already-downloaded files are skipped/resumed.

Examples:
  python -m qwen3_tts_gen.download                  # the default set (all variants)
  python -m qwen3_tts_gen.download --set base       # 1.7B + 0.6B base only
  python -m qwen3_tts_gen.download --models 1.7b-base 0.6b-base
  python -m qwen3_tts_gen.download --list           # print the registry and exit
"""
from __future__ import annotations

import argparse

from .config import configure_hf_cache
from .models import DOWNLOAD_SETS, MODELS, TOKENIZER, resolve_model


def _dir_size_bytes(path) -> int:
    from pathlib import Path

    total = 0
    for p in Path(path).rglob("*"):
        try:
            # HF snapshots are symlinks into blobs/; is_file()/stat() follow the
            # link, so we count the real file size exactly once.
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}TB"


def _list_registry() -> None:
    print(f"tokenizer (always): {TOKENIZER}")
    print("\nmodels:")
    for alias, spec in MODELS.items():
        print(f"  {alias:<18} {spec.repo_id:<42} {spec.params:<5} {spec.summary}")
    print("\ndownload sets:")
    for name, aliases in DOWNLOAD_SETS.items():
        print(f"  {name:<10} {', '.join(aliases)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="qwen3-tts-download",
        description="Download Qwen3-TTS weights into the shared scratch cache.",
    )
    ap.add_argument("--set", dest="set_name", default="all", choices=sorted(DOWNLOAD_SETS),
                    help="named group of models to fetch (default: all)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="explicit model aliases or repo ids (overrides --set)")
    ap.add_argument("--cache", default=None, help="override cache dir (QWEN3_TTS_CACHE)")
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    args = ap.parse_args(argv)

    if args.list:
        _list_registry()
        return 0

    cache = configure_hf_cache(args.cache)

    selected = args.models if args.models else DOWNLOAD_SETS[args.set_name]
    repo_ids = [TOKENIZER] + [resolve_model(m) for m in selected]
    # De-duplicate while preserving order.
    repo_ids = list(dict.fromkeys(repo_ids))

    print(f"[cache] {cache}")
    print(f"[plan]  {len(repo_ids)} repo(s): {', '.join(repo_ids)}")

    from huggingface_hub import snapshot_download

    grand_total = 0
    for i, repo_id in enumerate(repo_ids, 1):
        print(f"\n[{i}/{len(repo_ids)}] {repo_id}")
        local_dir = snapshot_download(repo_id=repo_id)
        size = _dir_size_bytes(local_dir)
        grand_total += size
        print(f"    -> {local_dir}  ({_human(size)})")

    print(f"\n[done] {len(repo_ids)} repo(s) ready in {cache}  (total ~{_human(grand_total)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
