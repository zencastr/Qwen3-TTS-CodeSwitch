#!/usr/bin/env python3
"""Thin wrapper around `python -m qwen3_tts_gen.download`.

Run inside the project venv (`source .venv/bin/activate` or `uv run`):
    python scripts/download_models.py --set all
See `python -m qwen3_tts_gen.download --help` for all options.
"""
from qwen3_tts_gen.download import main

if __name__ == "__main__":
    raise SystemExit(main())
