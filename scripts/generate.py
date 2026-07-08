#!/usr/bin/env python3
"""Thin wrapper around `python -m qwen3_tts_gen.generate`.

Run inside the project venv (`source .venv/bin/activate` or `uv run`):
    python scripts/generate.py manifests/example.jsonl --out-dir outputs/demo
See `python -m qwen3_tts_gen.generate --help` for all options.
"""
from qwen3_tts_gen.generate import main

if __name__ == "__main__":
    raise SystemExit(main())
