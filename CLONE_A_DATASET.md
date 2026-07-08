# Voice-cloning a speech dataset with Qwen3-TTS

A reusable recipe (hand this to Claude Code) for cloning any speech dataset —
regenerating each clip's utterance in the original speaker's voice — using this repo.

**GOAL:** Voice-clone a speech dataset with Qwen3-TTS.

**EXISTING TOOL (reuse, don't rebuild):** `~/qwen3_tts` — a uv-managed repo wrapping
Qwen3-TTS (package `qwen3_tts_gen`) for batch, manifest-driven voice cloning on the
B300 SLURM cluster. It already handles weight download, generation, resumable SLURM
job arrays, and the Blackwell cuDNN-SDPA fix.

## Steps

1. **Weights** are cached at `/scratch/ick/qwen3_tts/cache`. If missing:
   `uv run python -m qwen3_tts_gen.download --set all`
2. **LANGUAGE LIMIT:** Qwen3-TTS supports ONLY 10 languages — Chinese, English,
   French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish.
   Filter the dataset to these and map its language codes → these Qwen names.
3. **Build a JSONL manifest**, one line per clip:
   `{"id","mode":"clone","ref_audio","ref_text","text","language"}`
   where `ref_audio` = the source clip (absolute path OK), `ref_text` = its
   transcript, `text` = the same transcript (to re-speak it) or any target text,
   `language` = the Qwen language name.
   COPY + ADAPT `~/longtail-lid/scripts/build_fleurs_manifest.py` (change the
   code→Qwen-name map and the metadata reader for the new dataset).
4. **Generate at scale** (one single-GPU job per shard, resumable via `--skip-existing`):
   ```bash
   cd ~/qwen3_tts
   slurm/submit.sh array 8 manifests/<name>.jsonl --out-dir /scratch/ick/qwen3_tts/<name>_clone
   ```
   Outputs land as `<out-dir>/<id>.wav`.

## Gotchas

- Qwen3-TTS output is **24 kHz mono**. If a downstream model needs 16 kHz, resample
  (see `~/longtail-lid/scripts/resample_clone_16k.py`).
- All GPU work runs under **SLURM (`-p gpu`)** on the B300 (sm_103) nodes; the repo
  already disables the broken cuDNN SDPA backend.
- **SMOKE-TEST FIRST:** build a 1-clip-per-language manifest (`--per-lang 1`) and run
  a short debug-GPU job before launching the full array.

The only real per-dataset work is one manifest-builder script; everything else is reused.

---
*Example run: FLEURS → 10 langs × 300 clips = 3,000 cloned utterances at
`/scratch/ick/qwen3_tts/fleurs_clone`. See `manifests/fleurs_full.jsonl` and
`~/longtail-lid/scripts/build_fleurs_manifest.py`.*
