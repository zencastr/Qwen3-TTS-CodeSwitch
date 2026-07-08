# qwen3-tts-gen

Batch **voice-cloning speech generation** with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS),
built to run at scale under SLURM. You hand it a manifest of *(reference clip,
reference transcript, text-to-speak)* rows; it loads the model once and writes
one WAV per row, with sharding and resume for large jobs.

Qwen3-TTS itself is a dependency (the `qwen-tts` package) — this repo is the
manifest/batch/SLURM layer around it. Environment managed with **uv**.

- **Voice cloning** from a reference clip + its transcript (the default workflow)
- Also supports **custom-voice** (preset speakers) and **voice-design** (free-form voice prompts)
- 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- Manifests in **JSONL** or **CSV/TSV**; per-item failures are isolated, not fatal
- SLURM **single-job** and **resumable job-array** scripts, parameterised by [config/cluster.env](config/cluster.env)

---

## Layout

```
qwen3_tts/
├── pyproject.toml            # uv project; Python 3.12; Blackwell (cu128) torch index
├── config/cluster.env        # SLURM + cache + model defaults (edit for your site)
├── src/qwen3_tts_gen/        # the package
│   ├── config.py             # env-driven runtime config + HF cache wiring
│   ├── models.py             # registry of the Qwen3-TTS checkpoints + aliases
│   ├── manifest.py           # JSONL/CSV parsing, validation, sharding, resume
│   ├── backend.py            # thin wrapper over qwen_tts (load + synthesize + write)
│   ├── generate.py           # batch generation CLI (the main entrypoint)
│   ├── download.py           # weight downloader / initializer
│   └── build_manifest.py     # build a manifest from a folder of clips + texts
├── scripts/                  # thin `python scripts/...` wrappers
├── slurm/                    # submit.sh + download/generate/array sbatch files
├── manifests/example.jsonl   # example manifest (one remote-ref row works out of the box)
└── examples/                 # drop your local reference clips here
```

---

## Setup

Requires `uv` (already on this cluster) and runs on the **B300** GPU nodes.

```bash
cd ~/qwen3_tts
uv sync                       # creates .venv (Python 3.12) and installs everything
```

`uv sync` pulls a CUDA-12.8 build of PyTorch for the Blackwell (B300, sm_100)
GPUs — see [Blackwell notes](#blackwell--gpu-notes) if torch can't see the GPU.

### Download the weights (the "initialize the model" step)

Weights are cached on scratch (`/scratch/ick/qwen3_tts/cache` by default) so they
download once and are shared by every job. The 12 Hz tokenizer is always fetched.

```bash
# all variants (1.7B/0.6B Base + CustomVoice + VoiceDesign):
uv run python -m qwen3_tts_gen.download --set all
uv run python -m qwen3_tts_gen.download --list      # show the registry
```

This is pure network I/O (no GPU), so it's fine to run on the login node, or as
a batch job: `slurm/submit.sh download --set all`.

---

## Manifest format

One row per utterance. JSONL (one JSON object per line), or CSV/TSV with a
header. For the default **clone** mode:

```json
{"id": "u001", "ref_audio": "examples/alice.wav", "ref_text": "transcript of the clip", "text": "new words to speak in Alice's voice", "language": "English", "out": "alice/u001.wav"}
```

| field          | required | notes |
|----------------|----------|-------|
| `text`         | ✅       | the text to synthesize (synonyms: `script`, `target_text`) |
| `ref_audio`    | clone    | reference clip: local path **or** URL (`reference_audio`, `prompt_audio`, `audio`) |
| `ref_text`     | clone    | exact transcript of `ref_audio` (`reference_text`, `prompt_text`) |
| `id`           |          | unique id; defaults to `item_000001…`; also the default output name |
| `language`     |          | one of the 10 supported; defaults to `--language` |
| `mode`         |          | `clone` (default) · `custom` · `design` |
| `out`          |          | output path, relative to `--out-dir` unless absolute |
| `speaker`      | custom   | preset speaker name (mode=`custom`) |
| `voice_prompt` | design   | natural-language voice description (mode=`design`) |
| `instruct`     |          | optional style instruction (custom/design) |

Build one from a folder of reference clips (each with a sidecar `.txt`
transcript) and a file of target lines:

```bash
uv run python -m qwen3_tts_gen.build_manifest \
    --refs-dir examples/ --texts-file lines.txt \
    --language English --out manifests/run.jsonl
```

---

## Generate (local / interactive)

GPU work must run under SLURM on this cluster — never on the login node. Grab an
interactive shell on the `debug` partition for quick tries:

```bash
srun --partition=debug --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:20:00 --pty bash
source .venv/bin/activate

# validate the manifest without loading the model:
python -m qwen3_tts_gen.generate manifests/example.jsonl --dry-run

# generate:
python -m qwen3_tts_gen.generate manifests/example.jsonl --out-dir outputs/demo
```

Outputs land in `--out-dir` (one WAV per row) alongside a `report.jsonl`
recording per-item status, errors, and timing.

Useful flags (`--help` for all): `--model 0.6b-base`, `--language German`,
`--audio-root DIR` (resolve relative `ref_audio`), `--skip-existing` (resume),
`--limit N`, `--batch-size N` (best-effort batching; falls back to per-item).

---

## Generate at scale (SLURM)

`slurm/submit.sh` reads [config/cluster.env](config/cluster.env) and passes the
right flags to `sbatch`. Logs go to `logs/`.

**One job, one GPU, whole manifest:**

```bash
slurm/submit.sh generate manifests/run.jsonl --out-dir outputs/run
```

**Job array — shard a big manifest across N tasks (N GPUs in parallel):**

```bash
slurm/submit.sh array 8 manifests/run.jsonl --out-dir outputs/run
```

This submits `--array=0-7`; task *i* processes shard *i* of 8 (round-robin over
rows) and runs with `--skip-existing`, so **requeued or failed tasks resume
automatically**. Each shard writes its own `report.shardN.jsonl`. Tune resources
inline:

```bash
TIME=08:00:00 MEM=64G slurm/submit.sh array 16 manifests/run.jsonl --out-dir outputs/run
```

The cluster default is **4 GPUs/user** on the `gpu` partition (8 B300s total), so
arrays larger than that will queue and drain as GPUs free up — which is fine.

---

## Example: voice-cloned copies of FLEURS

[scripts/build_fleurs_manifest.py](scripts/build_fleurs_manifest.py) turns the
FLEURS dataset at `/scratch/ick/LID/fleurs` into manifests where each clip is
re-spoken in its own (cloned) voice using its own transcript. Only the **10
languages Qwen3-TTS supports** are included (`en_us`, `cmn_hans_cn`, `fr_fr`,
`de_de`, `it_it`, `ja_jp`, `ko_kr`, `pt_br`, `ru_ru`, `es_419` → 300 clips each,
3000 total); the other 73 FLEURS languages are skipped (Qwen3-TTS rejects them).

```bash
# full manifest (3000 clips) + a 1-per-language smoke manifest
uv run python scripts/build_fleurs_manifest.py --out manifests/fleurs_full.jsonl
uv run python scripts/build_fleurs_manifest.py --per-lang 1 --out manifests/fleurs_smoke.jsonl

# scale across GPUs (outputs land as <lang>/<id>.wav, resumable):
slurm/submit.sh array 8 manifests/fleurs_full.jsonl --out-dir /scratch/ick/qwen3_tts/fleurs_clone
```

Verified: a 1-per-language smoke run produced valid audio for all 10 languages
(`ok=10 err=0`, RTF ≈ 1.4 on one B300 without flash-attn).

## Configuration

Defaults live in [config/cluster.env](config/cluster.env) and can be overridden
by exporting before `submit.sh`. Runtime knobs are also plain env vars:

| variable | default | meaning |
|----------|---------|---------|
| `QWEN3_TTS_CACHE` | `/scratch/ick/qwen3_tts/cache` | weight cache (sets `HF_HOME`) |
| `QWEN3_TTS_MODEL` | `1.7b-base` | model alias or HF repo id |
| `QWEN3_TTS_DTYPE` | `bfloat16` | `bfloat16` · `float16` · `float32` |
| `QWEN3_TTS_ATTN`  | `sdpa` | `sdpa` · `flash_attention_2` · `eager` |
| `QWEN3_TTS_DEVICE`| `cuda:0` | torch device |
| `PARTITION` / `GRES` / `TIME` / `MEM` / `CPUS_PER_TASK` / `ACCOUNT` | see file | sbatch flags |

Model aliases: `1.7b-base`, `0.6b-base`, `1.7b-customvoice`, `0.6b-customvoice`,
`1.7b-voicedesign` (or pass a full `Qwen/...` repo id).

---

## Blackwell / GPU notes

- The B300s are **Blackwell Ultra (sm_103)** and need a **CUDA 12.8+** PyTorch
  build. [pyproject.toml](pyproject.toml) pins the `cu128` index (verified:
  `torch 2.11.0+cu128`, cuDNN 9.19, sees the B300). If `torch.cuda.is_available()`
  is `False` or you hit a `no kernel image` / `sm_103` error, bump `cu128` →
  `cu129`/`cu130` in pyproject and re-run `uv lock && uv sync`.
- **cuDNN SDPA is disabled by default** (`QWEN3_TTS_CUDNN_SDP=0`). On sm_103 the
  cuDNN 9.x attention backend has no valid execution plan and crashes inside the
  Mimi audio codec (`cuDNN Frontend error: No valid execution plans built`).
  Disabling just that backend (done automatically in `load_model`) lets SDPA use
  the mem-efficient/math backends; cuDNN convolutions stay on. Do **not** work
  around it with `torch.backends.cudnn.enabled=False` — that segfaults here.
- **flash-attn is opt-in.** It has no prebuilt Blackwell wheels and the login
  node has no `nvcc`, so the default attention is PyTorch **SDPA**
  (`QWEN3_TTS_ATTN=sdpa`). If you build flash-attn yourself, set
  `QWEN3_TTS_ATTN=flash_attention_2`.

## Troubleshooting

- **`SoX could not be found!` at import** — harmless for this pipeline. `qwen_tts`
  imports `sox` in its *25 Hz* tokenizer path, which the 12 Hz models here never
  use; reference audio is loaded via `soundfile`/`librosa`. Ignore it (or install
  the `sox` CLI to silence it).
- **`Warning: flash-attn is not installed. Will only run the manual PyTorch
  version.`** — expected. flash-attn is opt-in (see above); the PyTorch fallback
  is the default and works fine.
- **`NotImplementedError: ... has no 'generate_voice_design'`** — the installed
  `qwen-tts` names that method differently. Check the real API and update the
  mapping: `uv run python -m qwen3_tts_gen.backend --inspect`, then edit
  `_MODE_METHODS` in [src/qwen3_tts_gen/backend.py](src/qwen3_tts_gen/backend.py).
  (Clone and custom-voice are the verified paths.)
- **Items error with missing `ref_audio`** — paths are relative to where you run
  unless you pass `--audio-root`. Use `--dry-run` to list resolved paths and
  flag missing local references before submitting.
- **Re-running** — `--skip-existing` (always on for array jobs) skips rows whose
  output WAV already exists; delete the WAV to regenerate.

## License

This wrapper: see repository. Qwen3-TTS models and the `qwen-tts` package are
Apache-2.0 (© Alibaba Cloud / the Qwen team).
