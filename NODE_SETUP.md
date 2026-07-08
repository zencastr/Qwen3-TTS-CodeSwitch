# New-node setup: qwen3_tts

Handoff checklist for bootstrapping this workspace on a fresh node. Run top
to bottom; each step is safe to skip if already done.

## 1. Clone the repo

```bash
git clone https://github.com/zencastr/Qwen3-TTS-CodeSwitch.git ~/qwen3_tts
cd ~/qwen3_tts
```

## 2. Python environment

```bash
uv sync              # creates .venv/ from pyproject.toml + uv.lock
source .venv/bin/activate
```

## 3. Cluster config

Check `config/cluster.env` matches the new cluster (partition names, GRES
syntax, `TIME` budgets). On a Blackwell (sm_103) box, `QWEN3_TTS_CUDNN_SDP=0`
must stay disabled — the cuDNN SDPA backend has no valid execution plan on
that architecture and crashes inside the audio codec.

```bash
cat config/cluster.env
# override anything cluster-specific in config/cluster.local.env (gitignored)
```

## 4. Download model weights

```bash
slurm/submit.sh download --set all
# or directly:
python scripts/download_models.py --set all
```

Weights land under `$QWEN3_TTS_CACHE` (default `/scratch/ick/qwen3_tts/cache`,
~5GB). Do not copy these from the old node — just re-download.

## 5. Download source datasets

The committed manifests (`manifests/*.jsonl`) are ready to use as-is — you
only need the *source* audio/text they point at, materialized on this node.

**FLEURS** (source for `fleurs_full.jsonl`, ~9.4GB, expected at
`/scratch/ick/LID/fleurs` with a `manifest.tsv` + `<lang>/` clip dirs — this
exact layout is what `scripts/build_fleurs_manifest.py --fleurs-root` reads):

```bash
# materialized by the sibling whisper-codeswitching repo's LID pipeline
cd ~/whisper-codeswitching
uv sync --extra lid
python experiments/lid/build_eval_set.py --dataset fleurs
# then symlink/copy its fleurs output into /scratch/ick/LID/fleurs in the
# manifest.tsv + <lang>/ layout build_fleurs_manifest.py expects
```

**CodemixQA** (source for `codemixqa_full.jsonl`): a parquet expected at
`/scratch/ick/qwen3_tts/codemixqa/codemixqa_24k.parquet`
(`scripts/build_codemixqa_manifest.py --codemixqa-parquet` default). Pull the
original CodemixQA HF dataset and filter to the 6 Qwen3-TTS-supported
languages the same way the existing parquet was built (see the CodemixQA
dataset memory for which 6 of 16 languages overlap).

Re-run the manifest builders only if you need to regenerate the `.jsonl`
files themselves (different language subset, per-lang caps, etc).

## 6. Sanity check before a big run

```bash
python -m qwen3_tts_gen.generate manifests/example.jsonl --dry-run
```

## 7. Generated clone datasets (previous node's outputs)

The actual synthesized audio (`*_clone/` dirs, several GB each) is **not**
in git — it's compute-expensive GPU output, not something to regenerate by
default. It should arrive as a tarball transferred separately (see the
background compression task run on the old node). Once you have it:

```bash
tar xzf codemixqa_clone_and_friends.tar.gz -C /scratch/ick/qwen3_tts/
```

## 8. Resuming in-flight jobs

Check `logs/` (gitignored, not transferred) is gone — that's fine, it's just
SLURM stdout. To resume any interrupted generation run, reruns with
`--skip-existing` will pick up where the old node left off, as long as the
output directory (`/scratch/.../*_clone`) came along in the tarball:

```bash
slurm/submit.sh array 16 manifests/codemixqa_full.jsonl \
  --out-dir /scratch/ick/qwen3_tts/codemixqa_clone --skip-existing \
  --max-new-tokens 600 --repetition-penalty 1.3
```

## Known gotchas (see repo history / commit messages for detail)

- `language="auto"` + default sampling can non-deterministically hang on
  code-switched text (see the "Diagnose and mitigate auto-language runaway
  decoding" commit). Always pass `--max-new-tokens` and consider
  `--repetition-penalty 1.3` for CodemixQA-style generation.
- Static batching (`--batch-size > 1`) gates the whole batch on its slowest
  sequence — the `max_new_tokens` cap matters even more at batch size > 1.
