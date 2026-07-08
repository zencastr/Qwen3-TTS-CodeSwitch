#!/bin/bash
# Sharded, resumable generation across a SLURM job array. Each array task
# processes shard $SLURM_ARRAY_TASK_ID of N and skips outputs that already
# exist, so requeued/failed tasks pick up where they left off.
#
# Submit via:  slurm/submit.sh array N MANIFEST [args...]
# (submit.sh sets --array=0-(N-1) and passes N as the first argument here.)
#SBATCH --job-name=qwen3tts-arr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
set -euo pipefail

ROOT="${REPO_ROOT:-$HOME/qwen3_tts}"
cd "$ROOT"
mkdir -p logs
# shellcheck source=/dev/null
source config/cluster.env
export QWEN3_TTS_CACHE QWEN3_TTS_MODEL QWEN3_TTS_DTYPE QWEN3_TTS_ATTN QWEN3_TTS_CUDNN_SDP
# shellcheck source=/dev/null
source .venv/bin/activate

NUM_SHARDS="${1:?array script needs NUM_SHARDS as arg 1 (use slurm/submit.sh array N ...)}"
shift
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

echo "[slurm] job=$SLURM_JOB_ID task=$SHARD/$NUM_SHARDS host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none}"
srun python -m qwen3_tts_gen.generate "$@" \
    --num-shards "$NUM_SHARDS" --shard-index "$SHARD" --skip-existing
