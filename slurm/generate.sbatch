#!/bin/bash
# Single-GPU generation over a whole manifest. Submit via:
#   slurm/submit.sh generate MANIFEST [args...]
#SBATCH --job-name=qwen3tts-gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail

ROOT="${REPO_ROOT:-$HOME/qwen3_tts}"
cd "$ROOT"
mkdir -p logs
# shellcheck source=/dev/null
source config/cluster.env
export QWEN3_TTS_CACHE QWEN3_TTS_MODEL QWEN3_TTS_DTYPE QWEN3_TTS_ATTN QWEN3_TTS_CUDNN_SDP
# shellcheck source=/dev/null
source .venv/bin/activate

echo "[slurm] job=$SLURM_JOB_ID host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none}"
srun python -m qwen3_tts_gen.generate "$@"
