#!/bin/bash
# Download Qwen3-TTS weights into the scratch cache. Pure network I/O, so this
# runs on the CPU partition (no --gres). Submit via:  slurm/submit.sh download
#SBATCH --job-name=qwen3tts-dl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail

ROOT="${REPO_ROOT:-$HOME/qwen3_tts}"
cd "$ROOT"
mkdir -p logs
# shellcheck source=/dev/null
source config/cluster.env
export QWEN3_TTS_CACHE
# shellcheck source=/dev/null
source .venv/bin/activate

# Default to all variants; pass e.g. `--set base` through submit.sh to narrow.
ARGS=("$@")
[[ ${#ARGS[@]} -eq 0 ]] && ARGS=(--set all)
srun python -m qwen3_tts_gen.download "${ARGS[@]}"
