#!/usr/bin/env bash
# Submit a qwen3-tts-gen job, applying the settings in config/cluster.env as
# sbatch CLI flags (which override the #SBATCH defaults baked into each script).
#
# Usage:
#   slurm/submit.sh download [download args...]
#   slurm/submit.sh generate MANIFEST [generate args...]
#   slurm/submit.sh array N MANIFEST [generate args...]   # shard across N tasks
#
# Examples:
#   slurm/submit.sh download --set all
#   slurm/submit.sh generate manifests/example.jsonl --out-dir outputs/demo
#   slurm/submit.sh array 8 manifests/big.jsonl --out-dir outputs/big
#   TIME=08:00:00 GRES=gpu:1 slurm/submit.sh array 16 manifests/big.jsonl
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
# shellcheck source=/dev/null
source "$ROOT/config/cluster.env"
mkdir -p "$ROOT/logs"

# GPU sbatch flags (used by generate / array). CLI flags beat #SBATCH lines.
gpu_args=(--partition="$PARTITION" --gres="$GRES"
          --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM" --time="$TIME")
acct_args=()
[[ -n "${ACCOUNT:-}" ]] && acct_args+=(--account="$ACCOUNT")

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  echo "usage: submit.sh {download|generate|array} ..." >&2
  exit 2
fi
shift

case "$cmd" in
  download)
    # Download is pure network I/O -> no GPU; the script targets the cpu partition.
    exec sbatch "${acct_args[@]}" "$ROOT/slurm/download_models.sbatch" "$@"
    ;;
  generate)
    [[ $# -ge 1 ]] || { echo "generate needs a MANIFEST" >&2; exit 2; }
    exec sbatch "${gpu_args[@]}" "${acct_args[@]}" \
         "$ROOT/slurm/generate.sbatch" "$@"
    ;;
  array)
    n="${1:?array needs N (number of shards)}"; shift
    [[ $# -ge 1 ]] || { echo "array needs a MANIFEST after N" >&2; exit 2; }
    exec sbatch "${gpu_args[@]}" "${acct_args[@]}" --array="0-$((n - 1))" \
         "$ROOT/slurm/generate_array.sbatch" "$n" "$@"
    ;;
  *)
    echo "unknown command: $cmd (use download|generate|array)" >&2
    exit 2
    ;;
esac
