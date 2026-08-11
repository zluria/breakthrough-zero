#!/usr/bin/env bash
#SBATCH --job-name=btz-sp-sweep
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --array=0-5%4
#SBATCH --output=logs/selfplay-sweep-%A_%a.out
#SBATCH --error=logs/selfplay-sweep-%A_%a.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"

: "${RUN_ID:?submit with --export=ALL,RUN_ID=<stable-run-id>}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"

readonly simulations_by_task=(16 32 64 16 32 64)
readonly rollout_by_task=(uniform uniform uniform tactical tactical tactical)
readonly simulations="${simulations_by_task[$SLURM_ARRAY_TASK_ID]}"
readonly rollout="${rollout_by_task[$SLURM_ARRAY_TASK_ID]}"
readonly label="${rollout}-s${simulations}"
readonly seed="$((202608120100 + SLURM_ARRAY_TASK_ID))"
readonly output="${data_root}/${RUN_ID}/${label}"

arguments=(
  "$output"
  --games 64
  --chunk-games 16
  --simulations "$simulations"
  --c-puct 1.5
  --sample-until-ply 6
  --temperature 1.0
  --rules mini
  --seed "$seed"
)
if [[ "$rollout" == tactical ]]; then
  arguments+=(--tactical-rollouts)
fi

echo "run_id=$RUN_ID task=$SLURM_ARRAY_TASK_ID label=$label seed=$seed"
"$python" scripts/generate_dummy_selfplay.py "${arguments[@]}"
