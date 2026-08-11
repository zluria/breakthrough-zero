#!/usr/bin/env bash
#SBATCH --job-name=btz-std-probe
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/standard-probe-%A_%a.out
#SBATCH --error=logs/standard-probe-%A_%a.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"

: "${RUN_ID:?submit with --export=ALL,RUN_ID=<stable-run-id>}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly simulations_by_task=(16 32 64)
readonly simulations="${simulations_by_task[$SLURM_ARRAY_TASK_ID]}"
readonly seed="$((202608120500 + SLURM_ARRAY_TASK_ID))"
readonly output="${data_root}/${RUN_ID}/tactical-s${simulations}"

echo "run_id=$RUN_ID task=$SLURM_ARRAY_TASK_ID simulations=$simulations seed=$seed"
"$python" scripts/generate_dummy_selfplay.py \
  "$output" \
  --games 8 \
  --chunk-games 4 \
  --simulations "$simulations" \
  --c-puct 1.5 \
  --sample-until-ply 12 \
  --temperature 1.0 \
  --rules standard \
  --tactical-rollouts \
  --seed "$seed"
