#!/usr/bin/env bash
#SBATCH --job-name=btz-mini-cpuct
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/mini-cpuct-%A_%a.out
#SBATCH --error=logs/mini-cpuct-%A_%a.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"

: "${RUN_ID:?submit with RUN_ID=<stable output directory>}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly cpuct_by_task=(0.25 0.75 1.5 3.0)
readonly cpuct="${cpuct_by_task[$SLURM_ARRAY_TASK_ID]}"
readonly output="${data_root}/${RUN_ID}/cpuct-${cpuct}"

test -z "$(git status --porcelain --untracked-files=normal)"
"$python" scripts/run_mini_tournament.py \
  "$output" \
  --rules mini \
  --pairs 48 \
  --opening-plies 4 \
  --move-seconds 0.02 \
  --time-tolerance-seconds 0.01 \
  --puct-c-puct "$cpuct" \
  --seed 202608111700
