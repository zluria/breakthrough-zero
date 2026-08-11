#!/usr/bin/env bash
#SBATCH --job-name=btz-native-mini-data
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/native-mini-data-%A_%a.out
#SBATCH --error=logs/native-mini-data-%A_%a.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"

: "${RUN_ID:?submit with RUN_ID=<stable output directory>}"
: "${C_PUCT:?submit with C_PUCT=<screened value>}"
readonly simulations="${SIMULATIONS:-32}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
printf -v shard "shard-%03d" "$SLURM_ARRAY_TASK_ID"
readonly output="${data_root}/${RUN_ID}/${shard}"
readonly seed="$((202608112100 + SLURM_ARRAY_TASK_ID))"

test -z "$(git status --porcelain --untracked-files=normal)"
"$python" scripts/generate_dummy_selfplay.py \
  "$output" \
  --games 64 \
  --chunk-games 16 \
  --simulations "$simulations" \
  --c-puct "$C_PUCT" \
  --sample-until-ply 4 \
  --temperature 1.0 \
  --rules mini \
  --tactical-rollouts \
  --seed "$seed"
"$python" scripts/summarize_selfplay.py "$output"
