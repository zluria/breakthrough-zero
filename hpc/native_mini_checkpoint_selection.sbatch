#!/usr/bin/env bash
#SBATCH --job-name=btz-native-select
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/native-select-%j.out
#SBATCH --error=logs/native-select-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"

: "${TRAIN_RUN_ID:?submit with TRAIN_RUN_ID=<training directory>}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly training_root="${data_root}/${TRAIN_RUN_ID}"

test -z "$(git status --porcelain --untracked-files=normal)"
"$python" scripts/select_training_checkpoints.py \
  "$training_root" \
  "$training_root/selection.json" \
  --expected-runs 6
