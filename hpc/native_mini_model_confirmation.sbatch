#!/usr/bin/env bash
#SBATCH --job-name=btz-native-model-confirm
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=7G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/native-model-confirm-%j.out
#SBATCH --error=logs/native-model-confirm-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

: "${TRAIN_RUN_ID:?submit with TRAIN_RUN_ID=<training directory>}"
: "${RUN_ID:?submit with RUN_ID=<arena output directory>}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly selection="${data_root}/${TRAIN_RUN_ID}/selection.json"

small="$(jq -er --arg key c32-b3-soft_z '.models[$key].path' "$selection")"
large="$(jq -er --arg key c64-b4-soft_z '.models[$key].path' "$selection")"
readonly small large
test -f "$small"
test -f "$large"

readonly output="${data_root}/${RUN_ID}"
test -z "$(git status --porcelain --untracked-files=normal)"
"$python" scripts/run_mini_tournament.py \
  "$output" \
  --rules mini \
  --pairs 64 \
  --opening-plies 4 \
  --move-seconds 0.05 \
  --time-tolerance-seconds 0.10 \
  --puct-c-puct 1.5 \
  --baselines strong \
  --matchups custom-relevant \
  --model "c32-b3-soft_z=${small}" \
  --model "c64-b4-soft_z=${large}" \
  --seed 202608112700
