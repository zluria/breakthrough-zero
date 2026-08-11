#!/usr/bin/env bash
#SBATCH --job-name=btz-native-model-screen
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=6G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=00:15:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/native-model-screen-%A_%a.out
#SBATCH --error=logs/native-model-screen-%A_%a.err

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
readonly labels=(
  c32-b3-outcome c32-b3-soft_z c32-b3-mixed_z_q
  c64-b4-outcome c64-b4-soft_z c64-b4-mixed_z_q
)
readonly label="${labels[$SLURM_ARRAY_TASK_ID]}"
readonly selection="${data_root}/${TRAIN_RUN_ID}/selection.json"
model="$(jq -er --arg key "$label" '.models[$key].path' "$selection")"
readonly model
readonly output="${data_root}/${RUN_ID}/${label}"

test -z "$(git status --porcelain --untracked-files=normal)"
test -n "$model"
test -f "$model"
"$python" scripts/run_mini_tournament.py \
  "$output" \
  --rules mini \
  --pairs 24 \
  --opening-plies 4 \
  --move-seconds 0.05 \
  --time-tolerance-seconds 0.10 \
  --puct-c-puct 1.5 \
  --baselines strong \
  --matchups custom-vs-baselines \
  --model "${label}=${model}" \
  --seed 202608112600
