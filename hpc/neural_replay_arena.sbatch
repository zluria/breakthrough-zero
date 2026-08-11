#!/usr/bin/env bash
#SBATCH --job-name=btz-r01-arena
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=00:30:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/neural-r01-arena-%A_%a.out
#SBATCH --error=logs/neural-r01-arena-%A_%a.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

: "${PRETRAINED_MODEL:?submit with PRETRAINED_MODEL=<model.keras>}"
: "${TRAINING_RUN:?submit with TRAINING_RUN=<training directory>}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly pretrained="pretrained=${PRETRAINED_MODEL}"
readonly outcome="r01-outcome=${TRAINING_RUN}/outcome/model.keras"
readonly soft_z="r01-soft-z=${TRAINING_RUN}/soft_z/model.keras"
readonly first_by_task=("$pretrained" "$pretrained" "$outcome")
readonly second_by_task=("$outcome" "$soft_z" "$soft_z")
readonly output="${data_root}/neural-r01-arena-${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"

test -z "$(git status --porcelain --untracked-files=normal)"
"$python" -m unittest discover -s tests
"$python" scripts/run_mini_tournament.py \
  "$output" \
  --rules standard \
  --pairs 32 \
  --opening-plies 4 \
  --move-seconds 0.05 \
  --time-tolerance-seconds 0.02 \
  --baselines none \
  --seed 202608121301 \
  --ensemble-model "${first_by_task[$SLURM_ARRAY_TASK_ID]}" \
  --ensemble-model "${second_by_task[$SLURM_ARRAY_TASK_ID]}"
