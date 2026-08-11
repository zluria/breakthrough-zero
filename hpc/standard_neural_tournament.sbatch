#!/usr/bin/env bash
#SBATCH --job-name=btz-std-arena
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/standard-arena-%j.out
#SBATCH --error=logs/standard-arena-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

: "${TRAINING_RUN:?submit with --export=ALL,TRAINING_RUN=<run-directory>}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly models="${data_root}/${TRAINING_RUN}"
readonly output="${data_root}/standard-neural-arena-${SLURM_JOB_ID}"

git diff --quiet
git diff --cached --quiet
"$python" -m unittest discover -s tests
"$python" scripts/run_mini_tournament.py \
  "$output" \
  --rules standard \
  --pairs 8 \
  --opening-plies 4 \
  --move-seconds 0.05 \
  --time-tolerance-seconds 0.10 \
  --seed 202608120901 \
  --model s32-outcome="${models}/s32-outcome/model.keras" \
  --model s32-soft-z="${models}/s32-soft_z/model.keras" \
  --model s64-outcome="${models}/s64-outcome/model.keras" \
  --model s64-soft-z="${models}/s64-soft_z/model.keras"
