#!/usr/bin/env bash
#SBATCH --job-name=btz-neural-arena
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/neural-arena-%j.out
#SBATCH --error=logs/neural-arena-%j.err

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
readonly output="${data_root}/mini-neural-arena-${SLURM_JOB_ID}"

test -z "$(git status --porcelain --untracked-files=normal)"
"$python" -m unittest discover -s tests
"$python" scripts/run_mini_tournament.py \
  "$output" \
  --pairs 16 \
  --opening-plies 4 \
  --move-seconds 0.05 \
  --time-tolerance-seconds 0.02 \
  --seed 202608120401 \
  --model tiny-outcome="${models}/c32-b3-outcome/model.keras" \
  --model tiny-soft-z="${models}/c32-b3-soft_z/model.keras" \
  --model base-outcome="${models}/c64-b4-outcome/model.keras" \
  --model base-soft-z="${models}/c64-b4-soft_z/model.keras"
