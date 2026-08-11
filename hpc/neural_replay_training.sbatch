#!/usr/bin/env bash
#SBATCH --job-name=btz-neural-train
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=00:20:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/neural-train-%A_%a.out
#SBATCH --error=logs/neural-train-%A_%a.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

: "${INPUT_A:?submit with INPUT_A=<first replay directory>}"
: "${INPUT_B:?submit with INPUT_B=<second replay directory>}"
: "${INITIAL_MODEL:?submit with INITIAL_MODEL=<model.keras>}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly targets=(outcome soft_z)
readonly target="${targets[$SLURM_ARRAY_TASK_ID]}"
readonly output="${data_root}/neural-training-${SLURM_ARRAY_JOB_ID}/${target}"

test -z "$(git status --porcelain --untracked-files=normal)"
"$python" -m unittest discover -s tests
"$python" scripts/train_pretraining.py \
  "$INPUT_A" \
  "$output" \
  --extra-input "$INPUT_B" \
  --initial-model "$INITIAL_MODEL" \
  --target "$target" \
  --channels 64 \
  --residual-blocks 4 \
  --value-hidden 64 \
  --epochs 5 \
  --batch-size 128 \
  --learning-rate 0.0003 \
  --l2 0.0001 \
  --validation-fraction 0.2 \
  --seed 202608121201
