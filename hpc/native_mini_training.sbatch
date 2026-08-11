#!/usr/bin/env bash
#SBATCH --job-name=btz-native-mini-train
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=00:08:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/native-mini-train-%A_%a.out
#SBATCH --error=logs/native-mini-train-%A_%a.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

: "${DATA_RUN_ID:?submit with DATA_RUN_ID=<audited corpus directory>}"
: "${RUN_ID:?submit with RUN_ID=<stable output directory>}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly input="${data_root}/${DATA_RUN_ID}"

readonly channels_by_task=(32 32 32 64 64 64)
readonly blocks_by_task=(3 3 3 4 4 4)
readonly targets_by_task=(outcome soft_z mixed_z_q outcome soft_z mixed_z_q)
readonly channels="${channels_by_task[$SLURM_ARRAY_TASK_ID]}"
readonly blocks="${blocks_by_task[$SLURM_ARRAY_TASK_ID]}"
readonly target="${targets_by_task[$SLURM_ARRAY_TASK_ID]}"
readonly label="c${channels}-b${blocks}-${target}"
readonly output="${data_root}/${RUN_ID}/${label}"

test -z "$(git status --porcelain --untracked-files=normal)"
"$python" scripts/train_pretraining.py \
  "$input" \
  "$output" \
  --target "$target" \
  --channels "$channels" \
  --residual-blocks "$blocks" \
  --value-hidden 64 \
  --epochs 1000 \
  --max-training-seconds 120 \
  --checkpoint-every 4 \
  --batch-size 256 \
  --learning-rate 0.001 \
  --l2 0.0001 \
  --seed 202608112301
