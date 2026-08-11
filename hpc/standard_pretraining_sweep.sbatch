#!/usr/bin/env bash
#SBATCH --job-name=btz-std-train
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=00:20:00
#SBATCH --array=0-3%2
#SBATCH --output=logs/standard-train-%A_%a.out
#SBATCH --error=logs/standard-train-%A_%a.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"

: "${RUN_ID:?submit with --export=ALL,RUN_ID=<stable-run-id>}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly source_run="${data_root}/standard-stage1-20260812-4b82411"

readonly simulations_by_task=(32 32 64 64)
readonly targets_by_task=(outcome soft_z outcome soft_z)
readonly simulations="${simulations_by_task[$SLURM_ARRAY_TASK_ID]}"
readonly target="${targets_by_task[$SLURM_ARRAY_TASK_ID]}"
readonly label="s${simulations}-${target}"
readonly input="${source_run}/tactical-s${simulations}"
readonly output="${data_root}/${RUN_ID}/${label}"

echo "run_id=$RUN_ID task=$SLURM_ARRAY_TASK_ID label=$label"
"$python" scripts/train_pretraining.py \
  "$input" \
  "$output" \
  --target "$target" \
  --channels 64 \
  --residual-blocks 4 \
  --value-hidden 64 \
  --epochs 40 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --l2 0.0001 \
  --max-train-positions 2500 \
  --max-validation-positions 600 \
  --seed 202608120801
