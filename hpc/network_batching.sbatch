#!/usr/bin/env bash
#SBATCH --job-name=btz-net-batch
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/network-batching-%j.out
#SBATCH --error=logs/network-batching-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

: "${MODEL_PATH:?submit with --export=ALL,MODEL_PATH=<model.keras>}"

test -z "$(git status --porcelain --untracked-files=normal)"
"$python" -m unittest discover -s tests
"$python" scripts/benchmark_network_batching.py "$MODEL_PATH" \
  --batch-sizes 1,4,8,16,32,64,128,256 \
  --seconds 2 \
  --seed 202608120902
