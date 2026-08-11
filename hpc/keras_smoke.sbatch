#!/bin/bash -l
#SBATCH --job-name=btz_smoke
#SBATCH --partition=rtx3070
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --output=logs/hpc_smoke_%j.out
#SBATCH --error=logs/hpc_smoke_%j.err

set -euo pipefail

echo "host=$(hostname)"
echo "started=$(date -Is)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

module load anaconda/anaconda
conda run --no-capture-output -n zur_env python scripts/hpc_keras_smoke.py

echo "finished=$(date -Is)"
