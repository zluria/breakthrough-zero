#!/usr/bin/env bash
#SBATCH --job-name=btz-neural-sp
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/neural-selfplay-%j.out
#SBATCH --error=logs/neural-selfplay-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

: "${MODEL_PATH:?submit with --export=ALL,MODEL_PATH=<model.keras>}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly output="${data_root}/neural-selfplay-pilot-${SLURM_JOB_ID}"

git diff --quiet
git diff --cached --quiet
"$python" -m unittest discover -s tests
"$python" scripts/generate_neural_selfplay.py "$output" \
  --model "$MODEL_PATH" \
  --rules standard \
  --games 64 \
  --chunk-games 64 \
  --batch-size 64 \
  --simulations 16 \
  --c-puct 1.5 \
  --sample-until-ply 12 \
  --temperature 1.0 \
  --noise-fraction 0 \
  --seed 202608120903
"$python" scripts/summarize_selfplay.py "$output"
