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
readonly use_ensemble="${SYMMETRY_ENSEMBLE:-0}"
readonly simulations="${SIMULATIONS:-16}"
if ! [[ "$simulations" =~ ^[1-9][0-9]*$ ]]; then
  echo "SIMULATIONS must be a positive integer" >&2
  exit 2
fi
ensemble_flag=()
if [[ "$use_ensemble" == "1" ]]; then
  ensemble_flag=(--symmetry-ensemble)
elif [[ "$use_ensemble" != "0" ]]; then
  echo "SYMMETRY_ENSEMBLE must be 0 or 1" >&2
  exit 2
fi
ablation_flags=()
if [[ "${UNIFORM_POLICY:-0}" == "1" ]]; then
  ablation_flags+=(--uniform-policy)
elif [[ "${UNIFORM_POLICY:-0}" != "0" ]]; then
  echo "UNIFORM_POLICY must be 0 or 1" >&2
  exit 2
fi
if [[ "${ZERO_VALUE:-0}" == "1" ]]; then
  ablation_flags+=(--zero-value)
elif [[ "${ZERO_VALUE:-0}" != "0" ]]; then
  echo "ZERO_VALUE must be 0 or 1" >&2
  exit 2
fi

git diff --quiet
git diff --cached --quiet
"$python" -m unittest discover -s tests
"$python" scripts/generate_neural_selfplay.py "$output" \
  --model "$MODEL_PATH" \
  --rules standard \
  --games 64 \
  --chunk-games 64 \
  --batch-size 64 \
  --simulations "$simulations" \
  --c-puct 1.5 \
  --sample-until-ply 12 \
  --temperature 1.0 \
  --noise-fraction 0 \
  "${ensemble_flag[@]}" \
  "${ablation_flags[@]}" \
  --seed 202608120903
"$python" scripts/summarize_selfplay.py "$output"
