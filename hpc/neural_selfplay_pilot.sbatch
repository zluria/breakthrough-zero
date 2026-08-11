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
readonly run_label="${RUN_LABEL:-neural-selfplay-pilot}"
readonly output="${data_root}/${run_label}-${SLURM_JOB_ID}"
readonly use_ensemble="${SYMMETRY_ENSEMBLE:-0}"
readonly simulations="${SIMULATIONS:-16}"
readonly sample_until_ply="${SAMPLE_UNTIL_PLY:-12}"
readonly games="${GAMES:-64}"
readonly chunk_games="${CHUNK_GAMES:-64}"
readonly batch_size="${BATCH_SIZE:-64}"
readonly master_seed="${MASTER_SEED:-202608120903}"
for value_name in simulations games chunk_games batch_size; do
  value="${!value_name}"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name^^} must be a positive integer" >&2
    exit 2
  fi
done
if ! [[ "$sample_until_ply" =~ ^[0-9]+$ ]]; then
  echo "SAMPLE_UNTIL_PLY must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$master_seed" =~ ^[0-9]+$ ]]; then
  echo "MASTER_SEED must be a non-negative integer" >&2
  exit 2
fi
if [[ "$run_label" == */* || -z "$run_label" ]]; then
  echo "RUN_LABEL must be one non-empty path component" >&2
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

test -z "$(git status --porcelain --untracked-files=normal)"
"$python" -m unittest discover -s tests
"$python" scripts/generate_neural_selfplay.py "$output" \
  --model "$MODEL_PATH" \
  --rules standard \
  --games "$games" \
  --chunk-games "$chunk_games" \
  --batch-size "$batch_size" \
  --simulations "$simulations" \
  --c-puct 1.5 \
  --sample-until-ply "$sample_until_ply" \
  --temperature 1.0 \
  --noise-fraction 0 \
  "${ensemble_flag[@]}" \
  "${ablation_flags[@]}" \
  --seed "$master_seed"
"$python" scripts/summarize_selfplay.py "$output"
