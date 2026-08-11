#!/usr/bin/env bash
#SBATCH --job-name=btz-native-explore
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/native-explore-%j.out
#SBATCH --error=logs/native-explore-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

: "${TRAIN_RUN_ID:?submit with TRAIN_RUN_ID=<training directory>}"
: "${DATA_RUN_ID:?submit with DATA_RUN_ID=<audited corpus directory>}"
: "${RUN_ID:?submit with RUN_ID=<new diagnostic directory>}"
readonly expected_model_hash="3bc3ca17393ce803d3d89ccc02f649cfb755b56b1af944c1f41c82db4114beb6"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly selection="${data_root}/${TRAIN_RUN_ID}/selection.json"
readonly training_data="${data_root}/${DATA_RUN_ID}"
model="$(jq -er --arg key c32-b3-soft_z '.models[$key].path' "$selection")"
readonly model
test -f "$model"
checksum_line="$(sha256sum -- "$model")"
actual_model_hash="${checksum_line%% *}"
readonly checksum_line actual_model_hash
if [[ "$actual_model_hash" != "$expected_model_hash" ]]; then
  echo "selected checkpoint hash does not match the preregistration" >&2
  exit 2
fi

readonly output="${data_root}/${RUN_ID}"
test ! -e "$output"
worktree_status="$(git status --porcelain --untracked-files=normal)"
readonly worktree_status
test -z "$worktree_status"
mkdir -p "$output"

"$python" -m unittest discover -s tests
"$python" scripts/diagnose_model_symmetry.py \
  "$model" \
  --states 512 \
  --batch-size 64 \
  --seed 202608113200 | tee "$output/model_symmetry.json"
"$python" scripts/diagnose_model_calibration.py \
  "$model" "$training_data" \
  --validation-fraction 0.2 \
  --split-seed 202608112301 \
  --batch-size 64 | tee "$output/model_calibration.json"

readonly games=256
readonly simulations=32
readonly batch_size=64
readonly sample_until_ply=4
readonly master_seed=202608113201
labels=(off moderate alpha_zero_mass sharp)
fractions=(0 0.10 0.25 0.25)
concentrations=(10 10 10 2.5)

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  fraction="${fractions[$index]}"
  concentration="${concentrations[$index]}"
  setting_output="${output}/${label}"

  "$python" scripts/generate_neural_selfplay.py "$setting_output" \
    --model "$model" \
    --rules mini \
    --games "$games" \
    --chunk-games 64 \
    --batch-size "$batch_size" \
    --simulations "$simulations" \
    --c-puct 1.5 \
    --sample-until-ply "$sample_until_ply" \
    --temperature 1.0 \
    --noise-fraction "$fraction" \
    --noise-total-concentration "$concentration" \
    --seed "$master_seed"

  "$python" scripts/audit_selfplay.py "$setting_output" \
    --expected-games "$games" \
    --expected-rules breakthrough-5x5-one-row-v1 \
    --expected-simulations "$simulations" \
    --expected-c-puct 1.5 \
    --expected-model-sha256 "$expected_model_hash" \
    --expected-noise-fraction "$fraction" \
    --expected-noise-total-concentration "$concentration" \
    --expected-sample-until-ply "$sample_until_ply" \
    --expected-batch-size "$batch_size" | tee "$setting_output/audit.json"
done

"$python" scripts/summarize_selfplay.py "$output" | tee "$output/summary.json"
for label in moderate alpha_zero_mass sharp; do
  "$python" scripts/compare_selfplay.py \
    "$output/off" "$output/$label" | tee "$output/off_vs_${label}.json"
done
