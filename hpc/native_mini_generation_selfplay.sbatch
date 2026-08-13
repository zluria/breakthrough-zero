#!/usr/bin/env bash
#SBATCH --job-name=btz-mini-gen-data
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=7G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/mini-gen-data-%j.out
#SBATCH --error=logs/mini-gen-data-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

: "${EXPECTED_COMMIT:?submit with the published Git commit}"
: "${MODEL_TRAIN_RUN_ID:?submit with the actor-model directory}"
: "${MODEL_KEY:?submit with the actor model key}"
: "${EXPECTED_MODEL_HASH:?submit with the pinned model SHA-256}"
: "${SELFPLAY_SEED:?submit with the preregistered self-play seed}"
: "${RUN_ID:?submit with a new self-play directory}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly model_manifest_name="${MODEL_MANIFEST_NAME:-actor.json}"
readonly model_manifest="${data_root}/${MODEL_TRAIN_RUN_ID}/${model_manifest_name}"
readonly output="${data_root}/${RUN_ID}"

current_commit="$(git rev-parse HEAD)"
worktree_status="$(git status --porcelain --untracked-files=normal)"
model="$(jq -er --arg key "$MODEL_KEY" '.models[$key].path' "$model_manifest")"
recorded_model_hash="$(jq -er --arg key "$MODEL_KEY" '.models[$key].sha256' "$model_manifest")"
readonly current_commit worktree_status model recorded_model_hash
[[ "$current_commit" == "$EXPECTED_COMMIT" ]]
test -z "$worktree_status"
test ! -e "$output"
test -f "$model"
[[ "$recorded_model_hash" == "$EXPECTED_MODEL_HASH" ]]
model_checksum="$(sha256sum -- "$model")"
readonly model_checksum
[[ "${model_checksum%% *}" == "$EXPECTED_MODEL_HASH" ]]

"$python" -m unittest discover -s tests
"$python" scripts/generate_neural_selfplay.py "$output" \
  --model "$model" \
  --rules mini \
  --games 256 \
  --chunk-games 64 \
  --batch-size 64 \
  --simulations 32 \
  --c-puct 1.5 \
  --sample-until-ply 4 \
  --temperature 1.0 \
  --noise-fraction 0 \
  --noise-total-concentration 10 \
  --seed "$SELFPLAY_SEED"

"$python" scripts/audit_selfplay.py "$output" \
  --expected-games 256 \
  --expected-rules breakthrough-5x5-one-row-v1 \
  --expected-simulations 32 \
  --expected-c-puct 1.5 \
  --expected-model-sha256 "$EXPECTED_MODEL_HASH" \
  --expected-noise-fraction 0 \
  --expected-noise-total-concentration 10 \
  --expected-sample-until-ply 4 \
  --expected-batch-size 64 | tee "$output/audit.json"
