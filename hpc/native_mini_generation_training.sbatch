#!/usr/bin/env bash
#SBATCH --job-name=btz-mini-gen-train
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=7G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/mini-gen-train-%j.out
#SBATCH --error=logs/mini-gen-train-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

: "${EXPECTED_COMMIT:?submit with the published Git commit}"
: "${PRETRAIN_RUN_ID:?submit with the rollout corpus directory}"
: "${SELFPLAY_RUN_ID:?submit with the neural self-play directory}"
: "${PARENT_TRAIN_RUN_ID:?submit with the parent training directory}"
: "${PARENT_MODEL_KEY:?submit with the parent selection key}"
: "${EXPECTED_PARENT_HASH:?submit with the pinned parent SHA-256}"
: "${TRAIN_SEED:?submit with the preregistered training seed}"
: "${RUN_ID:?submit with a new generation training directory}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly parent_selection="${data_root}/${PARENT_TRAIN_RUN_ID}/selection.json"
readonly pretraining="${data_root}/${PRETRAIN_RUN_ID}"
readonly selfplay="${data_root}/${SELFPLAY_RUN_ID}"
readonly output="${data_root}/${RUN_ID}"

current_commit="$(git rev-parse HEAD)"
worktree_status="$(git status --porcelain --untracked-files=normal)"
parent="$(jq -er --arg key "$PARENT_MODEL_KEY" '.models[$key].path' "$parent_selection")"
recorded_parent_hash="$(jq -er --arg key "$PARENT_MODEL_KEY" '.models[$key].sha256' "$parent_selection")"
readonly current_commit worktree_status parent recorded_parent_hash
[[ "$current_commit" == "$EXPECTED_COMMIT" ]]
test -z "$worktree_status"
test ! -e "$output"
test -f "$parent"
[[ "$recorded_parent_hash" == "$EXPECTED_PARENT_HASH" ]]
parent_checksum="$(sha256sum -- "$parent")"
readonly parent_checksum
[[ "${parent_checksum%% *}" == "$EXPECTED_PARENT_HASH" ]]

"$python" -m unittest discover -s tests
"$python" scripts/audit_selfplay.py "$selfplay" \
  --expected-games 256 \
  --expected-rules breakthrough-5x5-one-row-v1 \
  --expected-simulations 32 \
  --expected-c-puct 1.5 \
  --expected-model-sha256 "$EXPECTED_PARENT_HASH" \
  --expected-noise-fraction 0 \
  --expected-noise-total-concentration 10 \
  --expected-sample-until-ply 4 \
  --expected-batch-size 64

"$python" scripts/train_pretraining.py \
  "$pretraining" \
  "$output/child" \
  --extra-input "$selfplay" \
  --primary-loss-fraction 0.75 \
  --initial-model "$parent" \
  --target mixed_z_q \
  --channels 32 \
  --residual-blocks 3 \
  --value-hidden 64 \
  --epochs 1000 \
  --max-training-seconds 120 \
  --checkpoint-every 4 \
  --batch-size 256 \
  --learning-rate 0.0003 \
  --l2 0.0001 \
  --validation-fraction 0.2 \
  --seed "$TRAIN_SEED"

"$python" scripts/select_training_checkpoints.py \
  "$output" "$output/selection.json" --expected-runs 1
