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
: "${PARENT_MODEL_KEY:?submit with the parent model key}"
: "${EXPECTED_PARENT_HASH:?submit with the pinned parent SHA-256}"
: "${TRAIN_SEED:?submit with the preregistered training seed}"
: "${RUN_ID:?submit with a new generation training directory}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly parent_manifest_name="${PARENT_MANIFEST_NAME:-actor.json}"
readonly parent_manifest="${data_root}/${PARENT_TRAIN_RUN_ID}/${parent_manifest_name}"
readonly pretraining="${data_root}/${PRETRAIN_RUN_ID}"
readonly selfplay="${data_root}/${SELFPLAY_RUN_ID}"
readonly output="${data_root}/${RUN_ID}"

current_commit="$(git rev-parse HEAD)"
worktree_status="$(git status --porcelain --untracked-files=normal)"
parent="$(jq -er --arg key "$PARENT_MODEL_KEY" '.models[$key].path' "$parent_manifest")"
recorded_parent_hash="$(jq -er --arg key "$PARENT_MODEL_KEY" '.models[$key].sha256' "$parent_manifest")"
parent_state="$(jq -r --arg key "$PARENT_MODEL_KEY" '.models[$key].training_state // empty' "$parent_manifest")"
readonly current_commit worktree_status parent recorded_parent_hash parent_state
[[ "$current_commit" == "$EXPECTED_COMMIT" ]]
test -z "$worktree_status"
test ! -e "$output"
test -f "$parent"
[[ "$recorded_parent_hash" == "$EXPECTED_PARENT_HASH" ]]
parent_checksum="$(sha256sum -- "$parent")"
readonly parent_checksum
[[ "${parent_checksum%% *}" == "$EXPECTED_PARENT_HASH" ]]
resume_state=()
if [[ -n "$parent_state" ]]; then
  test -f "${parent_state}.index"
  resume_state=(--initial-training-state "$parent_state")
fi

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

"$python" scripts/train_replay.py \
  "$pretraining" \
  "$output" \
  --fresh-input "$selfplay" \
  --historical-fraction 0.25 \
  --initial-model "$parent" \
  "${resume_state[@]}" \
  --target mixed_z_q \
  --channels 32 \
  --residual-blocks 3 \
  --value-hidden 64 \
  --max-steps 100000 \
  --max-training-seconds 120 \
  --max-replay-consumption 4 \
  --evaluate-every 20 \
  --batch-size 256 \
  --learning-rate 0.0003 \
  --l2 0.0001 \
  --validation-fraction 0.2 \
  --seed "$TRAIN_SEED"

jq -e '.status == "complete" and
       .actor_key == "latest" and
       .evaluation_role == "diagnostic_only" and
       (.models.latest.run_step > 0)' \
  "$output/actor.json"
