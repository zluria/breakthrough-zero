#!/usr/bin/env bash
#SBATCH --job-name=btz-mini-large-data
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=01:20:00
#SBATCH --output=logs/mini-large-data-%j.out
#SBATCH --error=logs/mini-large-data-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

: "${EXPECTED_COMMIT:?submit with the published Git commit}"
: "${SELFPLAY_RUN_ID:?submit with a new self-play directory}"
: "${TRAIN_RUN_ID:?submit with a new training directory}"
: "${ARENA_RUN_ID:?submit with a new arena directory}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly parent_selection="${data_root}/native-mini-generation-1-20260813-v1/selection.json"
readonly pretraining="${data_root}/native-mini-data-20260811-c15-s32-v1"
readonly selfplay="${data_root}/${SELFPLAY_RUN_ID}"
readonly training="${data_root}/${TRAIN_RUN_ID}"
readonly arena="${data_root}/${ARENA_RUN_ID}"
readonly expected_parent_hash="1c68f6cd159ed6cd273f9703bfb2ff1848d0c2dde5fae13e7a0e786263781ce3"
readonly games=12288
readonly selfplay_seed=202608133501
readonly training_seed=202608135101
readonly arena_seed=202608135201

current_commit="$(git rev-parse HEAD)"
worktree_status="$(git status --porcelain --untracked-files=normal)"
parent="$(jq -er '.models.child.path' "$parent_selection")"
recorded_parent_hash="$(jq -er '.models.child.sha256' "$parent_selection")"
readonly current_commit worktree_status parent recorded_parent_hash
[[ "$current_commit" == "$EXPECTED_COMMIT" ]]
test -z "$worktree_status"
test ! -e "$selfplay"
test ! -e "$training"
test ! -e "$arena"
test -f "$parent"
[[ "$recorded_parent_hash" == "$expected_parent_hash" ]]
parent_checksum="$(sha256sum -- "$parent")"
readonly parent_checksum
[[ "${parent_checksum%% *}" == "$expected_parent_hash" ]]

"$python" -m unittest discover -s tests

"$python" scripts/generate_neural_selfplay.py "$selfplay" \
  --model "$parent" \
  --rules mini \
  --games "$games" \
  --chunk-games 64 \
  --batch-size 64 \
  --simulations 32 \
  --c-puct 1.5 \
  --sample-until-ply 4 \
  --temperature 1.0 \
  --noise-fraction 0 \
  --noise-total-concentration 10 \
  --seed "$selfplay_seed"

"$python" scripts/audit_selfplay.py "$selfplay" \
  --expected-games "$games" \
  --expected-rules breakthrough-5x5-one-row-v1 \
  --expected-simulations 32 \
  --expected-c-puct 1.5 \
  --expected-model-sha256 "$expected_parent_hash" \
  --expected-noise-fraction 0 \
  --expected-noise-total-concentration 10 \
  --expected-sample-until-ply 4 \
  --expected-batch-size 64 | tee "$selfplay/audit.json"
"$python" scripts/summarize_selfplay.py "$selfplay" \
  | tee "$selfplay/summary.json"

"$python" scripts/train_pretraining.py \
  "$pretraining" \
  "$training/child" \
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
  --seed "$training_seed"
"$python" scripts/select_training_checkpoints.py \
  "$training" "$training/selection.json" --expected-runs 1

child="$(jq -er '.models.child.path' "$training/selection.json")"
child_hash="$(jq -er '.models.child.sha256' "$training/selection.json")"
readonly child child_hash
test -f "$child"
child_checksum="$(sha256sum -- "$child")"
readonly child_checksum
[[ "${child_checksum%% *}" == "$child_hash" ]]
if [[ "$child_hash" == "$expected_parent_hash" ]]; then
  jq -n \
    --arg parent_sha256 "$expected_parent_hash" \
    --arg selected_sha256 "$child_hash" \
    '{status:"rollback", arena_skipped:true,
      reason:"selection returned the unchanged parent",
      parent_sha256:$parent_sha256, selected_sha256:$selected_sha256}' \
    | tee "$training/decision.json"
  exit 0
fi

"$python" scripts/run_mini_tournament.py "$arena" \
  --rules mini \
  --pairs 256 \
  --opening-plies 4 \
  --move-seconds 0.05 \
  --time-tolerance-seconds 0.10 \
  --puct-c-puct 1.5 \
  --baselines strong \
  --matchups custom-relevant \
  --model "generation-1=$parent" \
  --model "large-data=$child" \
  --seed "$arena_seed"
