#!/usr/bin/env bash
#SBATCH --job-name=btz-mini-gen-arena
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=7G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/mini-gen-arena-%j.out
#SBATCH --error=logs/mini-gen-arena-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

: "${EXPECTED_COMMIT:?submit with the published Git commit}"
: "${PARENT_TRAIN_RUN_ID:?submit with the parent training directory}"
: "${PARENT_MODEL_KEY:?submit with the parent selection key}"
: "${EXPECTED_PARENT_HASH:?submit with the pinned parent SHA-256}"
: "${PARENT_NAME:?submit with the parent arena name}"
: "${CHILD_TRAIN_RUN_ID:?submit with the child training directory}"
: "${CHILD_MODEL_KEY:?submit with the child selection key}"
: "${EXPECTED_CHILD_HASH:?submit with the pinned child SHA-256}"
: "${CHILD_NAME:?submit with the child arena name}"
: "${ARENA_SEED:?submit with the preregistered arena seed}"
: "${RUN_ID:?submit with a new arena directory}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly parent_selection="${data_root}/${PARENT_TRAIN_RUN_ID}/selection.json"
readonly child_selection="${data_root}/${CHILD_TRAIN_RUN_ID}/selection.json"
readonly output="${data_root}/${RUN_ID}"

current_commit="$(git rev-parse HEAD)"
worktree_status="$(git status --porcelain --untracked-files=normal)"
parent="$(jq -er --arg key "$PARENT_MODEL_KEY" '.models[$key].path' "$parent_selection")"
parent_hash="$(jq -er --arg key "$PARENT_MODEL_KEY" '.models[$key].sha256' "$parent_selection")"
child="$(jq -er --arg key "$CHILD_MODEL_KEY" '.models[$key].path' "$child_selection")"
child_hash="$(jq -er --arg key "$CHILD_MODEL_KEY" '.models[$key].sha256' "$child_selection")"
readonly current_commit worktree_status parent parent_hash child child_hash
[[ "$current_commit" == "$EXPECTED_COMMIT" ]]
test -z "$worktree_status"
test ! -e "$output"
test -f "$parent"
test -f "$child"
[[ "$parent_hash" == "$EXPECTED_PARENT_HASH" ]]
[[ "$child_hash" == "$EXPECTED_CHILD_HASH" ]]
parent_checksum="$(sha256sum -- "$parent")"
child_checksum="$(sha256sum -- "$child")"
readonly parent_checksum child_checksum
[[ "${parent_checksum%% *}" == "$EXPECTED_PARENT_HASH" ]]
[[ "${child_checksum%% *}" == "$EXPECTED_CHILD_HASH" ]]

"$python" scripts/run_mini_tournament.py "$output" \
  --rules mini \
  --pairs 64 \
  --opening-plies 4 \
  --move-seconds 0.05 \
  --time-tolerance-seconds 0.10 \
  --puct-c-puct 1.5 \
  --baselines strong \
  --matchups custom-relevant \
  --model "$PARENT_NAME=$parent" \
  --model "$CHILD_NAME=$child" \
  --seed "$ARENA_SEED"
