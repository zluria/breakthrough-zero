#!/usr/bin/env bash
#SBATCH --job-name=btz-mini-g1-arena
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=7G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/mini-g1-arena-%j.out
#SBATCH --error=logs/mini-g1-arena-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

: "${EXPECTED_COMMIT:?submit with the published Git commit}"
: "${PARENT_TRAIN_RUN_ID:?submit with the parent training directory}"
: "${CHILD_TRAIN_RUN_ID:?submit with the generation-1 training directory}"
: "${RUN_ID:?submit with a new arena directory}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly parent_selection="${data_root}/${PARENT_TRAIN_RUN_ID}/selection.json"
readonly child_selection="${data_root}/${CHILD_TRAIN_RUN_ID}/selection.json"
readonly output="${data_root}/${RUN_ID}"
readonly expected_parent_hash="3bc3ca17393ce803d3d89ccc02f649cfb755b56b1af944c1f41c82db4114beb6"

current_commit="$(git rev-parse HEAD)"
worktree_status="$(git status --porcelain --untracked-files=normal)"
parent="$(jq -er --arg key c32-b3-soft_z '.models[$key].path' "$parent_selection")"
child="$(jq -er --arg key child '.models[$key].path' "$child_selection")"
child_hash="$(jq -er --arg key child '.models[$key].sha256' "$child_selection")"
readonly current_commit worktree_status parent child child_hash
[[ "$current_commit" == "$EXPECTED_COMMIT" ]]
test -z "$worktree_status"
test ! -e "$output"
test -f "$parent"
test -f "$child"
parent_checksum="$(sha256sum -- "$parent")"
child_checksum="$(sha256sum -- "$child")"
readonly parent_checksum child_checksum
[[ "${parent_checksum%% *}" == "$expected_parent_hash" ]]
[[ "${child_checksum%% *}" == "$child_hash" ]]

"$python" scripts/run_mini_tournament.py "$output" \
  --rules mini \
  --pairs 64 \
  --opening-plies 4 \
  --move-seconds 0.05 \
  --time-tolerance-seconds 0.10 \
  --puct-c-puct 1.5 \
  --baselines strong \
  --matchups custom-relevant \
  --model "generation-0=$parent" \
  --model "generation-1=$child" \
  --seed 202608133401
