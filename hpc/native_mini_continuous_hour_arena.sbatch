#!/usr/bin/env bash
#SBATCH --job-name=btz-mini-cont-eval
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=7G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/mini-cont-eval-%j.out
#SBATCH --error=logs/mini-cont-eval-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

: "${EXPECTED_COMMIT:?submit with the published Git commit}"
: "${LOOP_RUN_ID:?submit with the completed continuous-run directory}"
: "${RUN_ID:?submit with a new diagnostic arena directory}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly progress="${data_root}/${LOOP_RUN_ID}/progress.json"
readonly output="${data_root}/${RUN_ID}"

current_commit="$(git rev-parse HEAD)"
worktree_status="$(git status --porcelain --untracked-files=normal)"
loop_status="$(jq -er '.status' "$progress")"
initial="$(jq -er '.initial_actor.path' "$progress")"
initial_hash="$(jq -er '.initial_actor.sha256' "$progress")"
latest="$(jq -er '.latest_actor.path' "$progress")"
latest_hash="$(jq -er '.latest_actor.sha256' "$progress")"
readonly current_commit worktree_status loop_status
readonly initial initial_hash latest latest_hash
[[ "$current_commit" == "$EXPECTED_COMMIT" ]]
test -z "$worktree_status"
test ! -e "$output"
[[ "$loop_status" == "complete" ]]
test -f "$initial"
test -f "$latest"
[[ "$initial_hash" != "$latest_hash" ]]
initial_checksum="$(sha256sum -- "$initial")"
latest_checksum="$(sha256sum -- "$latest")"
readonly initial_checksum latest_checksum
[[ "${initial_checksum%% *}" == "$initial_hash" ]]
[[ "${latest_checksum%% *}" == "$latest_hash" ]]

"$python" scripts/run_mini_tournament.py "$output" \
  --rules mini \
  --pairs 256 \
  --opening-plies 4 \
  --move-seconds 0.05 \
  --time-tolerance-seconds 0.10 \
  --puct-c-puct 1.5 \
  --baselines none \
  --matchups all \
  --model "initial=$initial" \
  --model "latest=$latest" \
  --seed 202608136201

jq -e '.failure_count == 0 and .matchup_count == 1' "$output/summary.json"
