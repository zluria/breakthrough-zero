#!/usr/bin/env bash
#SBATCH --job-name=btz-mini-continuous
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=01:15:00
#SBATCH --output=logs/mini-continuous-%j.out
#SBATCH --error=logs/mini-continuous-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=1

: "${EXPECTED_COMMIT:?submit with the published Git commit}"
: "${RUN_ID:?submit with a new continuous-run directory}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly initial_manifest="${data_root}/native-mini-generation-2-20260813-v1/selection.json"
readonly historical="${data_root}/native-mini-large-selfplay-20260813-v1"
readonly expected_initial_hash="fcab56f571c1392443871d6b012c2307ed42b493ed88c2fe3d7a4e35b1501d4d"
readonly output="${data_root}/${RUN_ID}"

current_commit="$(git rev-parse HEAD)"
worktree_status="$(git status --porcelain --untracked-files=normal)"
initial_model="$(jq -er '.models.child.path' "$initial_manifest")"
recorded_hash="$(jq -er '.models.child.sha256' "$initial_manifest")"
readonly current_commit worktree_status initial_model recorded_hash
[[ "$current_commit" == "$EXPECTED_COMMIT" ]]
test -z "$worktree_status"
test ! -e "$output"
test -d "$historical"
test -f "$initial_model"
[[ "$recorded_hash" == "$expected_initial_hash" ]]
checksum_line="$(sha256sum -- "$initial_model")"
readonly checksum_line
[[ "${checksum_line%% *}" == "$expected_initial_hash" ]]

"$python" -m unittest discover -s tests
"$python" scripts/run_continuous_training.py \
  "$historical" \
  "$output" \
  --initial-model "$initial_model" \
  --initial-model-sha256 "$expected_initial_hash" \
  --duration-seconds 3600 \
  --minimum-cycle-seconds 300 \
  --games-per-cycle 1024 \
  --fresh-window-cycles 4 \
  --training-seconds-per-cycle 120 \
  --max-replay-consumption 4 \
  --historical-fraction 0.25 \
  --seed 202608136101

jq -e '
  .status == "complete" and
  .phase == "complete" and
  .settings.root_noise_fraction == 0 and
  (.cycles | length) > 0 and
  .latest_actor.sha256 != .initial_actor.sha256 and
  .totals.fresh_positions > 0 and
  .totals.training_examples_consumed > 0
' "$output/progress.json"
