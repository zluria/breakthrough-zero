#!/usr/bin/env bash
#SBATCH --job-name=btz-mini-cpuct-confirm
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=logs/mini-cpuct-confirm-%j.out
#SBATCH --error=logs/mini-cpuct-confirm-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"

: "${RUN_ID:?submit with RUN_ID=<stable output directory>}"
: "${CANDIDATE_A:?submit with CANDIDATE_A=<first c_puct>}"
: "${CANDIDATE_B:?submit with CANDIDATE_B=<second c_puct>}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"
readonly output="${data_root}/${RUN_ID}"

test -z "$(git status --porcelain --untracked-files=normal)"
"$python" scripts/run_mini_tournament.py \
  "$output" \
  --rules mini \
  --pairs 96 \
  --opening-plies 4 \
  --move-seconds 0.05 \
  --time-tolerance-seconds 0.05 \
  --baselines none \
  --tactical-puct "cpuct-${CANDIDATE_A}=${CANDIDATE_A}" \
  --tactical-puct "cpuct-${CANDIDATE_B}=${CANDIDATE_B}" \
  --seed 202608112000
