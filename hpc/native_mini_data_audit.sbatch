#!/usr/bin/env bash
#SBATCH --job-name=btz-native-mini-audit
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/native-mini-audit-%j.out
#SBATCH --error=logs/native-mini-audit-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"

: "${DATA_RUN_ID:?submit with DATA_RUN_ID=<corpus directory>}"
: "${C_PUCT:?submit with C_PUCT=<expected value>}"
readonly simulations="${SIMULATIONS:-32}"
readonly data_root="${BTZ_DATA_ROOT:-/home/zurlu/breakthrough-zero-data}"

test -z "$(git status --porcelain --untracked-files=normal)"
"$python" scripts/audit_selfplay.py \
  "${data_root}/${DATA_RUN_ID}" \
  --expected-games 512 \
  --expected-rules breakthrough-5x5-one-row-v1 \
  --expected-simulations "$simulations" \
  --expected-c-puct "$C_PUCT"
