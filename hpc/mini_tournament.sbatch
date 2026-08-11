#!/usr/bin/env bash
#SBATCH --job-name=btz-mini-arena
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs/mini-arena-%j.out
#SBATCH --error=logs/mini-arena-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
source /home/zurlu/.venvs/btz-py311-tf221/bin/activate
export PYTHONPATH="$PWD/src"

git diff --quiet
git diff --cached --quiet
python -m unittest discover -s tests
python scripts/run_mini_tournament.py \
  "results/mini-hpc-${SLURM_JOB_ID}" \
  --pairs 32 \
  --opening-plies 6 \
  --opening-simulations 8 \
  --move-seconds 0.05 \
  --time-tolerance-seconds 0.02 \
  --seed 20260811
