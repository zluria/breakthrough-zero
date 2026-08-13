#!/usr/bin/env bash
#SBATCH --job-name=btz-net-smoke
#SBATCH --partition=rtx3070
#SBATCH --constraint=rtx3070
#SBATCH --exclude=HPC-RTX3070-06,HPC-RTX3070-07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:rtx_3070:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/network-smoke-%j.out
#SBATCH --error=logs/network-smoke-%j.err

set -euo pipefail

cd /home/zurlu/breakthrough-zero
readonly python=/home/zurlu/.venvs/btz-py311-tf221/bin/python
export PYTHONPATH="$PWD/src"

test -z "$(git status --porcelain --untracked-files=normal)"
"$python" -m unittest discover -s tests

readonly root="${SLURM_TMPDIR:-/tmp}/btz-network-smoke-${SLURM_JOB_ID}"
"$python" scripts/generate_dummy_selfplay.py \
  "$root/data" \
  --games 16 \
  --chunk-games 8 \
  --simulations 16 \
  --rules mini \
  --tactical-rollouts \
  --seed 202608120201
"$python" scripts/train_pretraining.py \
  "$root/data" \
  "$root/training" \
  --target outcome \
  --channels 8 \
  --residual-blocks 1 \
  --value-hidden 8 \
  --epochs 5 \
  --batch-size 64 \
  --seed 202608120202

# Exercise online fine-tuning's rollback boundary with a real saved model.
"$python" scripts/train_pretraining.py \
  "$root/data" \
  "$root/replay/child" \
  --initial-model "$root/training/model.keras" \
  --target outcome \
  --channels 8 \
  --residual-blocks 1 \
  --value-hidden 8 \
  --epochs 2 \
  --checkpoint-every 4 \
  --batch-size 64 \
  --seed 202608120203
jq -e \
  '.history[0].epoch == 0 and
   .history[0].checkpoint_reasons == ["initial_model"] and
   .history[0].checkpoint.path != null' \
  "$root/replay/child/run.json"
"$python" scripts/select_training_checkpoints.py \
  "$root/replay" "$root/replay/selection.json" --expected-runs 1
