# Phase 23 review: detect the first self-play regression

## Risk being tested

A common AlphaZero failure is that a rollout-MCTS teacher is strongest, its
pretrained policy/value student is weaker, and the first self-play-trained
student is weaker still. Training loss cannot diagnose playing strength, and
losses for outcome and soft-Z targets are not numerically comparable.

The immediate question is therefore narrow: did either five-epoch update beat
the unchanged pretrained checkpoint under the evaluator used to make its
data?

## Fair arena design

Compare three agents:

1. the selected pretrained 64x4 soft-Z checkpoint;
2. generation-1 fine-tuning with final outcome values;
3. generation-1 fine-tuning with soft-Z values.

All three use exact four-symmetry inference averaging, `c_puct=1.5`, no search
noise, and the same 50 ms wall-clock move budget plus 100 ms scheduler grace.
Warm-up happens outside the clock. Use one immutable suite of 32 standard-board
openings made by exactly four uniform-random plies. Every matchup uses every
opening twice with colors reversed.

Run the three pairwise comparisons as independent Slurm array tasks on three
GPUs. They use the same opening seed and 64 games per pair, so splitting them
changes only scheduling, not the comparison. The full test suite runs before
each task. Preserve every move time, terminal status, work count, opening, and
agent seed.

## Interpretation and stop rule

This is a screening arena, but 32 paired openings is enough to expose a large
regression. Do not generate generation 2 if both updated agents lose clearly
to pretraining. Instead inspect policy symmetry, illegal masking, target
calibration, epoch choice, and replay mix. If one update is at least
competitive, compare it next against alpha-beta and tactical rollout PUCT
using the same wall-clock protocol before promoting it.
