# Phase 26 review: native mini gate and search screen

## What is authorized

Two cheap, independent checks may run after the audited branch is clean on the
cluster:

1. One RTX 3070 executes every test, including real native 5x5 and standard
   8x8 TensorFlow build/train/save/load boundaries.
2. Four CPU-only arena tasks screen tactical-rollout PUCT with `c_puct` values
   `0.25, 0.75, 1.5, 3.0` at identical 20 ms move budgets and the same 48
   paired openings.

Parent-Q first-play urgency remains fixed. The screen changes no training data,
noise, temperature, architecture, or value target.

## Why this screen comes before more data

The pretraining labels come from MCTS. Generating hundreds of games with an
untested search constant would scale an assumption. The existing `1.5` result
is useful but was never compared with lower values; resource-efficient search
work explicitly warns that the useful scale can be setup-dependent.

This four-task run is a disaster screen, not a final tune. It compares each
PUCT variant to the same alpha-beta, random, and rollout anchors and records
paired uncertainty. If two settings are plausibly competitive, confirm those
two on fresh openings before choosing the 512-game data-generator setting.

## Failure checks

- Any nonterminal ply-limit result, timeout, illegal move, dirty worktree, or
  model/rules mismatch fails the phase.
- Ratings are not compared across tasks without inspecting their direct
  paired results against the immutable anchors.
- A point estimate with overlapping uncertainty does not select a winner.
- CPU arena tasks request no GPU; the TensorFlow gate requests exactly one.
