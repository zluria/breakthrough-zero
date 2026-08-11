# Phase 13 review: first 8x8 self-play probe

## Purpose

Mini-board timings do not determine a sensible 8x8 search budget.  This probe
measures complete-game length, positions per second, storage, value dispersion,
and failure behavior on standard Breakthrough before we choose any real
pretraining count.

Three independent CPU-only array tasks generate eight games each at 16, 32,
and 64 simulations per move.  All use tactical rollouts, `c_puct=1.5`, twelve
plies of visit sampling, no root noise, immutable chunks, and distinct seeds.

## Why tactical only

On the mini board, tactical rollout PUCT was much stronger than uniform
rollouts at equal wall-clock time.  In the new generation sweep it also had
essentially the same throughput and lower mean absolute disagreement between
root soft-Z and the final result at all three search budgets.  Repeating the
uniform branch on 8x8 would spend compute without answering the immediate
scaling question.  A later controlled strength check can reopen it.

## Parallelism and resource use

Each task requests one CPU, one GB, and no GPU.  The tasks have disjoint output
directories and seeds, so they can run concurrently with neural evaluation.
Eight games per setting are a timing probe, not a learning dataset or outcome-
balance estimate.

## Predictions and decision rule

- Throughput will fall more sharply with simulations than on 5x5 because each
  rollout is longer.
- Thirty-two simulations is still the provisional compromise, but the next
  stage will use measured positions per CPU-hour and fixed-data learning, not
  this prediction.
- We will next generate a modest staged dataset at one or at most two budgets.
  There is no fixed target game count.

Stop on any timeout, incomplete manifest, replay failure, or more than one GB
of memory.  Do not scale from a job that has only a successful Slurm status;
load and replay every resulting chunk first.
