# Phase 9 review: bounded parallel dummy-MCTS sweep

## Why this job exists

The assignment's suggested game count is not a target in its own right.  The
next decision is how much useful policy and value information one unit of CPU
time buys on the 5x5 debug game.  This sweep measures that frontier before any
larger pretraining run.

The network is intentionally absent.  Every leaf uses the already-tested
uniform policy and an absolute Player-1 random-rollout value.  This keeps the
experiment focused on two generator choices:

- 16, 32, or 64 simulations per move;
- uniform rollouts or the small "win, then capture" tactical preference.

`c_puct=1.5`, visit sampling, and zero root noise are fixed.  Changing them in
the same sweep would make a six-task timing result impossible to interpret.

## Parallelization design

The six configurations are independent Slurm array tasks, capped at four
concurrent tasks.  Each task has one CPU, its own master seed, its own output
directory, and writes immutable checksummed chunks.  No task shares a writable
file with another task.  This is the simplest useful form of self-play
parallelism and should remain the default until neural inference needs GPU
batching.

A separate CPU-only preflight job runs all tests and generates two disposable
games.  The array has an `afterok` dependency on that job.  A bad checkout or
environment therefore consumes no sweep allocation.

## Predictions registered before submission

1. Throughput should fall approximately with the simulation count, although
   fixed game and serialization costs will soften the ratio.
2. Tactical rollouts should be slightly slower.  The local six-game timing
   suggested about six percent, but that estimate is noisy.
3. Tactical rollouts are the likely data-generator default if their overhead
   remains modest: in the fixed-opening tournament, tactical rollout PUCT beat
   plain rollout PUCT 55-9 at equal wall-clock time.
4. Thirty-two simulations is the likely first training-data budget.  Sixteen
   may be too noisy and 64 may not provide twice the useful target information.
   This is a prediction, not a preset conclusion.

## What this sweep can and cannot decide

Sixty-four games per configuration are enough for throughput, record-size,
game-length, outcome-balance, and target-dispersion diagnostics.  They are not
enough to claim an Elo effect.  Architecture, value target, `c_puct`, replay
age, and root noise remain separate experiments.

The next data stage will be selected from measured learning curves on these
records.  We will stop generating data when held-out losses and paired Elo no
longer improve enough to justify the CPU time; there is no commitment to
10,000 games.

## Stop conditions

- The preflight does not pass every test.
- Any task writes an incomplete chunk or a checksum/replay validation fails.
- A task exceeds 15 minutes or one GB of memory on the mini game.
- Seeds, configurations, or output directories overlap.

