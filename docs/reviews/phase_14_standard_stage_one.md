# Phase 14 review: first standard-board fixed-data stage

## Evidence entering this stage

The eight-game 8x8 probe produced approximately 46, 30, and 19 positions per
second at 16, 32, and 64 simulations.  Search depth increased visit-policy
entropy and modestly reduced soft-Z disagreement with the final result, but the
sample is too small to price that improvement reliably.

Only three of the 24 probe games were won by Player 1.  Two independent checks
argue against immediately blaming the absolute-Q implementation:

- 5,000 uniform and 5,000 tactical raw rollouts were nearly color-balanced;
- a deterministic whole-search symmetry test maps every root child visit under
  player swap and negates every Q exactly.

The outcome skew remains a diagnostic to measure with more games.  It is not a
conclusion about Breakthrough or the training target.

## Jobs

Two independent CPU-only tasks generate 64 tactical-rollout games:

- 32 simulations per move;
- 64 simulations per move.

Both keep `c_puct=1.5`, twelve plies of visit sampling, no root noise, distinct
seeds, and 16-game immutable chunks.  The 16-simulation setting is dropped:
its targets were least diverse and disagreed most with outcomes, while its
throughput advantage is not needed for this small stage.

## Decision after generation

All chunks will be checksum-loaded and replayed.  We will then train the same
selected compact architecture on equal numbers of positions from each search
budget.  The decision metric is paired equal-time neural PUCT Elo, with policy
loss, value calibration, entropy, and inference simulations as diagnostics.

Sixty-four games per setting is deliberately modest.  No further standard
self-play is authorized until at least one resulting model learns useful
policy and value signals and survives the arena regression ladder.
