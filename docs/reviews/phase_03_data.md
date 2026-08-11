# Phase 3 review: preserve the expensive search data

## Principle

Self-play is much more expensive than encoding or gradient updates. The saved
unit is therefore an unaugmented game in absolute coordinates, not a ready-made
tensor batch. Encoding, augmentation, target selection, and train/validation
splitting happen later.

## What one searched position retains

- Both bitboards, player to move, and ply.
- Every legal absolute move.
- Per-move immutable network prior, possibly noised search prior, visits,
  absolute value sum, and squared-value sum.
- The selected move.
- Root visits, absolute value sum, squared-value sum, and raw leaf evaluation.
- The value reached by following the most-visited path through the saved tree.
- Whether this was a full policy search and its training sample weight.

The completed game adds the final absolute result and generation seed. The
manifest records schema and rules versions, search/model configuration, code
revision, counts, and a file checksum.

This supports final-result, soft-Z, best-child/A0C, played-child, greedy-backup,
mixture, future-root/horizon, and variance-aware experiments without regenerating
games. Consecutive positions also support an opponent-next-policy head.

## Deliberate storage tradeoff

We save all root edges but not the entire search tree. The scalar greedy-backup
value is computed while the tree exists. Saving every internal node would make
the format and data volume much larger for little expected benefit. If a later
hypothesis truly requires a different traversal of the old tree, it must first
justify that cost in a small pilot.

## Correctness checks

1. Save/load is exact and rejects a changed checksum or schema.
2. Unsigned 64-bit bitboards preserve square 63.
3. Stored moves remain absolute and legal after loading.
4. Every target is absolute Player 1 value.
5. Player-swap augmentation negates first moments and scalar values but not
   visits, priors, or squared moments.
6. All four symmetries transform the selected move and every root edge.
7. Validation splitting is by whole game, never by position.

## Experiment order after pretraining

The immutable pretraining chunks become a cheap offline laboratory. We first
compare value targets, compact CNN architectures, optimizer schedules, and
manual input features at equal wall-clock budgets. Search constants are then
compared in a wall-clock arena using the same trained checkpoints. Only the
small set of winners enters expensive online self-play experiments.
