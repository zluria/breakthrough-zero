# Phase 18 review: diagnose the neural self-play color imbalance

## Trigger

The first batched neural pilot passed rules, terminal, search, checksum, and
replay validation, but Player 1 won only 9 of 64 standard games.  This is far
from the roughly balanced uniform-rollout sanity check and the 45.3% Player-1
rate in the 64-simulation dummy-MCTS pretraining games.  The pilot data is
quarantined from learning until the cause is understood.

## What is already ruled out

- Optimized legal moves match the literal reference through complete games.
- Terminal moves retain the last mover and an absolute winner.
- PUCT values are never negated; only selection maximizes or minimizes for the
  parent's current player.
- Whole PUCT searches are exactly antisymmetric under player swap when the
  evaluator is antisymmetric.
- Batched and scalar orchestration save exactly equal seeded games with a
  deterministic evaluator, including root noise and slot refill.

These tests make a learned evaluator bias more likely than a rules or backup
sign bug, but do not prove it.

## Diagnostic order

1. Measure the selected model on exact player-swapped and left-right-reflected
   random state pairs.  Report absolute-value residuals, mapped policy
   divergence, and top-move agreement.
2. Re-audit the pretraining raw target distribution by player to move and by
   the four augmentation transforms.  A transformed target must negate exactly
   when players are swapped.
3. Measure the pilot's trajectory diversity and repeat a small seed only if the
   observed 9/64 result remains plausible after model diagnostics.
4. Compare a symmetry-averaged evaluator only as a controlled diagnostic.  Do
   not silently promote it: two network calls have a real throughput cost.

## Decision boundary

If the model is materially non-antisymmetric, fix training balance or add an
explicit symmetry-consistency loss and retrain from the saved pretraining data.
If the model is symmetric but the pilot repeats the color skew, inspect move
sampling and search trajectories position by position.  Expensive self-play
and replay-buffer tuning remain paused either way.
