# Phase 32 review: fixed-model exploration diagnostic

## Common-sense review before compute

The selected checkpoint is not yet permission for neural self-play. Three
diagnostic gaps would make a noise experiment misleading:

1. `diagnose_model_symmetry.py` was hard-coded to the 8x8 rules and therefore
   could not inspect the native 5x5 model. It now derives the ruleset from the
   model input and tests exact-zero policy probabilities safely.
2. Neural generation did not request deterministic TensorFlow operations or
   retain GPU and Slurm identity in chunk metadata. It now does both and records
   search-generation seconds and positions per second.
3. The noise design promised low-network-prior and immediate-win measurements,
   but the summarizer reported only entropy and uniqueness. It now reports the
   search-prior/network-prior KL, visits and visit mass on moves below half the
   uniform prior, and whether immediate winning moves receive a top visit and
   are selected.

These are boundary corrections, not a reason to start a larger run. The
corrected tests and one native TensorFlow smoke must pass from a clean,
published commit first.

## Question

Does ordinary root Dirichlet noise address low-policy-prior blind spots in the
selected native-mini model without visibly damaging tactical reliability or
generation throughput?

This is a fixed-model diagnostic. It may reject settings or justify one small
learning ablation; it cannot establish that noise improves Elo.

## Frozen setup

- Model: 32x3 soft-Z epoch 84, SHA-256
  `3bc3ca17393ce803d3d89ccc02f649cfb755b56b1af944c1f41c82db4114beb6`.
- Rules: native 5x5, 75 actions.
- Search: 32 simulations, `c_puct=1.5`, parent-Q FPU, absolute Player-1 values.
- Move sampling: temperature 1 through ply 4 (two moves per side), then
  deterministic most-visited selection.
- 256 games per setting, batch size 64, identical game seeds, one GPU node.
- Four settings: off; fraction 0.10/total concentration 10; fraction 0.25/10;
  fraction 0.25/2.5.
- No symmetry ensemble, archive starts, playout-cap randomization, target
  change, or retraining. They are separate mechanisms.

Equal simulations isolate the search-prior intervention. This is not a rated
arena: wall time and positions/second are still reported so a setting cannot
hide a material generation cost.

## Measurements and decision rule

For each setting report all fail-closed corpus checks plus:

- search-prior/network-prior KL;
- normalized network and visit-policy entropy;
- fraction of low-network-prior actions visited and their visit mass;
- immediate-win top-visit and selection rates;
- unique positions, trajectories, and prefixes;
- game length, outcome balance, generation seconds, and positions/second.

Use paired per-game comparisons because seeds are shared. Reject any setting
with invalid games, inconsistent hashes/configuration, a clear immediate-win
regression, or a material throughput loss unexplained by longer games. Do not
promote noise because it creates more unique trajectories: that is expected
and does not answer the blind-spot question.

Advance at most one noisy setting, preferring the smaller 0.10 fraction when
effects are similar. It advances only if it materially increases low-prior
coverage without a tactical or throughput warning. The next test would then be
a small equal-self-play-time, equal-learner-time comparison with fresh paired
Elo. If the diagnostic shows no useful separation, retain no noise and avoid
spending training compute on this mechanism now.

## Stop condition

No job may be submitted while the corrections above exist only in a dirty or
unpublished checkout. This protects the experiment/commit identity that the
earlier adversarial audit required.
