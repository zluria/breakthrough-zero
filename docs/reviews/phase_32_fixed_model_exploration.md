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

The single-node sequential Slurm driver is
`hpc/native_mini_exploration_diagnostic.sbatch`. Keeping all four settings on
one GPU avoids turning node-to-node throughput variation into a noise result;
the settings still share game seeds for paired diagnostics.

Before generating those games, the same job evaluates the selected checkpoint
on the original game-level validation split. It reports policy KL/top-move
agreement, value error against both final outcome and root Q, fixed-bin outcome
calibration, absolute-player breakdowns, and policy mass on immediate wins.
This is the required calibration gate, not another training pass.

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

## Execution and result

Job 33603 first passed the published-commit boundary smoke on RTX3070-08: all
102 tests plus native generation, deterministic Keras training, checkpointing,
and GPU execution completed in 17 seconds. Job 33604 then completed the four
settings sequentially on the same GPU in 2 minutes 58 seconds. All four
fail-closed audits passed; each setting contains 256 games searched at 32
simulations from the same per-game seeds.

| Setting | Low-prior actions visited | Low-prior visit mass | Immediate win selected | Positions/s | Unique trajectories |
| --- | ---: | ---: | ---: | ---: | ---: |
| Off | 58.0% | 2.53% | 95.5% | 97.3 | 253 |
| Fraction 0.10, concentration 10 | 69.5% | 3.31% | 93.4% | 96.2 | 251 |
| Fraction 0.25, concentration 10 | 75.6% | 3.57% | 91.4% | 94.9 | 255 |
| Fraction 0.25, concentration 2.5 | 63.9% | 3.08% | 89.2% | 96.7 | 256 |

Moderate noise changed 196 of 256 full trajectories relative to no noise, but
P1 win fraction moved only from 63.7% to 64.5% (35 games changed from P2 to P1
and 33 changed from P1 to P2). The heavier ordinary setting increased coverage
further at a larger tactical and throughput cost. The sharp setting produced
the most unique trajectories but worse coverage than ordinary 0.25 noise and
the lowest immediate-win selection rate; uniqueness would have selected the
wrong intervention.

Held-out diagnostics on the original 102-game validation split (1,381
positions) reported policy KL 0.370, policy top-target agreement 36.2%, root-Q
MAE 0.180, final-outcome MAE 0.733, and final-outcome sign accuracy 73.6%.
This is consistent with the selected soft-Z model fitting its search target
more closely than the noisy final result; it does not validate soft-Z for
online bootstrapped replay.

The 512-state symmetry diagnostic also found substantial learned residuals:
mean policy L1 was 0.232 under player swap and 0.240 under left-right reflection,
with top-move agreement 51.0% and 50.0% respectively. Mean absolute value
residuals were 0.138 and 0.124. This warrants the already planned equal-time
symmetry experiment; it does not justify silently enabling fourfold inference.

## Decision

Advance only fraction 0.10 / total concentration 10 to a small end-to-end
learning ablation. It materially improved the intended low-prior coverage
metric, had the smallest throughput cost (about 1.1%), and incurred the
smallest observed immediate-win reduction. **Do not adopt it as the baseline
yet.** The next comparison must give noise-off and moderate-noise actors equal
self-play wall time, give their learners equal wall time, and use fresh paired
noise-free Elo. Reject the two 0.25 settings for now.

An operational near miss preceded the valid smoke: job 33602 was submitted on
the old commit because an explicit fetch populated `FETCH_HEAD`, while a
semicolon-separated remote command continued after the named tracking-ref
merge failed. It finished its 18-second temporary smoke before cancellation
arrived and is inadmissible. Job 33603 used `FETCH_HEAD`, verified commit
`9ff98d9`, and is the only qualifying smoke. Remote submission gates must use
fail-fast chaining or an explicit shell script; a clean-check alone is not
enough when command sequencing can continue after failure.
