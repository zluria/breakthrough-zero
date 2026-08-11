# Phase 27 review: staged native-mini pretraining data

## Decision and scope

After the corrected `c_puct` screen and any required finalist confirmation,
generate 512 new 5x5 games as eight immutable 64-game shards. At most four
one-CPU tasks run concurrently; the job requests no GPU. The raw schema stores
absolute states, legal moves, visits, priors, root/action value sums and
squares, selected moves, final results, seeds, and search metadata. It is
therefore reusable by every native CNN and all currently planned value
targets.

This is the first baseline stage, not a declaration that 512 games or 32
simulations is optimal. The older 64-game sweep found tactical rollouts at 32
simulations attractive for throughput and visit-target entropy, but that was
not a downstream Elo ablation. The new stage is large enough to make native
architecture and value-target failures visible while remaining cheap to
replace or extend.

The phase-28 direct confirmation did not distinguish `c_puct=1.5` from 3.0,
so its preregistered simplicity rule selected 1.5 for this corpus.

## Fixed choices

- Native 5x5 rules and compact 75-action neural boundary downstream.
- Tactical rollout evaluator: immediate win, then capture, otherwise random.
- Parent-Q first-play urgency and absolute Player-1 values.
- The screened `c_puct`, passed explicitly at submission.
- 32 simulations per move unless the submission records a deliberate
  `SIMULATIONS` override.
- Visit sampling at temperature 1 through ply 4: two moves per side.
- Deterministic most-visited selection afterward.
- No Dirichlet noise. Rollout randomness and the four sampled opening plies
  already provide stochasticity; noise remains a later controlled ablation.

Training applies the four exact symmetries lazily in a balanced four-epoch
cycle. Augmented tensors are not stored as if they were independent games.

## Gates before training

1. Every array task must finish normally on the same clean Git commit.
2. Every manifest checksum and full trajectory must reload successfully.
3. Exactly 512 game seeds must exist and all must be unique.
4. Rules, simulations, `c_puct`, sampling, temperature, tactical-rollout, and
   noise metadata must agree across shards; only master seed and chunk range
   may differ.
5. All games must terminate naturally within the rules-derived 40-ply proof
   bound. Report game-length percentiles, outcome balance, unique positions,
   trajectory/prefix diversity, visit entropy, and root-value calibration.

Do not train if any check fails. Do not expand the corpus merely because the
jobs are cheap. Train the preregistered small comparison, run fresh paired
arenas, and expand only if the resulting learning curve is plausibly
data-limited.

The postflight is a separate one-CPU Slurm job rather than work on the login
node. The six GPU training tasks receive an `afterok` dependency on that audit,
so a corrupt or mixed corpus allocates no accelerator time.

## Resource judgment

The native TensorFlow gate already ran the complete suite on the exact code.
Repeating all 88 tests in every identical CPU shard would spend more allocation
without adding independent evidence, so shard jobs use a clean-worktree gate
and rely on that recorded prerequisite. Each shard writes only to its own
directory and can be retried without overwriting a valid chunk.
