# Phase 36: continuous replay learner

## Why rewrite instead of patch

The previous loop mixed three concerns: learning, checkpoint diagnosis, and
actor selection. Validation could roll back to the starting model, after which
the arena compared that same file under two names. It also reset Adam between
updates and implemented a 75/25 source mixture with extreme per-position loss
weights. On the large corpus, one historical position carried about 71 times
the weight of a fresh position.

The rewrite keeps one short contract:

```text
immutable self-play -> stratified replay -> optimizer update -> latest actor
                              |
                              +-> diagnostic snapshots and arenas
```

This is not merely a stylistic preference. AlphaZero states that, unlike
AlphaGo Zero's 55% best-player gate, it maintains one continually updated
network, uses its latest parameters for self-play, and omits the evaluation and
best-player selection step.

Source: [Silver et al., AlphaZero, lines 29--35 of the arXiv text](https://arxiv.org/abs/1712.01815).

## Implemented decisions

1. `ReplaySampler` draws exact historical and fresh quotas. Every example has
   weight one; aggregate source influence is visible from counts.
2. Optimizer examples per newest training position are capped. Wall time and
   optimizer steps are independent hard caps and both are recorded.
3. A position receives one symmetry each time it is presented and cycles
   through all four exact transforms. Validation is unaugmented.
4. Adam moments and iteration count are saved with each model and strictly
   restored on the next update. Manifests distinguish the local `run_step`
   from the monotonic global `optimizer_step`.
5. Validation is reported separately by source and simple board-scaled game
   phase. `best_validation` is saved only so a regression can be inspected.
6. `actor.json` always points `models.latest` at the newest completed
   checkpoint. No acceptance, rejection, promotion, authorization, champion,
   or rollback state exists.
7. Arenas use pre-generated paired openings, early opening randomness, color
   reversal, and distinct checkpoint hashes. They measure Elo; they do not
   control self-play.
8. Policy cross-entropy covers the complete action head. Illegal target entries
   are zero and therefore train impossible logits down. Legal masking remains
   mandatory in search and is also reported as a diagnostic loss.

## Adversarial checks before a real cycle

- A unit test makes validation worsen at the final checkpoint and verifies that
  the final checkpoint is still the actor.
- Replay tests verify exact quotas, no repeated position within a source batch,
  four-symmetry cycling, and the consumption-to-step calculation.
- The network continuation test performs one Adam update, saves, restores, then
  proves that the next update is numerically identical.
- The existing tournament boundary rejects two labels with the same model
  hash before it creates an arena output directory.
- HPC job 33624 ran the tiny learner for two updates, inspected `actor.json`,
  strictly restored the published Adam state, completed a third update, and
  ran `bash -n` on all three active Slurm wrappers. It passed in 18 seconds on
  an RTX 3070.

## Deliberately not implemented

- No validation early stopping in the actor path.
- No Elo threshold, confidence-bound promotion, champion, or rollback.
- No priority replay, recency weighting, auxiliary heads, or Gumbel search in
  this baseline. Each would be a separate measured experiment.
- No automatic response to a regression. The loop records enough evidence for
  a human-readable diagnosis; changing the algorithm remains an explicit
  experiment.
