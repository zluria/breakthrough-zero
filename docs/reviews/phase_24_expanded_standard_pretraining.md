# Phase 24 review: withdrawn 8x8 expansion proposal

## Status

This proposal was reviewed and withdrawn before submission. Its executable
Slurm script was removed so that a future operator cannot mistake an obsolete
plan for the current gate.

The useful diagnosis remains: the existing 64-game standard dataset and the
model trained on only 2,500 positions are a pipeline pilot, not adequate
pretraining evidence. That checkpoint is named `bootstrap-v0`; neither it nor
the padded-mini experiment selects a standard architecture or target.

## Why the proposed run was rejected

The draft jumped directly from 64 to 2,048 games while calling the jump
"successive doubling." It also fixed 64 simulations and `c_puct=1.5` before
the new 5x5 tuning sandbox had tested those assumptions. Although the raw data
would have been reusable, this was still an unjustified commitment of cluster
time and would not have produced the learning curve needed to choose scale.

## Replacement gate

1. Pass the native 5x5/8x8 TensorFlow boundary test.
2. Use native 5x5 experiments to establish informed starting ranges for
   search, architecture, targets, exploration, and training.
3. Generate data in auditable stages, beginning with 512 mini games and
   expanding only when fresh evaluation indicates data limitation.
4. Freeze a coherent baseline and at most three variants, obtain another
   independent three-risk audit, and only then confirm settings locally on
   8x8.
5. If 8x8 data generation is authorized, preserve nested corpus prefixes so
   64/128/256/512/... learning curves can be measured from the same archive.

This document is retained because the rejected plan is itself an important
management lesson: a large round number plus a reusable-data argument is not a
substitute for a staged decision rule.
