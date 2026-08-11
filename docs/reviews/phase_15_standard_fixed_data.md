# Phase 15 review: standard-board fixed-data training

## Experiment

The first standard-board neural comparison is a 2x2 design:

| Search data | Final-result value | Soft-Z value |
| --- | --- | --- |
| 32 simulations | task 0 | task 1 |
| 64 simulations | task 2 | task 3 |

All four jobs use the 64-channel, four-block residual CNN.  The architecture is
held fixed because the mini experiment showed similar policy losses and no
reliable valid Elo separation yet; architecture tuning is not the question in
this stage.

## Equal data and augmentation

Game-level train/validation splits are made before position sampling.  Each job
then deterministically selects exactly 2,500 training and 600 validation
positions from its split.  This equalizes examples, batches, and optimizer
updates despite different game lengths.  Every training draw still samples one
of the four exact augmentations; validation stays identity-only.

The complete raw games remain available.  Position limiting is an experiment
control, not discarded data, and selected positions are reproducible from the
stored input checksums and seed.

## Predictions

1. Soft-Z will again be much easier to fit than the final outcome.
2. The 64-simulation policy target may learn faster because it has higher visit
   entropy, but a diffuse target can also make top-move accuracy lower.
3. If 64-simulation data does not improve paired equal-time Elo, 32 simulations
   should be preferred for its roughly 65 percent higher generation throughput.
4. No result is promoted from loss alone.  The arena must have zero abnormal
   terminations and report actual simulations per move.

## Budget

Each task gets one RTX3070, two CPUs, four GB, and at most 20 minutes.  Two may
run concurrently.  Forty epochs correspond to the same 800 optimizer batches
per job.  No additional 8x8 self-play is authorized by this phase.
