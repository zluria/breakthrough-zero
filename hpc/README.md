# HPC jobs

These files are reproducible experiment records. Read the matching note in
`docs/reviews/` before rerunning a historical job: many scripts pin a checkpoint
hash, seed, data directory, and Git commit for one completed experiment.

The useful entry points are:

- `create_environment.sh`: create the Python environment.
- `project_environment_smoke.sbatch`: check the checkout and test suite.
- `network_boundary_smoke.sbatch`: exercise native 5x5/8x8 Keras boundaries.
- `native_mini_generation_selfplay.sbatch`: generate and audit one immutable
  self-play archive from an explicit actor manifest.
- `native_mini_generation_training.sbatch`: update the actor from pinned
  historical and fresh archives, then publish `actor.json`.
- `native_mini_generation_arena.sbatch`: compare two distinct, hash-verified
  checkpoints on paired openings. Its result is diagnostic only.
- `native_mini_continuous_hour.sbatch`: run the preregistered one-hour,
  noise-off self-play/replay-update health validation.
- `native_mini_continuous_hour_arena.sbatch`: after that loop completes,
  compare its initial and final actors on 256 paired openings. It cannot alter
  the loop or its actor.

The active loop always consumes `models.latest` from `actor.json`. Validation
also records `models.best_validation`, but no arena or validation result can
accept, reject, promote, roll back, or authorize an actor. For the one-time
bootstrap from an older artifact, set `PARENT_MANIFEST_NAME=selection.json`;
new runs default to `actor.json`.

The one-hour driver writes an atomically replaced `progress.json`. It is the
live monitoring surface for cycle phase, fresh positions, throughput, replay
consumption, source composition, losses, policy surprise, and actor identity.

All other `.sbatch` files preserve historical screens or diagnostics. They are
evidence for the experiment ledger, not recommended defaults. In particular,
`native_mini_large_data_diagnosis.sbatch` is the corrected record of job 33611,
not the next job to submit.
