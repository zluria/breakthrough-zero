# HPC jobs

These files are reproducible experiment records, not a single automatic
training pipeline. Read the matching note in `docs/reviews/` before rerunning a
job: many scripts pin a checkpoint hash, seed, data directory, and Git commit
for one completed experiment.

The useful entry points are:

- `create_environment.sh`: create the Python environment.
- `project_environment_smoke.sbatch`: check the checkout and test suite.
- `network_boundary_smoke.sbatch`: exercise native 5x5/8x8 Keras boundaries.
- `native_mini_generation_selfplay.sbatch`: generate and audit one immutable
  self-play archive.
- `native_mini_generation_training.sbatch`: train one candidate from pinned
  pretraining and self-play archives.
- `native_mini_generation_arena.sbatch`: compare two distinct, hash-verified
  checkpoints on paired openings.

All other `.sbatch` files preserve historical screens or diagnostics. They are
evidence for the experiment ledger, not recommended defaults. In particular,
`native_mini_large_data_diagnosis.sbatch` is the corrected record of job 33611,
not the next job to submit.
