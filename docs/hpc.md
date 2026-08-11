# HPC operations

The cluster is a measured-compute backend, not a place to debug interactively.
Every job gets a phase review, explicit resource limits, a log, and a local
test or smoke gate before its budget increases.

## Current safe resources

The live inventory and maintenance exclusions are recorded in
[`reviews/phase_05_hpc_smoke.md`](reviews/phase_05_hpc_smoke.md). As of
2026-08-11, use only the `rtx3070` partition and request the named GPU resource
`gpu:rtx_3070:1`. Do not use RTX2080, GTX1080Ti, L40S, RTX3070-06, or
RTX3070-07 until the recorded reasons change.

## Smoke submission

From a clean repository checkout on the login node:

```bash
mkdir -p logs
sbatch hpc/keras_smoke.sbatch
squeue -u "$USER"
```

The job uses the existing `zur_env` read-only to diagnose the GPU stack. It is
not a training environment and does not install or retain anything. A passing
log ends with one JSON object whose `status` is `pass`, followed by a finish
timestamp. Both standard output and error are retained in `logs/`.

That diagnostic passed as job `33475`; details are in the phase review. The
project environment is deliberately separate:

```bash
bash hpc/create_environment.sh
mkdir -p logs
sbatch hpc/project_environment_smoke.sbatch
```

Environment creation installs the pinned direct requirements but does not
import TensorFlow on the login node. The Slurm gate runs dependency checks, all
repository tests, and the Keras GPU smoke. See
[`reviews/phase_06_hpc_environment.md`](reviews/phase_06_hpc_environment.md)
for the rationale and stop conditions.

The project gate passed as job `33476`. Exact package versions are in
`requirements-hpc-lock.txt`; raw job logs remain in the ignored HPC `logs/`
directory, and the durable summary is in
[`benchmarks/hpc_smoke_20260811.md`](benchmarks/hpc_smoke_20260811.md).

Never run TensorFlow workloads on the login node. Use `sacct` after completion
to record state, elapsed time, allocated resources, and exit code before
increasing a job budget.

## CPU-only baseline tournament

The cluster has no separate CPU partition. `hpc/mini_tournament.sbatch` uses
one CPU and 2 GB on an available RTX3070 host but deliberately requests no
GPU resource. Slurm can therefore leave the accelerator available to another
job. It runs all tests before 32 paired openings per matchup and saves every
game, seed, timing, work count, termination reason, and rating summary.

```bash
mkdir -p logs
sbatch hpc/mini_tournament.sbatch
```

This tournament is a baseline measurement, not a TensorFlow workload. The
review and pre-registered predictions are in
[`reviews/phase_08_pre_tournament_predictions.md`](reviews/phase_08_pre_tournament_predictions.md).
