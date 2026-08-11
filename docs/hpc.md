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

Never run TensorFlow workloads on the login node. Use `sacct` after completion
to record state, elapsed time, allocated resources, and exit code before
increasing a job budget.
