# Phase 5 review: prove the HPC path before spending the HPC budget

## Live inventory

Checked on 2026-08-11:

- Eight RTX3070 nodes are idle. Each has one `gpu:rtx_3070`, 16 CPU cores,
  and 30 GB RAM.
- RTX3070-07 is running an unrelated user job and must not be disturbed.
- RTX3070-06 and the L40S node are down.
- The administrator notice says RTX2080 nodes are under maintenance even
  though Slurm reports them idle. We will not use them.
- The existing `zur_env` has Python 3.9.18, TensorFlow 2.14.0 built for CUDA
  11.8, cuDNN 8.9.7, and NumPy 1.23.5.

The existing environment is useful for a read-only GPU-stack diagnostic, but
it is not the project environment: this repository targets Python 3.11. We do
not mutate `zur_env` or weaken the project requirement to fit it.

## Scope and resource request

The smoke job requests one RTX3070, two CPUs, 8 GB RAM, and at most ten
minutes. It excludes nodes 06 and 07 explicitly. It does not generate games,
run architecture trials, or retain a model checkpoint.

The job must prove:

1. Slurm grants the named GPU resource on an allowed RTX3070 node.
2. TensorFlow sees exactly the allocated GPU and was built with CUDA support.
3. An explicitly placed matrix multiplication executes on the GPU.
4. A small Keras CNN with policy and absolute-value heads completes one
   synthetic training batch with finite losses and changed weights.
5. The model saves to node-local temporary storage, reloads, and reproduces
   predictions.

## Stop conditions

Any missing GPU, CPU placement, non-finite loss, unchanged weights, save/load
mismatch, or nonzero job status stops the phase. Preserve the log and diagnose
the first failure; do not start a resubmission loop and do not install packages
into `zur_env`.

## Next decision

If this passes, define a dedicated Python 3.11 environment and run the same
smoke plus the repository tests there. Only that second gate can authorize
Keras model development or training jobs.
