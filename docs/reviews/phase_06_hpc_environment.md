# Phase 6 review: build a reproducible project environment

## Decision

Use Python 3.11 with the current stable TensorFlow 2.21.0 Linux GPU package.
TensorFlow's official installation guide recommends a virtual environment and
`pip`, not conda, and its tested-build table pairs TensorFlow 2.21 with CUDA
12.5 and cuDNN 9.3. NVIDIA documents CUDA 12.x minor compatibility for Linux
drivers from 525 through 579; the cluster driver is 550.54.15.

Direct requirements live in `requirements-hpc.txt`. The private environment
path is `$HOME/.venvs/btz-py311-tf221`. Its name makes accidental reuse after a
version change conspicuous.

## Installation discipline

- Build under a unique `.building.<pid>` path and move into the final path only
  after `pip check` succeeds.
- Refuse to overwrite an existing final environment.
- Preserve the exact `pip freeze` inside the environment.
- Do not import TensorFlow or run numerical workloads on the login node.
- Do not modify the existing `zur_env`.
- If installation fails, retain the uniquely named build for diagnosis rather
  than silently retrying or blessing a partial environment.

## Project-environment gate

The follow-up Slurm job has the same ten-minute resource ceiling and node
exclusions as the first smoke. It must run:

1. `pip check` in the new environment.
2. The complete repository unit-test suite.
3. The Keras GPU placement, train, weight-change, save, and reload smoke.

Only an exit-code-zero job with all three checks passing authorizes this
environment for model work. Installation success alone is insufficient.
