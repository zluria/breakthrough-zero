#!/bin/bash -l
# Create the private HPC Python environment without running TensorFlow.

set -euo pipefail

readonly environment_parent="${BTZ_ENV_PARENT:-$HOME/.venvs}"
readonly environment="$environment_parent/btz-py311-tf221"
readonly building="$environment.building.$$"

if [[ -e "$environment" ]]; then
    echo "refusing to overwrite existing environment: $environment" >&2
    exit 1
fi

module load anaconda/anaconda
if [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.11" ]]; then
    echo "the Anaconda module did not provide Python 3.11" >&2
    exit 1
fi

mkdir -p "$environment_parent"
python -m venv "$building"
"$building/bin/python" -m pip install --upgrade pip
"$building/bin/python" -m pip install --requirement requirements-hpc-lock.txt
"$building/bin/python" -m pip check
"$building/bin/python" -m pip freeze --all > "$building/requirements.freeze.txt"
mv "$building" "$environment"

echo "created=$environment"
echo "TensorFlow has not been imported; run the Slurm environment smoke next."
