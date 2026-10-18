#!/bin/sh
# Runs a CPU and a GPU container (gpuaas.py runs this once to check the installation).
set -e
docker run --rm ubuntu sh -c 'echo "CPU: hello from a container with $(nproc) CPUs"'
docker run --rm --gpus all ubuntu nvidia-smi -L
