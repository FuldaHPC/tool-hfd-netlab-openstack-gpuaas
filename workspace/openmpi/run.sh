#!/bin/sh
# Builds and runs the examples (gpuaas.py runs this once to check the installation).
set -e
cd "$(dirname "$0")"
. /usr/share/modules/init/sh
module purge  # without the modules loaded in the calling shell
module load openmpi
make run
