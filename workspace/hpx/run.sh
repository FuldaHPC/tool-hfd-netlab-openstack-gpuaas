#!/bin/sh
# Builds and runs the examples (gpuaas.py runs this once to check the installation).
set -e
cd "$(dirname "$0")"
. /usr/share/modules/init/sh
module purge  # without the modules loaded in the calling shell
module load hpx
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
for program in pi_seq pi_cpu pi_gpu matmul_seq matmul_cpu matmul_gpu; do ./build/$program; done
