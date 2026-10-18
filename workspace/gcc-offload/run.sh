#!/bin/sh
# Builds and runs the C and the Fortran examples (gpuaas.py runs this once to check the installation).
set -e
cd "$(dirname "$0")"
# Without the modules loaded in the calling shell (their CUDA or compilers would be used instead).
if [ -f /usr/share/modules/init/sh ]; then . /usr/share/modules/init/sh; module purge; fi
make run
make -C fortran run
