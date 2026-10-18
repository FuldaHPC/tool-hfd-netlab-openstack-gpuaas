#!/bin/sh
# Runs the examples (gpuaas.py runs this once to check the installation).
set -e
cd "$(dirname "$0")"
export PATH="$HOME/.juliaup/bin:$PATH"
. /usr/share/modules/init/sh
module purge  # without the modules loaded in the calling shell
module load cuda
julia pi_seq.jl
julia -t auto pi_cpu.jl
for variant in arrays kernel ka dagger_kernel dagger_manual dagger_datadeps dagger_darray; do
  julia pi_gpu_$variant.jl
done
julia matmul_seq.jl
julia -t auto matmul_cpu.jl
for variant in arrays kernel ka dagger_kernel dagger_manual; do
  julia matmul_gpu_$variant.jl
done
# With one thread, Dagger moves the matrices of several GPUs one after the other (DArray gains on one GPU too).
julia -t auto matmul_gpu_dagger_datadeps.jl
julia -t auto matmul_gpu_dagger_darray.jl
