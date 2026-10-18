#!/bin/bash
# Phase A: the multi-GPU programs with 1, 2, 3 and 4 visible GPUs (MPI with one rank per GPU), and where
# Dagger puts the Datadeps tasks with 4 GPUs. Runs on a copy of the current workspace in /tmp/gpuaas-4gpu.
set -u
W=/tmp/gpuaas-4gpu
rm -rf "$W" && mkdir -p "$W" && tar -C "$W" -xf /tmp/gpuaas-4gpu-workspace.tar 2>/dev/null; cd "$W/workspace" || exit 1
. /usr/share/modules/init/sh
export PATH="$HOME/.juliaup/bin:$PATH"
COUNTS="0 0,1 0,1,2 0,1,2,3"
( module purge; module load openmpi && cd openmpi && make -s pi_gpu matmul_gpu > /dev/null || exit 1
  for g in $COUNTS; do np=$(echo "$g" | tr , '\n' | wc -l)
    echo "== MPI, GPUs $g, $np ranks"
    CUDA_VISIBLE_DEVICES=$g mpirun --oversubscribe -np "$np" ./pi_gpu
    CUDA_VISIBLE_DEVICES=$g mpirun --oversubscribe -np "$np" ./matmul_gpu
  done
  echo "== MPI, make run uses: $(make -n run | grep -c 'np 4 ./pi_gpu') x np 4 for pi_gpu" )
( module purge; module load chapel && cd chapel && make -s pi_gpus matmul_gpus > /dev/null || exit 1
  for g in $COUNTS; do
    echo "== Chapel, GPUs $g"
    CUDA_VISIBLE_DEVICES=$g ./pi_gpus
    CUDA_VISIBLE_DEVICES=$g ./matmul_gpus
  done )
( module purge; module load cuda && cd julia && cp /tmp/diag_datadeps.jl . || exit 1
  for n in 2048 8192; do echo "== Datadeps placement, 4 GPUs, n = $n"; julia -t auto diag_datadeps.jl "$n" auto; done
  for g in 0,1,2 0,1,2,3; do
    echo "== Julia, GPUs $g"
    CUDA_VISIBLE_DEVICES=$g julia pi_gpu_dagger_datadeps.jl
    CUDA_VISIBLE_DEVICES=$g julia -t auto matmul_gpu_dagger_datadeps.jl
    CUDA_VISIBLE_DEVICES=$g julia pi_gpu_dagger_darray.jl
    CUDA_VISIBLE_DEVICES=$g julia -t auto matmul_gpu_dagger_darray.jl
  done )
echo "== done"
