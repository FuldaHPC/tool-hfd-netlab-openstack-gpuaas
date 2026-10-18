#!/bin/bash
# MPI matrix product at n = 8192 on 4 GPUs: CUDA-aware (as now), host buffers, and CUDA-aware without smcuda.
set -u
W=/tmp/gpuaas-4gpu-mpiexp
rm -rf "$W" && mkdir -p "$W" && tar -C "$W" -xf /tmp/gpuaas-4gpu-workspace.tar 2>/dev/null && cd "$W/workspace/openmpi" || exit 1
cp /tmp/matmul_gpu_host.cu .
. /usr/share/modules/init/sh; module purge; module load openmpi
for n in 8192 2048; do
  sed -i -E "s/^#define MATRIX_N [0-9]+L/#define MATRIX_N ${n}L/" common.h
  rm -f matmul_gpu matmul_gpu_host && make -s matmul_gpu > /dev/null && nvcc -ccbin mpicxx -O3 -Xcompiler -march=x86-64-v3 -arch=native -o matmul_gpu_host matmul_gpu_host.cu || exit 1
  for r in 1 2 3; do
    echo "MX $n cuda-aware $r $(mpirun --oversubscribe -np 4 ./matmul_gpu 2>&1 | grep -E 'matrix product|rror' | tail -1)"
    echo "MX $n host-buffers $r $(mpirun --oversubscribe -np 4 ./matmul_gpu_host 2>&1 | grep -E 'matrix product|rror' | tail -1)"
    echo "MX $n cuda-aware-btl-sm $r $(mpirun --oversubscribe -np 4 --mca btl self,sm ./matmul_gpu 2>&1 | grep -E 'matrix product|rror' | tail -1)"
  done
done
echo "== done"
