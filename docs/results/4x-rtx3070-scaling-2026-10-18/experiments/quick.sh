#!/bin/bash
set -u
cd /tmp/gpuaas-4gpu/workspace/julia && cp /tmp/darray_layouts.jl . || exit 1
. /usr/share/modules/init/sh; module purge; module load cuda
export PATH="$HOME/.juliaup/bin:$PATH"
for prog in pi_gpu_dagger_datadeps.jl pi_gpu_dagger_darray.jl; do
  for t in "" "-t auto"; do
    for r in 1 2 3; do echo "QUICK $prog ${t:--t1} $r $(julia $t $prog 2>&1 | grep -E '^GPU|rror' | tail -1)"; done
  done
done
echo "== DArray layouts, n = 2048"; julia -t auto darray_layouts.jl 2048 2>&1 | grep -E "matrix product|rror"
echo "== DArray layouts, n = 8192"; julia -t auto darray_layouts.jl 8192 2>&1 | grep -E "matrix product|rror"
echo "== done"
