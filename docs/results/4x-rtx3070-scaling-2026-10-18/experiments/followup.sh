#!/bin/bash
# (1) DArray π with one block per GPU instead of four fixed blocks, on 1-4 GPUs; (2) Dagger.Kernel and the manual
# Dagger variant (programs for one GPU) with one and with four visible GPUs. Output lines: "FU <test> <gpus> <run> <result>".
set -u
W=/tmp/gpuaas-4gpu-followup
rm -rf "$W" && mkdir -p "$W" && tar -C "$W" -xf /tmp/gpuaas-4gpu-workspace.tar 2>/dev/null && cd "$W/workspace/julia" || exit 1
sed 's|Blocks(cld(chunks, 4))|Blocks(cld(chunks, CUDA.ndevices()))|' pi_gpu_dagger_darray.jl > pi_gpu_dagger_darray_g.jl
grep -c "CUDA.ndevices()))" pi_gpu_dagger_darray_g.jl
. /usr/share/modules/init/sh; module purge; module load cuda
export PATH="$HOME/.juliaup/bin:$PATH"
res() { grep -E "^GPU|rror" | tail -1 | sed -E 's/, NVIDIA GeForce RTX 3070//; s/ \(error [^)]*\)//'; }
for g in 0 0,1 0,1,2 0,1,2,3; do
  for r in 1 2 3; do echo "FU darray_pi_blocks=g $g $r $(CUDA_VISIBLE_DEVICES=$g julia pi_gpu_dagger_darray_g.jl 2>&1 | res)"; done
done
for prog in matmul_gpu_dagger_kernel.jl matmul_gpu_dagger_manual.jl; do
  for g in 0 0,1,2,3; do
    for r in 1 2 3; do echo "FU $prog $g $r $(CUDA_VISIBLE_DEVICES=$g julia $prog 2>&1 | res)"; done
  done
done
echo "== done"
