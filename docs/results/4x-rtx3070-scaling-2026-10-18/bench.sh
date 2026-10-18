#!/bin/bash
# Scaling over one to four GPUs on a g1-4x3070.xlarge: π and the matrix product (n = 2048 and 8192) of the
# programs for several GPUs with CUDA_VISIBLE_DEVICES = 0 / 0,1 / 0,1,2 / 0,1,2,3 (MPI with one rank per
# GPU), and the single-GPU programs they are compared with, on one GPU. Every program runs RUNS times in a
# temporary copy of the workspace; runs.csv gets problem,n,program,gpus,run,seconds,gflops,status
# (gpus = number of visible GPUs), all output goes to stdout.
set -u
W=${W:-/tmp/gpuaas-4gpu-scaling}
RUNS=${RUNS:-3}
NS=${NS:-"2048 8192"}
rm -rf "$W" && mkdir -p "$W" && tar -C "$W" -xf /tmp/gpuaas-4gpu-workspace.tar 2>/dev/null && cd "$W/workspace" || exit 1
. /usr/share/modules/init/sh
export PATH="$HOME/.juliaup/bin:$PATH"
csv=$W/runs.csv
echo "problem,n,program,gpus,run,seconds,gflops,status" > "$csv"

record() {  # problem n program gpus run output-file
  local line secs gf st err
  line=$(grep -E " in [0-9.]+ s \([0-9.]+ GFLOP/s\)$" "$6" | tail -1)
  if [ -z "$line" ]; then
    echo "$1,$2,\"$3\",$4,$5,,,failed" >> "$csv"
    return
  fi
  secs=$(echo "$line" | sed -E 's/.* in ([0-9.]+) s .*/\1/')
  gf=$(echo "$line" | sed -E 's/.*\(([0-9.]+) GFLOP\/s\)$/\1/')
  if [ "$1" = matmul ]; then
    case $line in *" correct in "*) st=ok ;; *) st=wrong ;; esac
  else
    err=$(echo "$line" | sed -E 's/.*\(error ([0-9.e+-]+)\).*/\1/')
    st=$(awk -v e="$err" 'BEGIN { print (e + 0 < 1e-9) ? "ok" : "wrong" }')
  fi
  echo "$1,$2,\"$3\",$4,$5,$secs,$gf,$st" >> "$csv"
}

run() {  # problem n program gpu-list command ...
  local problem=$1 n=$2 prog=$3 list=$4 count r out
  count=$(echo "$list" | tr , '\n' | wc -l)
  shift 4
  for r in $(seq 1 "$RUNS"); do
    out=$(mktemp)
    echo "== $problem n=$n | $prog | $count GPU(s) | run $r | $(date +%T)"
    CUDA_VISIBLE_DEVICES=$list timeout 900 "$@" > "$out" 2>&1
    grep -E " in [0-9.]+ s \(|rror|Abort" "$out" | tail -2
    record "$problem" "$n" "$prog" "$count" "$r" "$out"
    rm -f "$out"
  done
}

COUNTS="0 0,1 0,1,2 0,1,2,3"
build() {  # n: set the matrix size and build the compiled programs
  sed -i -E "s/^#define MATRIX_N [0-9]+L/#define MATRIX_N ${1}L/" cuda/common.h openmpi/common.h
  sed -i -E "s/^const MATRIX_N = [0-9]+/const MATRIX_N = $1/" julia/common.jl
  ( module purge; export PATH=/usr/local/cuda/bin:$PATH; cd cuda && rm -f pi_gpu matmul_gpu && make -s pi_gpu matmul_gpu > /dev/null )
  ( module purge; module load openmpi; cd openmpi && rm -f pi_gpu matmul_gpu && make -s pi_gpu matmul_gpu > /dev/null )
}
( module purge; module load chapel; cd chapel && make -s pi_gpu pi_gpus matmul_gpu matmul_gpus > /dev/null ) || echo "Chapel build failed"

# π (the same size in every run)
echo "######## pi ($(date +%T))"
build 2048
( cd cuda && run pi 0 "CUDA kernel" 0 ./pi_gpu )
( module purge; module load chapel; cd chapel && run pi 0 "Chapel pi_gpu" 0 ./pi_gpu )
( module purge; module load cuda; cd julia && run pi 0 "CUDA.jl kernel" 0 julia pi_gpu_kernel.jl )
for g in $COUNTS; do
  np=$(echo "$g" | tr , '\n' | wc -l)
  ( module purge; module load openmpi; cd openmpi && run pi 0 "MPI" "$g" mpirun --oversubscribe -np "$np" ./pi_gpu )
  ( module purge; module load chapel; cd chapel && run pi 0 "Chapel pi_gpus" "$g" ./pi_gpus )
  ( module purge; module load cuda; cd julia && run pi 0 "Dagger Datadeps" "$g" julia pi_gpu_dagger_datadeps.jl &&
    run pi 0 "Dagger DArray" "$g" julia pi_gpu_dagger_darray.jl )
done

for n in $NS; do
  echo "######## matmul n = $n ($(date +%T))"
  build "$n"
  ( cd cuda && run matmul "$n" "CUDA kernel" 0 ./matmul_gpu )
  ( module purge; module load chapel; cd chapel && run matmul "$n" "Chapel matmul_gpu" 0 ./matmul_gpu --matrixN="$n" )
  ( module purge; module load cuda; cd julia && run matmul "$n" "CUDA.jl kernel" 0 julia matmul_gpu_kernel.jl &&
    run matmul "$n" "CUDA.jl cuBLAS" 0 julia matmul_gpu_arrays.jl )
  for g in $COUNTS; do
    np=$(echo "$g" | tr , '\n' | wc -l)
    ( module purge; module load openmpi; cd openmpi && run matmul "$n" "MPI" "$g" mpirun --oversubscribe -np "$np" ./matmul_gpu )
    ( module purge; module load chapel; cd chapel && run matmul "$n" "Chapel matmul_gpus" "$g" ./matmul_gpus --matrixN="$n" )
    ( module purge; module load cuda; cd julia && run matmul "$n" "Dagger Datadeps" "$g" julia -t auto matmul_gpu_dagger_datadeps.jl &&
      run matmul "$n" "Dagger DArray" "$g" julia -t auto matmul_gpu_dagger_darray.jl )
  done
done
echo "######## done ($(date +%T))"
