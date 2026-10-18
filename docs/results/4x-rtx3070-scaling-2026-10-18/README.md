# Several GPUs: one to four RTX 3070

The programs for several GPUs with one, two, three and four visible GPUs (`CUDA_VISIBLE_DEVICES=0`, `0,1`, `0,1,2`, `0,1,2,3`), each run three times, for π (10⁹ steps) and for the matrix product with n = 2048 (the size of the workspace) and n = 8192, together with the programs for one GPU they are compared with.
Measured on 2026-10-18 with `bench.sh` on a `g1-4x3070.xlarge` (four RTX 3070, 4 vCPUs of an Intel Xeon Cascade Lake, 16 GB of RAM) started from an HPC image; the GPUs share PCIe through the host without peer-to-peer access (`nvidia-smi topo -p2p r`: chipset not supported).
MPI runs one rank per GPU (`make run` does the same), the Julia matrix products run with `-t auto` (as `julia/run.sh` does), and all 177 runs were correct; as everywhere in the workspace, the times include GPU memory allocation and all transfers.
The DArray π rows come from a second run (`experiments/followup.sh`) after `pi_gpu_dagger_darray.jl` got one block per GPU instead of four fixed blocks, which had left one GPU with twice the work on three GPUs (0.070 s instead of 0.051 s).

Plots: [scaling.pdf](scaling.pdf) (page 1 the speed-up over one GPU, page 2 GFLOP/s).

### π

| Program | 1 GPU | 2 GPUs | 3 GPUs | 4 GPUs | speed-up 1 → 4 |
|---|---:|---:|---:|---:|---:|
| MPI, one rank per GPU | 0.087 s | 0.047 s | 0.032 s | 0.024 s | 3.62× |
| Chapel *_gpus | 0.088 s | 0.052 s | 0.038 s | 0.033 s | 2.67× |
| Dagger Datadeps | 0.099 s | 0.058 s | 0.048 s | 0.062 s | 1.60× |
| Dagger DArray | 0.097 s | 0.056 s | 0.051 s | 0.048 s | 2.02× |
| CUDA kernel (one GPU) | 0.089 s | | | | |
| Chapel pi_gpu (one GPU) | 0.088 s | | | | |
| CUDA.jl kernel (one GPU) | 0.085 s | | | | |

### C = A B, n = 2048

| Program | 1 GPU | 2 GPUs | 3 GPUs | 4 GPUs | speed-up 1 → 4 |
|---|---:|---:|---:|---:|---:|
| MPI, one rank per GPU | 0.037 s | 0.048 s | 0.054 s | 0.052 s | 0.71× |
| Chapel *_gpus | 0.040 s | 0.028 s | 0.023 s | 0.021 s | 1.90× |
| Dagger Datadeps | 0.053 s | 0.052 s | 0.050 s | 0.090 s | 0.59× |
| Dagger DArray | 0.059 s | 0.083 s | 0.065 s | 0.071 s | 0.83× |
| CUDA kernel (one GPU) | 0.030 s | | | | |
| Chapel matmul_gpu (one GPU) | 0.040 s | | | | |
| CUDA.jl kernel (one GPU) | 0.024 s | | | | |
| CUDA.jl cuBLAS (one GPU) | 0.013 s | | | | |

### C = A B, n = 8192

| Program | 1 GPU | 2 GPUs | 3 GPUs | 4 GPUs | speed-up 1 → 4 |
|---|---:|---:|---:|---:|---:|
| MPI, one rank per GPU | 1.159 s | 0.978 s | 0.942 s | 0.963 s | 1.20× |
| Chapel *_gpus | 5.333 s | 2.699 s | 1.823 s | 1.392 s | 3.83× |
| Dagger Datadeps | 1.245 s | 0.901 s | 0.740 s | 0.715 s | 1.74× |
| Dagger DArray | 1.013 s | 1.028 s | 1.166 s | 1.124 s | 0.90× |
| CUDA kernel (one GPU) | 1.026 s | | | | |
| Chapel matmul_gpu (one GPU) | 5.228 s | | | | |
| CUDA.jl kernel (one GPU) | 0.980 s | | | | |
| CUDA.jl cuBLAS (one GPU) | 0.243 s | | | | |

## Findings

- π computes much and moves almost no data: MPI gets 3.6 times as fast on four GPUs, Chapel 2.7, DArray 2.0 and Datadeps 1.6 times. Chapel and Dagger lose part of the gain to their fixed costs per run (task start, small transfers), which do not shrink with more GPUs; for Dagger these are a few tens of milliseconds, a large part of a 0.05 s run.
- The matrix product with n = 2048 moves 48 MB between host and GPUs, and every GPU needs all of B: only Chapel gains (1.9× on four GPUs); MPI and both Dagger variants are slower on four GPUs than on one, because each further GPU adds 16 MB of transfers through the host but saves only a few milliseconds of computing.
- With n = 8192 the work grows 64 times and the data 16 times. Chapel then gets 3.8 times as fast on four GPUs and Datadeps 1.7 times (Dagger puts the four tasks on four different GPUs, but copies A to them one after the other).
- MPI stays at 1.2× even with n = 8192: each GPU computes its quarter in about 0.25 s, but rank 0 sends about 1 GB to the other ranks (rows of A, all of B to each, rows of C back), which takes about 0.7 s. This Open MPI (Spack's default build) has no single-copy mechanism for shared memory (no CMA, XPMEM or KNEM), so messages between ranks are copied twice; collectives on host buffers instead of the CUDA-aware ones, or without `smcuda`, were not faster (`experiments/mpiexp.sh`).
- DArray's `A * B` multiplies 2 × 2 blocks with cuBLAS and moves blocks between the GPUs through the host; with more GPUs it moves more, so it stays as fast as on one GPU. Other block layouts were slower: one column block of B per GPU took 2.23 s, 4 × 4 blocks 1.58 s, 2 × 2 blocks 1.18 s at n = 8192 (`experiments/darray_layouts.jl`).
- Speed-up is not speed: Chapel scales best, but its matrix kernel reaches only about 200 GFLOP/s per GPU at n = 8192, so it needs 1.39 s on four GPUs, where the CUDA kernel needs 1.03 s on one GPU and cuBLAS 0.24 s.
- The Dagger programs for one GPU (`Dagger.Kernel`, manual) vary from 0.038 to 0.088 s for n = 2048, with one visible GPU as with four (`experiments/followup.sh`), so the visible GPUs do not matter for them.

## Files

- `bench.sh` (the measurement), `bench.log` (its output), `runs.csv` (every run: `problem,n,program,gpus,run,seconds,gflops,status`), `results.csv` (median, fastest and slowest run per problem, size, program and number of GPUs).
- `evaluate.py` writes `results.csv`, `scaling.tsv` and the tables above; `gnuplot plot_scaling.gp` draws `scaling.pdf`.
- `experiments/`: the correctness check with 1–4 GPUs and the Datadeps placement (`check4.sh`, `diag_datadeps.jl`), threads for π and the DArray layouts (`quick.sh`, `darray_layouts.jl`), the DArray block count and the Dagger programs for one GPU (`followup.sh`), and the MPI variants (`mpiexp.sh`, `matmul_gpu_host.cu`), each with its log.
