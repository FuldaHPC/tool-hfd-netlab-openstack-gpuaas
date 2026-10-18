# Workspace

Examples for the software that `gpuaas.py` installed on this instance; only the folders of installed packages are present.
Every folder except `docker/` computes the same two problems in its programming model:

- `pi_*`: π as the integral of 4/(1+x²) over [0, 1] with the midpoint rule, 10⁹ steps in double precision.
- `matmul_*`: the matrix product C = A B of two 2048 × 2048 matrices in single precision, with the same simple algorithm everywhere (no BLAS, except where a Julia program says so), so that the models are compared and not libraries.

`*_seq` runs on one CPU core as reference, written as close as possible to the parallel version (a plain loop, Kokkos `Serial`, HPX `seq`, Chapel `for`), `*_cpu` uses all CPU cores, and `*_gpu` the GPU; where a folder has several GPU models, each has its own program (e.g. `*_acc` for OpenACC).

Each parallel and GPU program runs once on a small problem (warm-up: threads, GPU, compilers), then times the real one; the sequential programs time their first run (nothing to start), and the Julia programs warm up with the full problem until a run compiles no more code, as Julia compiles code while it runs.
GPU times include allocating GPU memory and copying the data to the GPU and back.
Each program prints its result, the time in seconds and the rate in GFLOP/s (6 floating-point operations per step of π, 2n³ for C = A B), and fails if the result is wrong or the GPU was not used.

| Folder          | Sequential and CPU                                   | GPU                                                                  | Build         |
|-----------------|------------------------------------------------------|----------------------------------------------------------------------|---------------|
| `cuda/`         | C++ loop (no parallel CPU version)                   | CUDA kernels                                                         | Makefile      |
| `gcc-offload/`  | C and Fortran: loops, OpenMP (GCC, gfortran)         | OpenMP `target` (C, Fortran), OpenACC (Fortran)                      | Makefile      |
| `llvm-offload/` | C and Fortran: loops, OpenMP (Clang, Flang)          | OpenMP `target` (C, Fortran)                                         | Makefile      |
| `nvhpc/`        | Fortran: loops, OpenMP (nvfortran)                   | OpenMP `target`, OpenACC, `do concurrent`, CUDA Fortran              | Makefile      |
| `openmpi/`      | MPI ranks                                            | MPI ranks with CUDA kernels, CUDA-aware MPI                          | Makefile      |
| `kokkos/`       | `parallel_*` in OpenMP (seq: Serial)                 | the same in the CUDA execution space                                 | CMake         |
| `hpx/`          | tasks, `for_loop(par)` (seq: `seq`)                  | CUDA kernels through an HPX executor                                 | CMake         |
| `chapel/`       | `forall` loops (seq: `for`)                          | `forall` loops in `on here.gpus[0]`                                  | Makefile      |
| `julia/`        | loops, Dagger tasks on threads                       | CUDA.jl arrays and kernels, KernelAbstractions, four ways of Dagger  | none          |
| `docker/`       | Ubuntu container                                     | container with `--gpus all`                                          | `run.sh` only |

Each folder has a `README.md` with the commands and a `run.sh` that builds and runs its programs; `gpuaas.py` ran every `run.sh` once after the installation, on a copy of the folder, to check that everything works (several at once on large flavors, so the times in that log are not meaningful).
The sizes and checks are in `common.h` (C, C++), `common.f90`, `Common.chpl` and `common.jl`, so the programs show only the model.
C, C++ and Fortran are compiled with `-O3` for `x86-64-v3` (AVX2 and FMA; as fast as `-march=native` on the cloud's hosts, and the programs still run after a snapshot is started on another host); Julia and Chapel compile for the CPU they run on.

## Running all examples

```sh
./run_all.sh              # all folders, or e.g. ./run_all.sh cuda julia
./run_all.sh -n 3         # every folder three times; the CSV, table and plot show the median run
```

`run_all.sh` runs the `run.sh` of every folder (the output goes to `results/<folder>.log`), collects all results in `results/results.csv` and plots them with gnuplot into `results/results.pdf`: one page for π and one for the matrix product, with a bar per program on a logarithmic GFLOP/s axis, grouped by folder.
It installs gnuplot (`gnuplot-nox`) with sudo when it is missing, and exits with 1 if a folder failed; programs with a wrong result are listed in the CSV but not plotted.
All folders together take about 10 minutes per run, most of it Julia compiling.
With `-n`, `results/runs.csv` has every run and `results/results.csv` the run with the median time of each program, with the number of runs and the fastest and slowest time (short GPU times vary by 10–30 % from run to run).

## Results

Measured on 2026-10-18 with `./run_all.sh -n 3` on a `g1-4x3070.xlarge` (4 vCPUs of an Intel Xeon Cascade Lake, four RTX 3070): the median of three runs of each program; the sequential versions use one core, the parallel CPU versions all four, the programs for one GPU the first GPU, and those for several GPUs ("on 4 GPUs") all four.
Single runs vary more than the medians (e.g. `Dagger.Kernel`'s matrix product took 0.060–0.088 s), and CPU times by up to 20 % between instances (other instances on the same host); `results/results.csv` has the fastest and slowest run of each program.
The CSV files, logs and plots of this run are in the repository under [docs/results/4x-rtx3070-2026-10-18/](../docs/results/4x-rtx3070-2026-10-18/).

| Folder          | Kind | Model                                                            |       π | GFLOP/s | C = A B | GFLOP/s |
|-----------------|------|------------------------------------------------------------------|--------:|--------:|--------:|--------:|
| `cuda/`         | seq  | C++                                                              | 1.441 s |     4.2 | 2.059 s |     8.3 |
|                 | GPU  | CUDA kernel                                                      | 0.093 s |    64.4 | 0.026 s |     649 |
| `gcc-offload/`  | seq  | GCC                                                              | 1.494 s |     4.0 | 2.037 s |     8.4 |
|                 | CPU  | GCC OpenMP, 4 threads                                            | 0.461 s |    13.0 | 0.909 s |    18.9 |
|                 | GPU  | GCC OpenMP target                                                | 0.092 s |    65.2 | 0.052 s |     328 |
|                 | seq  | gfortran                                                         | 1.430 s |     4.2 | 2.099 s |     8.2 |
|                 | CPU  | gfortran OpenMP, 4 threads                                       | 0.460 s |    13.0 | 0.892 s |    19.3 |
|                 | GPU  | gfortran OpenMP target                                           | 0.092 s |    65.1 | 0.050 s |     344 |
|                 | GPU  | gfortran OpenACC                                                 | 0.085 s |    70.7 | 0.055 s |     314 |
| `llvm-offload/` | seq  | Clang                                                            | 1.429 s |     4.2 | 2.396 s |     7.2 |
|                 | CPU  | Clang OpenMP, 4 threads                                          | 0.460 s |    13.0 | 0.842 s |    20.4 |
|                 | GPU  | Clang OpenMP target                                              | 0.144 s |    41.6 | 0.069 s |     248 |
|                 | seq  | flang                                                            | 1.427 s |     4.2 | 2.351 s |     7.3 |
|                 | CPU  | flang OpenMP, 4 threads                                          | 0.475 s |    12.6 | 0.865 s |    19.9 |
|                 | GPU  | flang OpenMP target                                              | 0.093 s |    64.2 | 0.291 s |    59.0 |
| `nvhpc/`        | seq  | nvfortran                                                        | 0.813 s |     7.4 | 2.314 s |     7.4 |
|                 | CPU  | nvfortran OpenMP, 4 threads                                      | 0.246 s |    24.4 | 0.802 s |    21.4 |
|                 | GPU  | nvfortran OpenMP target                                          | 0.141 s |    42.6 | 0.037 s |     458 |
|                 | GPU  | nvfortran OpenACC                                                | 0.085 s |    70.8 | 0.035 s |     490 |
|                 | GPU  | nvfortran do concurrent                                          | 0.086 s |    70.0 | 0.086 s |     201 |
|                 | GPU  | nvfortran CUDA Fortran                                           | 0.087 s |    68.8 | 0.061 s |     282 |
| `openmpi/`      | seq  | GCC                                                              | 1.450 s |     4.1 | 2.113 s |     8.1 |
|                 | CPU  | MPI, 4 ranks                                                     | 0.471 s |    12.7 | 1.017 s |    16.9 |
|                 | GPU  | MPI, 4 ranks on 4 GPUs with CUDA kernels, CUDA-aware             | 0.024 s |     249 | 0.056 s |     306 |
| `kokkos/`       | seq  | Kokkos Serial                                                    | 2.813 s |     2.1 | 2.319 s |     7.4 |
|                 | CPU  | Kokkos OpenMP, 4 threads                                         | 0.899 s |     6.7 | 0.831 s |    20.7 |
|                 | GPU  | Kokkos Cuda                                                      | 0.090 s |    66.4 | 0.027 s |     647 |
| `hpx/`          | seq  | HPX for_loop seq                                                 | 1.443 s |     4.2 | 2.752 s |     6.2 |
|                 | CPU  | HPX, 4 worker threads, 16 tasks / HPX for_loop, 4 worker threads | 0.458 s |    13.1 | 0.901 s |    19.1 |
|                 | GPU  | CUDA kernel via HPX cuda_executor                                | 0.093 s |    64.2 | 0.031 s |     555 |
| `chapel/`       | seq  | Chapel for                                                       | 1.445 s |     4.2 | 2.385 s |     7.2 |
|                 | CPU  | Chapel forall, 4 tasks                                           | 0.462 s |    13.0 | 0.793 s |    21.7 |
|                 | GPU  | Chapel forall on here.gpus[0]                                    | 0.094 s |    63.6 | 0.037 s |     470 |
|                 | GPU  | Chapel forall on 4 GPUs                                          | 0.032 s |     188 | 0.023 s |     759 |
| `julia/`        | seq  | Julia                                                            | 1.452 s |     4.1 | 2.264 s |     7.6 |
|                 | CPU  | Dagger, 4 threads, 16 tasks                                      | 0.417 s |    14.4 | 0.728 s |    23.6 |
|                 | GPU  | CUDA.jl arrays: A * B with cuBLAS                                | 0.086 s |    69.7 | 0.012 s |    1383 |
|                 | GPU  | CUDA.jl kernel                                                   | 0.085 s |    70.4 | 0.024 s |     714 |
|                 | GPU  | KernelAbstractions                                               | 0.085 s |    70.3 | 0.025 s |     701 |
|                 | GPU  | Dagger.Kernel                                                    | 0.099 s |    60.9 | 0.062 s |     276 |
|                 | GPU  | Dagger manual on CUDABackend                                     | 0.100 s |    59.8 | 0.082 s |     210 |
|                 | GPU  | Dagger Datadeps on 4 GPUs                                        | 0.056 s |     108 | 0.049 s |     351 |
|                 | GPU  | Dagger DArray on 4 GPUs: A * B with cuBLAS                       | 0.048 s |     126 | 0.056 s |     308 |

- With 4 cores, the parallel CPU versions are about 3 times faster than the sequential ones for π and 2.5 times for C = A B, which is limited by memory bandwidth.
- π on one GPU is limited by double precision, which the RTX 3070 computes at 1/64 of its single-precision speed; most models reach the same 0.085–0.094 s, while Clang's and nvfortran's OpenMP `target` (0.14 s) and the Dagger variants for one GPU (0.10 s) take longer. On four GPUs, MPI (0.024 s) and Chapel (0.032 s) are 3–4 times as fast, Dagger's Datadeps and DArray (0.048–0.056 s) about twice as fast.
- In the matrix product on one GPU, hand-written kernels are fastest (CUDA, Kokkos, HPX, CUDA.jl, KernelAbstractions: 0.024–0.031 s), followed by nvfortran's OpenMP and OpenACC (0.035–0.037 s) and Chapel (0.037 s); GCC's OpenMP and OpenACC take about twice as long, the Dagger kernels two to three times. cuBLAS (`julia/` arrays) is faster still (0.012 s), being a tuned library and not the simple algorithm. With n = 2048 only Chapel gains from four GPUs (0.023 s); for the others the transfers to four GPUs cost as much as the computing they save (see "Several GPUs").
- How a model maps loops to the GPU matters: GCC's OpenMP needs `simd` (30 times slower without), Clang's is fastest without it, a Chapel `forall` over a 2D domain was 60 times slower than over all n·n elements, and Flang (LLVM 22) needs 0.29 s, four times as long as Clang and eight times as long as nvfortran, which compile the same algorithm (nvfortran the same Fortran source).
- nvfortran vectorizes the π sum and adds in another order (GCC and Clang keep the order of the source), so its CPU versions of π are about twice as fast.
- Kokkos's CPU versions of π are about twice as slow as plain loops: in a build with CUDA, `KOKKOS_LAMBDA` makes the loop body a `__host__ __device__` lambda, which nvcc hands to the host compiler behind a wrapper that is not inlined, so every one of the 10⁹ steps is a function call; with a plain host lambda (`[=]`) the same programs are as fast as plain loops. The matrix product calls the lambda once per row, so it is as fast as plain loops.
- CUDA Fortran's matrix product (0.061 s) loses its time in the copies between host and GPU, not in the kernel (see `nvhpc/README.md`).
- Every timed result buffer is written once before the timer starts: the page faults of a fresh 16 MB buffer would add about 8 ms to the GPU times.

## Several GPUs

On flavors with several GPUs (e.g. `g1-4x3070.xlarge`), these programs use all of them; on one GPU they work as before:

- `openmpi/*_gpu.cu`: `make run` starts one MPI rank per GPU, and every rank takes its own GPU (round robin if there are more ranks); the matrix product splits the rows as evenly as possible, so three GPUs work too.
- `chapel/*_gpus.chpl`: a `coforall` over `here.gpus`, each GPU computes its share (the `*_gpu.chpl` programs stay on one GPU, because the general version's matrix kernel is 8 % slower on one GPU).
- `julia/*_gpu_dagger_datadeps.jl` (one task per GPU) and `julia/*_gpu_dagger_darray.jl` (DArray blocks spread over the GPUs), both in the scope `cuda_gpus=:`.

The other programs use one GPU; spreading them needs explicit device handling.
Measured on 2026-10-18 on a `g1-4x3070.xlarge` (four RTX 3070 without peer-to-peer access, PCIe through the host), the median of three runs with one and with four visible GPUs (`CUDA_VISIBLE_DEVICES`); two and three GPUs, and plots, are in [docs/results/4x-rtx3070-scaling-2026-10-18/](../docs/results/4x-rtx3070-scaling-2026-10-18/):

| Program                                      | π, 1 GPU | 4 GPUs | C = A B, 1 GPU | 4 GPUs | n = 8192, 1 GPU |  4 GPUs |
|----------------------------------------------|---------:|-------:|---------------:|-------:|----------------:|--------:|
| `openmpi/` MPI, one rank per GPU             |  0.087 s | 0.024 s |       0.037 s | 0.052 s |         1.159 s | 0.963 s |
| `chapel/` `*_gpus.chpl`                      |  0.088 s | 0.033 s |       0.040 s | 0.021 s |         5.333 s | 1.392 s |
| `julia/` Dagger Datadeps                     |  0.099 s | 0.062 s |       0.053 s | 0.090 s |         1.245 s | 0.715 s |
| `julia/` Dagger DArray (C = A B with cuBLAS) |  0.097 s | 0.048 s |       0.059 s | 0.071 s |         1.013 s | 1.124 s |

- π, which computes much and moves almost no data, gets 1.6 (Datadeps) to 3.6 (MPI) times as fast on four GPUs.
- The matrix product with n = 2048 moves 48 MB between host and GPUs, and every GPU needs all of B: only Chapel gains (1.9×); MPI and Dagger are slower on four GPUs than on one.
- With n = 8192 (64 times the work, 16 times the data) Chapel gets 3.8 times as fast and Datadeps 1.7 times. MPI stays at 1.2×: rank 0 sends about 1 GB to the other ranks, which this Open MPI copies twice through shared memory (it has no single-copy mechanism such as CMA; host buffers instead of CUDA-aware transfers were slower). DArray moves its blocks between the GPUs through the host and stays as fast as on one GPU.
- Chapel scales best, but its kernel is slow at this size: on four GPUs it needs 1.39 s, the CUDA kernel 1.03 s on one GPU, and cuBLAS 0.24 s.

## Environment Modules

The HPC software (Open MPI, Kokkos, HPX, Chapel, LLVM, the CUDA of the HPC stack) and the NVIDIA HPC SDK are available as Environment Modules:

```sh
module avail          # installed HPC packages
module load kokkos    # puts Kokkos (and the CUDA it needs) into the environment
module list           # what is loaded
module purge          # unload everything
```

Every `run.sh` starts with `module purge` and loads the modules it needs in its own shell, so modules loaded in your shell do not get in its way and running it leaves your shell unchanged.
`module` works in bash and, with the `zsh` package, in zsh.
Modules you load yourself stay loaded until `module purge` or logout. Some get in each other's way (another CUDA; `nvhpc` with its own `nvcc` and `mpirun`), so the commands in every folder's README start with `module purge`.
Three things to know:

- `module load cuda` (also loaded by `openmpi`, `kokkos`, `hpx` and `chapel`) puts the CUDA 13.0 of the HPC stack in front of the CUDA 13.4 of the `cuda` package in `/usr/local/cuda`; `nvcc --version` shows which one is active, and `module purge` returns to 13.4. Both work for the RTX 3070 and 3080.
- `module load chapel` also loads `llvm`, so `clang` and `flang` are then LLVM 22 with OpenMP offloading.
- `module load nvhpc` puts the NVIDIA HPC SDK first in the PATH, with its own `nvcc` and `mpirun`; unload it (`module unload nvhpc`) before using the `cuda` or `openmpi` examples in the same shell.

Files that you change here are kept when the instance is saved with `gpuaas.py snapshot` and started again.
