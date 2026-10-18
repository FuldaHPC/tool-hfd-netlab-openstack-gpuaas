# Julia with Dagger, CUDA.jl and KernelAbstractions

π with the midpoint rule (10⁹ steps, double precision) and the matrix product C = A B (2048 × 2048, `Float32`), each sequential, on the CPU threads, and on the GPU in the common Julia ways, with and without Dagger.
Every program prints its result, its time in seconds and its rate in GFLOP/s.

CPU:

- `pi_seq.jl`, `matmul_seq.jl`: plain loops on one thread, as reference.
- `pi_cpu.jl`, `matmul_cpu.jl`: the same loops in Dagger tasks on all threads, four tasks per thread; the matrix product gives each task a block of columns of C (Julia stores matrices by columns, so this matches the rows of the C versions).

GPU without Dagger:

| Variant  | `pi_gpu_*.jl`                                                | `matmul_gpu_*.jl`                               |
|----------|--------------------------------------------------------------|-------------------------------------------------|
| `arrays` | CUDA.jl array programming: broadcast into a `CuArray`, `sum` | `A * B` on `CuArray`s, which calls cuBLAS       |
| `kernel` | hand-written CUDA.jl kernel (`@cuda`)                        | hand-written CUDA.jl kernel (`@cuda`)           |
| `ka`     | KernelAbstractions.jl kernel on `CUDABackend()`              | KernelAbstractions.jl kernel on `CUDABackend()` |

GPU with Dagger, the four approaches of Dagger's documentation; `Dagger.Kernel` and the manual approach run in the scope of the first CUDA GPU (`Dagger.scope(cuda_gpu=1)`), Datadeps and DArrays in the scope of all GPUs of the instance (`cuda_gpus=:`), where Datadeps starts one task per GPU and the DArray blocks spread over the GPUs:

| Variant           | What it does                                                                                                              |
|-------------------|---------------------------------------------------------------------------------------------------------------------------|
| `dagger_kernel`   | `Dagger.Kernel` launches the KernelAbstractions kernel as a task on arrays from `Dagger.@mutable`                         |
| `dagger_manual`   | a task asks `Dagger.gpu_kernel_backend()` for the backend of its processor and launches the kernel itself                 |
| `dagger_datadeps` | inside `spawn_datadeps`, a task marked `Out(...)`/`In(...)` runs the kernel; Dagger moves host arrays to the GPU and back |
| `dagger_darray`   | `DArray`s in GPU memory with ordinary array operations (`map`, `sum`; `A * B` multiplies the blocks with cuBLAS)          |

All π kernels sum one chunk of the steps per GPU thread with the same `pi_chunk_sum`, all hand-written matrix kernels compute one element of C per thread with the same `matmul_entry` (both in `common.jl`), and the hand-written kernels use the same workgroups (256 threads for π, 16 × 16 for C), except `Dagger.Kernel`.
`Dagger.Kernel` has no option for the workgroup size and uses KernelAbstractions' default, with which the matrix product took about 1.6 times as long.
`matmul_gpu_arrays.jl` and `matmul_gpu_dagger_darray.jl` use cuBLAS, a tuned library, instead of the simple algorithm; compare them with each other, not with the kernels.
The GPU times include allocating GPU memory and copying the data there and back.
The versions without Dagger and the Datadeps version copy C into an array that was written before (a new array would add about 7 ms of page faults); with `Dagger.Kernel`, the manual approach and DArrays, Dagger returns C as a new array, which belongs to these approaches.
On several GPUs, every Datadeps task needs its own arrays for its columns of B and C (Datadeps cannot move views to a GPU: "Aliasing info … has different number of spans"); they are written before the timer as well, because the page faults of new ones inside the timed run made several GPUs slower than one.
On an instance with several GPUs, `Dagger.@mutable` needs the same scope as the kernel task (`scope=gpu`), otherwise it may create the array on another GPU, which the kernel cannot read (P2P between the cloud's GPUs is not possible; the result was wrong).
Datadeps and DArrays also move more data than the others: Datadeps copies the `Out` array to the GPU as well (C, or the chunk sums of π, which are then added on the CPU), and the DArray π copies an array of chunk indices to the GPU.

Run one program, or all with `./run.sh`:

```sh
module purge          # first unload other modules (another CUDA, MPI or compiler)
module load cuda
julia pi_seq.jl
julia -t auto pi_cpu.jl    # likewise matmul_cpu.jl
julia pi_gpu_kernel.jl     # likewise pi_gpu_arrays.jl, pi_gpu_ka.jl, pi_gpu_dagger_*.jl, matmul_seq.jl, matmul_gpu_*.jl
julia -t auto matmul_gpu_dagger_datadeps.jl   # likewise matmul_gpu_dagger_darray.jl
```

Each program first runs with the full problem until two runs in a row compile no code (`warm_up` in `common.jl`, usually 2–3 runs), then once more with time measurement, so that the time includes neither compiling nor garbage from the warm-up (a `GC.gc()` collects it before the timed run).
Julia compiles code just before it first runs it, and which code a run needs can depend on timing: a small problem missed some, and even a third full-size run of `pi_gpu_dagger_kernel.jl` sometimes compiled for 0.4 s; a garbage collection once added 0.1 s to `matmul_gpu_ka.jl`.
Loading the packages and compiling make every start take a while (about 5 s to load CUDA.jl and 15 s to compile Dagger and the GPU code), and the warm-up runs add two or three times the measured time.
In the same Julia session, later runs take a fraction of a second, so for experiments start `julia -t auto` once and use `include("pi_gpu_kernel.jl")` as often as you like.
`-t auto` uses all CPU cores; without it Julia runs on one thread.
The Datadeps and DArray matrix products run with `-t auto`: with one thread, Dagger moves the matrices for several GPUs one after the other, and DArray is faster with threads on one GPU too; for π, which moves almost no data, threads did not help (Datadeps on four GPUs: 0.060 s with one thread, 0.067 s with four).
`common.jl` holds the sizes, the checks and the shared per-thread functions.

Julia comes from juliaup (`juliaup update` fetches newer releases), and Dagger, CUDA.jl and KernelAbstractions are in the default environment of user `ubuntu`.
CUDA.jl uses the CUDA 13.0 of the HPC stack (`module load cuda`) instead of downloading its own.
GPU support of Dagger is part of Dagger itself (`using Dagger, CUDA`); the old DaggerGPU package is no longer needed.
