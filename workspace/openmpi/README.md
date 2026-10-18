# Open MPI (CUDA-aware)

Six programs, each printing its result, its time in seconds and its rate; by default the MPI programs for the CPU use one rank per CPU core, those for the GPU one rank per GPU:

- `pi_seq.c`: π with the midpoint rule (10⁹ steps, double precision) in a plain loop on one CPU core, without MPI, as reference.
- `pi_cpu.c`: the same with MPI; every rank sums a contiguous block of the steps, `MPI_Reduce` adds the parts.
- `pi_gpu.cu`: every rank sums its block in a CUDA kernel on one of the instance's GPUs (round robin by the rank on the node, so with one rank per GPU each rank has a GPU of its own), and `MPI_Allreduce` adds the partial sums directly in GPU memory.
- `matmul_seq.c`: the matrix product C = A B (2048 × 2048, single precision) in plain loops on one CPU core, without MPI.
- `matmul_cpu.c`: the same split by rows: `MPI_Scatter` sends each rank its rows of A, `MPI_Bcast` sends B, `MPI_Gather` collects C; the time includes the communication.
- `matmul_gpu.cu`: the same with a CUDA kernel per rank: `MPI_Scatterv` puts the rows of A into GPU buffers, rank 0 copies B to its GPU and `MPI_Bcast` sends it from GPU to GPU, and `MPI_Gatherv` collects C from the GPU buffers; the time includes this and allocating the GPU buffers. `Scatterv` and `Gatherv` split the rows as evenly as possible, so any number of ranks works (e.g. three GPUs).

Build and run (`make run NP=2 NP_GPU=1` sets the number of ranks for the CPU and the GPU programs; for `matmul_cpu` it must divide 2048):

```sh
module purge          # first unload other modules (another CUDA, MPI or compiler)
module load openmpi
make run
```

`module load openmpi` also loads the CUDA of the HPC stack; `nvcc -ccbin mpicxx` compiles the kernels and lets the MPI compiler wrapper build the host code.
The GPU programs need a CUDA-aware MPI and stop at the start if `MPIX_Query_cuda_support()` says otherwise.
`--oversubscribe` lets `make run NP=8` run on fewer cores (the ranks then share them).
`common.h` holds the sizes, the timer and the checks.
