# CUDA

Four programs, each printing its result, its time in seconds and its rate:

- `pi_seq.cpp`: π with the midpoint rule (10⁹ steps, double precision) in a plain C++ loop on one CPU core (sequential).
- `pi_gpu.cu`: the same in a CUDA kernel; each thread sums every (grid size)-th step, each block adds up its threads in shared memory and adds that to the result with one `atomicAdd`.
- `matmul_seq.cpp`: the matrix product C = A B (2048 × 2048, single precision) in C++ loops on one CPU core.
- `matmul_gpu.cu`: the same in a CUDA kernel with one thread per element of C; the time includes copying A and B to the GPU and C back.

Build and run:

```sh
module purge            # if you loaded HPC modules (their CUDA or compilers would be used)
make run
```

CUDA programs the GPU only, so there is no parallel CPU version here (see the other folders).
`make` alone only builds; `nvcc -arch=native` compiles for the GPU of this instance.
`common.h` holds the sizes, the timer and the checks (π within 10⁻⁹; for C, C x = A (B x) exactly).
The RTX 3070 and 3080 compute double precision at 1/64 of their single-precision speed, which limits `pi_gpu`.
