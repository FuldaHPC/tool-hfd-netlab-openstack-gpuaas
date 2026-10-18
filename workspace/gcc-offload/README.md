# GCC and gfortran with OpenMP and OpenACC offloading

Six C programs, each printing its result, its time in seconds and its rate, and eight Fortran programs in `fortran/` (the same six plus OpenACC, below):

- `pi_seq.c`: π with the midpoint rule (10⁹ steps, double precision) in a plain loop on one CPU core, as reference.
- `pi_cpu.c`: π with the midpoint rule (10⁹ steps, double precision) in an OpenMP `parallel for` reduction on the CPU threads.
- `pi_gpu.c`: the same in a `target teams distribute parallel for simd` region on the GPU.
- `matmul_seq.c`: the matrix product C = A B (2048 × 2048, single precision) in plain loops on one CPU core.
- `matmul_cpu.c`: the same with the rows of C distributed over the CPU threads.
- `matmul_gpu.c`: the same on the GPU: the rows over teams and threads, each row over the 32 lanes of a warp (`simd`); `map(to:)`/`map(from:)` copy the matrices, and the time includes it.

Build and run:

```sh
module purge            # if you loaded HPC modules (their CUDA or compilers would be used)
make run                # the C programs
make -C fortran run     # the Fortran programs; ./run.sh runs both
```

The GPU programs differ from `llvm-offload/`: GCC maps OpenMP threads to whole warps, so only `simd` spreads the work over the 32 lanes of a warp; without it they ran about 30 times slower.
Clang is faster without `simd` and with `collapse(2)` for the matrix product.

`gcc-offload` is a wrapper for `gcc -fopenmp -foffload=nvptx-none -foffload-options=-fcf-protection=none -foffload-options=-Wa,--no-verify -fno-stack-protector`.
Ubuntu's GCC enables `-fcf-protection` and the stack protector by default, which the NVIDIA offload compiler cannot handle.
The wrapper turns `-fcf-protection` off for the GPU code only, and the stack protector off for the whole program, because the GPU code inherits the host setting.
`--no-verify` stops the NVPTX assembler from checking the code with a CUDA toolkit's `ptxas`, whose CUDA 13 version rejects the default architecture of GCC 13 (Ubuntu 24.04); the driver compiles the code for the GPU at run time (compiling for the RTX 3070's generation with `-foffload-options=-misa=sm_80` made no difference).
`OMP_TARGET_OFFLOAD=MANDATORY` makes a program fail instead of silently running the target region on the CPU.
`common.h` holds the sizes, the timer and the checks.

## Fortran

`fortran/` computes the same with gfortran, GCC's Fortran compiler, which uses the same offload compiler:

- `pi_seq.f90`, `matmul_seq.f90`: plain loops on one CPU core, as reference.
- `pi_cpu.f90`, `matmul_cpu.f90`: OpenMP `parallel do` on the CPU threads.
- `pi_gpu.f90`, `matmul_gpu.f90`: OpenMP `target teams distribute parallel do`, with `simd` as in C.
- `pi_acc.f90`, `matmul_acc.f90`: OpenACC `parallel loop` (gangs and vector lanes) on the GPU.

Fortran stores matrices by columns, so the Fortran programs distribute the columns of C where the C programs distribute rows; both let neighbouring GPU threads take neighbouring elements in memory.
`gfortran-offload` is a wrapper for gfortran with the same options as `gcc-offload`; with `-fopenacc` it compiles OpenACC instead of OpenMP, because GCC's offload compiler takes only one of them per program.
OpenACC needs no environment variable: the programs fail unless `acc_get_device_type()` reports an NVIDIA GPU.
`common.f90` holds the sizes, the timer and the checks; `../nvhpc/` has the same programs for NVIDIA's nvfortran (with the OpenMP GPU programs of Flang in `../llvm-offload/fortran/`, without `simd`), plus `do concurrent` and CUDA Fortran.
