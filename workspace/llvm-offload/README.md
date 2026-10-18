# LLVM Clang and Flang with OpenMP offloading

The same six programs as in `gcc-offload/`, compiled with Clang 22 from the Spack HPC stack; the sequential and CPU programs are identical, the GPU programs use the mapping Clang is fastest with:

- `pi_seq.c`: π with the midpoint rule (10⁹ steps, double precision) in a plain loop on one CPU core, as reference.
- `pi_cpu.c`: π with the midpoint rule (10⁹ steps, double precision) in an OpenMP `parallel for` reduction on the CPU threads.
- `pi_gpu.c`: the same in a `target teams distribute parallel for` region on the GPU.
- `matmul_seq.c`: the matrix product C = A B (2048 × 2048, single precision) in plain loops on one CPU core.
- `matmul_cpu.c`: the same with the rows of C distributed over the CPU threads.
- `matmul_gpu.c`: the same on the GPU with one iteration per element of C (`collapse(2)`); the time includes copying the matrices.

Build and run:

```sh
module purge          # first unload other modules (another CUDA, MPI or compiler)
module load llvm cuda
make run                # the C programs
make -C fortran run     # the Fortran programs; ./run.sh runs both
```

`clang -fopenmp --offload-arch=native` compiles the target regions for the GPU of this instance; `--offload-arch=sm_86` names the architecture explicitly (RTX 3070 and 3080).
Clang needs `ptxas` from the CUDA toolkit for the GPU code, hence `module load cuda`.
Clang gives the programs an rpath to LLVM's OpenMP runtime (`libomp.so`, `libomptarget.so`), so they also run without any module loaded.
`OMP_TARGET_OFFLOAD=MANDATORY` makes a program fail instead of silently running the target region on the CPU.
The same LLVM is also the one Chapel uses, so only one LLVM is installed.
`common.h` holds the sizes, the timer and the checks.

## Fortran

`fortran/` computes the same with Flang, LLVM's Fortran compiler: `pi_seq.f90`, `pi_cpu.f90` (OpenMP `parallel do`), `pi_gpu.f90` (OpenMP `target teams distribute parallel do`), and the same three for the matrix product (`collapse(2)` on the GPU).
The sources are the same as in `../nvhpc/`; Flang cannot offload OpenACC yet, so there is no `*_acc` program.
`flang -fopenmp --offload-arch=native` compiles them like Clang, with the same rpath to LLVM's OpenMP runtime, and `common.f90` holds the sizes, the timer and the checks.
Flang's GPU matrix product takes 0.29 s on an RTX 3070, four times as long as Clang's (0.07 s) and eight times as long as nvfortran's from the same source (0.037 s): the kernel itself is slower (`LIBOMPTARGET_PROFILE=prof.json` shows the times), with the same launch configuration (`LIBOMPTARGET_INFO=-1`); `contiguous` or explicit-shape arrays did not change it.
Flang's OpenMP offloading is still young; π runs as fast as with the other compilers.
