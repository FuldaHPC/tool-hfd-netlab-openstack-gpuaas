# NVIDIA HPC SDK: Fortran on the GPU

π with the midpoint rule (10⁹ steps, double precision) and the matrix product C = A B (2048 × 2048, single precision) in Fortran with nvfortran, in every GPU model of the NVIDIA HPC SDK.
Every program prints its result, its time in seconds and its rate in GFLOP/s.

| Programs                             | Model                                                                   |
|--------------------------------------|-------------------------------------------------------------------------|
| `pi_seq.f90`, `matmul_seq.f90`       | plain loops on one CPU core, as reference                               |
| `pi_cpu.f90`, `matmul_cpu.f90`       | OpenMP `parallel do` on the CPU threads (`-mp`)                         |
| `pi_gpu.f90`, `matmul_gpu.f90`       | OpenMP `target teams distribute parallel do` on the GPU (`-mp=gpu`)     |
| `pi_acc.f90`, `matmul_acc.f90`       | OpenACC `parallel loop` on the GPU (`-acc=gpu`)                         |
| `pi_stdpar.f90`, `matmul_stdpar.f90` | standard Fortran `do concurrent`, run on the GPU by `-stdpar=gpu`       |
| `pi_cuf.cuf`, `matmul_cuf.cuf`       | CUDA Fortran kernels (`-cuda`), the counterpart of `../cuda/`           |

Build and run:

```sh
module purge          # first unload other modules (another CUDA, MPI or compiler)
module load nvhpc
make run
```

- The sequential, CPU and OpenACC sources are the same as in `../gcc-offload/fortran/`; the OpenMP GPU programs leave out the `simd` that gfortran needs, and the matrix product uses `collapse(2)` as with Clang.
- `do concurrent` has no directives at all; with `-stdpar=gpu` nvfortran puts allocatable arrays into CUDA managed memory, which the driver moves to the GPU when a kernel touches them and back when the CPU does; `matmul_stdpar.f90` sums C on the CPU before it stops the time, so the time includes moving C back, like the copies of the other GPU programs.
- CUDA Fortran writes the kernels by hand (`attributes(global)`, launched with `<<<blocks, threads>>>`) and copies with assignments between host arrays and arrays with the `device` attribute.
  Its matrix product (0.061 s) is slower than CUDA C's (0.026 s) because of these copies, not the kernel (13 ms in both): Nsight Systems showed a pause of 16 ms between the two uploads and 22 ms for the download, instead of 3 ms each in CUDA C.
- `-gpu=ccnative` compiles for the GPU of this instance (cc86 on an RTX 3070 or 3080), and `-tp=x86-64-v3` for the CPU as in the other folders.
- nvfortran vectorizes the π sum also in the sequential and CPU programs and so adds in another order (gfortran and Flang keep the order of the source unless told otherwise), which makes its CPU versions of π about twice as fast.
- `OMP_TARGET_OFFLOAD=MANDATORY` makes the OpenMP GPU programs fail instead of silently running on the CPU; the OpenACC programs check `acc_get_device_type()`.

The SDK (version 26.9) is in `/opt/nvidia/hpc_sdk` with its own CUDA 13.3 and MPI; `module load nvhpc` puts `nvfortran`, `nvc`, `nvc++`, its `nvcc` and `mpirun` first in the PATH, and `module unload nvhpc` removes them again.
`module avail nvhpc` also lists variants without MPI (`nvhpc-nompi`) and with HPC-X.
`common.f90` holds the sizes, the timer and the checks.
