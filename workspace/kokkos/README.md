# Kokkos (CUDA and OpenMP backends)

Six programs, each printing its result, its time in seconds and its rate; they differ mainly in the execution space:

- `pi_seq.cpp`: π with the midpoint rule (10⁹ steps, double precision) with `Kokkos::parallel_reduce` in the `Serial` execution space (one CPU core), as reference.
- `pi_cpu.cpp`: the same in the default host execution space, OpenMP in this build (all CPU cores).
- `pi_gpu.cpp`: the same in the default execution space, CUDA in this build; the program fails otherwise.
- `matmul_seq.cpp`: the matrix product C = A B (2048 × 2048, single precision) with `Kokkos::parallel_for` over the rows of C in `Serial`, the matrices as `Kokkos::View`s in host memory.
- `matmul_cpu.cpp`: the same in the OpenMP execution space.
- `matmul_gpu.cpp`: the same with an `MDRangePolicy` over all elements of C on the GPU; the matrices are Views in GPU memory and in host memory, and `Kokkos::deep_copy` moves them between the two (included in the time).

Build and run:

```sh
module purge          # first unload other modules (another CUDA, MPI or compiler)
module load kokkos
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/pi_seq   # likewise pi_cpu, pi_gpu, matmul_seq, matmul_cpu, matmul_gpu
```

`run.sh` does all of the above; `OMP_NUM_THREADS` sets the number of OpenMP threads.
Kokkos was built with CMake's CUDA language support (no `nvcc_wrapper`): a project enables the `CUDA` language and compiles its Kokkos sources as CUDA, see `CMakeLists.txt`.
`KOKKOS_LAMBDA` then makes every loop body a `__host__ __device__` lambda, which nvcc passes to the host compiler behind a wrapper that is not inlined; π, with one call per step, therefore takes twice as long on the CPU as with a plain host lambda `[=]` (2.8 s instead of 1.4 s sequentially), while the matrix product (one call per row) is not affected.
The GPU architecture is the one of the instance on which Kokkos was built; `common.h` holds the sizes, the timer and the checks.
