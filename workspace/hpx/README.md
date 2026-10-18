# HPX (with CUDA support)

Six programs, each printing its result, its time in seconds and its rate:

- `pi_seq.cpp`: π with the midpoint rule (10⁹ steps, double precision) in HPX's `for_loop` with the sequential policy `hpx::execution::seq` and a reduction, as reference.
- `pi_cpu.cpp`: the same in asynchronous HPX tasks, four per worker thread; `hpx::when_all` combines their futures.
- `pi_gpu.cu`: the same in a CUDA kernel; the memset, the launch and the copy back go through an `hpx::cuda::experimental::cuda_executor`, and the copy returns an `hpx::future`.
- `matmul_seq.cpp`: the matrix product C = A B (2048 × 2048, single precision) with `for_loop(hpx::execution::seq, ...)` over the rows of C (one CPU core).
- `matmul_cpu.cpp`: the same with the parallel policy `hpx::execution::par`.
- `matmul_gpu.cu`: the same in a CUDA kernel driven by the executor; copies and kernel run in one CUDA stream, and the time includes the copies.

Build and run:

```sh
module purge          # first unload other modules (another CUDA, MPI or compiler)
module load hpx
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/pi_seq   # likewise pi_cpu, pi_gpu, matmul_seq, matmul_cpu, matmul_gpu
```

`hpx_main.hpp` with the target `HPX::wrap_main` starts the HPX runtime around `main()`; `--hpx:threads=N` sets the number of worker threads.
`run.sh` does all of the above; `common.h` holds the sizes, the timer and the checks.
