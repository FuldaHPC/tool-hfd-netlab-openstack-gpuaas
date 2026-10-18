# Chapel (GPU support)

Eight programs, each printing its result, its time in seconds and its rate:

- `pi_seq.chpl`: π with the midpoint rule (10⁹ steps, double precision) in a serial `for` loop on one CPU core, as reference.
- `pi_cpu.chpl`: the same in a `forall` loop with a `+ reduce` intent on the CPU cores.
- `pi_gpu.chpl`: the same inside `on here.gpus[0]`, where the `forall` loop becomes a GPU kernel; each GPU thread sums one chunk of the steps, and `+ reduce` adds the chunks on the GPU.
- `matmul_seq.chpl`: the matrix product C = A B (2048 × 2048, `real(32)`) in serial `for` loops.
- `matmul_cpu.chpl`: the same with a `forall` loop over the rows of C.
- `matmul_gpu.chpl`: the same on the GPU: arrays declared inside `on here.gpus[0]` live in GPU memory, assigning host arrays copies them (included in the time), and a `forall` over all elements of C runs with one GPU thread per element.
- `pi_gpus.chpl`, `matmul_gpus.chpl`: the same on all GPUs of the instance: a `coforall` over `here.gpus` starts one task per GPU, and each GPU computes its share of the chunks or its block of rows of C (on one GPU the matrix product's kernel takes about 8 % longer than `matmul_gpu.chpl`'s, because of the row offset).

Build and run:

```sh
module purge          # first unload other modules (another CUDA, MPI or compiler)
module load chapel
make run
```

`@assertOnGpu` makes compilation fail if a loop would not become a GPU kernel.
A `forall` over the 2D domain of C was 60 times slower on the GPU than the loop over all n·n elements used here (Chapel 2.10).
`Common.chpl` holds the sizes and the checks; compiling takes about half a minute per program.
Chapel was built with `CHPL_LOCALE_MODEL=gpu` and CUDA for the GPU of the instance; programs without `on here.gpus[...]` run on the CPU as usual.
