# Benchmark results

All results were measured on 2026-10-18 on one instance with four RTX 3070 (`g1-4x3070.xlarge`: 4 vCPUs of an Intel Xeon Cascade Lake, 16 GB of RAM; the GPUs share PCIe through the host without peer-to-peer access), started from an HPC image with all packages.

| Folder | What |
| --- | --- |
| [4x-rtx3070-2026-10-18/](4x-rtx3070-2026-10-18/) | `~/workspace/run_all.sh -n 3` (see [workspace/README.md](../../workspace/README.md)): all 86 programs, the ones for one GPU on the first GPU, the ones for several GPUs on all four; the source of the tables in workspace/README.md |
| [4x-rtx3070-scaling-2026-10-18/](4x-rtx3070-scaling-2026-10-18/) | the programs for several GPUs with one, two, three and four GPUs, for π and for the matrix product with n = 2048 and 8192 (its README explains it) |

All results were correct in every run.

Files of a `run_all.sh` run:

- `results.pdf`: one page for π and one for the matrix product, GFLOP/s on a logarithmic scale.
- `results.csv`: the run with the median time of each program, with the number of runs and the fastest and slowest time (`folder,problem,kind,model,seconds,gflops,status,runs,min_seconds,max_seconds`).
- `runs.csv`: every run.
- `<folder>.log`: the output of each workspace folder's `run.sh`, including the compiler output.
- `pi.tsv`, `matmul.tsv`: the plot data.
