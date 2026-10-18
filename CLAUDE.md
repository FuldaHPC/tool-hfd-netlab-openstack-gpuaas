# CLAUDE.md

Guidance for Claude Code in this repository. User documentation: `README.md` (short) and `docs/guide.md` (details). Personal settings of a maintainer go into the gitignored `CLAUDE.local.md`.

## What this is

`gpuaas.py` (one argparse file, PEP 723 script run through uv) starts, lists, snapshots, shelves, renames and deletes GPU instances (RTX 3070/3080) on the NetLab OpenStack of HS Fulda and installs software through cloud-init fragments in `cloud-init/packages/`. `workspace/` holds the examples copied to the instances; `tools/` the offline checks and the Spack binary cache tools.

## Working rules

- Run `./tools/check.py` after every change to gpuaas.py, a package file or the workspace (user data of all package combinations, README Usage block = `COLUMNS=100 ./gpuaas.py help`, guide matrix = `help start`, identical copies, workspace scripts). After changing a script's dependency header: `uv lock --script FILE`.
- Cloud calls create real instances on a shared, scarce GPU pool: test on `g1-3070.small` (`.medium`/`.xlarge` only when needed), delete test instances right after, and only touch your own test instances.
- Test the wizard in-process with stubbed `questionary` prompts that answer "No" to "Start this instance?"/"Delete …?"; never send blind Enters (the start confirmation defaults to Yes).
- Test every command with a saved token and with a password login, with `XDG_CACHE_HOME` pointing at a scratch directory; read user-visible output unfiltered.
- The API is only reachable from the HS Fulda network or the VPN profile "05)_AI-NetLab-Pro".

## Pitfalls in gpuaas.py

- A saved-token connection is a hand-built keystoneauth `v3.Token` session without `conn.config.auth`: never read settings from there (username via `get_access().username`). `ask_password` reads `OS_PASSWORD` itself (openstacksdk moves `OS_*` into another cloud).
- `TOKEN_MINUTES` = `WAIT_MINUTES` + 5, so a token cannot expire during a wait; `start` makes no API calls after ACTIVE.
- `clouds.yaml` needs `verify: false`: the service endpoints use certificates of the cloud's own CA; urllib3's warning is silenced.
- The wizard asks the software before the image (images are rated by OS compatibility); an `--image` is looked up first and its packages show as "in the image". Packages with a `job` (gpu-burn) are never preselected from an image.
- Nova accepts at most 64 KiB of base64 user data: the workspace goes in as tar.xz and the whole user data is gzipped when needed; all packages use 63.1 KB, so larger workspace additions need trimming. `workspace/README.md` counts too.
- `snapshot --share` must remove `authorized_keys` (cloud-init only appends keys) before an image is shared.

## Packages and checks (cloud-init)

- Package files: cloud-config plus a `gpuaas:` block (`title`, `group`, `order` > its `requires`, `os` matrix, `disk`, `long`, `build`, `check`, `job`); fragments merge (lists appended, dicts merged, conflicting scalars are an error).
- `gpuaas-finish` runs builds and jobs each in a stand-alone `set -e` subshell (inside `( … ) || …` set -e would be ignored), checks in parallel (`xargs -P`, one per core and 4 GB), then writes `ready` or `failed: <pkg>` to `/var/lib/gpuaas/phase`.
- A package without a `check` runs its `workspace/<name>/run.sh` as ubuntu in a fresh copy of the unchanged workspace via `runuser -u` (not `runuser -l`, whose PAM session hides the output from the journal).
- Driver: headless packages chosen from `ubuntu-drivers list --recommended` (`ubuntu-drivers install` pulls in X and Mesa). CUDA: individual 13.4 packages, not `cuda-toolkit` (Nsight pulls in Mesa).
- `gcc-offload` wrapper flags exist for a reason: `-foffload-options=-fcf-protection=none` (Ubuntu's CET default breaks nvptx), global `-fno-stack-protector` (offload code inherits it), `-Wa,--no-verify` (CUDA 13's ptxas rejects GCC 13's sm_50).

## Workspace

- Every folder computes the same two problems (π midpoint rule 10⁹ steps double, |error| < 1e-9; C = A B n = 2048 float, A(i,j) = (i+2j)%8, B(i,j) = (3i+j)%8, checked by C x == A (B x) with x = 1..n) sequentially, on the CPU cores and on the GPU, prints `<CPU|GPU> [sequential] (<model>): ... in X s (R GFLOP/s)` and exits 1 if wrong or not on the GPU.
- Fairness: one warm-up (Julia: until two full runs compile nothing, then `GC.gc()`), GPU times include allocation and transfers, result buffers are touched before the timer, C/C++/Fortran use `-march=x86-64-v3`.
- `common.h` (6 copies), `common.f90` (3) and the shared Fortran sources must stay identical (check.py). READMEs and `run.sh` start with `module purge`.
- Multi-GPU (no P2P): `openmpi/Makefile` starts the GPU programs with one rank per GPU (`NP_GPU`, from `nvidia-smi -L`), ranks pick `local_rank % devices`, `matmul_gpu.cu` splits uneven rows with `MPI_Scatterv`/`Gatherv` (3 GPUs); Chapel `*_gpus.chpl` use `here.gpus`, Dagger Datadeps/DArray use `cuda_gpus=:`; `Dagger.@mutable` needs the kernel's `scope`, otherwise arrays land on the other GPU. Results on 4x RTX 3070 (`docs/results/4x-rtx3070-scaling-2026-10-18/`): π scales (MPI 3.6×, Chapel 2.7×, DArray 2.0×, Datadeps 1.6×); the matrix product only with larger n (n³ work vs n² data): at n = 8192 Chapel 3.8×, Datadeps 1.7×, MPI 1.2× (rank 0 sends ~1 GB; Spack's Open MPI has no CMA/XPMEM, so shared memory copies twice; host buffers or `--mca btl self,sm` were not faster), DArray 0.9× (blocks move through the host; other block layouts were slower). Datadeps cannot move views, so its per-GPU arrays must be written before the timer (new ones inside it made several GPUs slower than one); the multi-GPU Dagger matrix programs run with `julia -t auto` (one thread serializes the GPUs' copies; π gains nothing); DArray π uses one block per GPU (four fixed blocks unbalanced three GPUs).
- `run_all.sh -n 3` writes medians to `results/`; complete runs are kept in `docs/results/`.

## HPC stack (Spack, Ubuntu 24.04)

- Spack 1.2.2, recipes pinned to spack-packages `cd2e1efc`; chapel 2.10.0, kokkos 5.2.2, hpx 1.11.0, openmpi 5.0.11, llvm 22.1.8, cuda 13.0.2. Chapel 2.10 caps LLVM at 22 and CUDA at 13.0, and CUDA ≤ 13.0 conflicts with Ubuntu 26.04's glibc, hence 24.04. Upgrades follow Chapel (checklist in `tools/spack-cache/README.md`).
- One LLVM for Chapel and llvm-offload (variants fixed by a `require`); environment `target=x86_64_v3`, `unify`, `reuse`.
- Overlay repo (in `spack.yaml`): llvm adds the NVPTX OpenMP device runtime and flang-rt, and writes the rpath to `libomp` into `bin/x86_64-unknown-linux-gnu.cfg` (not `clang.cfg`, which the builtin recipe rewrites later; not `LD_LIBRARY_PATH`, whose `libgomp.so.1` alias breaks GCC programs); openmpi 5.0.11; hpx with header-only Boost.
- Binary cache: public Swift container `gpuaas-spack-cache`, unsigned, no TLS verification. Instances only use binaries with matching hashes, so rebuild it (`tools/spack-cache/update.sh`, needs `GPUAAS_BUILD_HOST`) after changing the pin, a spec, a recipe, the externals or the apt list; the cache build concretizes with `reuse: false`.

## Cloud facts

- Flavors `g1-3070.{small,medium,xlarge}`, `g1-3080.{small,medium}`, multi-GPU `g1-2x3070.{small,medium,xlarge}`, `g1-3x3070.xlarge`, `g1-4x3070.xlarge`, `g1-2x3080.{small,medium,xlarge}`, `g1-3x3080.xlarge`, `g1-4x3080.xlarge` (small 1 vCPU/2 GB/20 GB, medium 2/4/40, xlarge 4/16/160, also with several GPUs); RTX 2080 Ti flavors exist but are unsupported (`SUPPORTED_GPUS`). The GPUs of a multi-GPU host share PCIe through the CPU's host bridge without peer-to-peer access (`nvidia-smi topo -p2p r`: chipset not supported, on the 4x3070 and the 2x3080 hosts).
- "No valid host was found" means no free GPU of that flavor (users cannot query free GPUs). Instances are shelved after one week. Snapshots keep their instance's disk size (xlarge images need xlarge flavors).
- Each account has the network `<username>-net`; default image `ubuntu-26.04-resolute-server-cloud-image-amd64v3`, HPC on `ubuntu-24.04-noble-server-cloud-image-amd64`.
