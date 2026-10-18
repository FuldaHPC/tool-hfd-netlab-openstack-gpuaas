# Guide

Details of `gpuaas.py`; the [README](../README.md) has the short version and the full `./gpuaas.py help`.

## Setup

You need macOS or Linux (on Windows, WSL should work but is untested) with `ssh` and `ssh-keygen`.

1. Network: the Hochschule Fulda network, or the VPN profile "05)_AI-NetLab-Pro" from outside.
2. Install [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh` (on macOS also `brew install uv`).
   On the first run uv downloads Python and the dependencies (openstacksdk, questionary, PyYAML; versions pinned in `gpuaas.py.lock`) into its cache, with no virtualenv and no admin rights.
3. Get the code: `git clone https://github.com/FuldaHPC/tool-hfd-netlab-openstack-gpuaas.git && cd tool-hfd-netlab-openstack-gpuaas`
4. Run any command, e.g. `./gpuaas.py list`: the first run asks for your OpenStack username and password, lets you choose your project and writes `clouds.yaml` next to `gpuaas.py` (gitignored, without the password).
   The password is not stored: each login saves only its token (in `~/.cache/gpuaas/`, readable only by you), which the cloud accepts for one hour, so the password is asked for about once an hour.
   `./gpuaas.py logout` deletes the token, e.g. on a shared computer; for scripts, set `OS_PASSWORD` in the environment, e.g. from a password manager.
   `./gpuaas.py setup` creates the file again, e.g. for another user or project.
   By hand, copy `clouds.yaml.example` to `clouds.yaml` and fill in username and project ID; a `clouds.yaml` downloaded from Horizon works as well if you remove the password and add `verify: false` as in the template (the cloud's service endpoints use certificates of its own CA).

## Commands

`./gpuaas.py help <command>` shows the options of a command, for `start` including the software packages.
`./gpuaas.py` runs through uv (shebang); `uv run gpuaas.py ...` works as well.
`ssh`, `logs`, `snapshot` and `delete` only act on your own instances, even if other users share the project.
`delete` asks before it deletes an instance with its disk (`--yes` skips the question).
`rename` changes the name of an instance and its hostname, which the prompt shows at the next login.

## What `start` asks

1. Name (default `<username>-gpu`; names already used in the project are refused).
2. Flavor, read live from the cloud with GPU model and count, vCPUs, RAM and disk.
   Supported are the RTX 3070 (default and test card, `g1-3070.xlarge`) and the RTX 3080; the old RTX 2080 Ti flavors are shown but not supported.
3. Software (checkboxes, grouped, see below); dependencies are added automatically.
   With `--image`, the software already in that image is shown ticked and marked "in the image"; tick only what you want in addition.
4. Image: only images whose Ubuntu version supports all chosen software can be chosen, and the best public one is preselected (e.g. Ubuntu 24.04 for the HPC packages); also your project's snapshots, images shared with you that were made with `gpuaas.py` (marked with their visibility and the project that provides them), or any other image by name or ID.
5. For gpu-burn the duration.
6. SSH key: your keys in `~/.ssh` (`id_ed25519.pub` preselected), or a new key generated locally as `~/.ssh/gpuaas_ed25519`.

Command-line options preselect the answers.
With `--yes`, or without a terminal, nothing is asked and the preselected answers are used.
The instance is attached to `<username>-net`, the network that NetLab sets up for each account; `--network` chooses another one (`start` lists the available networks if it does not exist).
After a summary and your confirmation, `start` creates the instance, removes the old host key of the reused floating IP from `~/.ssh/known_hosts`, and waits until the software is installed (`--no-wait` skips this).

## Software packages

`./gpuaas.py help start` lists the packages with their compatibility:

```text
software packages (--packages) and compatibility (- = not supported):
                                                             Ubuntu 26.04  Ubuntu 24.04
  Basics
    nvidia-driver  NVIDIA driver (always)                         tested        tested
    cuda           CUDA toolkit 13.4                              tested        tested
    gcc-offload    GCC and gfortran GPU offloading                tested        tested
    nvhpc          NVIDIA HPC SDK (nvc, nvfortran)                tested        tested
  Containers
    docker         Docker with NVIDIA container toolkit           tested        tested
    gpu-burn       gpu-burn stress test                           tested        tested
  HPC (Spack, Ubuntu 24.04)
    spack          Spack with Environment Modules                      -        tested
    openmpi        Open MPI 5, CUDA-aware                              -        tested
    kokkos         Kokkos with CUDA and OpenMP                         -        tested
    hpx            HPX with CUDA support                               -        tested
    chapel         Chapel with GPU support                             -        tested
    julia          Julia with Dagger and CUDA.jl                       -        tested
    llvm-offload   LLVM Clang/Flang with OpenMP offloading             -        tested
  Shell
    zsh            zsh with Oh My Zsh, fzf and zoxide             tested        tested
```

The first boot installs all updates and the packages, then reboots once (driver, new kernel).
After the reboot the HPC packages are built, every package is checked, and the instance is marked ready; `start` prints `nvidia-smi` at that point.
If a check fails, `start` names the package; `./gpuaas.py logs NAME` shows the details.

Basics and containers:

- `nvidia-driver` (always): the headless form of the driver that `ubuntu-drivers` recommends: compute libraries, `nvidia-smi` and prebuilt kernel modules, without X server, Mesa or OpenGL.
- `cuda`: CUDA toolkit 13.4 from NVIDIA's apt repository: compiler, libraries and command-line tools, without the graphical Nsight profilers, which would pull in EGL and Mesa (`sudo apt install cuda-nsight-systems-13-4` adds them).
- `gcc-offload`: GCC and gfortran with OpenMP and OpenACC offloading to the GPU from the Ubuntu archive (needs only the driver).
  The wrapper `gcc-offload prog.c -o prog` runs `gcc -fopenmp -foffload=nvptx-none -foffload-options=-fcf-protection=none -foffload-options=-Wa,--no-verify -fno-stack-protector`; Ubuntu's default `-fcf-protection` and stack protector would otherwise break the offload compiler, and CUDA 13's `ptxas` rejects GCC 13's default architecture.
  `gfortran-offload` does the same for Fortran, and with `-fopenacc` it compiles OpenACC instead of OpenMP.
- `nvhpc`: the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) 26.9 from NVIDIA's apt repository: nvfortran, nvc and nvc++ with OpenMP and OpenACC offloading, `do concurrent` on the GPU (`-stdpar=gpu`) and CUDA Fortran.
  It brings its own CUDA and MPI, so it is only in the PATH after `module load nvhpc`; it needs about 15 GB of disk, i.e. a medium or larger flavor.
- `docker`: Docker with the NVIDIA container toolkit (`--gpus all`); user `ubuntu` may use it without sudo.
- `gpu-burn`: the [gpu-burn](https://github.com/wilicc/gpu-burn) stress test, runs once when the instance is ready and writes `/home/ubuntu/gpu-burn/gpu-burn-results_<time>.log`; its build pulls several GB of CUDA images, so use a medium or larger flavor.

HPC packages (Ubuntu 24.04, newest stable versions):

- `spack`: [Spack](https://spack.io) in `/opt/spack` with [Environment Modules](https://modules.readthedocs.io): `module avail` lists the HPC packages, `module load kokkos` loads one.
  The package recipes are pinned to a tested spack-packages commit; Ubuntu packages serve as externals, so Spack builds as little as possible.
- `openmpi`: Open MPI 5.0, CUDA-aware.
- `kokkos`: Kokkos 5.2 with the CUDA backend (built for the instance's GPU), OpenMP for the CPU cores and Serial.
- `hpx`: HPX 1.11 with CUDA support.
- `chapel`: Chapel 2.10 with GPU support, built on the same LLVM 22 as `llvm-offload`.
- `julia`: Julia (newest stable release via juliaup) with Dagger and CUDA.jl; CUDA.jl uses the stack's CUDA instead of downloading its own.
- `llvm-offload`: LLVM 22 with Clang and Flang and OpenMP offloading to the GPU (`module load llvm cuda`, then `clang -fopenmp --offload-arch=native` or `flang -fopenmp --offload-arch=native`), in addition to `gcc-offload`.
  Spack's LLVM recipe leaves out the OpenMP device runtime for NVIDIA GPUs and Flang's runtime library, so `spack.yaml` adds a small overlay recipe that builds both; Chapel uses the same LLVM, so only one LLVM is installed.
  The recipe also gives `clang` an rpath to LLVM's OpenMP runtime, so its programs run without the module and without `LD_LIBRARY_PATH`, which would break GCC's OpenMP programs (LLVM installs a `libgomp.so.1` alias).

Shell:

- `zsh`: zsh with [Oh My Zsh](https://ohmyz.sh) as login shell of user `ubuntu`: theme robbyrussell, autosuggestions, syntax highlighting, fzf, `z` (zoxide) to jump to frequent directories, colored `cat` and man pages, case-insensitive completion, make targets and ssh hosts from `~/.ssh/config`.
  `~/.zshrc` is a template from `cloud-init/packages/zsh.yaml` that you can change; Environment Modules (`module load`) and the PATH of CUDA and Julia work in zsh as well.

The HPC packages share one CUDA 13.0, the newest CUDA that Chapel's GPU support allows.
Ubuntu 26.04 needs CUDA 13.1 or newer because of its glibc, so the HPC packages run on Ubuntu 24.04.
After the reboot Spack installs them from a binary cache for the RTX 3070 and 3080 (a public container in the NetLab cloud, built with [tools/spack-cache/](../tools/spack-cache/README.md)), which takes about 15 minutes on a medium flavor; whatever the cache lacks is built from source, which took about an hour on an xlarge flavor (4 vCPUs).

Each package is a cloud-init fragment in `cloud-init/packages/` (see [Adding a package](#adding-a-package)); its examples are in `workspace/<name>/`.
`--userdata FILE` replaces the packages with your own cloud-init file.

## Workspace

`workspace/` in this repository has a folder per package (`cuda`, `gcc-offload`, `llvm-offload`, `nvhpc`, `docker`, `openmpi`, `kokkos`, `hpx`, `chapel`, `julia`) with programs in the package's programming models, a Makefile or CMakeLists.txt, a README with the commands, and a `run.sh` that builds and runs them (`docker/` only shows containers with and without GPU).
Every folder computes the same two problems sequentially, in parallel on the CPU cores and on the GPU (`cuda/` has no parallel CPU version): π with the midpoint rule (10⁹ steps, double precision) and the matrix product of two 2048 × 2048 matrices (single precision, the same simple algorithm everywhere, no BLAS).
Where a package offers several GPU models, each has its own program: Fortran with OpenMP and OpenACC (gfortran, Flang, nvfortran), `do concurrent` and CUDA Fortran (nvfortran), and seven GPU variants in Julia (CUDA.jl arrays and kernels, KernelAbstractions, and the four ways of Dagger).
Each program checks its result, fails if it is wrong or the GPU was not used, and prints its time and rate in GFLOP/s, so the programming models can be compared; [workspace/README.md](../workspace/README.md) has tables of results, and [docs/results/](results/README.md) the measurements behind them.
`~/workspace/run_all.sh` runs all folders on the instance, collects the results in `results/results.csv` and plots them with gnuplot into `results/results.pdf` (one page for π, one for the matrix product; it installs gnuplot when it is missing).
`start` copies the workspace files and the folders of the chosen packages to `~/workspace` on the instance (inside the cloud-init data, so no extra copy step), and runs every `run.sh` once as the package's check, on a fresh copy of the folder (so that your edits, kept in snapshots, cannot make a check fail): a ready instance has shown that all its software works.
The checks run in parallel, one per CPU core and 4 GB of memory (four on an xlarge flavor, one at a time on small and medium ones); `gpuaas.py logs` shows the output of each check as one block when it ends; the times in it are not meaningful, because the checks share CPU and GPU (`~/workspace/run_all.sh` measures).
Files changed on the instance are kept when it is saved with `snapshot` and started again.

## Snapshots and shared images

Installing takes time, so save a ready instance with `./gpuaas.py snapshot my-gpu my-image`.
`snapshot` refuses instances whose setup failed or is still running, because the image would list software that does not work (`--force` overrides this).
The image remembers its packages: it appears in the image menu of `start`, and those packages are not installed again (only checked).
To offer such an image to everyone, its owner or a NetLab admin can make it shared or community-visible, e.g. `openstack image set --community my-image`, if the cloud's policy allows it.
An image for others needs `--share`, because a snapshot keeps everything: cloud-init adds the key of each new instance to `~/.ssh/authorized_keys` but keeps the old ones, so your key would open every instance started from the image.

```sh
gpuaas.py snapshot my-gpu my-image --share   # removes your SSH keys, shell histories and cloud-init state first
gpuaas.py delete my-gpu                      # you cannot log in to my-gpu any more
```

Remove tokens, keys or data of your own from the home directory yourself before that.

## HPC image

An instance with all packages is ready about 28 minutes after the start on an xlarge flavor.
A snapshot of such an instance, an HPC image, saves most of that time: an instance started from it installs nothing and only runs the checks, so it is ready after about 15 minutes (about half of that is creating the instance from the 42 GB image).
To make one with all packages except gpu-burn (which would run again at every start), and start from it:

```sh
./gpuaas.py start hpc-build --flavor g1-3070.xlarge --image ubuntu-24.04-noble-server-cloud-image-amd64 \
    --packages cuda,gcc-offload,nvhpc,docker,openmpi,kokkos,hpx,chapel,julia,llvm-offload,zsh --yes
./gpuaas.py snapshot hpc-build hpc-image --share
./gpuaas.py delete hpc-build --yes
./gpuaas.py start my-hpc --flavor g1-3070.xlarge --image hpc-image
```

The wizard shows the image's software as already ticked, so Enter at each question is enough (`--yes` skips the questions).
The image needs an xlarge flavor, because a snapshot keeps the disk size of its instance (160 GB); making it takes about 45 minutes (28 for the instance, 18 for the snapshot).
After changes to the packages, the workspace or the binary cache, make a new image and delete the old one.
The maintainer's HPC image can be shared with other projects on request.

## Limits

Instances are shelved automatically one week after launch so that others can use the GPUs; `list` shows the remaining days.
A shelved instance keeps its disk (files and software) as an image and its IP address, but releases its GPU.
`./gpuaas.py unshelve NAME` resumes it if a GPU of its flavor is free (otherwise try again later), and `./gpuaas.py ssh NAME` offers to do this when the instance is shelved.
`./gpuaas.py shelve NAME` shelves an instance yourself, e.g. over a break, so that its GPU is free in the meantime.
Shelving took 6 minutes and unshelving 5.5 minutes for a small flavor in a test; `list` counts the week from the last start, so it shows about 7 days again after unshelving.
The project quota (instances, vCPUs, RAM) is shown by `list`.

## Security notes

- Only the cloud's login URL has a publicly trusted certificate; its service endpoints (compute, images, object storage) use certificates of the cloud's own CA, which is not published.
  `clouds.yaml` therefore has `verify: false`, and `gpuaas.py` silences urllib3's warning about it; once the CA certificate is available, `cacert: <file>` replaces it.
- For the same reason the Spack binary cache is fetched without TLS verification; it is also unsigned, and only its maintainer can write to its container.
- Your password is never stored; the saved login token in `~/.cache/gpuaas/` is readable only by you and expires after an hour.
- A snapshot keeps everything on the instance; share images only after `snapshot --share` (see above).

## Alternatives

Instances can also be started with the Horizon web interface (https://private-cloud.informatik.hs-fulda.de/), the [OpenStack CLI](https://docs.openstack.org/python-openstackclient/latest/), the [Terraform](https://registry.terraform.io/providers/terraform-provider-openstack/openstack/latest/docs) or [Pulumi](https://www.pulumi.com/registry/packages/openstack/) providers, or [Apache libcloud](https://libcloud.apache.org/).

## Repository layout

- `gpuaas.py`: the command-line tool, one file; its dependencies are declared inline and pinned in `gpuaas.py.lock`.
- `clouds.yaml.example`: the template for your `clouds.yaml`.
- `cloud-init/packages/`: one cloud-init fragment per software package.
- `workspace/`: the examples that `start` copies to the instances; [workspace/README.md](../workspace/README.md) explains them and has the results.
- `docs/`: this guide; `docs/results/`: the measurements behind those results ([README](results/README.md)).
- `tools/check.py`: offline checks; `tools/spack-cache/`: building and uploading the Spack binary cache ([README](../tools/spack-cache/README.md)).

## Contributing

Run `./tools/check.py` after changing `gpuaas.py`, a package file or the workspace; it needs no cloud access.
It checks the cloud-init data of every package, every pair of packages and all packages (YAML, shell syntax, Nova's size limit), that the Usage block in README.md matches `COLUMNS=100 ./gpuaas.py help` and the package matrix in this guide `COLUMNS=100 ./gpuaas.py help start`, the files in `workspace/` that must stay identical, and the workspace scripts.
After changing the dependency header of a script, update its lock file with `uv lock --script FILE`.

### Adding a package

A package is a file `cloud-init/packages/<name>.yaml`: a cloud-config (`packages`, `write_files`, `runcmd`, ...) plus a `gpuaas:` block that describes it, for example:

```yaml
gpuaas:
  title: My tool                 # text in the menu
  group: Basics                  # menu section
  order: 35                      # install order: higher than that of every package in requires
  requires: [cuda]               # optional: packages it needs
  disk: 2                        # approximate disk space in GB
  os:                            # compatibility: tested, untested or "no: <reason>"
    ubuntu-26.04: tested
    ubuntu-24.04: untested
  check: my-tool --version       # optional: without it, workspace/<name>/run.sh is the check
runcmd:
  - DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y my-tool
```

Further keys: `required` (always installed), `long` (built with Spack, so `start` waits longer), `build` (runs after the reboot, with the driver loaded) and `job` (runs once the instance is ready, like gpu-burn).
Lists from several packages are appended in install order, dictionaries are merged, and two packages that set the same value differently are an error.
Put examples into `workspace/<name>/` with a `run.sh`, then run `./tools/check.py`.

## Licenses of the installed software

The instances download further software, each under its own license:

| Software | Source | License |
| --- | --- | --- |
| NVIDIA driver, CUDA toolkit, HPC SDK, container toolkit | Ubuntu and NVIDIA apt repositories | NVIDIA's licenses |
| GCC, Docker, zsh, fzf, gnuplot and other Ubuntu packages | Ubuntu archive | their own licenses |
| Spack and its package recipes | GitHub | Apache-2.0 or MIT |
| Open MPI, Kokkos, HPX, Chapel, LLVM, Boost and their dependencies | built by Spack from source (or its binary cache) | their own open-source licenses |
| Julia (juliaup), Dagger.jl, CUDA.jl | GitHub, Julia package registry | MIT |
| gpu-burn | GitHub | BSD-2-Clause |
| Oh My Zsh, zsh-autosuggestions | GitHub | MIT |
| zsh-syntax-highlighting | GitHub | BSD-3-Clause |
