# Spack binary cache

The HPC packages are built from source with Spack.
A binary cache for the RTX 3070/3080 (sm_86) makes instances install them in minutes instead.
It lives in the public Swift container `gpuaas-spack-cache` of the NetLab cloud (`https://10.32.4.25/swift/v1/gpuaas-spack-cache`), which `cloud-init/packages/spack.yaml` configures as mirror.
Instances only use binaries whose Spack hashes match exactly, so rebuild the cache after changing anything that affects the HPC stack: the spack-packages commit, a Spack spec, a recipe in the overlay repository, the Spack externals or apt packages in `spack.yaml`, or Ubuntu updates of the externals.
The container concretizes with `reuse: false`, so a changed recipe gets a new hash and is rebuilt; with reuse, as on the instances, Spack would keep taking the old binary from the cache.

The cache is unsigned and read without TLS verification, because the cloud's internal CA is not available (see "Security notes" in [docs/guide.md](../../docs/guide.md)).
Anyone can read the container, but only its owner can write to it, so updating it needs that project's OpenStack login; for a cache of your own, create a public container in your project and change its name in `prune.py` and the mirror URL in `cloud-init/packages/spack.yaml`.

## Update

```sh
export GPUAAS_BUILD_HOST=myhost        # the ssh host that builds
tools/spack-cache/update.sh            # rebuild what changed, upload the new files
tools/spack-cache/update.sh --prune    # the same, then remove the files the new index no longer uses
tools/spack-cache/update.sh --fresh    # start with an empty mirror/ (after changing the spack-packages commit)
```

`update.sh` runs locally with your OpenStack login (like `gpuaas.py`; `OS_PASSWORD` works too) and does all steps over ssh on the build host `GPUAAS_BUILD_HOST`, in the folder `GPUAAS_BUILD_DIR` (default `gpuaas-cachebuild`, which keeps `mirror/` between runs).
The build host needs Linux on x86-64 with Docker usable without sudo, `python3` and `curl`, free disk for an LLVM build (plan for several tens of GB), and as many cores as possible (16 cores: about 2 hours from scratch); no GPU.
The steps:

1. `generate.py` prints `build.sh`, which builds the HPC stack in an `ubuntu:24.04` Docker container exactly like an instance (apt packages, files, runcmd lines and build steps from `cloud-init/packages/`; only `nvidia-smi` is faked, for sm_86), compiles the workspace examples and runs the CPU ones.
2. The container builds only what the cache lacks: from scratch about 2 hours, most of it LLVM with Flang and MLIR (84 minutes) and Chapel; with the cache in place about 15 minutes.
   Every finished package is pushed to `mirror/` at once, and a rerun after a failure installs from there, so a failing build does not cost the hours of the earlier ones.
   `--fresh` deletes `mirror/` first, so that its index lists only the current specs.
3. It prints the time per package and the workspace tests from `build.log`; `WORKSPACE FAIL chapel` is expected, because Chapel cannot link without the NVIDIA driver (`-lcuda`).
   Any other failure stops the script before the upload.
4. It uploads the files of the new index that Swift does not have yet (`mirror_files.py` lists them; curl on the build host with a fresh token from `prune.py --token`, valid for at least 55 minutes) and stops if a file did not get `201`.
   Packages whose manifest Swift already has are skipped with their tarballs: a package that the build installed from the cache and pushed again gets a new tarball with another checksum, which Swift's manifest does not use.
5. With `--prune`, `prune.py` removes the spec manifests and blobs that the current index no longer uses (the upload never deletes; on 2026-10-17 22.8 GB of 27.8 GB were unused) and checks that all kept files are still there.

`prune.py` also runs on its own: without `--delete` it only lists what it would remove.

## Upgrading the HPC stack

Chapel decides the LLVM and CUDA versions of the whole stack, so upgrades follow new Chapel releases.

1. Check what is new: the release notes of Chapel, Kokkos, HPX, Open MPI and Spack, and in [spack-packages](https://github.com/spack/spack-packages) the `develop` versions of `chapel`, `llvm`, `cuda`, `kokkos`, `hpx` and `openmpi` and Chapel's `depends_on("llvm@…")` and `depends_on("cuda@…")` lines.
2. In `cloud-init/packages/spack.yaml`: the Spack tag (`git clone --branch`), the spack-packages commit (`spack repo update --commit`), and the `require` lines of `llvm` (e.g. `@23`) and `cuda` (e.g. `@13.1`).
3. The overlay repository in `spack.yaml`: remove the `openmpi` version once the recipes have 5.0.11 or newer; check whether `hpx` still asks for `Boost.with_default_variants` (otherwise drop that overlay) and whether `llvm` still needs the GPU runtime target and `flang-rt`.
4. If Chapel allows CUDA 13.1 or newer, the HPC stack can move to Ubuntu 26.04: set the `os:` lines of the HPC package files and the CUDA versions in the workspace READMEs accordingly.
5. `tools/check.py`, then `tools/spack-cache/update.sh --fresh --prune`.
6. Start a fresh instance with all packages, check that nothing was built from source (`gpuaas.py logs`: only "fetching from build cache"), run `~/workspace/run_all.sh -n 3`, copy its `results/` to `docs/results/`, and update the tables in `workspace/README.md` and the versions in docs/guide.md and CLAUDE.md.
7. Rebuild the HPC image (docs/guide.md, "HPC image") and delete the old one.
