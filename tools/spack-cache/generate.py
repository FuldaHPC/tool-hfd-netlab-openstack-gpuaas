#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6"]
# ///
"""Print build.sh for an ubuntu:24.04 container that builds the HPC stack exactly like an instance:
the apt packages, files, runcmd lines and build steps come from cloud-init/packages/. Only nvidia-smi
is faked (no GPU in the container), so that gpuaas-spack builds for sm_86 (RTX 3070/3080). After the
builds the script compiles the workspace examples, runs the CPU examples that need no GPU, and pushes
everything to the binary cache in /work/mirror. Each finished build is pushed at once, and a rerun installs
what is already in /work/mirror, so a failing build does not cost the hours of the earlier ones (LLVM).
See README.md for the whole procedure.

Usage: tools/spack-cache/generate.py > build.sh
"""

import shlex
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SPACK_PACKAGES = ["openmpi", "kokkos", "hpx", "chapel", "llvm-offload"]  # julia is not built with Spack
CUDA_ARCH = "8.6"  # RTX 3070/3080, the only supported GPUs
SM = CUDA_ARCH.replace(".", "")  # as in sm_86


def fragment(name):
    return yaml.safe_load((REPO / "cloud-init" / "packages" / f"{name}.yaml").read_text())


def main():
    configs = {name: fragment(name) for name in ["spack", *SPACK_PACKAGES]}
    out = ["#!/bin/bash", "set -e", "export DEBIAN_FRONTEND=noninteractive",
           'stamp() { echo "=== $(date +%H:%M:%S) $*"; }',
           "stamp apt", "apt-get update -qq && apt-get upgrade -y -qq >/dev/null",
           # Cloud images already have curl, ca-certificates and gnupg; the Docker image does not.
           "apt-get install -y -qq curl ca-certificates gnupg "
           + " ".join(p for config in configs.values() for p in config.get("packages", [])) + " >/dev/null"]
    for config in configs.values():
        for f in config.get("write_files", []):
            out += [f"mkdir -p {shlex.quote(str(Path(f['path']).parent))}",
                    f"cat > {shlex.quote(f['path'])} <<'GPUAAS_EOF'", f["content"].rstrip("\n"), "GPUAAS_EOF"]
            if f.get("permissions"):
                out.append(f"chmod {f['permissions']} {shlex.quote(f['path'])}")
    for name, config in configs.items():
        if config.get("runcmd"):
            out += [f"stamp runcmd {name}", *config["runcmd"]]
    # Concretize fresh: with reuse (as on the instances) Spack would take outdated binaries from the
    # cache after a recipe change instead of rebuilding them. The pushed index then lists only these
    # specs, so instances cannot pick outdated ones either.
    out.append("HOME=/root /opt/spack/bin/spack -e /opt/spack-env config add concretizer:reuse:false")
    # A rerun after a failure (README: without deleting mirror/) installs the finished builds from there.
    out.append("[ -f /work/mirror/v3/manifests/index/index.manifest.json ] && HOME=/root /opt/spack/bin/spack"
               " -e /opt/spack-env mirror add --type binary --unsigned local file:///work/mirror")
    push = ("HOME=/root /opt/spack/bin/spack -e /opt/spack-env buildcache push --unsigned --update-index /work/mirror"
            " >/dev/null && chown -R $(stat -c %u:%g /work) /work/mirror")
    out += ["cat > /usr/local/bin/nvidia-smi <<'GPUAAS_EOF'", "#!/bin/sh", f"echo {CUDA_ARCH}", "GPUAAS_EOF",
            "chmod 0755 /usr/local/bin/nvidia-smi"]
    for name in SPACK_PACKAGES:
        out += [f"stamp 'build {name}'", "start=$(date +%s)",
                "(", configs[name]["gpuaas"]["build"].rstrip("\n"), ")",
                f'echo "TIME {name} $(( $(date +%s) - start )) s"', push]
    out += ["stamp 'workspace compile test'",
            "rm -rf /tmp/ws && cp -r /work/workspace /tmp/ws && . /usr/share/modules/init/sh",
            "export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1",
            # Only the CPU examples can run without a GPU; GPU code is compiled for sm_86 explicitly.
            f'(cd /tmp/ws/openmpi && module load openmpi && make all CUDA_ARCH=sm_{SM}'
            ' && mpirun --oversubscribe -np 2 ./pi_cpu && mpirun --oversubscribe -np 2 ./matmul_cpu)'
            ' && echo "WORKSPACE OK openmpi (built all, ran CPU)" || echo "WORKSPACE FAIL openmpi"',
            # Kokkos::initialize needs the GPU of the CUDA backend, so the Kokkos programs are only built.
            '(cd /tmp/ws/kokkos && module load kokkos && cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null'
            ' && cmake --build build) && echo "WORKSPACE OK kokkos (built all)" || echo "WORKSPACE FAIL kokkos"',
            '(cd /tmp/ws/hpx && module load hpx && cmake -B build -DCMAKE_BUILD_TYPE=Release'
            f' -DCMAKE_CUDA_ARCHITECTURES={SM} >/dev/null && cmake --build build'
            ' && ./build/pi_cpu && ./build/matmul_cpu) && echo "WORKSPACE OK hpx (built all, ran CPU)"'
            ' || echo "WORKSPACE FAIL hpx"',
            # Chapel links -lcuda, which only the NVIDIA driver provides, so this fails without a GPU.
            '(cd /tmp/ws/chapel && module load chapel && make all) && echo "WORKSPACE OK chapel (built all)"'
            ' || echo "WORKSPACE FAIL chapel (expected without a driver: -lcuda)"',
            # The CPU examples run without LD_LIBRARY_PATH, which tests the rpath of LLVM's OpenMP runtime.
            f'(cd /tmp/ws/llvm-offload && module load llvm cuda && make all OFFLOAD_ARCH=sm_{SM}'
            ' && ./pi_cpu && ./matmul_cpu) && echo "WORKSPACE OK llvm-offload (built all, ran CPU)"'
            ' || echo "WORKSPACE FAIL llvm-offload"',
            f'(cd /tmp/ws/llvm-offload/fortran && module load llvm cuda && make all OFFLOAD_ARCH=sm_{SM}'
            ' && ./pi_cpu && ./matmul_cpu) && echo "WORKSPACE OK llvm-offload Fortran (built all, ran CPU)"'
            ' || echo "WORKSPACE FAIL llvm-offload Fortran"',
            "stamp 'buildcache push'",
            push,
            "stamp done"]
    print("\n".join(out))


if __name__ == "__main__":
    main()
