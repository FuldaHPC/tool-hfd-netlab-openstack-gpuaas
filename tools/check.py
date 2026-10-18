#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["openstacksdk>=4", "pyyaml>=6", "questionary>=2"]
# ///
"""Offline checks of the repository, without any cloud call: run ./tools/check.py after changing
gpuaas.py, a package file or the workspace. Exits with 1 if a check fails.

- the cloud-init user data of every package, every pair of packages and all packages: valid YAML,
  shell syntax of the generated finish script and of the embedded #!/bin/sh files, Nova's size limit
- the Usage block in README.md and the package matrix in docs/guide.md equal the output of gpuaas.py help
  and gpuaas.py help start (COLUMNS=100)
- copies that must stay identical (common.h, common.f90, the Fortran sources shared by three folders)
- shell syntax of every workspace run.sh and of run_all.sh, and that every run.sh is executable
"""
import itertools
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"
SHELLS = [s for s in ("dash", "bash") if shutil.which(s)] or ["sh"]
failures = 0


def report(name, problems):
    global failures
    if problems:
        failures += 1
        print(f"FAIL {name}")
        for problem in problems[:10]:
            print(f"     {problem}")
    else:
        print(f"ok   {name}")


def shell_errors(script, label):
    errors = []
    for shell in SHELLS:
        result = subprocess.run([shell, "-n"], input=script, text=True, capture_output=True)
        if result.returncode:
            errors.append(f"{label} ({shell}): {result.stderr.strip()}")
    return errors


def load_gpuaas():
    sys.dont_write_bytecode = True  # no __pycache__ in the repository
    sys.path.insert(0, str(ROOT))
    import gpuaas
    return gpuaas


def check_userdata(g):
    catalog = g.load_packages()
    names = list(catalog)
    combinations = [[n] for n in names] + [list(c) for c in itertools.combinations(names, 2)] + [names]
    problems, largest = [], 0
    for wanted in combinations:
        packages = g.resolve_packages(catalog, wanted)
        data = g.build_userdata(catalog, packages, set(), "60")
        try:
            fitted = g.fit_userdata(data)
        except SystemExit as e:
            problems.append(f"{','.join(wanted)}: {e}")
            continue
        largest = max(largest, len(g.base64.b64encode(fitted if isinstance(fitted, bytes) else fitted.encode())))
        config = yaml.safe_load(data)
        for item in config["write_files"]:
            if item.get("content", "").startswith("#!/bin/sh"):
                problems += shell_errors(item["content"], f"{','.join(wanted)}: {item['path']}")
    report(f"user data of {len(combinations)} package combinations (largest {largest / 1000:.1f} of "
           f"{g.USERDATA_LIMIT / 1000:.1f} KB base64)", problems)


def readme_block(text, start, end="\n```"):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


def check_readme():
    env = dict(os.environ, COLUMNS="100")
    run = lambda *args: subprocess.run([sys.executable, str(ROOT / "gpuaas.py"), *args], env=env,
                                       capture_output=True, text=True, check=True).stdout
    readme = (ROOT / "README.md").read_text()
    guide = (ROOT / "docs" / "guide.md").read_text()
    usage = run("help").rstrip("\n")
    matrix = run("help", "start")
    matrix = matrix[matrix.index("software packages (--packages)"):matrix.index("\n\nDependencies")]
    problems = []
    if usage not in readme:
        problems.append("the Usage block differs from 'COLUMNS=100 ./gpuaas.py help'")
    if matrix not in guide:
        problems.append("the package matrix in docs/guide.md differs from 'COLUMNS=100 ./gpuaas.py help start'")
    report("help blocks in README.md and docs/guide.md", problems)


def check_copies():
    groups = [sorted(WORKSPACE.glob("*/common.h"))]
    fortran = [WORKSPACE / "gcc-offload" / "fortran", WORKSPACE / "llvm-offload" / "fortran", WORKSPACE / "nvhpc"]
    for name in ("common.f90", "pi_seq.f90", "pi_cpu.f90", "matmul_seq.f90", "matmul_cpu.f90"):
        groups.append([folder / name for folder in fortran])
    for name in ("pi_acc.f90", "matmul_acc.f90"):
        groups.append([fortran[0] / name, fortran[2] / name])  # gfortran and nvfortran
    for name in ("pi_gpu.f90", "matmul_gpu.f90"):
        groups.append([fortran[1] / name, fortran[2] / name])  # Flang and nvfortran
    problems = []
    for files in groups:
        texts = {f: f.read_bytes() for f in files if f.exists()}
        missing = [str(f.relative_to(ROOT)) for f in files if not f.exists()]
        if missing:
            problems.append("missing: " + ", ".join(missing))
        if len(set(texts.values())) > 1:
            problems.append("differ: " + ", ".join(str(f.relative_to(ROOT)) for f in texts))
    report(f"identical copies ({len(groups)} groups)", problems)


def check_scripts():
    problems = []
    scripts = sorted(WORKSPACE.glob("*/run.sh")) + [WORKSPACE / "run_all.sh"]
    for script in scripts:
        if not os.access(script, os.X_OK):
            problems.append(f"{script.relative_to(ROOT)} is not executable")
        problems += shell_errors(script.read_text(), str(script.relative_to(ROOT)))
    for folder in sorted(p for p in WORKSPACE.iterdir() if p.is_dir() and p.name != "results"):
        if not (folder / "run.sh").exists():
            problems.append(f"{folder.relative_to(ROOT)} has no run.sh")
    report(f"workspace scripts ({len(scripts)})", problems)


def main():
    g = load_gpuaas()
    check_userdata(g)
    check_readme()
    check_copies()
    check_scripts()
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
