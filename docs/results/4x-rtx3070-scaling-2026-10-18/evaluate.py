#!/usr/bin/env python3
"""Evaluates runs.csv of bench.sh: writes results.csv (median, fastest and slowest run per problem, size,
program and number of GPUs), scaling.tsv for plot_scaling.gp, and prints the speed-up tables as Markdown.
Usage, in this folder: python3 evaluate.py"""
import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PANELS = [("pi", 0, "π"), ("matmul", 2048, "C = A·B, n = 2048"), ("matmul", 8192, "C = A·B, n = 8192")]
MULTI = [("MPI", "MPI, one rank per GPU"), ("Chapel *", "Chapel *_gpus"),
         ("Dagger Datadeps", "Dagger Datadeps"), ("Dagger DArray", "Dagger DArray")]
REFERENCES = ["CUDA kernel", "Chapel *", "CUDA.jl kernel", "CUDA.jl cuBLAS"]


runs, bad = defaultdict(list), defaultdict(set)
for row in csv.DictReader(open(HERE / "runs.csv")):
    key = (row["problem"], int(row["n"]), row["program"], int(row["gpus"]))
    if row["status"] == "ok":
        runs[key].append((float(row["seconds"]), float(row["gflops"])))
    else:
        bad[key].add(row["status"])

with open(HERE / "results.csv", "w", newline="") as f:
    out = csv.writer(f, lineterminator="\n")
    out.writerow(["problem", "n", "program", "gpus", "runs", "median_seconds", "median_gflops", "min_seconds",
                  "max_seconds", "status"])
    for key in sorted(set(runs) | set(bad)):
        t = sorted(runs[key])
        m = t[(len(t) - 1) // 2] if t else ("", "")  # the lower middle run, as in run_all.sh
        out.writerow([*key, len(t), m[0], m[1], t[0][0] if t else "", t[-1][0] if t else "",
                      "/".join(sorted(bad.get(key, ()))) or "ok"])


def median(problem, n, program, gpus):
    t = sorted(runs.get((problem, n, program, gpus), []))
    return t[(len(t) - 1) // 2] if t else None


def chapel(problem, several):
    """'Chapel *' stands for pi_gpus/matmul_gpus (several GPUs) or pi_gpu/matmul_gpu (one GPU)."""
    return f"Chapel {problem}_gpus" if several else f"Chapel {problem}_gpu"


# scaling.tsv: blocks (separated by two blank lines) 0-11: panel p, program j of the programs for several GPUs
# (index 4p + j), rows for 1-4 GPUs; blocks 12-23: panel p, reference r on one GPU (index 12 + 4p + r).
blocks = []
for problem, n, _ in PANELS:
    for pattern, _ in MULTI:
        program = chapel(problem, True) if pattern == "Chapel *" else pattern
        one = median(problem, n, program, 1)
        rows = []
        for g in range(1, 5):
            m = median(problem, n, program, g)
            rows.append(f"{g}\t{m[0]}\t{m[1]}\t{one[0] / m[0]:.3f}" if m and one else f"{g}\tNaN\tNaN\tNaN")
        blocks.append(rows)
for problem, n, _ in PANELS:
    for pattern in REFERENCES:
        program = chapel(problem, False) if pattern == "Chapel *" else pattern
        m = median(problem, n, program, 1)
        blocks.append([f"1\t{m[0]}\t{m[1]}\tNaN" if m else "1\tNaN\tNaN\tNaN"])
(HERE / "scaling.tsv").write_text("# gpus\tseconds\tgflops\tspeedup\n" + "\n\n\n".join("\n".join(b) for b in blocks) + "\n")

# Markdown tables for the README
for problem, n, title in PANELS:
    print(f"\n{title}\n\n| Program | 1 GPU | 2 GPUs | 3 GPUs | 4 GPUs | speed-up 1 → 4 |\n|---|---:|---:|---:|---:|---:|")
    for pattern, label in MULTI:
        program = chapel(problem, True) if pattern == "Chapel *" else pattern
        ms = [median(problem, n, program, g) for g in range(1, 5)]
        cells = [f"{m[0]:.3f} s" if m else "–" for m in ms]
        speedup = f"{ms[0][0] / ms[3][0]:.2f}×" if ms[0] and ms[3] else ""
        print(f"| {label} | " + " | ".join(cells) + f" | {speedup} |")
    for pattern in REFERENCES:
        program = chapel(problem, False) if pattern == "Chapel *" else pattern
        m = median(problem, n, program, 1)
        if m:
            print(f"| {program} (one GPU) | {m[0]:.3f} s | | | | |")
print("\nnot ok:", {k: v for k, v in bad.items() if v} or "none")
