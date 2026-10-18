#!/bin/sh
# Runs the examples of every folder (its run.sh), collects their results in results/results.csv and
# plots them with gnuplot into results/results.pdf, one page for pi and one for the matrix product.
# The logs of each folder are in results/<folder>.log. Run it as ./run_all.sh, optionally with
# folder names (./run_all.sh cuda julia) to run only those, and with -n N to run every folder N times:
# results/runs.csv then has every run, results/results.csv the run with the median time of each program.
# It exits with 1 if a folder failed.
set -u
runs=1
if [ "${1:-}" = "-n" ]; then runs=${2:-}; shift 2 2>/dev/null || shift; fi
case $runs in ''|*[!0-9]*|0) echo "usage: ./run_all.sh [-n RUNS] [FOLDER ...]" >&2; exit 2 ;; esac
cd "$(dirname "$0")"
mkdir -p results
csv=results/results.csv
raw=results/runs.csv
# All folders with a run.sh, in this order (and any others after them):
order="cuda gcc-offload llvm-offload nvhpc openmpi kokkos hpx chapel julia docker"
all=$(for d in */; do [ -x "$d/run.sh" ] && echo "${d%/}"; done)
folders=${*:-$(for d in $order; do echo "$all" | grep -qx "$d" && echo "$d"; done
               echo "$all" | grep -vxF "$(echo "$order" | tr " " "\n")")}

echo "folder,problem,kind,model,seconds,gflops,status,run" > "$raw"
failed=0
for folder in $folders; do
  folder=${folder%/}
  printf '%-14s' "$folder"
  start=$(date +%s)
  result=ok
  : > "results/$folder.log"
  run=1
  while [ $run -le "$runs" ]; do
    [ "$runs" -gt 1 ] && echo "== run $run" >> "results/$folder.log"
    "./$folder/run.sh" >> "results/$folder.log" 2>&1 || { result="FAILED, see results/$folder.log"; failed=1; }
    run=$((run + 1))
  done
  echo "$(( $(date +%s) - start )) s, $result"
  # Result lines: <CPU|GPU> [sequential] (<model>): pi = ... (error E) in T s (R GFLOP/s)
  #           or: <CPU|GPU> [sequential] (<model>): NxN matrix product correct in T s (R GFLOP/s)
  awk -v folder="$folder" '
    /^== run [0-9]+$/ { run = $3 }
    / s \([0-9.]+ GFLOP\/s\)$/ && (/: pi = / || / matrix product /) {
      label = $0; sub(/\): (pi = | *[0-9]+x[0-9]+ matrix product).*/, ")", label)
      kind = label ~ /^CPU sequential/ ? "seq" : label ~ /^CPU/ ? "cpu" : label ~ /^GPU/ ? "gpu" : "host"
      model = label; sub(/^[^(]*\(/, "", model); sub(/\)$/, "", model); sub(/, NVIDIA [^,]*$/, "", model)
      gsub(/["\t]/, "", model)
      problem = / matrix product / ? "matmul" : "pi"
      match($0, / in [0-9.]+ s /); seconds = substr($0, RSTART + 4, RLENGTH - 7)
      match($0, /\([0-9.]+ GFLOP\/s\)$/); gflops = substr($0, RSTART + 1, RLENGTH - 10)
      status = "ok"
      if (problem == "matmul" && $0 !~ / correct in /) status = "wrong"
      if (problem == "pi") { match($0, /error [0-9.e+-]+/); if (substr($0, RSTART + 6, RLENGTH - 6) + 0 >= 1e-9) status = "wrong" }
      if (kind == "host") status = "not on the GPU"
      printf "%s,%s,%s,\"%s\",%s,%s,%s,%d\n", folder, problem, kind, model, seconds, gflops, status, run ? run : 1
    }' "results/$folder.log" >> "$raw"
done

# One row per program: the run with the median time (the lower one of the two middle runs for an even
# number), the worst status of all runs, the number of runs and the fastest and slowest time.
awk '
  NR == 1 { print "folder,problem,kind,model,seconds,gflops,status,runs,min_seconds,max_seconds"; next }
  {
    q = index($0, ",\""); key = substr($0, 1, index(substr($0, q + 2), "\"") + q + 1)
    split(substr($0, length(key) + 2), f, ",")
    if (!(key in n)) order[++keys] = key
    i = ++n[key]; t[key, i] = f[1] + 0; r[key, i] = f[2]; text[key, i] = f[1]
    if (f[3] != "ok" && !(key in bad)) bad[key] = f[3]
  }
  END {
    for (k = 1; k <= keys; k++) {
      key = order[k]; m = n[key]
      for (i = 1; i <= m; i++) idx[i] = i
      for (i = 2; i <= m; i++)  # insertion sort of the runs by time
        for (j = i; j > 1 && t[key, idx[j]] < t[key, idx[j - 1]]; j--) { x = idx[j]; idx[j] = idx[j - 1]; idx[j - 1] = x }
      if (m < runs) printf "warning: %s has only %d of %d results (see its log)\n", key, m, runs > "/dev/stderr"
      mid = idx[int((m + 1) / 2)]
      printf "%s,%s,%s,%s,%d,%s,%s\n", key, text[key, mid], r[key, mid], (key in bad) ? bad[key] : "ok", m,
             text[key, idx[1]], text[key, idx[m]]
    }
  }' runs="$runs" "$raw" > "$csv"
echo "$(($(wc -l < "$csv") - 1)) results in $csv$([ "$runs" -gt 1 ] && echo ", medians of $runs runs (all runs in $raw)")"

# Layout for the plot: per problem one row per correct result, grouped by folder with a header row
# and a band behind every other group (columns: type, y, value, seconds, kind, text).
layout() {
  awk -v problem="$1" '
    NR > 1 {
      q1 = index($0, "\""); rest = substr($0, q1 + 1); q2 = index(rest, "\"")
      model = substr(rest, 1, q2 - 1)
      split(substr($0, 1, q1 - 2), a, ","); split(substr(rest, q2 + 2), b, ",")
      if (a[2] != problem || b[3] != "ok") next
      if (a[1] != folder) {
        if (folder != "") { printf "band\t%g\t%g\t\t\t%d\n", top, y - 0.5, groups % 2; y += 0.6 }
        folder = a[1]; groups++; top = y - 0.5
        printf "header\t%g\t\t\t\t%s\n", y, folder; y++
      }
      printf "bar\t%g\t%s\t%s\t%s\t%s\n", y, b[2], b[1], a[3], model; y++
    }
    END { if (folder != "") printf "band\t%g\t%g\t\t\t%d\n", top, y - 0.5, groups % 2 }' "$csv"
}

if ! command -v gnuplot > /dev/null; then
  echo "installing gnuplot for the plot: sudo apt-get install gnuplot-nox"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q gnuplot-nox > /dev/null 2>&1 || echo "could not install gnuplot, so there is no plot"
fi
if command -v gnuplot > /dev/null; then
  problems=
  for problem in pi matmul; do
    layout $problem > results/$problem.tsv
    grep -q '^bar' results/$problem.tsv && problems="$problems $problem"  # a page only with correct results
  done
  rm -f results/results.pdf
  gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
  gpu=$(echo "$gpus" | head -n 1); count=$(echo "$gpus" | grep -c .)
  [ "$count" -gt 1 ] && gpu="${count}x $gpu"  # e.g. 2x NVIDIA GeForce RTX 3080
  machine="${gpu:-no GPU found}, $(nproc) CPU cores, $(date +%Y-%m-%d)$([ "$runs" -gt 1 ] && echo ", median of $runs runs")"
  if [ -n "$problems" ]; then
    gnuplot -e "dir='results'; machine='$machine'; problems='$problems'" plot_results.gp && echo "plot in results/results.pdf"
  else
    echo "no correct results, so there is no plot"
  fi
fi
exit $failed
