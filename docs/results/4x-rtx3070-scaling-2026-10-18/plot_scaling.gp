# Plots scaling.tsv (python3 evaluate.py) into scaling.pdf: page 1 the speed-up over one GPU, page 2 the rate,
# each with one panel for π and two for the matrix product. Usage, in this folder: gnuplot plot_scaling.gp
set terminal pdfcairo size 29.7cm,21cm font "Sans,9" enhanced butt linewidth 1
set output "scaling.pdf"
set datafile separator "\t"

ink = 0x333333; muted = 0x666666; light = 0x888888; faint = 0xbbbbbb; grid = 0xe8e8e8
# One colour per program, in fixed order (checked for colour-blind separation; every line also has its own
# marker and a value label, as two of the colours have less than 3:1 contrast on white).
array colors[4] = [0x2a78d6, 0xeb6834, 0x1baf7a, 0xeda100]
array pts[4] = [7, 5, 9, 13]
array names[4] = ["MPI, one rank per GPU", "Chapel *\\_gpus", "Dagger Datadeps", "Dagger DArray"]
array keynames[4] = ["MPI, one rank per GPU", "Chapel *_gpus", "Dagger Datadeps", "Dagger DArray"]
array titles[3] = ["π (10^9 steps, double)", "C = A·B, n = 2048 (single)", "C = A·B, n = 8192 (single)"]
data = "scaling.tsv"  # blocks 4p + j: panel p, program j for 1-4 GPUs; blocks 12 + 4p + r: reference r, one GPU
machine = "4 × NVIDIA GeForce RTX 3070 without peer-to-peer access (g1-4x3070.xlarge, 4 vCPUs), 2026-10-18; median of 3 runs; MPI with one rank per GPU"
note = "Times include allocating GPU memory and all transfers between host and GPUs; one GPU means CUDA\\_VISIBLE\\_DEVICES=0, two GPUs 0,1, and so on."

panel_x(p) = 0.06 + p * 0.31
header(sub) = sprintf("set label 1 \"{/:Bold=15 Programs for several GPUs on one to four RTX 3070}\" at screen 0.05, 0.955 left textcolor rgb ink; set label 2 \"%s\" at screen 0.05, 0.925 left textcolor rgb muted font \"Sans,10\"; set label 3 machine at screen 0.05, 0.902 left textcolor rgb light font \"Sans,8\"; set label 4 note at screen 0.05, 0.03 left textcolor rgb light font \"Sans,8\"", sub)
fmt(v) = sprintf(" %.1f×", v)

# ---------------------------------------------------------------- page 1: speed-up
set multiplot
eval header("Speed-up over the same program on one GPU; the dashed line is the ideal (n GPUs, n times as fast)")
set key at screen 0.05, 0.885 left top horizontal Left reverse samplen 2.5 width 3 textcolor rgb ink noenhanced font "Sans,9"
set xrange [0.8:4.6]; set xtics 1, 1, 4 nomirror textcolor rgb ink
set yrange [0:4.4]; set ytics 0, 1, 4 format "%.0f×" nomirror textcolor rgb ink
set grid ytics linecolor rgb grid linewidth 0.8
set border 3 linecolor rgb faint
set tics font "Sans,8"
do for [p=0:2] {
  set lmargin at screen panel_x(p) + 0.02; set rmargin at screen panel_x(p) + 0.28
  set bmargin at screen 0.12; set tmargin at screen 0.80
  set label 10 sprintf("{/:Bold %s}", titles[p + 1]) at graph 0.02, 0.97 left textcolor rgb ink
  set xlabel "GPUs" textcolor rgb muted font "Sans,8"
  set ylabel (p == 0 ? "speed-up" : "") textcolor rgb muted font "Sans,8"
  if (p > 0) { unset key; unset label 1; unset label 2; unset label 3; unset label 4 }
  plot x with lines linecolor rgb light dashtype 2 notitle, \
       for [j=1:4] data index 4 * p + j - 1 using 1:4 with linespoints linecolor rgb colors[j] linewidth 2 \
           pointtype pts[j] pointsize 0.9 title keynames[j], \
       for [j=1:4] data index 4 * p + j - 1 every ::3::3 using 1:4:(fmt($4)) with labels left offset 0.5, 0 \
           textcolor rgb ink font "Sans,8" notitle
}
unset multiplot

# ---------------------------------------------------------------- page 2: rate
reset
set datafile separator "\t"
set multiplot
eval header("Rate in GFLOP/s on a logarithmic axis; grey lines for comparison: the CUDA kernel (dashed) and cuBLAS (dotted) on one GPU")
set key at screen 0.05, 0.885 left top horizontal Left reverse samplen 2.5 width 3 textcolor rgb ink noenhanced font "Sans,9"
set xrange [0.8:4.6]; set xtics 1, 1, 4 nomirror textcolor rgb ink
set logscale y 10
set ytics nomirror textcolor rgb ink
set border 3 linecolor rgb faint
set grid ytics linecolor rgb grid linewidth 0.8
set tics font "Sans,8"
array ymin[3] = [40, 100, 150]
array ymax[3] = [400, 3000, 8000]
do for [p=0:2] {
  set lmargin at screen panel_x(p) + 0.02; set rmargin at screen panel_x(p) + 0.28
  set bmargin at screen 0.12; set tmargin at screen 0.80
  set yrange [ymin[p + 1]:ymax[p + 1]]
  set label 10 sprintf("{/:Bold %s}", titles[p + 1]) at graph 0.02, 0.97 left textcolor rgb ink
  set xlabel "GPUs" textcolor rgb muted font "Sans,8"
  set ylabel (p == 0 ? "GFLOP/s" : "") textcolor rgb muted font "Sans,8"
  if (p > 0) { unset key; unset label 1; unset label 2; unset label 3; unset label 4 }
  # References: the CUDA kernel (r = 1) and cuBLAS (r = 4) on one GPU as horizontal lines, named in the legend
  plot data index 12 + 4 * p using (0.8):3:(3.8):(0) with vectors nohead linecolor rgb light dashtype 2 notitle, \
       data index 12 + 4 * p + 3 using (0.8):3:(3.8):(0) with vectors nohead linecolor rgb light dashtype 3 notitle, \
       for [j=1:4] data index 4 * p + j - 1 using 1:3 with linespoints linecolor rgb colors[j] linewidth 2 \
           pointtype pts[j] pointsize 0.9 title keynames[j], \
       NaN with lines linecolor rgb light dashtype 2 title "CUDA kernel on one GPU", \
       NaN with lines linecolor rgb light dashtype 3 title "cuBLAS on one GPU"
}
unset multiplot
