# Plots the results of run_all.sh into <dir>/results.pdf: one A4 page per problem with a horizontal bar
# per program (GFLOP/s on a logarithmic axis, the time next to the bar), grouped by folder and coloured
# by sequential, parallel CPU and GPU. Usage: gnuplot -e "dir='results'; machine='...'" plot_results.gp
# (problems='pi' plots only one page; run_all.sh leaves out problems without correct results).
if (!exists("dir")) dir = "results"
if (!exists("machine")) machine = ""
if (!exists("problems")) problems = "pi matmul"

# butt, not rounded: the zero-length y tics of the header and gap rows would show as dots.
set terminal pdfcairo size 21cm,29.7cm font "Sans,8" enhanced butt linewidth 0.8
set output dir . "/results.pdf"
set datafile separator "\t"

# Colours: grey for sequential, blue for the CPU cores, orange for the GPU.
color(kind) = kind eq "seq" ? 0x9a9a9a : kind eq "cpu" ? 0x4c72b0 : 0xdd8452
rate(v) = v >= 100 ? sprintf("%.0f", v) : v >= 1 ? sprintf("%.1f", v) : sprintf("%.2f", v)
band = 0xf4f4f4
ink = 0x333333

do for [problem in problems] {
  reset
  set datafile separator "\t"
  file = dir . "/" . problem . ".tsv"
  stats file using (strcol(1) eq "bar" ? $3 : NaN) nooutput
  vmin = STATS_min; vmax = STATS_max
  stats file using 2 nooutput
  rows = STATS_max + 1
  xmin = 10**floor(log10(vmin / 1.5)); xmax = vmax * 4

  set multiplot
  set label 1 (problem eq "pi" ? "{/:Bold=15 π by the midpoint rule}" : "{/:Bold=15 Matrix product C = A·B}") \
    at screen 0.04, 0.965 left textcolor rgb ink
  set label 2 (problem eq "pi" ? "10^9 steps in double precision, 6 floating-point operations per step" \
                               : "two 2048 × 2048 matrices in single precision, the same simple algorithm everywhere") \
    at screen 0.04, 0.947 left textcolor rgb 0x666666 font "Sans,9"
  set label 3 machine at screen 0.04, 0.933 left textcolor rgb 0x888888 font "Sans,8"
  set label 4 "GPU times include allocating GPU memory and the transfers to and from the GPU." \
    at screen 0.93, 0.022 right textcolor rgb 0x888888 font "Sans,7"
  set xlabel "GFLOP/s (logarithmic)" textcolor rgb 0x666666 font "Sans,8" offset 0, 0.3

  set lmargin at screen 0.30; set rmargin at screen 0.93
  set tmargin at screen 0.905; set bmargin at screen 0.06
  set logscale x
  set xrange [xmin:xmax]
  set yrange [rows - 0.4:-0.6]
  set format x "%.0f"
  set xtics nomirror textcolor rgb ink scale 0.5
  set mxtics 10
  set x2tics nomirror format "%.0f" textcolor rgb 0x888888 scale 0.5
  set x2range [xmin:xmax]; set logscale x2
  set ytics scale 0 nomirror textcolor rgb ink font "Sans,7.5" noenhanced
  set grid xtics mxtics lc rgb 0xe2e2e2 lw 0.6, lc rgb 0xeeeeee lw 0.4 back
  set border 0
  unset key
  set style fill solid 1.0 noborder

  plot file using (strcol(1) eq "band" && $6 == 1 ? xmin : NaN):(($2 + $3) / 2):(xmin):(xmax):($2):($3) \
         with boxxyerror fc rgb band, \
       file using (strcol(1) eq "bar" ? xmin : NaN):2:(xmin):3:($2 - 0.34):($2 + 0.34):(color(strcol(5))): \
         ytic(strcol(1) eq "bar" ? strcol(6) : "") with boxxyerror fc rgb variable, \
       file using (strcol(1) eq "bar" ? $3 * 1.07 : NaN):2:(rate($3) . " GFLOP/s, " . sprintf("%.3f s", $4)) \
         with labels left font "Sans,6.5" textcolor rgb 0x555555 noenhanced, \
       file using (strcol(1) eq "header" ? xmin : NaN):2:(sprintf("{/:Bold %s}", strcol(6))) \
         with labels left offset character -33, 0 font "Sans,8.5" textcolor rgb ink

  # Legend in the free space at the top right
  unset label; unset grid; unset border; unset tics; unset logscale; unset x2tics; unset xlabel
  set lmargin at screen 0.62; set rmargin at screen 0.96; set tmargin at screen 0.975; set bmargin at screen 0.925
  set xrange [0:1]; set yrange [0:1]
  set key at screen 0.95, 0.968 right top vertical samplen 1.2 spacing 1.15 font "Sans,8" textcolor rgb ink noautotitle
  plot NaN with boxes fc rgb color("seq") title "sequential (1 core)", \
       NaN with boxes fc rgb color("cpu") title "parallel on the CPU cores", \
       NaN with boxes fc rgb color("gpu") title "GPU"
  unset multiplot
}
