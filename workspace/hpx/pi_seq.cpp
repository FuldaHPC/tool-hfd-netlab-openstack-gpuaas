// pi with the midpoint rule in HPX's for_loop with the sequential execution policy hpx::execution::seq
// (one CPU core) and a reduction, as reference for pi_cpu.cpp (tasks) and pi_gpu.cu.
#include "common.h"
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/hpx_main.hpp>

double pi_seq(long n) {
  const double h = 1.0 / n;
  double sum = 0.0;
  hpx::experimental::for_loop(hpx::execution::seq, 0L, n, hpx::experimental::reduction_plus(sum),
                              [=](long i, double &s) {
                                const double x = (i + 0.5) * h;
                                s += 4.0 / (1.0 + x * x);
                              });
  return h * sum;
}

int main() {
  double start = seconds_now();
  double pi = pi_seq(PI_STEPS);
  return pi_report("CPU sequential (HPX for_loop seq)", pi, PI_STEPS, seconds_now() - start);
}
