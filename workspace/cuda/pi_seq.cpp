// pi with the midpoint rule in a plain C++ loop on one CPU core (sequential), as reference for pi_gpu.cu.
#include "common.h"

int main() {
  const long n = PI_STEPS;
  const double h = 1.0 / n;
  double start = seconds_now(), sum = 0.0;
  for (long i = 0; i < n; ++i) {
    double x = (i + 0.5) * h;
    sum += 4.0 / (1.0 + x * x);
  }
  return pi_report("CPU sequential (C++)", h * sum, n, seconds_now() - start);
}
