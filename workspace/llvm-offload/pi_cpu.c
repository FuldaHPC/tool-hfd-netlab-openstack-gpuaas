// pi with the midpoint rule in an OpenMP parallel loop on the CPU threads.
#include "common.h"
#include <omp.h>

static double pi_cpu(long n) {
  const double h = 1.0 / n;
  double sum = 0.0;
  #pragma omp parallel for reduction(+ : sum)
  for (long i = 0; i < n; ++i) {
    double x = (i + 0.5) * h;
    sum += 4.0 / (1.0 + x * x);
  }
  return h * sum;
}

int main(void) {
  pi_cpu(1000);  // warm-up: starts the OpenMP threads
  double start = seconds_now();
  double pi = pi_cpu(PI_STEPS);
  double seconds = seconds_now() - start;
  char label[100];
  snprintf(label, sizeof label, "CPU (%s OpenMP, %d threads)", COMPILER, omp_get_max_threads());
  return pi_report(label, pi, PI_STEPS, seconds);
}
