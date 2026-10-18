// pi with the midpoint rule in an OpenMP target region on the GPU.
// Run with OMP_TARGET_OFFLOAD=MANDATORY, so that the program fails instead of falling back to the CPU.
#include "common.h"
#include <omp.h>

static double pi_gpu(long n) {
  const double h = 1.0 / n;
  double sum = 0.0;
  #pragma omp target teams distribute parallel for reduction(+ : sum) map(tofrom : sum)
  for (long i = 0; i < n; ++i) {
    double x = (i + 0.5) * h;
    sum += 4.0 / (1.0 + x * x);
  }
  return h * sum;
}

int main(void) {
  int on_host = 1;
  #pragma omp target map(tofrom : on_host)
  on_host = omp_is_initial_device();
  pi_gpu(1000);  // warm-up: initializes the GPU and loads the GPU code
  double start = seconds_now();
  double pi = pi_gpu(PI_STEPS);
  double seconds = seconds_now() - start;
  char label[100];
  if (on_host)
    snprintf(label, sizeof label, "Host (no offloading!)");
  else
    snprintf(label, sizeof label, "GPU (%s OpenMP target)", COMPILER);
  return pi_report(label, pi, PI_STEPS, seconds) || on_host;
}
