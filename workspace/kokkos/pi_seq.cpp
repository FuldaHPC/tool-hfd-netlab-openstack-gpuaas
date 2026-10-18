// pi with the midpoint rule and Kokkos::parallel_reduce in the Serial execution space (one CPU core),
// as reference for pi_cpu.cpp (OpenMP) and pi_gpu.cpp (CUDA): only the execution space differs.
#include "common.h"
#include <Kokkos_Core.hpp>

using Space = Kokkos::Serial;

double pi_kokkos(long n) {
  const double h = 1.0 / n;
  double sum = 0.0;
  Kokkos::parallel_reduce(
      "pi", Kokkos::RangePolicy<Space>(0, n),
      KOKKOS_LAMBDA(const long i, double &s) {
        const double x = (i + 0.5) * h;
        s += 4.0 / (1.0 + x * x);
      },
      sum);
  return h * sum;
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  int status;
  {
    double start = seconds_now();
    double pi = pi_kokkos(PI_STEPS);
    double seconds = seconds_now() - start;
    char label[100];
    snprintf(label, sizeof label, "CPU sequential (Kokkos %s)", Space::name());
    status = pi_report(label, pi, PI_STEPS, seconds);
  }
  Kokkos::finalize();
  return status;
}
