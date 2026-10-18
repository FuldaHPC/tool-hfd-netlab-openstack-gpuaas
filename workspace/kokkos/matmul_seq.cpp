// Matrix product C = A B with Kokkos::parallel_for over the rows of C in the Serial execution space
// (one CPU core), as reference for matmul_cpu.cpp (OpenMP): only the execution space differs. The
// matrices are Kokkos Views in host memory; the loop order i-k-j lets the innermost loop run along
// rows of B and C.
#include "common.h"
#include <Kokkos_Core.hpp>

using Space = Kokkos::Serial;
using Matrix = Kokkos::View<float **, Kokkos::LayoutRight, Space>;

void matmul(const Matrix &a, const Matrix &b, const Matrix &c) {
  const long n = a.extent(0);
  Kokkos::parallel_for(
      "matmul", Kokkos::RangePolicy<Space>(0, n), KOKKOS_LAMBDA(const long i) {
        for (long j = 0; j < n; ++j) c(i, j) = 0.0f;
        for (long k = 0; k < n; ++k) {
          const float aik = a(i, k);
          for (long j = 0; j < n; ++j) c(i, j) += aik * b(k, j);
        }
      });
  Kokkos::fence();
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  int status;
  {
    const long n = MATRIX_N;
    Matrix a("A", n, n), b("B", n, n), c("C", n, n);
    matrix_init(a.data(), b.data(), n);
    double start = seconds_now();
    matmul(a, b, c);
    double seconds = seconds_now() - start;
    char label[100];
    snprintf(label, sizeof label, "CPU sequential (Kokkos %s)", Space::name());
    status = matrix_report(label, a.data(), b.data(), c.data(), n, seconds);
  }
  Kokkos::finalize();
  return status;
}
