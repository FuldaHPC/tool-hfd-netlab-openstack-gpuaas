// Matrix product C = A B in plain C++ loops on one CPU core (sequential), as reference for matmul_gpu.cu.
// The loop order i-k-j lets the innermost loop run along rows of B and C.
#include "common.h"
#include <vector>

int main() {
  const long n = MATRIX_N;
  std::vector<float> a(n * n), b(n * n), c(n * n, 0.0f);
  matrix_init(a.data(), b.data(), n);
  double start = seconds_now();
  for (long i = 0; i < n; ++i) {
    for (long j = 0; j < n; ++j) c[i * n + j] = 0.0f;
    for (long k = 0; k < n; ++k) {
      const float aik = a[i * n + k];
      for (long j = 0; j < n; ++j) c[i * n + j] += aik * b[k * n + j];
    }
  }
  return matrix_report("CPU sequential (C++)", a.data(), b.data(), c.data(), n, seconds_now() - start);
}
