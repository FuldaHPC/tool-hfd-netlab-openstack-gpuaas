// Matrix product C = A B in plain loops on one CPU core (sequential), as reference for matmul_cpu.c
// and matmul_gpu.c. The loop order i-k-j lets the innermost loop run along rows of B and C.
#include "common.h"

int main(void) {
  const long n = MATRIX_N;
  float *a = alloc_floats(n * n), *b = alloc_floats(n * n), *c = alloc_floats(n * n);
  matrix_init(a, b, n);
  double start = seconds_now();
  for (long i = 0; i < n; ++i) {
    for (long j = 0; j < n; ++j) c[i * n + j] = 0.0f;
    for (long k = 0; k < n; ++k) {
      const float aik = a[i * n + k];
      for (long j = 0; j < n; ++j) c[i * n + j] += aik * b[k * n + j];
    }
  }
  double seconds = seconds_now() - start;
  int status = matrix_report("CPU sequential (" COMPILER ")", a, b, c, n, seconds);
  free(a);
  free(b);
  free(c);
  return status;
}
