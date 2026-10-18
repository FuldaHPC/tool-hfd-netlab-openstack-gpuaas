// Matrix product C = A B with an OpenMP parallel loop over the rows of C on the CPU threads.
// The loop order i-k-j lets the innermost loop run along rows of B and C.
#include "common.h"
#include <omp.h>

static void matmul_cpu(long n, const float *a, const float *b, float *c) {
  #pragma omp parallel for
  for (long i = 0; i < n; ++i) {
    for (long j = 0; j < n; ++j) c[i * n + j] = 0.0f;
    for (long k = 0; k < n; ++k) {
      const float aik = a[i * n + k];
      for (long j = 0; j < n; ++j) c[i * n + j] += aik * b[k * n + j];
    }
  }
}

int main(void) {
  const long n = MATRIX_N;
  float *a = alloc_floats(n * n), *b = alloc_floats(n * n), *c = alloc_floats(n * n);
  matrix_init(a, b, n);
  matmul_cpu(16, a, b, c);  // warm-up: starts the OpenMP threads
  double start = seconds_now();
  matmul_cpu(n, a, b, c);
  double seconds = seconds_now() - start;
  char label[100];
  snprintf(label, sizeof label, "CPU (%s OpenMP, %d threads)", COMPILER, omp_get_max_threads());
  int status = matrix_report(label, a, b, c, n, seconds);
  free(a);
  free(b);
  free(c);
  return status;
}
