// Matrix product C = A B in an OpenMP target region on the GPU: the rows of C are distributed over the
// teams and threads, and "simd" spreads each row over the 32 lanes of a warp (GCC maps OpenMP threads
// to whole warps; Clang is faster with collapse(2) instead, see llvm-offload/).
// map(to:) copies A and B to the GPU and map(from:) copies C back; the time includes it.
// Run with OMP_TARGET_OFFLOAD=MANDATORY, so that the program fails instead of falling back to the CPU.
#include "common.h"
#include <omp.h>

static void matmul_gpu(long n, const float *a, const float *b, float *c) {
  #pragma omp target teams distribute parallel for map(to : a[0 : n * n], b[0 : n * n]) map(from : c[0 : n * n])
  for (long i = 0; i < n; ++i)
    #pragma omp simd
    for (long j = 0; j < n; ++j) {
      float s = 0.0f;
      for (long k = 0; k < n; ++k) s += a[i * n + k] * b[k * n + j];
      c[i * n + j] = s;
    }
}

int main(void) {
  const long n = MATRIX_N;
  int on_host = 1;
  #pragma omp target map(tofrom : on_host)
  on_host = omp_is_initial_device();
  float *a = alloc_floats(n * n), *b = alloc_floats(n * n), *c = alloc_floats(n * n);
  matrix_init(a, b, n);
  matmul_gpu(16, a, b, c);  // warm-up: initializes the GPU and loads the GPU code
  double start = seconds_now();
  matmul_gpu(n, a, b, c);
  double seconds = seconds_now() - start;
  char label[100];
  if (on_host)
    snprintf(label, sizeof label, "Host (no offloading!)");
  else
    snprintf(label, sizeof label, "GPU (%s OpenMP target)", COMPILER);
  int status = matrix_report(label, a, b, c, n, seconds) || on_host;
  free(a);
  free(b);
  free(c);
  return status;
}
