// Shared by the C and C++ examples of this folder: problem sizes, test matrices, timer and result
// checks, so that the programs themselves only show the programming model.
#ifndef GPUAAS_COMMON_H
#define GPUAAS_COMMON_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PI_STEPS 1000000000L  // steps of the midpoint rule for pi
#define PI_FLOPS_PER_STEP 6   // x = (i + 0.5) * h; sum += 4 / (1 + x * x): 3 adds, 2 multiplies, 1 division
#define MATRIX_N 2048L        // the matrices are MATRIX_N x MATRIX_N floats

#ifdef __clang__
#define COMPILER "Clang"
#else
#define COMPILER "GCC"
#endif

static inline double seconds_now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

// pi is the integral of 4/(1+x^2) over [0,1]; the midpoint rule sums 4/(1+x^2) at x = (i+0.5)/n.
// Rounding makes the error about 1e-13; one missing step would add at least 2e-9.
static inline int pi_report(const char *label, double pi, long n, double seconds) {
  double error = fabs(pi - M_PI);
  printf("%s: pi = %.12f (error %.1e) in %.3f s (%.1f GFLOP/s)\n", label, pi, error, seconds,
         (double)PI_FLOPS_PER_STEP * n / seconds / 1e9);
  return error < 1e-9 ? 0 : 1;
}

// Allocates count floats and writes them once: the system maps fresh memory only when it is first
// written, and these page faults (about 8 ms for 16 MB) must not fall into a timed region.
static inline float *alloc_floats(long count) {
  float *p = (float *)malloc(count * sizeof(float));
  memset(p, 0, count * sizeof(float));
  return p;
}

// Small integers, so that every product and sum in C = A B is exact in single precision.
static inline void matrix_init(float *a, float *b, long n) {
  for (long i = 0; i < n; ++i)
    for (long j = 0; j < n; ++j) {
      a[i * n + j] = (float)((i + 2 * j) % 8);
      b[i * n + j] = (float)((3 * i + j) % 8);
    }
}

// Checks C x == A (B x) for x = (1, 2, ..., n) in O(n^2); exact, as all values are integers below 2^53.
// x has no zero, so that every element of C counts.
static inline int matrix_correct(const float *a, const float *b, const float *c, long n) {
  double *bx = (double *)malloc(n * sizeof(double));
  int ok = 1;
  for (long k = 0; k < n; ++k) {
    double s = 0.0;
    for (long j = 0; j < n; ++j) s += (double)b[k * n + j] * (j + 1);
    bx[k] = s;
  }
  for (long i = 0; i < n && ok; ++i) {
    double cx = 0.0, abx = 0.0;
    for (long j = 0; j < n; ++j) {
      cx += (double)c[i * n + j] * (j + 1);
      abx += (double)a[i * n + j] * bx[j];
    }
    ok = cx == abx;
  }
  free(bx);
  return ok;
}

static inline int matrix_report(const char *label, const float *a, const float *b, const float *c, long n,
                                double seconds) {
  int ok = matrix_correct(a, b, c, n);
  printf("%s: %ldx%ld matrix product %s in %.3f s (%.1f GFLOP/s)\n", label, n, n, ok ? "correct" : "WRONG", seconds,
         2.0 * n * n * n / seconds / 1e9);
  return ok ? 0 : 1;
}

#ifdef __CUDACC__
#include <cuda_runtime.h>
// Stops with the reason if a CUDA call or kernel failed (a kernel's error shows up in a later call),
// instead of printing a wrong result.
static inline void cuda_check(void) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    exit(1);
  }
}
#endif

#endif
