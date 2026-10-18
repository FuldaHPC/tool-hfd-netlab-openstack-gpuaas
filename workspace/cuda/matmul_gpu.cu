// Matrix product C = A B in a CUDA kernel with one thread per element of C. The time includes
// allocating GPU memory, copying A and B to the GPU and C back.
#include "common.h"
#include <cuda_runtime.h>
#include <vector>

__global__ void matmul_kernel(long n, const float *a, const float *b, float *c) {
  long i = blockIdx.y * (long)blockDim.y + threadIdx.y, j = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < n && j < n) {
    float s = 0.0f;
    for (long k = 0; k < n; ++k) s += a[i * n + k] * b[k * n + j];
    c[i * n + j] = s;
  }
}

static void matmul_gpu(long n, const float *a, const float *b, float *c) {
  size_t bytes = n * n * sizeof(float);
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
  dim3 block(16, 16), grid((n + 15) / 16, (n + 15) / 16);
  matmul_kernel<<<grid, block>>>(n, d_a, d_b, d_c);
  cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);  // waits for the kernel
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cuda_check();
}

int main() {
  const long n = MATRIX_N;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    printf("No CUDA device found\n");
    return 1;
  }
  std::vector<float> a(n * n), b(n * n), c(n * n);
  matrix_init(a.data(), b.data(), n);
  matmul_gpu(16, a.data(), b.data(), c.data());  // warm-up: CUDA context and kernel
  double start = seconds_now();
  matmul_gpu(n, a.data(), b.data(), c.data());
  double seconds = seconds_now() - start;
  char label[300];
  snprintf(label, sizeof label, "GPU (CUDA kernel, %s)", prop.name);
  return matrix_report(label, a.data(), b.data(), c.data(), n, seconds);
}
