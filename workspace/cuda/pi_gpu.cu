// pi with the midpoint rule in a CUDA kernel: every thread sums every (grid size)-th step, each
// block adds up its threads in shared memory and adds that to the result with one atomicAdd.
#include "common.h"
#include <cuda_runtime.h>

__global__ void pi_kernel(long n, double h, double *sum) {
  __shared__ double block[256];
  double s = 0.0;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
    double x = (i + 0.5) * h;
    s += 4.0 / (1.0 + x * x);
  }
  block[threadIdx.x] = s;
  __syncthreads();
  for (int half = blockDim.x / 2; half > 0; half /= 2) {
    if (threadIdx.x < half) block[threadIdx.x] += block[threadIdx.x + half];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(sum, block[0]);
}

static double pi_gpu(long n) {
  const double h = 1.0 / n;
  double sum, *d_sum;
  cudaMalloc(&d_sum, sizeof(double));
  cudaMemset(d_sum, 0, sizeof(double));
  pi_kernel<<<1024, 256>>>(n, h, d_sum);
  cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);  // waits for the kernel
  cudaFree(d_sum);
  cuda_check();
  return h * sum;
}

int main() {
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    printf("No CUDA device found\n");
    return 1;
  }
  pi_gpu(1000);  // warm-up: creates the CUDA context and loads the kernel
  double start = seconds_now();
  double pi = pi_gpu(PI_STEPS);
  double seconds = seconds_now() - start;
  char label[300];
  snprintf(label, sizeof label, "GPU (CUDA kernel, %s)", prop.name);
  return pi_report(label, pi, PI_STEPS, seconds);
}
