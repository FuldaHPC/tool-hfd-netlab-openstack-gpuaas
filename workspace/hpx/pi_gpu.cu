// pi with the midpoint rule in a CUDA kernel driven by HPX: the memset, the kernel launch and the copy
// back go through an hpx::cuda::experimental::cuda_executor, and the copy returns an hpx::future
// (pattern of HPX's own async_cuda tests). The kernel is the same as in cuda/pi_gpu.cu.
#include "common.h"
#include <hpx/future.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/modules/async_cuda.hpp>
#include <cstddef>

namespace cu = hpx::cuda::experimental;

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

double pi_hpx_gpu(cu::cuda_executor &exec, long n) {
  double h = 1.0 / n, sum, *d_sum;
  cudaMalloc(reinterpret_cast<void **>(&d_sum), sizeof(double));
  void *args[] = {&n, &h, &d_sum};
  hpx::post(exec, cudaMemsetAsync, d_sum, 0, sizeof(double));
  hpx::post(exec, cudaLaunchKernel<void>, reinterpret_cast<void *>(&pi_kernel), dim3(1024), dim3(256), args,
            std::size_t(0));
  hpx::future<void> copied = hpx::async(exec, cudaMemcpyAsync, &sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
  copied.get();
  cudaFree(d_sum);
  cuda_check();
  return h * sum;
}

int main() {
  cu::enable_user_polling poll("default");  // lets HPX complete futures of GPU work
  cu::cuda_executor exec(0, cu::event_mode{});
  pi_hpx_gpu(exec, 1000);  // warm-up: CUDA context and kernel
  double start = seconds_now();
  double pi = pi_hpx_gpu(exec, PI_STEPS);
  double seconds = seconds_now() - start;
  return pi_report("GPU (CUDA kernel via HPX cuda_executor)", pi, PI_STEPS, seconds);
}
