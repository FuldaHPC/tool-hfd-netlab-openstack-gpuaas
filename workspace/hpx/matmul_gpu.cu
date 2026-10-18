// Matrix product C = A B in a CUDA kernel driven by HPX: the copies of A and B to the GPU, the kernel
// launch and the copy of C back go through an hpx::cuda::experimental::cuda_executor in one CUDA
// stream; the last copy returns an hpx::future. The time includes allocating GPU memory. The kernel
// is the same as in cuda/matmul_gpu.cu.
#include "common.h"
#include <hpx/future.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/modules/async_cuda.hpp>
#include <cstddef>
#include <vector>

namespace cu = hpx::cuda::experimental;

__global__ void matmul_kernel(long n, const float *a, const float *b, float *c) {
  long i = blockIdx.y * (long)blockDim.y + threadIdx.y, j = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < n && j < n) {
    float s = 0.0f;
    for (long k = 0; k < n; ++k) s += a[i * n + k] * b[k * n + j];
    c[i * n + j] = s;
  }
}

void matmul_hpx_gpu(cu::cuda_executor &exec, long n, const float *a, const float *b, float *c) {
  const std::size_t bytes = n * n * sizeof(float);
  float *d_a, *d_b, *d_c;
  cudaMalloc(reinterpret_cast<void **>(&d_a), bytes);
  cudaMalloc(reinterpret_cast<void **>(&d_b), bytes);
  cudaMalloc(reinterpret_cast<void **>(&d_c), bytes);
  void *args[] = {&n, &d_a, &d_b, &d_c};
  hpx::post(exec, cudaMemcpyAsync, d_a, a, bytes, cudaMemcpyHostToDevice);
  hpx::post(exec, cudaMemcpyAsync, d_b, b, bytes, cudaMemcpyHostToDevice);
  hpx::post(exec, cudaLaunchKernel<void>, reinterpret_cast<void *>(&matmul_kernel),
            dim3((n + 15) / 16, (n + 15) / 16), dim3(16, 16), args, std::size_t(0));
  hpx::future<void> copied = hpx::async(exec, cudaMemcpyAsync, c, d_c, bytes, cudaMemcpyDeviceToHost);
  copied.get();
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cuda_check();
}

int main() {
  cu::enable_user_polling poll("default");  // lets HPX complete futures of GPU work
  cu::cuda_executor exec(0, cu::event_mode{});
  const long n = MATRIX_N;
  std::vector<float> a(n * n), b(n * n), c(n * n);
  matrix_init(a.data(), b.data(), n);
  matmul_hpx_gpu(exec, 16, a.data(), b.data(), c.data());  // warm-up: CUDA context and kernel
  double start = seconds_now();
  matmul_hpx_gpu(exec, n, a.data(), b.data(), c.data());
  double seconds = seconds_now() - start;
  return matrix_report("GPU (CUDA kernel via HPX cuda_executor)", a.data(), b.data(), c.data(), n, seconds);
}
