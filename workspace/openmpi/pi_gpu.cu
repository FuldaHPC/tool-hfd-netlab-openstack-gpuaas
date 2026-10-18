// pi with the midpoint rule split over the MPI ranks, each rank's block summed by a CUDA kernel on one
// of the node's GPUs (round robin; make run starts one rank per GPU). MPI_Allreduce adds the partial sums directly in GPU memory, which needs a
// CUDA-aware MPI; the program checks that with MPIX_Query_cuda_support().
#include "common.h"
#include <cuda_runtime.h>
#include <mpi.h>
#include <mpi-ext.h>

__global__ void pi_kernel(long begin, long end, double h, double *sum) {
  __shared__ double block[256];
  double s = 0.0;
  for (long i = begin + blockIdx.x * (long)blockDim.x + threadIdx.x; i < end; i += (long)gridDim.x * blockDim.x) {
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

static double pi_mpi_gpu(long n, int rank, int size) {
  const double h = 1.0 / n;
  double sum, *d_part, *d_sum;
  cudaMalloc(&d_part, sizeof(double));
  cudaMalloc(&d_sum, sizeof(double));
  cudaMemset(d_part, 0, sizeof(double));
  pi_kernel<<<1024, 256>>>(rank * n / size, (rank + 1) * n / size, h, d_part);
  cudaDeviceSynchronize();
  MPI_Allreduce(d_part, d_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  // GPU buffers
  cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_part);
  cudaFree(d_sum);
  cuda_check();
  return h * sum;
}

int main(int argc, char **argv) {
  int rank, size, cuda_aware = 0, status = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  cuda_aware = MPIX_Query_cuda_support();
#endif
  if (!cuda_aware) {  // before any GPU buffer reaches MPI, which would crash
    if (rank == 0) printf("This MPI is not CUDA-aware.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  // Every rank uses one GPU of its node, round robin, so that several GPUs share the work.
  MPI_Comm node;
  int local_rank, devices = 0;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
  MPI_Comm_rank(node, &local_rank);
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    if (rank == 0) printf("No CUDA device found\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  cudaSetDevice(local_rank % devices);
  const int gpus = devices < size ? devices : size;
  pi_mpi_gpu(1000, rank, size);  // warm-up: CUDA context, kernel, MPI connections
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();
  double pi = pi_mpi_gpu(PI_STEPS, rank, size);
  double seconds = MPI_Wtime() - start;
  if (rank == 0) {
    char label[100];
    snprintf(label, sizeof label, "GPU (MPI, %d rank%s on %d GPU%s with CUDA kernels, CUDA-aware)", size,
             size > 1 ? "s" : "", gpus, gpus > 1 ? "s" : "");
    status = pi_report(label, pi, PI_STEPS, seconds);
  }
  MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return status;
}
