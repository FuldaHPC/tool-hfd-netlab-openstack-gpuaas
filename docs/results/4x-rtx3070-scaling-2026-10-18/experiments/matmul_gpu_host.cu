// Matrix product C = A B split over the MPI ranks by rows, each rank's rows computed by a CUDA kernel on
// one of the node's GPUs (round robin; make run starts one rank per GPU). The CUDA-aware MPI moves the data directly to and from GPU memory:
// MPI_Scatterv puts the rows of A into GPU buffers, MPI_Bcast sends B from GPU memory to GPU memory,
// and MPI_Gatherv collects C from the GPU buffers on rank 0 (Scatterv and Gatherv, so that the number of
// ranks need not divide n, e.g. three GPUs). The time includes all of it, also allocating the GPU buffers.
#include "common.h"
#include <cuda_runtime.h>
#include <mpi.h>
#include <mpi-ext.h>

__global__ void matmul_kernel(long rows, long n, const float *a, const float *b, float *c) {
  long i = blockIdx.y * (long)blockDim.y + threadIdx.y, j = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (i < rows && j < n) {
    float s = 0.0f;
    for (long k = 0; k < n; ++k) s += a[i * n + k] * b[k * n + j];
    c[i * n + j] = s;
  }
}

static void matmul_mpi_gpu(long n, int rank, int size, const float *a, const float *b, float *c) {
  // Experiment: the collectives on host buffers, one cudaMemcpy per rank and direction.
  int *counts = (int *)malloc(size * sizeof(int)), *offsets = (int *)malloc(size * sizeof(int));
  for (int r = 0; r < size; ++r) {
    offsets[r] = (int)(r * n / size * n);
    counts[r] = (int)(((r + 1) * n / size - r * n / size) * n);
  }
  const long rows = counts[rank] / n;
  float *a_rows = (float *)malloc(rows * n * sizeof(float)), *c_rows = (float *)malloc(rows * n * sizeof(float));
  float *b_all = rank == 0 ? (float *)b : (float *)malloc(n * n * sizeof(float));
  float *d_a_rows, *d_b, *d_c_rows;
  cudaMalloc(&d_a_rows, rows * n * sizeof(float));
  cudaMalloc(&d_b, n * n * sizeof(float));
  cudaMalloc(&d_c_rows, rows * n * sizeof(float));
  MPI_Scatterv(a, counts, offsets, MPI_FLOAT, a_rows, counts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(b_all, n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  cudaMemcpy(d_a_rows, a_rows, rows * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b_all, n * n * sizeof(float), cudaMemcpyHostToDevice);
  dim3 block(16, 16), grid((n + 15) / 16, (rows + 15) / 16);
  matmul_kernel<<<grid, block>>>(rows, n, d_a_rows, d_b, d_c_rows);
  cudaMemcpy(c_rows, d_c_rows, rows * n * sizeof(float), cudaMemcpyDeviceToHost);
  MPI_Gatherv(c_rows, counts[rank], MPI_FLOAT, c, counts, offsets, MPI_FLOAT, 0, MPI_COMM_WORLD);
  cudaFree(d_a_rows);
  cudaFree(d_b);
  cudaFree(d_c_rows);
  free(a_rows);
  free(c_rows);
  if (rank != 0) free(b_all);
  free(counts);
  free(offsets);
  cuda_check();
}

int main(int argc, char **argv) {
  const long n = MATRIX_N;
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
  float *a = NULL, *b = NULL, *c = NULL;
  if (rank == 0) {
    a = (float *)malloc(n * n * sizeof(float));
    b = (float *)malloc(n * n * sizeof(float));
    c = alloc_floats(n * n);
    matrix_init(a, b, n);
  }
  matmul_mpi_gpu(64, rank, size, a, b, c);  // warm-up: CUDA context, kernel, MPI
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();
  matmul_mpi_gpu(n, rank, size, a, b, c);
  double seconds = MPI_Wtime() - start;
  if (rank == 0) {
    char label[100];
    snprintf(label, sizeof label, "GPU (MPI, %d rank%s on %d GPU%s with CUDA kernels, host buffers)", size,
             size > 1 ? "s" : "", gpus, gpus > 1 ? "s" : "");
    status = matrix_report(label, a, b, c, n, seconds);
  }
  MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
  free(a);
  free(b);
  free(c);
  MPI_Finalize();
  return status;
}
