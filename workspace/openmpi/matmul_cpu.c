// Matrix product C = A B split over the MPI ranks by rows: MPI_Scatter sends each rank its rows of A,
// MPI_Bcast sends B to all, every rank computes its rows of C, and MPI_Gather collects C on rank 0.
// The time includes the communication.
#include "common.h"
#include <mpi.h>

static void matmul_mpi(long n, int size, const float *a, float *b, float *c, float *a_rows, float *c_rows) {
  const long rows = n / size;
  MPI_Scatter(a, rows * n, MPI_FLOAT, a_rows, rows * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(b, n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
  for (long i = 0; i < rows; ++i) {
    for (long j = 0; j < n; ++j) c_rows[i * n + j] = 0.0f;
    for (long k = 0; k < n; ++k) {
      const float aik = a_rows[i * n + k];
      for (long j = 0; j < n; ++j) c_rows[i * n + j] += aik * b[k * n + j];
    }
  }
  MPI_Gather(c_rows, rows * n, MPI_FLOAT, c, rows * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
  const long n = MATRIX_N;
  int rank, size, status = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (n % size != 0) {
    if (rank == 0) printf("The number of ranks must divide %ld.\n", n);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  float *a = NULL, *c = NULL, *b = alloc_floats(n * n);
  float *a_rows = alloc_floats(n * n / size), *c_rows = alloc_floats(n * n / size);
  if (rank == 0) {
    a = alloc_floats(n * n);
    c = alloc_floats(n * n);
    matrix_init(a, b, n);
  }
  matmul_mpi(64, size, a, b, c, a_rows, c_rows);  // warm-up: sets up the communication between the ranks
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();
  matmul_mpi(n, size, a, b, c, a_rows, c_rows);
  double seconds = MPI_Wtime() - start;
  if (rank == 0) {
    char label[100];
    snprintf(label, sizeof label, "CPU (MPI, %d rank%s)", size, size > 1 ? "s" : "");
    status = matrix_report(label, a, b, c, n, seconds);
  }
  MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
  free(a);
  free(b);
  free(c);
  free(a_rows);
  free(c_rows);
  MPI_Finalize();
  return status;
}
