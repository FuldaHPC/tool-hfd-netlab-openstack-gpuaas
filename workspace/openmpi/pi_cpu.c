// pi with the midpoint rule split over the MPI ranks: every rank sums a contiguous block of the
// steps, and MPI_Reduce adds the partial sums on rank 0.
#include "common.h"
#include <mpi.h>

static double pi_mpi(long n, int rank, int size) {
  const double h = 1.0 / n;
  const long begin = rank * n / size, end = (rank + 1) * n / size;
  double part = 0.0, sum = 0.0;
  for (long i = begin; i < end; ++i) {
    double x = (i + 0.5) * h;
    part += 4.0 / (1.0 + x * x);
  }
  MPI_Reduce(&part, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  return h * sum;
}

int main(int argc, char **argv) {
  int rank, size, status = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  pi_mpi(1000, rank, size);  // warm-up: sets up the communication between the ranks
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();
  double pi = pi_mpi(PI_STEPS, rank, size);
  double seconds = MPI_Wtime() - start;
  if (rank == 0) {
    char label[100];
    snprintf(label, sizeof label, "CPU (MPI, %d rank%s)", size, size > 1 ? "s" : "");
    status = pi_report(label, pi, PI_STEPS, seconds);
  }
  MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Finalize();
  return status;
}
