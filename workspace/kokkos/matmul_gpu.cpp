// Matrix product C = A B with Kokkos::parallel_for over all elements of C (MDRangePolicy) in the
// default execution space, which is CUDA in this build. The matrices are Views in host memory and in
// GPU memory; Kokkos::deep_copy moves A and B to the GPU and C back. The time includes allocating
// the GPU Views and the copies.
#include "common.h"
#include <Kokkos_Core.hpp>
#include <string>

using Space = Kokkos::DefaultExecutionSpace;
using Matrix = Kokkos::View<float **, Kokkos::LayoutRight, Space::memory_space>;
using HostMatrix = Kokkos::View<float **, Kokkos::LayoutRight, Kokkos::HostSpace>;

// Iterate::Right makes j the fastest index, like the LayoutRight Views, so that neighbouring GPU threads
// access neighbouring elements; the default for CUDA (Left) was 4 times slower.
using Policy = Kokkos::MDRangePolicy<Space, Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;

void matmul(const Matrix &a, const Matrix &b, const Matrix &c) {
  const long n = a.extent(0);
  Kokkos::parallel_for(
      "matmul", Policy({0, 0}, {n, n}), KOKKOS_LAMBDA(const long i, const long j) {
        float s = 0.0f;
        for (long k = 0; k < n; ++k) s += a(i, k) * b(k, j);
        c(i, j) = s;
      });
}

// Allocates the GPU Views (without zero-filling them, like cudaMalloc), copies A and B there, computes C
// and copies it back into the host View c; the time includes all of it.
void matmul_gpu(const HostMatrix &a, const HostMatrix &b, const HostMatrix &c) {
  const long n = a.extent(0);
  Matrix d_a(Kokkos::view_alloc(Kokkos::WithoutInitializing, "A"), n, n),
      d_b(Kokkos::view_alloc(Kokkos::WithoutInitializing, "B"), n, n),
      d_c(Kokkos::view_alloc(Kokkos::WithoutInitializing, "C"), n, n);
  Kokkos::deep_copy(d_a, a);
  Kokkos::deep_copy(d_b, b);
  matmul(d_a, d_b, d_c);
  Kokkos::deep_copy(c, d_c);  // waits for the kernel
}

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  int status;
  {
    const long n = MATRIX_N;
    HostMatrix a("A", n, n), b("B", n, n), c("C", n, n);
    matrix_init(a.data(), b.data(), n);
    matmul_gpu(HostMatrix("a", 16, 16), HostMatrix("b", 16, 16), HostMatrix("c", 16, 16));  // warm-up: GPU, kernel
    double start = seconds_now();
    matmul_gpu(a, b, c);
    double seconds = seconds_now() - start;
    char label[100];
    snprintf(label, sizeof label, "GPU (Kokkos %s)", Space::name());
    status = matrix_report(label, a.data(), b.data(), c.data(), n, seconds) || std::string(Space::name()) != "Cuda";
  }
  Kokkos::finalize();
  return status;
}
