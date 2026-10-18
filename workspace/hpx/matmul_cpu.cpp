// Matrix product C = A B with HPX's parallel algorithm for_loop over the rows of C on the CPU worker
// threads (execution policy hpx::execution::par). The loop order i-k-j lets the innermost loop run
// along rows of B and C.
#include "common.h"
#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/runtime.hpp>
#include <vector>

void matmul(long n, const float *a, const float *b, float *c) {
  hpx::experimental::for_loop(hpx::execution::par, 0L, n, [=](long i) {
    for (long j = 0; j < n; ++j) c[i * n + j] = 0.0f;
    for (long k = 0; k < n; ++k) {
      const float aik = a[i * n + k];
      for (long j = 0; j < n; ++j) c[i * n + j] += aik * b[k * n + j];
    }
  });
}

int main() {
  const long n = MATRIX_N;
  std::vector<float> a(n * n), b(n * n), c(n * n);
  matrix_init(a.data(), b.data(), n);
  matmul(16, a.data(), b.data(), c.data());  // warm-up
  double start = seconds_now();
  matmul(n, a.data(), b.data(), c.data());
  double seconds = seconds_now() - start;
  char label[100];
  snprintf(label, sizeof label, "CPU (HPX for_loop, %zu worker threads)",
           static_cast<std::size_t>(hpx::get_os_thread_count()));
  return matrix_report(label, a.data(), b.data(), c.data(), n, seconds);
}
