// pi with the midpoint rule in asynchronous HPX tasks on the CPU worker threads: the steps are split
// into four tasks per worker thread, and hpx::when_all combines their futures.
#include "common.h"
#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <vector>

double pi_hpx(long n, long tasks) {
  const double h = 1.0 / n;
  std::vector<hpx::future<double>> parts;
  for (long t = 0; t < tasks; ++t) {
    const long begin = t * n / tasks, end = (t + 1) * n / tasks;
    parts.push_back(hpx::async([=] {
      double s = 0.0;
      for (long i = begin; i < end; ++i) {
        const double x = (i + 0.5) * h;
        s += 4.0 / (1.0 + x * x);
      }
      return s;
    }));
  }
  double sum = 0.0;
  for (auto &part : hpx::when_all(parts).get()) sum += part.get();
  return h * sum;
}

int main() {
  const long threads = hpx::get_os_thread_count(), tasks = 4 * threads;
  pi_hpx(1000, tasks);  // warm-up
  double start = seconds_now();
  double pi = pi_hpx(PI_STEPS, tasks);
  double seconds = seconds_now() - start;
  char label[100];
  snprintf(label, sizeof label, "CPU (HPX, %ld worker threads, %ld tasks)", threads, tasks);
  return pi_report(label, pi, PI_STEPS, seconds);
}
