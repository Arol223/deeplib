#pragma once
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <thread>
#include <vector>

namespace deeplib {
inline int num_threads() {
  static const int n = [] {
    if (const char *s = std::getenv("DEEPLIB_THREADS"))
      return std::max(1, std::atoi(s));
    return std::min(
        8, static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
  }();
  return n;
}

// Split [begin, end) into contiguous chunks and run fn(chunk_begin, chunk_end)
// on each, in parallel. Returns once every chunk has finished.
//
// Contract for fn:
//   - it must only write memory no other chunk writes (reads may overlap)
//   - it must not call Graph::make, which is not thread-safe
//   - it must not throw
template <typename F>
void parallel_for(int begin, int end, F fn, int min_per_thread = 1) {
  const int total = end - begin;
  if (total <= 0)
    return;

  const int n_threads =
      std::min(num_threads(), std::max(1, total / min_per_thread));

  if (n_threads == 1) {
    fn(begin, end);
    return;
  }

  const int chunk = (total + n_threads - 1) / n_threads; // ceiling division
  std::vector<std::jthread> workers;
  workers.reserve(n_threads - 1);
  for (int t = 1; t < n_threads; t++) {
    const int lo = begin + t * chunk;
    const int hi = std::min(end, lo + chunk);
    if (lo >= hi)
      break;
    workers.emplace_back(fn, lo, hi);
  }
  fn(begin, std::min(end, begin + chunk)); // this thread takes chunk 0
} // Each jthread joins in its destructor, here

} // namespace deeplib