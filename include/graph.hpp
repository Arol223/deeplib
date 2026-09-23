#pragma once
#include "tensor.hpp"
#include <cstddef>
#include <vector>

namespace deeplib {

class Graph {
public:
  Tensor *make(std::vector<int> shape, bool requires_grad = false);
  ~Graph();
  void reset() { next = 0; }
  void clear();

  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;
  Graph() = default;

private:
  std::vector<Tensor *> owned;
  size_t next = 0;
};
} // namespace deeplib