#include "graph.hpp"
#include "tensor.hpp"

namespace deeplib {

Tensor *Graph::make(std::vector<int> shape, bool requires_grad) {
  // reusable tensor
  if (next < owned.size() && owned[next]->shape == shape) {
    Tensor *t = owned[next++];
    t->requires_grad = requires_grad;
    std::fill(t->grad.begin(), t->grad.end(), 0.0f);
    t->fill(0.0f);
    t->parents.clear();
    t->backward_fn = nullptr;
    return t;
  }

  // No reusable slot: allocate and take over position
  Tensor *t = new Tensor(shape, requires_grad);
  if (next < owned.size()) {
    delete owned[next];
    owned[next] = t;
  } else {
    owned.push_back(t);
  }

  next++;
  return t;
}

void Graph::clear() {
  for (Tensor *t : owned) {
    delete t;
  }
  owned.clear();
  next = 0;
}

Graph::~Graph() { clear(); }
} // namespace deeplib