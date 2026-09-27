#include "activations.hpp"
#include "graph.hpp"
#include "parallel.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cmath>
#include <iterator>

namespace deeplib {
Tensor *relu(Graph *g, Tensor *x) {
  Tensor *out = g->make(x->shape, x->requires_grad, false);

  int n = x->size();
  parallel_for(
      0, n,
      [&](int lo, int hi) {
        for (int i = 0; i < std::ssize(x->data); i++) {
          out->data[i] = std::max(0.0f, x->data[i]);
        }
      },
      32768);

  out->parents = {x};
  out->backward_fn = [x, n, out]() {
    parallel_for(
        0, n,
        [&](int lo, int hi) {
          for (int i = lo; i < hi; i++) {
            x->grad[i] +=
                (x->data[i] > 0.0f) ? out->grad[i] : 0.0f; // out->grad[i];
          }
        },
        32768);
  };
  return out;
}

Tensor *sigmoid(Graph *g, Tensor *x) {
  Tensor *out = g->make(x->shape, x->requires_grad);

  for (int i = 0; i < std::ssize(x->data); i++) {
    out->data[i] = 1 / (1 + std::exp(-x->data[i]));
  }

  out->parents = {x};
  out->backward_fn = [x, out]() {
    for (int i = 0; i < std::ssize(x->grad); i++) {
      x->grad[i] += out->grad[i] * out->data[i] * (1 - out->data[i]);
    }
  };
  return out;
}

Tensor *tanh(Graph *g, Tensor *x) {
  Tensor *out = g->make(x->shape, x->requires_grad);

  for (int i = 0; i < std::ssize(x->data); i++) {
    out->data[i] = std::tanh(x->data[i]);
  }

  out->parents = {x};
  out->backward_fn = [x, out] {
    for (int i = 0; i < std::ssize(x->grad); i++) {
      x->grad[i] += out->grad[i] * (1 - out->data[i] * out->data[i]);
    }
  };
  return out;
}
} // namespace deeplib