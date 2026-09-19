#include "losses.hpp"
#include "graph.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>

namespace deeplib {

Tensor *mse(Graph *g, Tensor *pred, Tensor *target) {

  assert(pred->shape == target->shape);
  Tensor *out = g->make({1}, pred->requires_grad);

  const int n = pred->size();
  float total = 0;
  for (int i = 0; i < n; i++) {
    float res = (pred->data[i] - target->data[i]);
    total += res * res;
  }
  out->data[0] = total / n;
  out->parents = {pred};

  out->backward_fn = [pred, target, out, n]() {
    const float scale = 2.0f / n;
    for (int i = 0; i < std::ssize(pred->grad); i++) {
      pred->grad[i] += out->grad[0] * scale * (pred->data[i] - target->data[i]);
    }
  };
  return out;
}

Tensor *softmax_cross_entropy(Graph *g, Tensor *logits, Tensor *target) {
  assert(logits->data.size() == target->data.size());

  Tensor *out = g->make({1}, logits->requires_grad);

  // Forward pass
  int n = std::ssize(logits->data);
  std::vector<float> p(n);
  float m = *std::max_element(logits->data.begin(), logits->data.end());

  float partition = 0.0f;

  for (int i = 0; i < n; i++) {
    p[i] = std::exp(logits->data[i] - m);
    partition += p[i];
  }
  for (int i = 0; i < n; i++) {
    p[i] /= partition;
  }
  float log_partition = std::log(partition);
  float tot = 0;
  for (int i = 0; i < n; i++) {
    tot -= target->data[i] * (logits->data[i] - m - log_partition);
  }
  out->data[0] = tot;

  // Backward pass
  out->parents = {logits};
  out->backward_fn = [logits, target, out, p, n]() {
    for (int i = 0; i < n; i++) {
      logits->grad[i] += out->grad[0] * (p[i] - target->data[i]);
    }
  };

  return out;
}

} // namespace deeplib