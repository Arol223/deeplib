#include "losses.hpp"
#include "graph.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <vector>

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
  assert(logits->shape.size() == 2);
  assert(logits->shape == target->shape);

  Tensor *out = g->make({1}, logits->requires_grad);
  out->data[0] = 0.0f;
  // Forward pass
  int b = logits->shape[0]; // batch size
  int n = logits->shape[1]; // n_logits

  std::vector<float> p(n * b); // probabilities
  for (int s = 0; s < b; s++) {
    float m = *std::max_element(logits->data.begin() + s * n, // max logit
                                logits->data.begin() + (s + 1) * n);

    float partition = 0.0f; // softmax partition function

    for (int i = 0; i < n; i++) {
      p[s * n + i] = std::exp(logits->at(s, i) - m);
      partition += p[s * n + i];
    }
    for (int i = 0; i < n; i++) {
      p[s * n + i] /= partition;
    }
    float log_partition = std::log(partition);
    float tot = 0;
    for (int i = 0; i < n; i++) {
      tot -= target->at(s, i) * (logits->at(s, i) - m - log_partition);
    }
    out->data[0] += tot;
  }
  out->data[0] /= b;
  // Backward pass
  out->parents = {logits};
  out->backward_fn = [logits, target, out, p, b, n]() {
    const float scale = 1.0f / b;
    for (int s = 0; s < b; s++) {
      for (int i = 0; i < n; i++) {
        logits->grad_at(s, i) +=
            out->grad[0] * scale * (p[s * n + i] - target->at(s, i));
      }
    }
  };

  return out;
}

} // namespace deeplib