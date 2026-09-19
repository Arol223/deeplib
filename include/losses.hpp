#pragma once
#include "graph.hpp"
#include "tensor.hpp"

namespace deeplib {

Tensor *mse(Graph *g, Tensor *pred, Tensor *target);

Tensor *softmax_cross_entropy(Graph *g, Tensor *logits, Tensor *target);

} // namespace deeplib