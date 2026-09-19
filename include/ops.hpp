#pragma once
#include "graph.hpp"
#include "tensor.hpp"

namespace deeplib {

Tensor *add(Graph *g, Tensor *a, Tensor *b);

Tensor *sub(Graph *g, Tensor *a, Tensor *b);

Tensor *mul(Graph *g, Tensor *a, Tensor *b);

Tensor *mul_scalar(Graph *g, Tensor *a, float s);

Tensor *matmul(Graph *g, Tensor *a, Tensor *b);

Tensor *transpose(Graph *g, Tensor *a);

Tensor *sum(Graph *g, Tensor *a);

} // namespace deeplib