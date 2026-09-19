#include "graph.hpp"
#include "tensor.hpp"

namespace deeplib {

Tensor *relu(Graph *g, Tensor *x);

Tensor *sigmoid(Graph *g, Tensor *x);

Tensor *tanh(Graph *g, Tensor *x);
} // namespace deeplib