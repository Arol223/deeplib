#pragma once
#include "graph.hpp"
#include "tensor.hpp"

namespace deeplib {

void grad_check(Graph *g, Tensor *x,
                std::function<Tensor *(Graph *, Tensor *)> f, float eps = 1e-4f,
                float tol = 2e-2f);
} // namespace deeplib