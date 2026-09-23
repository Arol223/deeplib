#pragma once
#include "graph.hpp"
#include "tensor.hpp"
#include <vector>

namespace deeplib {

Tensor *add(Graph *g, Tensor *a, Tensor *b);

Tensor *sub(Graph *g, Tensor *a, Tensor *b);

Tensor *mul(Graph *g, Tensor *a, Tensor *b);

Tensor *mul_scalar(Graph *g, Tensor *a, float s);

Tensor *matmul(Graph *g, Tensor *a, Tensor *b);

Tensor *transpose(Graph *g, Tensor *a);

Tensor *sum(Graph *g, Tensor *a);

Tensor *add_bias(Graph *g, Tensor *x, Tensor *b);

Tensor *flatten(Graph *g, Tensor *x);

Tensor *conv2d(Graph *g, Tensor *x, Tensor *k, int stride = 1, int padding = 0);

Tensor *add_channel_bias(Graph *g, Tensor *x, Tensor *b); // x (N,C,H,W), b (C)

Tensor *im2col(Graph *g, Tensor *x, int Kh, int Kw, int stride = 1,
               int padding = 0);

Tensor *reshape_op(Graph *g, Tensor *x, const std::vector<int> &new_shape);

Tensor *permute_nhwc_to_nchw(Graph *g, Tensor *x);

Tensor *conv2d_im2col(Graph *g, Tensor *x, Tensor *k, int stride = 1,
                      int padding = 0);

Tensor *maxpool2d(Graph *g, Tensor *x, int Kh, int Kw, int stride);

} // namespace deeplib