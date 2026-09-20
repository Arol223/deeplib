#include "ops.hpp"
#include "graph.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cstddef>
#include <iterator>
#include <vector>

namespace deeplib {

Tensor *add(Graph *g, Tensor *a, Tensor *b) {
  assert(a->shape == b->shape);
  bool requires_grad = a->requires_grad || b->requires_grad;
  Tensor *out = g->make(a->shape, requires_grad);

  const size_t n = a->data.size();
  for (size_t i = 0; i < n; i++) {
    out->data[i] = a->data[i] + b->data[i];
  }
  out->parents = {a, b};

  out->backward_fn = [a, b, out]() {
    for (size_t i = 0; i < out->grad.size(); i++) {
      a->grad[i] += out->grad[i];
      b->grad[i] += out->grad[i];
    }
  };
  return out;
}

Tensor *sub(Graph *g, Tensor *a, Tensor *b) {
  assert(a->shape == b->shape);
  bool requires_grad = a->requires_grad || b->requires_grad;
  Tensor *out = g->make(a->shape, requires_grad);

  const size_t n = a->data.size();

  for (size_t i = 0; i < n; i++) {
    out->data[i] = a->data[i] - b->data[i];
  }
  out->parents = {a, b};

  out->backward_fn = [a, b, out]() {
    for (size_t i = 0; i < out->grad.size(); i++) {
      a->grad[i] += out->grad[i];
      b->grad[i] -= out->grad[i];
    }
  };
  return out;
}

Tensor *mul(Graph *g, Tensor *a, Tensor *b) {
  assert(a->shape == b->shape);
  bool requires_grad = a->requires_grad || b->requires_grad;
  Tensor *out = g->make(a->shape, requires_grad);

  const size_t n = a->data.size();

  for (size_t i = 0; i < n; i++) {
    out->data[i] = a->data[i] * b->data[i];
  }

  out->parents = {a, b};

  out->backward_fn = [a, b, out]() {
    for (size_t i = 0; i < out->grad.size(); i++) {
      a->grad[i] += out->grad[i] * b->data[i];
      b->grad[i] += out->grad[i] * a->data[i];
    }
  };
  return out;
}

Tensor *mul_scalar(Graph *g, Tensor *a, float s) {
  bool requires_grad = a->requires_grad;
  Tensor *out = g->make(a->shape, requires_grad);

  const size_t n = a->data.size();

  for (size_t i = 0; i < n; i++) {
    out->data[i] = a->data[i] * s;
  }
  out->parents = {a};
  out->backward_fn = [a, s, out]() {
    for (size_t i = 0; i < out->grad.size(); i++) {
      a->grad[i] += out->grad[i] * s;
    }
  };
  return out;
}

Tensor *matmul(Graph *g, Tensor *a, Tensor *b) {
  assert(a->shape.size() == 2);
  assert(b->shape.size() == 2);
  assert(a->shape[1] == b->shape[0]);

  bool requires_grad = a->requires_grad || b->requires_grad;
  const int out_col = b->shape[1];
  const int out_row = a->shape[0];
  const int inner = a->shape[1];

  Tensor *out = g->make({a->shape[0], b->shape[1]}, requires_grad);

  for (int i = 0; i < out_row; i++) {
    float *__restrict out_row = out->data.data() + i * out->strides[0];
    for (int p = 0; p < inner; p++) {
      const float a_ip = a->at(i, p);
      const float *__restrict b_row = b->data.data() + p * b->strides[0];
#pragma GCC ivdep
      for (int j = 0; j < out_col; j++) {
        out_row[j] += a_ip * b_row[j];
      }
    }
  }

  out->parents = {a, b};
  out->backward_fn = [a, b, out]() {
    int m = a->shape[0];
    int k = a->shape[1];
    int n = b->shape[1];
    for (int i = 0; i < m; i++) {
      const float *__restrict g_row = out->grad.data() + i * out->strides[0];
      for (int p = 0; p < k; p++) {
        float acc = 0.0f;
        const float *__restrict b_row = b->data.data() + p * b->strides[0];
#pragma GCC ivdep
        for (int j = 0; j < n; j++) {
          acc += g_row[j] * b_row[j];
        }
        a->grad_at(i, p) += acc;
      }
    }

    for (int i = 0; i < m; i++) {
      const float *__restrict grad_row = out->grad.data() + i * out->strides[0];
      for (int p = 0; p < k; p++) {
        const float a_ip = a->at(i, p);
        float *__restrict bg_row = b->grad.data() + p * b->strides[0];
#pragma GCC ivdep
        for (int j = 0; j < n; j++) {
          bg_row[j] += a_ip * grad_row[j];
        }
      }
    }
  };

  return out;
}

Tensor *transpose(Graph *g, Tensor *a) {
  assert(a->shape.size() == 2);
  bool requires_grad = a->requires_grad;
  int out_rows = a->shape[1];
  int out_cols = a->shape[0];

  Tensor *out = g->make({out_rows, out_cols}, requires_grad);
  for (int i = 0; i < out_rows; i++) {
    for (int j = 0; j < out_cols; j++) {
      out->at(i, j) = a->at(j, i);
    }
  }
  out->parents = {a};
  out->backward_fn = [a, out]() {
    for (int i = 0; i < a->shape[0]; i++) {
      for (int j = 0; j < a->shape[1]; j++) {
        a->grad_at(i, j) += out->grad_at(j, i);
      }
    }
  };
  return out;
}

Tensor *sum(Graph *g, Tensor *a) {
  Tensor *out = g->make({1}, a->requires_grad);
  float s = 0;
  for (int i = 0; i < std::ssize(a->data); i++) {
    s += a->data[i];
  }
  out->data[0] = s;

  out->parents = {a};
  out->backward_fn = [a, out]() {
    for (size_t i = 0; i < a->grad.size(); i++) {
      a->grad[i] += out->grad[0];
    }
  };
  return out;
}

Tensor *add_bias(Graph *g, Tensor *x, Tensor *b) {
  assert(x->shape.size() == 2);
  assert(b->shape.size() == 1);
  assert(x->shape[1] == b->shape[0]);
  Tensor *out = g->make(x->shape, b->requires_grad || x->requires_grad);

  // forward pass
  for (int i = 0; i < x->shape[0]; i++) {
    for (int j = 0; j < x->shape[1]; j++) {
      out->at(i, j) = x->at(i, j) + b->data[j];
    }
  }
  out->parents = {x, b};
  out->backward_fn = [x, b, out]() {
    for (int j = 0; j < x->shape[1]; j++) {
      float partial_b = 0;
      for (int i = 0; i < x->shape[0]; i++) {
        partial_b += out->grad_at(i, j);
        x->grad_at(i, j) += out->grad_at(i, j);
      }
      b->grad[j] += partial_b;
    }
  };

  return out;
}

Tensor *flatten(Graph *g, Tensor *x) {
  int bs = x->shape[0];
  int flat_size = x->size() / bs;
  int batch_stride = x->strides[0];
  Tensor *out = g->make({bs, flat_size}, x->requires_grad);
  for (int b = 0; b < bs; b++) {
    for (int i = 0; i < flat_size; i++) {
      out->at(b, i) = x->data[b * batch_stride + i];
    }
  }
  out->parents = {x};
  out->backward_fn = [x, out, bs, flat_size, batch_stride]() {
    for (int b = 0; b < bs; b++) {
      for (int i = 0; i < flat_size; i++) {
        x->grad[b * batch_stride + i] += out->grad_at(b, i);
      }
    }
  };

  return out;
}

Tensor *conv2d(Graph *g, Tensor *x, Tensor *k) {
  assert(x->shape.size() == 4);
  assert(k->shape.size() == 4);
  assert(x->shape[0] == 1);
  // assert(x->shape[1] == 1);
  assert(k->shape[0] == 1);
  // assert(k->shape[1] == 1);

  int H = x->shape[2];
  int W = x->shape[3];
  int Kh = k->shape[2];
  int Kw = k->shape[3];
  int Oh = H - Kh + 1;
  int Ow = W - Kw + 1;
  Tensor *out = g->make({1, 1, Oh, Ow}, x->requires_grad || k->requires_grad);

  // Forward pass
  for ()
    for (int i = 0; i < Oh; i++) {
      for (int j = 0; j < Ow; j++) {
        float tot = 0.0f;
        for (int u = 0; u < Kh; u++) {
          for (int v = 0; v < Kw; v++) {
            tot += x->at(0, 0, i + u, j + v) * k->at(0, 0, u, v);
          }
        }
        out->at(0, 0, i, j) = tot;
      }
    }

  // Backward pass
  out->parents = {x, k};
  out->backward_fn = [x, k, out, Kh, Kw, Oh, Ow]() {
    for (int i = 0; i < Oh; i++) {
      for (int j = 0; j < Ow; j++) {
        for (int u = 0; u < Kh; u++) {
          for (int v = 0; v < Kw; v++) {
            k->grad_at(0, 0, u, v) +=
                out->grad_at(0, 0, i, j) * x->at(0, 0, i + u, j + v);
            x->grad_at(0, 0, i + u, j + v) +=
                out->grad_at(0, 0, i, j) * k->at(0, 0, u, v);
          }
        }
      }
    }
  };

  return out;
}

} // namespace deeplib