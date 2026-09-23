#include "ops.hpp"
#include "graph.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <limits>
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

  return reshape_op(g, x, {x->shape[0], x->size() / x->shape[0]});
}

Tensor *conv2d(Graph *g, Tensor *x, Tensor *k, int stride, int padding) {
  assert(x->shape.size() == 4);
  assert(k->shape.size() == 4);
  // assert(x->shape[0] == 1);
  // assert(x->shape[1] == 1);
  // assert(k->shape[0] == 1);
  // assert(k->shape[1] == 1);
  assert(x->shape[1] == k->shape[1]);
  assert(stride >= 1 && padding >= 0);

  int bs = x->shape[0]; // batch size
  int n_filters = k->shape[0];
  int H = x->shape[2];
  int W = x->shape[3];
  int Kh = k->shape[2];
  int Kw = k->shape[3];
  assert(H + 2 * padding >= Kh && W + 2 * padding >= Kw);
  int Oh = (H + 2 * padding - Kh) / stride + 1;
  int Ow = (W + 2 * padding - Kw) / stride + 1;
  int n_channels = k->shape[1];

  Tensor *out =
      g->make({bs, n_filters, Oh, Ow}, x->requires_grad || k->requires_grad);

  // Forward pass
  for (int n = 0; n < bs; n++) {
    for (int f = 0; f < n_filters; f++) {
      for (int i = 0; i < Oh; i++) {
        for (int j = 0; j < Ow; j++) {
          float tot = 0.0f;
          for (int c = 0; c < n_channels; c++) {
            for (int u = 0; u < Kh; u++) {
              int hi = stride * i + u - padding;
              if (hi < 0 || hi >= H)
                continue;
              for (int v = 0; v < Kw; v++) {
                int wi = stride * j + v - padding;
                if (wi < 0 || wi >= W)
                  continue;
                tot += x->at(n, c, hi, wi) * k->at(f, c, u, v);
              }
            }
          }
          out->at(n, f, i, j) = tot;
        }
      }
    }
  }
  // Backward pass
  out->parents = {x, k};
  out->backward_fn = [x, k, n_filters, n_channels, bs, out, Kh, Kw, Oh, Ow,
                      stride, padding, H, W]() {
    for (int n = 0; n < bs; n++) {
      for (int f = 0; f < n_filters; f++) {
        for (int i = 0; i < Oh; i++) {
          for (int j = 0; j < Ow; j++) {
            float out_grad = out->grad_at(n, f, i, j);
            for (int c = 0; c < n_channels; c++) {
              for (int u = 0; u < Kh; u++) {
                int hi = stride * i + u - padding;
                if (hi < 0 || hi >= H)
                  continue;
                for (int v = 0; v < Kw; v++) {
                  int wi = stride * j + v - padding;
                  if (wi < 0 || wi >= W)
                    continue;
                  k->grad_at(f, c, u, v) += out_grad * x->at(n, c, hi, wi);
                  x->grad_at(n, c, hi, wi) += out_grad * k->at(f, c, u, v);
                }
              }
            }
          }
        }
      }
    }
  };

  return out;
}

Tensor *add_channel_bias(Graph *g, Tensor *x, Tensor *b) {
  assert(x->shape[1] == b->shape[0]);
  Tensor *out = g->make(x->shape);

  int bs = x->shape[0];
  int channels = x->shape[1];
  int H = x->shape[2];
  int W = x->shape[3];

  // Forward pass
  for (int n = 0; n < bs; n++) {
    for (int c = 0; c < channels; c++) {
      for (int i = 0; i < H; i++) {
        float bc = b->data[c];
        for (int j = 0; j < W; j++) {
          out->at(n, c, i, j) = x->at(n, c, i, j) + bc;
        }
      }
    }
  }

  out->parents = {x, b};
  out->backward_fn = [x, b, out, bs, channels, H, W]() {
    for (int n = 0; n < bs; n++) {
      for (int c = 0; c < channels; c++) {
        float b_grad = 0.0f;
        for (int i = 0; i < H; i++) {
          for (int j = 0; j < W; j++) {
            b_grad += out->grad_at(n, c, i, j);
            x->grad_at(n, c, i, j) += out->grad_at(n, c, i, j);
          }
        }
        b->grad[c] += b_grad;
      }
    }
  };
  return out;
}

Tensor *im2col(Graph *g, Tensor *x, int Kh, int Kw, int stride, int padding) {
  assert(x->shape.size() == 4);

  assert(stride >= 1 && padding >= 0);

  int bs = x->shape[0]; // batch size
  int H = x->shape[2];
  int W = x->shape[3];
  assert(H + 2 * padding >= Kh && W + 2 * padding >= Kw);
  int Oh = (H + 2 * padding - Kh) / stride + 1;
  int Ow = (W + 2 * padding - Kw) / stride + 1;
  int C_in = x->shape[1];

  Tensor *out = g->make({bs * Oh * Ow, C_in * Kh * Kw}, x->requires_grad);
  for (int n = 0; n < bs; n++) {
    for (int i = 0; i < Oh; i++) {
      for (int j = 0; j < Ow; j++) {
        int r = n * Oh * Ow + i * Ow + j;
        for (int c = 0; c < C_in; c++) {
          for (int u = 0; u < Kh; u++) {
            int hi = stride * i + u - padding;
            for (int v = 0; v < Kw; v++) {
              int wi = stride * j + v - padding;
              int q = c * Kh * Kw + u * Kw + v;
              out->at(r, q) = (hi < 0 || hi >= H || wi < 0 || wi >= W)
                                  ? 0.0f
                                  : x->at(n, c, hi, wi);
            }
          }
        }
      }
    }
  }
  out->parents = {x};
  out->backward_fn = [x, out, Kh, Kw, stride, padding, bs, Oh, Ow, C_in, H,
                      W]() {
    for (int n = 0; n < bs; n++) {
      for (int i = 0; i < Oh; i++) {
        for (int j = 0; j < Ow; j++) {
          int r = n * Oh * Ow + i * Ow + j;
          for (int c = 0; c < C_in; c++) {
            for (int u = 0; u < Kh; u++) {
              int hi = stride * i + u - padding;
              for (int v = 0; v < Kw; v++) {
                int wi = stride * j + v - padding;
                int q = c * Kh * Kw + u * Kw + v;
                if (hi < 0 || hi >= H || wi < 0 || wi >= W)
                  continue;
                x->grad_at(n, c, hi, wi) += out->grad_at(r, q);
              }
            }
          }
        }
      }
    }
  };
  return out;
}

Tensor *reshape_op(Graph *g, Tensor *x, const std::vector<int> &new_shape) {
  assert(x->size() == product(new_shape));
  Tensor *out = g->make(new_shape, x->requires_grad);

  // forward pass
  std::copy(x->data.begin(), x->data.end(), out->data.begin());

  // backward pass
  out->parents = {x};
  out->backward_fn = [x, out]() {
    const int n = x->size();
    for (int i = 0; i < n; i++) {
      x->grad[i] += out->grad[i];
    }
  };
  return out;
}

Tensor *permute_nhwc_to_nchw(Graph *g, Tensor *x) {
  assert(x->shape.size() == 4);

  int N = x->shape[0], H = x->shape[1], W = x->shape[2], C = x->shape[3];

  Tensor *out = g->make({N, C, H, W}, x->requires_grad);

  for (int n = 0; n < N; n++) {
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        for (int c = 0; c < C; c++) {
          out->at(n, c, h, w) = x->at(n, h, w, c);
        }
      }
    }
  }
  out->parents = {x};
  out->backward_fn = [x, out, N, C, H, W]() {
    for (int n = 0; n < N; n++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          for (int c = 0; c < C; c++) {
            x->grad_at(n, h, w, c) += out->grad_at(n, c, h, w);
          }
        }
      }
    }
  };
  return out;
}

Tensor *conv2d_im2col(Graph *g, Tensor *x, Tensor *k, int stride, int padding) {
  assert(x->shape.size() == 4);
  assert(k->shape.size() == 4);
  assert(x->shape[1] == k->shape[1]);
  assert(stride >= 1 && padding >= 0);

  int N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
  int F = k->shape[0], Kh = k->shape[2], Kw = k->shape[3];
  assert(H + 2 * padding >= Kh && W + 2 * padding >= Kw);

  int Oh = (H + 2 * padding - Kh) / stride + 1;
  int Ow = (W + 2 * padding - Kw) / stride + 1;

  Tensor *cols = im2col(g, x, Kh, Kw, stride, padding); // (N*Oh*Ow, C*Kh*Kw)
  Tensor *kf = reshape_op(g, k, {F, C * Kh * Kw});
  Tensor *kt = transpose(g, kf);   // (C*Kh*Kw, F)
  Tensor *r = matmul(g, cols, kt); // (N*Oh*Ow, F)
  Tensor *rr = reshape_op(g, r, {N, Oh, Ow, F});
  return permute_nhwc_to_nchw(g, rr);
}

Tensor *maxpool2d(Graph *g, Tensor *x, int Kh, int Kw, int stride) {
  assert(stride >= 1);

  int N = x->shape[0], C = x->shape[1], H = x->shape[2], W = x->shape[3];
  assert(H >= Kh && W >= Kw);
  int Oh = (H - Kh) / stride + 1;
  int Ow = (W - Kw) / stride + 1;

  Tensor *out = g->make({N, C, Oh, Ow}, x->requires_grad);
  std::vector<int> argmax(out->size());
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int i = 0; i < Oh; i++) {
        for (int j = 0; j < Ow; j++) {
          float running_max = -std::numeric_limits<float>::infinity();
          int amax; // local argmax
          for (int u = 0; u < Kh; u++) {
            int hi = stride * i + u;
            for (int v = 0; v < Kw; v++) {
              int wi = stride * j + v;
              float p = x->at(n, c, hi, wi);
              if (p > running_max) {
                running_max = p;
                amax = n * x->strides[0] + c * x->strides[1] +
                       hi * x->strides[2] + wi * x->strides[3];
                int out_idx = n * out->strides[0] + c * out->strides[1] +
                              i * out->strides[2] + j * out->strides[3];
                out->data[out_idx] = running_max;
                argmax[out_idx] = amax;
              }
            }
          }
        }
      }
    }
  }
  out->parents = {x};
  out->backward_fn = [x, out, argmax]() {
    for (size_t idx = 0; idx < argmax.size(); idx++) {
      x->grad[argmax[idx]] += out->grad[idx];
    }
  };
  return out;
}
} // namespace deeplib