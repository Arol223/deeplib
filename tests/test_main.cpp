#include "activations.hpp"
#include "dataset.hpp"
#include "grad_check.hpp"
#include "graph.hpp"
#include "layers.hpp"
#include "losses.hpp"
#include "ops.hpp"
#include "tensor.hpp"
#include <iostream>

using namespace deeplib;

void test_ops() {

  // 0. Create randomly initialized tensors x, b and graph g
  Tensor *x = new Tensor({3, 4}, true);
  x->randomize();
  Tensor *b = new Tensor({3, 4}, true);
  b->randomize();

  Graph g;
  // 1. Verify grad checker itself, L = sum(x^2), so dL/dx = 2x
  grad_check(
      &g, x, [](Graph *gr, Tensor *t) { return sum(gr, mul(gr, t, t)); },
      2e-4f);
  x->zero_grad();
  b->zero_grad();
  g.clear();

  // 2. add
  grad_check(
      &g, x, [b](Graph *gr, Tensor *t) { return sum(gr, add(gr, t, b)); },
      2e-4f);
  x->zero_grad();
  b->zero_grad();

  // 3. matmul - tricky one...
  Tensor *A = new Tensor({2, 3}, true);
  Tensor *B = new Tensor({3, 4}, true);
  A->randomize();
  B->randomize();

  grad_check(
      &g, A, [B](Graph *gr, Tensor *t) { return sum(gr, matmul(gr, t, B)); },
      2e-4f);
  A->zero_grad();
  B->zero_grad();

  grad_check(
      &g, B, [A](Graph *gr, Tensor *t) { return sum(gr, matmul(gr, A, t)); },
      2e-4f);

  // 4 Transpose
  Tensor *T = new Tensor({2, 5}, true);
  T->randomize();
  grad_check(&g, T,
             [](Graph *gr, Tensor *t) { return sum(gr, transpose(gr, t)); });

  // 7. Add bias
  Tensor *xb = new Tensor({3, 4}, true); // batch of 3
  Tensor *bb = new Tensor({4}, true);
  xb->randomize();
  bb->randomize();

  grad_check(&g, xb, [bb](Graph *gr, Tensor *t) {
    return sum(gr, add_bias(gr, t, bb));
  });
  grad_check(&g, bb, [xb](Graph *gr, Tensor *t) {
    return sum(gr, add_bias(gr, xb, t));
  });

  // 8. conv2d
  Tensor *in = new Tensor({1, 1, 5, 5});
  Tensor *k = new Tensor({1, 1, 3, 3});
  in->randomize();
  k->randomize();

  grad_check(&g, in,
             [k](Graph *gr, Tensor *t) { return sum(gr, conv2d(gr, t, k)); });
  grad_check(&g, k,
             [in](Graph *gr, Tensor *t) { return sum(gr, conv2d(gr, in, t)); });
}

void test_activations() {
  Graph g;
  // 5. activations — fresh tensors each, so no zero_grad needed
  Tensor *r = new Tensor({3, 4}, true);
  r->randomize();
  grad_check(&g, r, [](Graph *gr, Tensor *t) { return sum(gr, relu(gr, t)); });

  Tensor *s1 = new Tensor({3, 4}, true);
  s1->randomize();
  grad_check(&g, s1,
             [](Graph *gr, Tensor *t) { return sum(gr, sigmoid(gr, t)); });

  Tensor *t1 = new Tensor({3, 4}, true);
  t1->randomize();
  grad_check(&g, t1, [](Graph *gr, Tensor *t) { return sum(gr, tanh(gr, t)); });

  // 5. composed — incoming gradient is non-uniform, so a missing
  //    out->grad in a backward_fn can't hide behind sum's ones
  Tensor *s2 = new Tensor({3, 4}, true);
  s2->randomize();
  grad_check(&g, s2, [](Graph *gr, Tensor *t) {
    return sum(gr, mul(gr, sigmoid(gr, t), t));
  });

  Tensor *t2 = new Tensor({3, 4}, true);
  t2->randomize();
  grad_check(&g, t2, [](Graph *gr, Tensor *t) {
    return sum(gr, mul(gr, tanh(gr, t), t));
  });
}

void test_conv2d() {
  std::cout << "Testing conv2d gives expected shape\n";
  Tensor x({1, 3, 3, 3});
  for (int i = 0; i < 9; i++)
    x.data[i] = i + 1; // 1..9

  Tensor k({2, 3, 2, 2});
  k.fill(1.0f);

  Graph g;
  conv2d(&g, &x, &k)->print();
}

void test_conv2d_batch() {
  const int N = 2, C = 2, F = 2, H = 5, W = 5, K = 3;
  const int Oh = H - K + 1, Ow = W - K + 1;

  Tensor x({N, C, H, W});
  Tensor k({F, C, K, K});
  x.randomize();
  k.randomize();

  Graph g;
  Tensor *out = conv2d(&g, &x, &k); // (N, F, Oh, Ow)

  // 1. Consistency: each batched slice must equal the unbatched conv
  //    of that sample alone. The verified single-sample conv is the oracle.
  Tensor xs({1, C, H, W});
  bool ok = true;
  for (int n = 0; n < N; n++) {
    std::copy(x.data.begin() + n * x.strides[0],
              x.data.begin() + (n + 1) * x.strides[0], xs.data.begin());
    Tensor *ref = conv2d(&g, &xs, &k); // (1, F, Oh, Ow)

    for (int i = 0; i < ref->size(); i++) {
      float batched = out->data[n * out->strides[0] + i];
      if (std::abs(batched - ref->data[i]) > 1e-6f)
        ok = false;
    }
  }
  std::cout << "conv2d batch consistency: " << (ok ? "PASSED" : "FAILED")
            << "\n";

  // 2. Gradients, with the output weighted by a fixed random tensor.
  Tensor xg({N, C, H, W}, true);
  Tensor kg({F, C, K, K}, true);
  Tensor w({N, F, Oh, Ow});
  xg.randomize();
  kg.randomize();
  w.randomize();

  Tensor *xp = &xg;
  Tensor *kp = &kg;
  Tensor *wp = &w;
  grad_check(&g, xp, [kp, wp](Graph *gr, Tensor *t) {
    return sum(gr, mul(gr, conv2d(gr, t, kp), wp));
  });
  grad_check(&g, kp, [xp, wp](Graph *gr, Tensor *t) {
    return sum(gr, mul(gr, conv2d(gr, xp, t), wp));
  });
}

void test_conv2d_channels() {
  // 2 input channels, 2 filters, 3x3 input, 2x2 kernels -> (1, 2, 2, 2)
  Tensor x({1, 2, 3, 3});
  for (int h = 0; h < 3; h++)
    for (int w = 0; w < 3; w++) {
      x.at(0, 0, h, w) = 1.0f;
      x.at(0, 1, h, w) = 2.0f;
    }

  // filter 0 reads channel 0 only; filter 1 reads both.
  // Asymmetric in (f, c) so an index swap changes the answer.
  Tensor k({2, 2, 2, 2});
  for (int u = 0; u < 2; u++)
    for (int v = 0; v < 2; v++) {
      k.at(0, 0, u, v) = 1.0f;
      k.at(0, 1, u, v) = 0.0f;
      k.at(1, 0, u, v) = 1.0f;
      k.at(1, 1, u, v) = 1.0f;
    }

  Graph g;
  Tensor *out = conv2d(&g, &x, &k);

  // expected at every spatial position: filter 0 -> 4*1 = 4,
  // filter 1 -> 4*1 + 4*2 = 12
  bool ok = true;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++) {
      if (out->at(0, 0, i, j) != 4.0f)
        ok = false;
      if (out->at(0, 1, i, j) != 12.0f)
        ok = false;
    }
  std::cout << "conv2d channels: " << (ok ? "PASSED" : "FAILED") << "\n";
  if (!ok)
    out->print();
}

namespace {
bool all_close(const Tensor *a, const Tensor *b, float tol = 1e-6f) {
  if (a->shape != b->shape)
    return false;
  for (size_t i = 0; i < a->data.size(); i++)
    if (std::abs(a->data[i] - b->data[i]) > tol)
      return false;
  return true;
}
} // namespace

void test_conv2d_stride_padding() {
  // Non-square everywhere, so an H/W or Kh/Kw swap can't hide.
  const int N = 2, C = 2, F = 2, H = 7, W = 8, Kh = 3, Kw = 2;

  Tensor x({N, C, H, W});
  Tensor k({F, C, Kh, Kw});
  x.randomize();
  k.randomize();
  Graph g;

  // 1. Stride: stride-2 output is the stride-1 output at even positions.
  {
    Tensor *s1 = conv2d(&g, &x, &k, 1, 0); // (2, 2, 5, 7)
    Tensor *s2 = conv2d(&g, &x, &k, 2, 0); // (2, 2, 3, 4)

    // Expected shape computed by hand: (7-3)/2+1 = 3, (8-2)/2+1 = 4
    bool ok = s2->shape[2] == 3 && s2->shape[3] == 4;
    for (int n = 0; n < N; n++)
      for (int f = 0; f < F; f++)
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 4; j++)
            if (std::abs(s2->at(n, f, i, j) - s1->at(n, f, 2 * i, 2 * j)) >
                1e-6f)
              ok = false;
    std::cout << "conv2d stride: " << (ok ? "PASSED" : "FAILED") << "\n";
  }

  // 2. Padding: conv with padding P equals unpadded conv on an
  //    explicitly zero-padded copy of the input.
  {
    const int P = 1;
    Tensor xpad({N, C, H + 2 * P, W + 2 * P}); // constructor zero-fills
    for (int n = 0; n < N; n++)
      for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
          for (int w = 0; w < W; w++)
            xpad.at(n, c, h + P, w + P) = x.at(n, c, h, w);

    Tensor *padded = conv2d(&g, &x, &k, 1, P);
    Tensor *manual = conv2d(&g, &xpad, &k, 1, 0);
    std::cout << "conv2d padding: "
              << (all_close(padded, manual) ? "PASSED" : "FAILED") << "\n";
  }

  // 3. Gradients with stride and padding together, weighted loss.
  {
    constexpr int S = 2, P = 1;
    const int Oh = (H + 2 * P - Kh) / S + 1; // (7+2-3)/2+1 = 4
    const int Ow = (W + 2 * P - Kw) / S + 1; // (8+2-2)/2+1 = 5

    Tensor xg({N, C, H, W}, true);
    Tensor kg({F, C, Kh, Kw}, true);
    Tensor wt({N, F, Oh, Ow});
    xg.randomize();
    kg.randomize();
    wt.randomize();

    Tensor *xgp = &xg;
    Tensor *kgp = &kg;
    Tensor *wp = &wt;
    grad_check(&g, xgp, [kgp, wp](Graph *gr, Tensor *t) {
      return sum(gr, mul(gr, conv2d(gr, t, kgp, S, P), wp));
    });
    grad_check(&g, kgp, [xgp, wp](Graph *gr, Tensor *t) {
      return sum(gr, mul(gr, conv2d(gr, xgp, t, S, P), wp));
    });
  }
}

void test_losses() {
  // 6. loss functions
  // a) MSE
  Graph g;
  Tensor *p = new Tensor({3, 4}, true);
  Tensor *y = new Tensor({3, 4}, false);
  set_seed(128);
  p->randomize();
  y->randomize();
  grad_check(&g, p, [y](Graph *gr, Tensor *t) { return mse(gr, t, y); });

  // b) softmax-crossentropy
  Tensor *z = new Tensor({1, 5}, true);
  z->randomize();
  Tensor *y_2 = new Tensor({1, 5}, false);
  y_2->data[2] = 1.0f; // rest are zero from the constructor

  grad_check(&g, z, [y_2](Graph *gr, Tensor *t) {
    return softmax_cross_entropy(gr, t, y_2);
  });

  Tensor *z_nd = new Tensor({3, 4}, true);
  z_nd->randomize();
  Tensor *yd = new Tensor({3, 4}, false);
  yd->at({0, 1}) = 1.0f;
  yd->at({1, 3}) = 1.0f;
  yd->at({2, 0}) = 1.0f;

  grad_check(&g, z_nd, [yd](Graph *gr, Tensor *t) {
    return softmax_cross_entropy(gr, t, yd);
  });
}

void test_layers() {
  Graph g;
  // 8. Test Linear
  Linear layer(3, 2);

  Tensor *x2 = new Tensor({1, 3}, false);
  x2->randomize();

  Tensor *out = layer.forward(&g, x2);

  Tensor *loss = sum(&g, out);
  loss->backward();

  out->print(); // expect shape (1, 2)

  for (Tensor *p : layer.parameters()) {
    p->print(); // grads should be non-zero
  }
}

void test_mnist() {
  try {
    Dataset train = load_mnist("data/train-images-idx3-ubyte",
                               "data/train-labels-idx1-ubyte");
    std::cout << "n=" << train.n << " image_size=" << train.n_features
              << " first_label=" << train.labels[0] << "\n";
    print_ascii(train, 0);
  } catch (const std::exception &e) {
    std::cerr << "MNIST test skipped: " << e.what() << "\n";
  }
}

int main() {
  test_ops(); // add, matmul, transpose, add_bias
  test_conv2d();
  test_conv2d_channels();
  test_conv2d_batch();
  test_conv2d_stride_padding();
  test_activations(); // relu, sigmoid, tanh, composed
  test_losses();      // mse, softmax_cross_entropy
  test_layers();      // Linear
  test_mnist();

  return 0;
}