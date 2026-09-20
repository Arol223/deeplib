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
  Tensor x({1, 1, 3, 3});
  for (int i = 0; i < 9; i++)
    x.data[i] = i + 1; // 1..9

  Tensor k({1, 1, 2, 2});
  k.fill(1.0f);

  Graph g;
  conv2d(&g, &x, &k)->print();
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
  test_activations(); // relu, sigmoid, tanh, composed
  test_losses();      // mse, softmax_cross_entropy
  test_layers();      // Linear
  test_mnist();

  return 0;
}