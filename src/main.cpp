#include "dataset.hpp"
#include "graph.hpp"
#include "layers.hpp"
#include "losses.hpp"
#include "ops.hpp"
#include "optimizers.hpp"
#include "tensor.hpp"
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace deeplib;

void train_xor() {
  Tensor *x = new Tensor({4, 2});
  Tensor *y = new Tensor({4, 1});

  // 0 XOR 0
  x->at({0, 0}) = 0;
  x->at({0, 1}) = 0;
  y->at({0, 0}) = 0;

  // 0 XOR 1
  x->at({1, 0}) = 0;
  x->at({1, 1}) = 1;
  y->at({1, 0}) = 1;

  // 1 XOR 0
  x->at({2, 0}) = 1;
  x->at({2, 1}) = 0;
  y->at({2, 0}) = 1;

  // 1 XOR 1
  x->at({3, 0}) = 1;
  x->at({3, 1}) = 1;
  y->at({3, 0}) = 0;

  // Building the network
  Sequential net;
  net.add(new Linear(2, 4));
  net.add(new Tanh());
  net.add(new Linear(4, 1));

  // Optimizer
  SGD opt(net.parameters(), 0.1f);
  Graph g;

  for (int epoch = 0; epoch < 2000; epoch++) {
    g.clear();
    opt.zero_grad();
    Tensor *out = net.forward(&g, x);
    Tensor *loss = mse(&g, out, y);
    loss->backward();
    opt.step();

    if (epoch % 200 == 0) {
      std::cout << "Epoch: " << epoch << ": " << "Loss: " << loss->data[0]
                << "\n";
    }
  }
  g.clear();
  net.forward(&g, x)->print();
}

void train_mnist() {
  Dataset train = load_mnist("data/train-images-idx3-ubyte",
                             "data/train-labels-idx1-ubyte");

  const int batch_size = 64;
  const int n_batches = train.n / batch_size; // Drop dangling batch at end

  Sequential net({new Linear(784, 128), new Relu(), new Linear(128, 10)});

  SGD opt(net.parameters(), 0.1f);
  Graph g;

  Tensor x({batch_size, 784});
  Tensor y({batch_size, 10});

  std::vector<int> order(train.n);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(0);

  for (int epoch = 0; epoch < 5; epoch++) {
    std::shuffle(order.begin(), order.end(), rng);
    float epoch_loss = 0.0f;

    for (int bi = 0; bi < n_batches; bi++) {
      std::vector<int> idx(order.begin() + bi * batch_size,
                           order.begin() + (bi + 1) * batch_size);

      make_batch(train, idx, &x, &y);

      g.clear();
      opt.zero_grad();
      Tensor *out = net.forward(&g, &x);
      Tensor *loss = softmax_cross_entropy(&g, out, &y);
      loss->backward();
      opt.step();

      epoch_loss += loss->data[0];
    }
    std::cout << "Epoch: " << epoch << " Loss: " << epoch_loss / n_batches
              << "\n";
  }
}

int main() {
  train_mnist();
  return 0;
}
