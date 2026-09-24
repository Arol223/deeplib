#include "dataset.hpp"
#include "graph.hpp"
#include "layers.hpp"
#include "losses.hpp"
#include "ops.hpp"
#include "optimizers.hpp"
#include "tensor.hpp"
#include <array>
#include <chrono>
#include <functional>
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
    auto t0 = std::chrono::steady_clock::now();
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
    auto t1 = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration<double>(t1 - t0).count() << "s\n";
    std::cout << "Epoch: " << epoch << " Loss: " << epoch_loss / n_batches
              << "\n";
  }
  Dataset test =
      load_mnist("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

  int correct = 0;
  for (int bi = 0; bi < test.n / batch_size; bi++) {
    std::vector<int> idx(batch_size);
    std::iota(idx.begin(), idx.end(), bi * batch_size);
    make_batch(test, idx, &x, &y);

    g.clear();
    Tensor *out = net.forward(&g, &x);

    for (int s = 0; s < batch_size; s++) {
      int pred = 0;
      for (int c = 1; c < 10; c++) {
        if (out->at(s, c) > out->at(s, pred))
          pred = c;
      }
      if (pred == test.labels[idx[s]])
        correct++;
    }
  }
  std::cout << "test accuracy: "
            << 100.0 * correct / (test.n / batch_size * batch_size) << "%\n";
}

void train_cnn_mnist() {
  Dataset train = load_mnist("data/train-images-idx3-ubyte",
                             "data/train-labels-idx1-ubyte");

  const int batch_size = 64;
  const int n_batches = train.n / batch_size; // Drop dangling batch at
                                              // end

  Sequential net({new Conv2d(1, 8, 3, 3), new Relu(), new MaxPool2d(2, 2, 2),
                  new Conv2d(8, 16, 3, 3), new Relu(), new MaxPool2d(2, 2, 2),
                  new Flatten(), new Linear(400, 10)});

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
    auto t0 = std::chrono::steady_clock::now();
    for (int bi = 0; bi < n_batches; bi++) {
      std::vector<int> idx(order.begin() + bi * batch_size,
                           order.begin() + (bi + 1) * batch_size);

      make_batch(train, idx, &x, &y);

      g.reset();
      opt.zero_grad();
      x.reshape({batch_size, 1, 28, 28});
      Tensor *out = net.forward(&g, &x);
      x.reshape({batch_size, 784});
      Tensor *loss = softmax_cross_entropy(&g, out, &y);
      loss->backward();
      opt.step();

      epoch_loss += loss->data[0];
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration<double>(t1 - t0).count() << "s\n";
    std::cout << "Epoch: " << epoch << " Loss: " << epoch_loss / n_batches
              << "\n";
  }
  Dataset test =
      load_mnist("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

  int correct = 0;
  for (int bi = 0; bi < test.n / batch_size; bi++) {
    std::vector<int> idx(batch_size);
    std::iota(idx.begin(), idx.end(), bi * batch_size);
    make_batch(test, idx, &x, &y);

    g.clear();
    x.reshape({batch_size, 1, 28, 28});
    Tensor *out = net.forward(&g, &x);
    x.reshape({batch_size, 784});
    for (int s = 0; s < batch_size; s++) {
      int pred = 0;
      for (int c = 1; c < 10; c++) {
        if (out->at(s, c) > out->at(s, pred))
          pred = c;
      }
      if (pred == test.labels[idx[s]])
        correct++;
    }
  }
  std::cout << "test accuracy: "
            << 100.0 * correct / (test.n / batch_size * batch_size) << "%\n";
}

void test_cifar10(Dataset &test) {}

void train_cifar10() {
  Dataset train = load_cifar10("data/cifar-10-batches-bin");
  Dataset test = load_cifar10("data/cifar-10-batches-bin", true);
  std::array<float, 3> mean;
  std::array<float, 3> std_dev;
  standardize(train, mean, std_dev, true);
  standardize(test, mean, std_dev, false);

  const int batch_size = 64;
  const int n_batches = train.n / batch_size;

  Sequential net({new Conv2d(3, 32, 3, 3, 1, 1), // 32x32 -> 32x32 -> 16x16
                  new Relu(), new MaxPool2d(2, 2, 2),
                  new Conv2d(32, 64, 3, 3, 1, 1), // 16x16 -> 16x16 -> 8x8
                  new Relu(), new MaxPool2d(2, 2, 2), new Flatten(),
                  new Linear(4096, 10)}); // 64x8x8 = 4096

  SGDMomentum opt(net.parameters(), 0.01f, 0.9f);
  Graph g;

  Tensor x({batch_size, 1024 * 3});
  Tensor y({batch_size, 10});

  auto eval = [&]() {
    int correct = 0;
    for (int bi = 0; bi < test.n / batch_size; bi++) {
      std::vector<int> idx(batch_size);
      std::iota(idx.begin(), idx.end(), bi * batch_size);
      make_batch(test, idx, &x, &y);

      g.reset();
      x.reshape({batch_size, 3, 32, 32});
      Tensor *out = net.forward(&g, &x);
      x.reshape({batch_size, 1024 * 3});
      for (int s = 0; s < batch_size; s++) {
        int pred = 0;
        for (int c = 1; c < 10; c++) {
          if (out->at(s, c) > out->at(s, pred))
            pred = c;
        }
        if (pred == test.labels[idx[s]])
          correct++;
      }
    }
    std::cout << "test accuracy: "
              << 100.0 * correct / (test.n / batch_size * batch_size) << "%\n";
  };

  std::vector<int> order(train.n);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(0);

  for (int epoch = 0; epoch < 20; epoch++) {
    std::shuffle(order.begin(), order.end(), rng);
    float epoch_loss = 0.0f;
    auto t0 = std::chrono::steady_clock::now();
    for (int bi = 0; bi < n_batches; bi++) {
      std::vector<int> idx(order.begin() + bi * batch_size,
                           order.begin() + (bi + 1) * batch_size);

      make_batch(train, idx, &x, &y);

      g.reset();
      opt.zero_grad();
      x.reshape({batch_size, 3, 32, 32});
      Tensor *out = net.forward(&g, &x);
      x.reshape({batch_size, 1024 * 3});
      Tensor *loss = softmax_cross_entropy(&g, out, &y);
      loss->backward();
      opt.step();

      epoch_loss += loss->data[0];
      if (epoch % 5 == 4) {
        eval();
      }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration<double>(t1 - t0).count() << "s\n";
    std::cout << "Epoch: " << epoch << " Loss: " << epoch_loss / n_batches
              << "\n";
  }
}
