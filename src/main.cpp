#include "graph.hpp"
#include "layers.hpp"
#include "losses.hpp"
#include "ops.hpp"
#include "optimizers.hpp"
#include "tensor.hpp"
#include <iostream>
#include <vector>

using namespace deeplib;

int main() {
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
  return 0;
}
