#pragma once
#include "activations.hpp"
#include "graph.hpp"
#include "ops.hpp"
#include "tensor.hpp"
#include <csignal>
#include <istream>
#include <ostream>
#include <vector>

namespace deeplib {

class Module {
public:
  virtual Tensor *forward(Graph *g, Tensor *x) = 0;
  virtual std::vector<Tensor *> parameters() = 0;
  virtual ~Module() = default;

  void save(std::ostream &f);
  void save(const std::string &path);
  void load(std::istream &f);
  void load(const std::string &path);
};

class Linear : public Module {
public:
  Linear(int in_features, int out_features);
  ~Linear() override;

  Tensor *forward(Graph *g, Tensor *x) override;
  std::vector<Tensor *> parameters() override;

  Linear(const Linear &) = delete;
  Linear &operator=(const Linear &) = delete;

private:
  Tensor *W; // (in, out)
  Tensor *b; // (out)
};

class Conv2d : public Module {
public:
  Conv2d(int in_channels, int out_channels, int kh, int kw, int stride = 1,
         int padding = 0);
  ~Conv2d() override;
  Tensor *forward(Graph *g, Tensor *x) override;
  std::vector<Tensor *> parameters() override;
  Conv2d(const Conv2d &) = delete;
  Conv2d &operator=(const Conv2d &) = delete;

private:
  Tensor *k; // (out, in, kh, kw)
  Tensor *b; // (out)
  int stride, padding;
};

class Sequential : public Module {
public:
  Sequential(std::vector<Module *> layers);
  Sequential() = default;
  ~Sequential() override;
  Sequential(const Sequential &) = delete;
  Sequential &operator=(const Sequential &) = delete;

  Tensor *forward(Graph *g, Tensor *x) override;
  std::vector<Tensor *> parameters() override;
  void add(Module *m);

private:
  std::vector<Module *> module_list;
};

class Tanh : public Module {
public:
  Tensor *forward(Graph *g, Tensor *x) override { return tanh(g, x); }
  std::vector<Tensor *> parameters() override { return {}; }
};

class Relu : public Module {
public:
  Tensor *forward(Graph *g, Tensor *x) override { return relu(g, x); }
  std::vector<Tensor *> parameters() override { return {}; }
};

class Sigmoid : public Module {
public:
  Tensor *forward(Graph *g, Tensor *x) override { return sigmoid(g, x); }
  std::vector<Tensor *> parameters() override { return {}; }
};

class Flatten : public Module {
public:
  Tensor *forward(Graph *g, Tensor *x) override { return flatten(g, x); }
  std::vector<Tensor *> parameters() override { return {}; }
};

class MaxPool2d : public Module {
public:
  MaxPool2d(int kh, int kw, int stride) : kh(kh), kw(kw), stride(stride){};
  Tensor *forward(Graph *g, Tensor *x) override {
    return maxpool2d(g, x, kh, kw, stride);
  }
  std::vector<Tensor *> parameters() override { return {}; }

private:
  int kh, kw, stride;
};
} // namespace deeplib