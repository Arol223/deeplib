#pragma once
#include "graph.hpp"
#include "tensor.hpp"
#include <vector>

namespace deeplib {

class Module {
public:
  virtual Tensor *forward(Graph *g, Tensor *x) = 0;
  virtual std::vector<Tensor *> parameters() = 0;
  virtual ~Module() = default;
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
} // namespace deeplib