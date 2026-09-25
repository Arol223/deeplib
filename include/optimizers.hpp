#pragma once
#include "tensor.hpp"
#include <iostream>
#include <vector>

namespace deeplib {

class Optimizer {
public:
  Optimizer(std::vector<Tensor *> params) : params(params) {}
  virtual void step() = 0;
  virtual ~Optimizer() = default;

  void zero_grad() {
    for (Tensor *p : params)
      p->zero_grad();
  }

  virtual void save_state(std::ostream &) {}
  virtual void load_state(std::istream &) {}

protected:
  std::vector<Tensor *> params;
};

class SGD : public Optimizer {
public:
  SGD(std::vector<Tensor *> params, float lr);
  void step() override;

private:
  float lr;
};

class SGDMomentum : public Optimizer {
public:
  SGDMomentum(std::vector<Tensor *> params, float lr, float momentum = 0.9f)
      : Optimizer(params), lr(lr), momentum(momentum) {
    for (Tensor *p : params)
      velocity.emplace_back(p->data.size(), 0.0f);
  }
  void step() override;
  void save_state(std::ostream &) override;
  void load_state(std::istream &) override;

private:
  float lr, momentum;
  std::vector<std::vector<float>> velocity;
};
} // namespace deeplib