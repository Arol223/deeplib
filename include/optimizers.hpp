#pragma once
#include "tensor.hpp"
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

} // namespace deeplib