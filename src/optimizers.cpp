#include "optimizers.hpp"
#include "tensor.hpp"
#include <cstddef>
#include <iterator>
#include <vector>

namespace deeplib {

SGD::SGD(std::vector<Tensor *> params, float lr) : Optimizer(params), lr(lr) {}

void SGD::step() {
  for (Tensor *p : params) {
    for (int i = 0; i < std::ssize(p->data); i++) {
      p->data[i] -= lr * p->grad[i];
    }
  }
}

void SGDMomentum::step() {
  for (size_t i = 0; i < params.size(); i++) {
    Tensor *p = params[i];
    std::vector<float> &v = velocity[i];
    for (size_t j = 0; j < p->data.size(); j++) {
      v[j] = momentum * v[j] - lr * p->grad[j];
      p->data[j] += v[j];
    }
  }
}

} // namespace deeplib