#include "optimizers.hpp"
#include "tensor.hpp"
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

} // namespace deeplib