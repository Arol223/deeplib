#include "tensor.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <random>
#include <set>
#include <vector>

namespace deeplib {

namespace {
std::mt19937 gen(42);
}

void set_seed(unsigned s) { gen.seed(s); }

Tensor::Tensor(std::vector<int> shape, bool requires_grad)
    : shape(shape), requires_grad(requires_grad) {
  compute_strides();
  data.resize(size(), 0.0f);
  grad.resize(size(), 0.0f);
};

void Tensor::compute_strides() {
  strides.resize(shape.size());
  int stride = 1;
  for (size_t i = shape.size(); i--;) {
    strides[i] = stride;
    stride *= shape[i];
  }
}

int Tensor::flat_index(const std::vector<int> &idx) const {
  assert(idx.size() == shape.size());
  int flat = 0;

  for (size_t i = 0; i < idx.size(); i++) {
    assert(idx[i] >= 0 && idx[i] < shape[i]);
    flat += idx[i] * strides[i];
  }
  return flat;
}

float &Tensor::at(const std::vector<int> &idx) {
  int flat = flat_index(idx);
  return data[flat];
}

const float &Tensor::at(const std::vector<int> &idx) const {
  int flat = flat_index(idx);
  return data[flat];
}

int Tensor::size() const { return product(shape); }

float &Tensor::grad_at(const std::vector<int> &idx) {
  int flat = flat_index(idx);
  return grad[flat];
}

void Tensor::print() const {
  std::cout << "Shape: (";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i > 0)
      std::cout << ", ";
    std::cout << shape[i];
  }
  std::cout << ")\n";

  std::cout << "Data: [";
  for (size_t i = 0; i < data.size(); i++) {
    if (i > 0)
      std::cout << ", ";
    std::cout << data[i];
  }
  std::cout << "]\n";

  std::cout << "Grad: [";
  for (size_t i = 0; i < grad.size(); i++) {
    if (i > 0)
      std::cout << ", ";
    std::cout << grad[i];
  }
  std::cout << "]\n";
}

void Tensor::fill(float value) {
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = value;
  }
}

void Tensor::randomize(float lo, float hi) {
  std::uniform_real_distribution<float> dist(lo, hi);
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = dist(gen);
  }
}

void Tensor::reshape(const std::vector<int> &new_shape) {
  assert(size() == product(new_shape));
  shape = new_shape;
  compute_strides();
}

namespace {
void build_topo(Tensor *t, std::vector<Tensor *> &topo,
                std::set<Tensor *> &visited) {
  if (visited.contains(t)) {
    return;
  }
  visited.insert(t);
  for (Tensor *p : t->parents) {
    build_topo(p, topo, visited);
  }
  topo.push_back(t);
}
} // namespace

void Tensor::backward() {
  assert(size() == 1);
  grad[0] = 1.0f;
  std::vector<Tensor *> topo;
  std::set<Tensor *> visited;
  build_topo(this, topo, visited);

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    if ((*it)->backward_fn) {
      (*it)->backward_fn();
    }
  }
}

void Tensor::zero_grad() { std::fill(grad.begin(), grad.end(), 0.0f); }
} // namespace deeplib
