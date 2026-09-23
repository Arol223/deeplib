#pragma once
#include "utils.hpp"
#include <functional>
#include <vector>

namespace deeplib {

class Tensor {
public:
  std::vector<float> data;
  std::vector<float> grad;
  std::vector<int> shape;
  std::vector<int> strides;
  bool requires_grad;

  std::vector<Tensor *> parents;
  std::function<void()> backward_fn;

  int size() const { return product(shape); }

  // 2D indexing
  float &at(const std::vector<int> &idx); // Nicer indexing into flat vector
  const float &at(const std::vector<int> &idx) const;
  float &at(int i, int j) { return data[i * strides[0] + j * strides[1]]; }
  const float &at(int i, int j) const {
    return data[i * strides[0] + j * strides[1]];
  }

  // 4D indexing, ready for conv2d
  float &at(int n, int c, int h, int w) {
    return data[n * strides[0] + c * strides[1] + h * strides[2] +
                w * strides[3]];
  }
  const float &at(int n, int c, int h, int w) const {
    return data[n * strides[0] + c * strides[1] + h * strides[2] +
                w * strides[3]];
  }
  float &grad_at(int n, int c, int h, int w) {
    return grad[n * strides[0] + c * strides[1] + h * strides[2] +
                w * strides[3]];
  }

  float &grad_at(int i, int j) { return grad[i * strides[0] + j * strides[1]]; }

  float &grad_at(const std::vector<int> &idx);

  void backward();

  Tensor(std::vector<int> shape, bool requires_grad = false);

  void print() const;

  void fill(float value);
  void randomize(float lo = -1.0f, float hi = 1.0f);

  void reshape(const std::vector<int> &new_shape);

  void zero_grad();

private:
  int flat_index(const std::vector<int> &idx) const;
  void compute_strides();
};

void set_seed(unsigned s);

} // namespace deeplib