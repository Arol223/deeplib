#pragma once
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

  int size() const;

  float &at(const std::vector<int> &idx); // Nicer indexing into flat vector
  const float &at(const std::vector<int> &idx) const;

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