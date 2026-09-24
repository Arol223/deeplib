#pragma once
#include "tensor.hpp"
#include <string>
#include <vector>

namespace deeplib {

struct Dataset {
  std::vector<float> images; // MNIST n*784
  std::vector<int> labels;   // n
  int n;
  int n_features; // 784
  int rows;
  int cols;
};

Dataset load_mnist(const std::string &image_path,
                   const std::string &label_path);

Dataset load_cifar10(const std::string &path, const bool test = false);

void standardize(Dataset &ds, std::array<float, 3> &mean,
                 std::array<float, 3> &std_dev, bool compute);

void print_ascii(const Dataset &ds, int idx);

void make_batch(const Dataset &ds, const std::vector<int> &indices,
                Tensor *x_out, Tensor *y_out);

} // namespace deeplib