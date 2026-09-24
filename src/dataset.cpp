#include "dataset.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs = std::filesystem;

namespace deeplib {

namespace {
uint32_t swap_endian(uint32_t v) {
  return ((v >> 24) & 0xFF) | ((v >> 8) & 0xFF00) | ((v << 8) & 0xFF0000) |
         ((v << 24) & 0xFF000000);
}

} // namespace
Dataset load_mnist(const std::string &image_path,
                   const std::string &label_path) {
  Dataset ds;

  std::ifstream fi(image_path, std::ios::binary);
  if (!fi) {
    throw std::runtime_error("Could not open " + image_path);
  };

  uint32_t magic, n_images, rows, cols;
  fi.read(reinterpret_cast<char *>(&magic), 4);
  magic = swap_endian(magic);

  if (magic != 0x803)
    throw std::runtime_error("Magic number check failed");

  fi.read(reinterpret_cast<char *>(&n_images), 4);
  n_images = swap_endian(n_images);
  fi.read(reinterpret_cast<char *>(&rows), 4);
  rows = swap_endian(rows);
  fi.read(reinterpret_cast<char *>(&cols), 4);
  cols = swap_endian(cols);

  ds.n = n_images;
  ds.n_features = rows * cols;
  ds.cols = cols;
  ds.rows = rows;
  // Read data into byte buffer

  std::vector<uint8_t> raw(n_images * rows * cols);
  fi.read(reinterpret_cast<char *>(raw.data()), raw.size());

  if (!fi)
    throw std::runtime_error("Unexpected end of " + image_path);

  ds.images.resize(raw.size());
  for (size_t i = 0; i < raw.size(); i++) {
    ds.images[i] = raw[i] / 255.0f;
  }

  // labels
  std::ifstream fl(label_path, std::ios::binary);
  if (!fl)
    throw std::runtime_error("Could not open " + label_path);
  uint32_t magic_l, n_labels;

  fl.read(reinterpret_cast<char *>(&magic_l), 4);
  magic_l = swap_endian(magic_l);
  if (magic_l != 0x801)
    throw std::runtime_error("Magic number validation failed for labels");

  fl.read(reinterpret_cast<char *>(&n_labels), 4);
  n_labels = swap_endian(n_labels);

  if (n_labels != n_images)
    throw std::runtime_error("Unexpected n_labels != n_images");

  std::vector<uint8_t> labels(n_labels);
  fl.read(reinterpret_cast<char *>(labels.data()), labels.size());
  if (!fl)
    throw std::runtime_error("unexpected end of " + label_path);

  ds.labels.resize(labels.size());
  for (size_t i = 0; i < labels.size(); i++) {
    ds.labels[i] = labels[i];
  }

  return ds;
}

void print_ascii(const Dataset &ds, int idx) {

  int label = ds.labels[idx];

  std::cout << "Label " << idx << " : " << label << "\n";

  int offset = idx * ds.n_features;
  std::cout << "ASCII rendering: \n";

  for (int i = 0; i < ds.n_features; i++) {
    if (i % ds.cols == 0) {
      std::cout << "\n";
    } //  new line

    std::cout << (ds.images[offset + i] < 0.5f ? ' ' : '#');
  }
}

Dataset load_cifar10(const std::string &dir_path, const bool test) {
  const int n_features = 3072;
  const int record = 1 + n_features; // label byte plus pixels
  std::vector<std::string> batch_names;

  if (test) {
    batch_names.push_back((fs::path(dir_path) / "test_batch.bin").string());
  } else {
    for (int i = 1; i <= 5; i++) {
      fs::path p =
          fs::path(dir_path) / ("data_batch_" + std::to_string(i) + ".bin");
      batch_names.push_back(p.string());
    }
  }

  Dataset ds;
  ds.rows = 32;
  ds.cols = 32;
  ds.n_features = n_features;
  ds.n = 0;

  if (!test) {
    ds.images.reserve(50000 * n_features);
    ds.labels.reserve(50000);
  } else {
    ds.images.reserve(10000 * n_features);
    ds.labels.reserve(10000);
  }

  for (const std::string &path : batch_names) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
      throw std::runtime_error("Could not open " + path);
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());

    if (raw.size() % record != 0)
      throw std::runtime_error("Unexpected size for " + path);
    int n_records = raw.size() / record;
    ds.n += n_records;
    for (int i = 0; i < n_records; i++) {
      const int base = i * record;
      ds.labels.push_back(raw[base]);
      for (int j = 0; j < n_features; j++) {
        ds.images.push_back(raw[base + 1 + j] / 255.0f);
      }
    }
  }

  return ds;
}

void standardize(Dataset &ds, std::array<float, 3> &mean,
                 std::array<float, 3> &std_dev, bool compute) {
  const int per_channel = ds.n_features / 3;
  if (compute) {
    for (int c = 0; c < 3; c++) {
      double sum = 0.0, sumsq = 0.0;
      for (int i = 0; i < ds.n; i++) {
        const int base = i * ds.n_features + c * per_channel;
        for (int j = 0; j < per_channel; j++) {
          float v = ds.images[base + j];
          sum += v;
          sumsq += v * v;
        }
      }
      double count = static_cast<double>(ds.n) * per_channel;
      mean[c] = sum / count;
      std_dev[c] = std::sqrt(sumsq / count - mean[c] * mean[c]);
    }
  }

  for (int c = 0; c < 3; c++) {
    std::cout << "c=" << c << " mean=" << mean[c] << " std=" << std_dev[c]
              << "\n";
    assert(std_dev[c] > 1e-8f);
    const float inv = 1.0f / std_dev[c];
    for (int i = 0; i < ds.n; i++) {
      const int base = i * ds.n_features + c * per_channel;
      for (int j = 0; j < per_channel; j++) {
        ds.images[base + j] = (ds.images[base + j] - mean[c]) * inv;
      }
    }
  }
}
void make_batch(const Dataset &ds, const std::vector<int> &indices,
                Tensor *x_out, Tensor *y_out) {
  assert(x_out->shape[0] == std::ssize(indices));
  assert(x_out->shape[1] == ds.n_features);
  assert(y_out->shape[0] == std::ssize(indices));

  y_out->fill(0.0f);
  for (size_t b = 0; b < indices.size(); b++) {
    int src = indices[b];
    int offset = src * ds.n_features;
    std::copy(ds.images.begin() + offset,
              ds.images.begin() + offset + ds.n_features,
              x_out->data.begin() + b * ds.n_features);
    y_out->at({static_cast<int>(b), ds.labels[src]}) = 1.0f;
  }
}

} // namespace deeplib