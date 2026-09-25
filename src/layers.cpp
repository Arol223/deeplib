#include "layers.hpp"
#include "graph.hpp"
#include "ops.hpp"
#include "tensor.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <vector>

namespace deeplib {

// Module
namespace {
constexpr uint32_t CKPT_MAGIC = 0x444C4350; // "DLCP"
constexpr uint32_t CKPT_VERSION = 1;
} // namespace

namespace {
void write_u32(std::ostream &f, uint32_t v) {
  f.write(reinterpret_cast<const char *>(&v), sizeof v);
}

uint32_t read_u32(std::istream &f) {
  uint32_t v;
  f.read(reinterpret_cast<char *>(&v), sizeof v);
  if (!f)
    throw std::runtime_error("checkpoint truncated");
  return v;
}
} // namespace

void Module::save(std::ostream &f) {
  static_assert(sizeof(int) == sizeof(int32_t), "format assumes 32-bit int");

  std::vector<Tensor *> params = parameters();

  write_u32(f, CKPT_MAGIC);
  write_u32(f, CKPT_VERSION);
  write_u32(f, static_cast<uint32_t>(params.size()));

  for (Tensor *p : params) {
    write_u32(f, static_cast<uint32_t>(p->shape.size()));
    f.write(reinterpret_cast<const char *>(p->shape.data()),
            p->shape.size() * sizeof(int32_t));
    f.write(reinterpret_cast<const char *>(p->data.data()),
            p->data.size() * sizeof(float));
  }
  if (!f)
    throw std::runtime_error("checkpoint write failed");
}

void Module::save(const std::string &path) {
  const std::string tmp = path + ".tmp";
  std::ofstream f(tmp, std::ios::binary);
  if (!f)
    throw std::runtime_error("could not open " + tmp);
  save(f);
  f.close();
  if (!f)
    throw std::runtime_error("failed to write " + tmp);
  std::filesystem::rename(tmp, path);
}

void Module::load(std::istream &f) {
  std::vector<Tensor *> params = parameters();
  uint32_t magic = read_u32(f);
  if (magic != CKPT_MAGIC)
    throw std::runtime_error(
        "checkpoint load failed, magic numbers do not match");
  uint32_t version = read_u32(f);
  if (version != CKPT_VERSION)
    throw std::runtime_error("checkpoint load failed, versions not matching");
  uint32_t count = read_u32(f);
  if (count != params.size())
    throw std::runtime_error("checkpoint load failed, not same number of "
                             "tensors in checkpoint and target");
  std::vector<std::vector<float>> loaded_params;
  loaded_params.reserve(params.size());
  for (Tensor *p : params) {
    uint32_t rank = read_u32(f);
    if (rank != p->shape.size())
      throw std::runtime_error(
          "checkpoint load failed, tensors must have the same dimensions");
    std::vector<int> dims(rank);
    f.read(reinterpret_cast<char *>(dims.data()), rank * sizeof(int32_t));
    if (!f)
      throw std::runtime_error("failed to read dims");
    if (dims != p->shape)
      throw std::runtime_error(
          "Dimension mismatch when reading parameter tensor");

    std::vector<float> loaded_data(p->data.size());
    f.read(reinterpret_cast<char *>(loaded_data.data()),
           loaded_data.size() * sizeof(float));
    if (!f)
      throw std::runtime_error("Failed to read parameters");
    loaded_params.push_back(std::move(loaded_data));
  }
  for (size_t i = 0; i < loaded_params.size(); i++) {
    params[i]->data.swap(loaded_params[i]);
  }
}

void Module::load(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Could not open " + path);
  load(f);
}
// ----------------Linear--------------------------
Linear::Linear(int in_features, int out_features) {
  W = new Tensor({in_features, out_features}, true);
  b = new Tensor({out_features}, true);

  float xavier_bound = std::sqrt(6.0f / (in_features + out_features));

  W->randomize(-xavier_bound, xavier_bound);
  // bias is zero
}

Tensor *Linear::forward(Graph *g, Tensor *x) {
  return add_bias(g, matmul(g, x, W), b);
}

Linear::~Linear() {
  delete W;
  delete b;
}

std::vector<Tensor *> Linear::parameters() { return {W, b}; }

// --------------- Conv2d ----------------------
Conv2d::Conv2d(int in_channels, int out_channels, int kh, int kw, int stride,
               int padding)
    : stride(stride), padding(padding) {
  k = new Tensor({out_channels, in_channels, kh, kw}, true);
  b = new Tensor({out_channels}, true);

  int fan_in = in_channels * kh * kw;
  float he_bound = std::sqrt(6.0f / fan_in);
  k->randomize(-he_bound, he_bound);
  // bias is 0
}

Tensor *Conv2d::forward(Graph *g, Tensor *x) {
  return add_channel_bias(g, conv2d_im2col(g, x, k, stride, padding), b);
}

Conv2d::~Conv2d() {
  delete k;
  delete b;
}

std::vector<Tensor *> Conv2d::parameters() { return {k, b}; }

// --------------- Sequential--------------------

// Constructors
Sequential::Sequential(std::vector<Module *> layers) : module_list(layers) {}

// Destructor
Sequential::~Sequential() {
  for (Module *m : module_list)
    delete m;
}

void Sequential::add(Module *m) { module_list.push_back(m); }

Tensor *Sequential::forward(Graph *g, Tensor *x) {
  Tensor *h = x;
  for (Module *m : module_list) {
    h = m->forward(g, h);
  }
  return h;
}

std::vector<Tensor *> Sequential::parameters() {
  std::vector<Tensor *> out;
  for (Module *m : module_list) {
    std::vector<Tensor *> p = m->parameters();
    out.insert(out.end(), p.begin(), p.end());
  }
  return out;
}

} // namespace deeplib