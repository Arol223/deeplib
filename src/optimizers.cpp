#include "optimizers.hpp"
#include "tensor.hpp"
#include <cstddef>
#include <cstdint>
#include <istream>
#include <iterator>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace deeplib {
namespace {
constexpr uint32_t OPT_MAGIC = 0x444C4F53; // "DLOS"
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

void SGDMomentum::save_state(std::ostream &f) {
  static_assert(sizeof(int) == sizeof(int32_t), "format assumes 32-bit int");
  write_u32(f, OPT_MAGIC);
  write_u32(f, CKPT_VERSION);
  write_u32(f, velocity.size());

  for (const std::vector<float> &v : velocity) {
    write_u32(f, static_cast<uint32_t>(v.size()));
    f.write(reinterpret_cast<const char *>(v.data()), sizeof(float) * v.size());
  }

  if (!f)
    throw std::runtime_error("Writing velocities failed");
}

void SGDMomentum::load_state(std::istream &f) {
  if (read_u32(f) != OPT_MAGIC)
    throw std::runtime_error("not an optimizer state section");
  if (read_u32(f) != CKPT_VERSION)
    throw std::runtime_error("optimizer state version mismatch");
  if (read_u32(f) != velocity.size())
    throw std::runtime_error("optimizer state count mismatch");

  std::vector<std::vector<float>> loaded;
  loaded.reserve(velocity.size());

  for (const std::vector<float> &v : velocity) {
    uint32_t n = read_u32(f);
    if (n != v.size())
      throw std::runtime_error("Velocity size mismatch");
    std::vector<float> buf(n);
    f.read(reinterpret_cast<char *>(buf.data()), n * sizeof(float));
    if (!f)
      throw std::runtime_error("optimizer state truncated");
    loaded.push_back(std::move(buf));
  }

  velocity.swap(loaded); // commit: one noexcept swap
}

} // namespace deeplib