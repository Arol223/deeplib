#include "layers.hpp"
#include "graph.hpp"
#include "ops.hpp"
#include "tensor.hpp"
#include <cmath>
#include <vector>

namespace deeplib {

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