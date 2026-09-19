#include "grad_check.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>

namespace deeplib {

void grad_check(Graph *g, Tensor *x,
                std::function<Tensor *(Graph *, Tensor *)> f, float eps,
                float tol) {
  x->zero_grad();

  Tensor *loss = f(g, x);
  loss->backward();
  std::vector<float> grad_analytic = x->grad;

  int failures = 0;
  // numeric grad, [f(x + eps) - f(x - eps)] / 2*eps
  for (size_t i = 0; i < x->data.size(); i++) {
    float orig = x->data[i];

    x->data[i] = orig + eps;
    float l_plus = f(g, x)->data[0];

    x->data[i] = orig - eps;
    float l_minus = f(g, x)->data[0];

    x->data[i] = orig;

    float numeric = (l_plus - l_minus) / (2 * eps);

    float denom =
        std::max(std::abs(numeric) + std::abs(grad_analytic[i]), 1e-8f);
    float rel_err = std::abs(numeric - grad_analytic[i]) / denom;
    float abs_err = std::abs(numeric - grad_analytic[i]);
    if (rel_err > tol && abs_err > 1e-3f) {
      failures++;
      std::cout << "FAIL at " << i << ": analytic " << grad_analytic[i]
                << ", numeric " << numeric << ", rel_err " << rel_err << "\n";
    }
    g->clear();
  }

  std::cout << (failures ? "FAILED" : "PASSED") << " (" << failures << "/"
            << x->data.size() << " bad)\n";
}
} // namespace deeplib