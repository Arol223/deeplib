#include "grad_check.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>

void grad_check(Tensor *x, std::function<Tensor *(Tensor *)> f, float eps)
{
    x->zero_grad();

    Tensor* loss = f(x);
    loss->backward();
    std::vector<float> grad_analytic = x->grad;
    
    int failures = 0;
    // numeric grad, [f(x + eps) - f(x - eps)] / 2*eps
    for (size_t i = 0; i < x->data.size(); i++){
        float orig = x->data[i];

        x->data[i] = orig + eps;
        float l_plus = f(x)->data[0];

        x->data[i] = orig - eps;
        float l_minus = f(x)->data[0];

        x->data[i] = orig;

        float numeric = (l_plus - l_minus) / (2 * eps);

        float denom = std::max(std::abs(numeric) + std::abs(grad_analytic[i]), 1e-8f);
        float rel_err = std::abs(numeric - grad_analytic[i]) / denom;
        
        if (rel_err > 1e-3f){
            failures++;
            std::cout << "FAIL at " << i << ": analytic " << grad_analytic[i]
            << ", numeric " << numeric << ", rel_err " << rel_err << "\n";
        }
    }
    std::cout << (failures ? "FAILED" : "PASSED") << " (" << failures
    << "/" << x->data.size() << " bad)\n";


}