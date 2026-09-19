#include "grad_check.hpp"
#include "ops.hpp"
#include "tensor.hpp"

int main() {

    // 0. Create randomly initialized tensors x, b
    Tensor* x = new Tensor({3,4}, true);
    x->randomize();
    Tensor* b = new Tensor({3,4}, true);
    b->randomize();
    // 1. Verify grad checker itself, L = sum(x^2), so dL/dx = 2x
    grad_check(x, [](Tensor* t) { return sum(mul(t, t)); }, 2e-2f);
    x->zero_grad();
    b->zero_grad();

    // 2. add
    grad_check(x, [b](Tensor* t) { return sum(add(t, b)); }, 2e-2f); 
    x->zero_grad();
    b->zero_grad();

    // 3. matmul - tricky one...
    Tensor* A = new Tensor({2, 3}, true);
    Tensor* B = new Tensor({3, 4}, true);
    A->randomize();
    B->randomize();

    grad_check(A, [B](Tensor* t) { return sum(matmul(t, B)); }, 2e-2f);
    A->zero_grad();
    B->zero_grad();


    grad_check(B, [A](Tensor* t) { return sum(matmul(A, t));}, 2e-2f); 

    return 0;
}