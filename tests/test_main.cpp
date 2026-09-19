#include "grad_check.hpp"
#include "graph.hpp"
#include "ops.hpp"
#include "tensor.hpp"

int main() {

    // 0. Create randomly initialized tensors x, b and graph g
    Tensor* x = new Tensor({3,4}, true);
    x->randomize();
    Tensor* b = new Tensor({3,4}, true);
    b->randomize();

    Graph g;
    // 1. Verify grad checker itself, L = sum(x^2), so dL/dx = 2x
    grad_check(&g, x, [](Graph* gr, Tensor* t) { return sum(gr, mul(gr, t, t)); }, 2e-4f);
    x->zero_grad();
    b->zero_grad();
    g.clear();

    // 2. add
    grad_check(&g, x, [b](Graph* gr, Tensor* t) { return sum(gr, add(gr,t, b)); }, 2e-4f); 
    x->zero_grad();
    b->zero_grad();

    // 3. matmul - tricky one...
    Tensor* A = new Tensor({2, 3}, true);
    Tensor* B = new Tensor({3, 4}, true);
    A->randomize();
    B->randomize();

    grad_check(&g, A, [B](Graph* gr, Tensor* t) { return sum(gr,matmul(gr, t, B)); }, 2e-4f);
    A->zero_grad();
    B->zero_grad();


    grad_check(&g, B, [A](Graph* gr, Tensor* t) { return sum(gr, matmul(gr, A, t));}, 2e-4f); 

    // 4 Transpose
    Tensor* T = new Tensor({2, 5}, true);
    T->randomize();
    grad_check(&g, T, [](Graph* gr, Tensor* t) { return sum(gr, transpose(gr, t)); });
    return 0;
}