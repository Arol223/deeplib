#include "tensor.hpp"
#include "ops.hpp"
#include <iostream>

int main() {
    Tensor* a = new Tensor({2, 2}, true);
    Tensor* b = new Tensor({2, 2}, true);
    a->fill(3.0f);
    b->fill(4.0f);

    Tensor* c = add(a, b);
    Tensor* loss = sum(c);
    loss->backward();
    a->print();
    b->print();
    c->print();
    return 0;
}
