#pragma once
#include "tensor.hpp"

Tensor* add(Tensor* a, Tensor* b);

Tensor* sub(Tensor* a, Tensor* b);

Tensor* mul(Tensor* a, Tensor* b);

Tensor* mul_scalar(Tensor* a, float s);

Tensor* matmul(Tensor* a, Tensor* b);

Tensor* transpose(Tensor* a);

Tensor* sum(Tensor* a);

