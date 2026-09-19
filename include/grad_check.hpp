#pragma once
#include "tensor.hpp"

void grad_check(Tensor* x, std::function<Tensor*(Tensor*)> f, float eps = 1e-4f);