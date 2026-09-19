#include "ops.hpp"
#include "tensor.hpp"
#include <cassert>
#include <cstddef>
#include <iterator>
#include <vector>

Tensor* add(Tensor* a, Tensor* b)
{
    assert(a->shape == b->shape);
    bool requires_grad = a->requires_grad || b->requires_grad;
    Tensor* out = new Tensor(a->shape, requires_grad);

    const size_t n = a->data.size();
    for (size_t i = 0; i < n; i++)
    {
        out->data[i] = a->data[i] + b->data[i];
    }
    out->parents = {a, b};

    out->backward_fn = [a, b, out](){
        for (size_t i = 0; i < out->grad.size(); i++)
        {
            a->grad[i] += out->grad[i];
            b->grad[i] += out->grad[i];
        }
       
    };
    return out;
}

Tensor* sub(Tensor* a, Tensor* b)
{
    assert(a->shape == b->shape);
    bool requires_grad = a->requires_grad || b->requires_grad;
    Tensor* out = new Tensor(a->shape, requires_grad);

    const size_t n = a->data.size();

    for (size_t i = 0; i < n; i++)
    {
        out->data[i] = a->data[i] - b->data[i];
    }
    out->parents = {a, b};

    out->backward_fn = [a, b, out](){
        for (size_t i = 0; i < out->grad.size(); i++)
        {
            a->grad[i] += out->grad[i];
            b->grad[i] -= out->grad[i];
        }
       
    };
    return out;
}

Tensor* mul(Tensor* a, Tensor* b)
{
    assert(a->shape == b->shape);
    bool requires_grad = a->requires_grad || b->requires_grad;
    Tensor* out = new Tensor(a->shape, requires_grad);

    const size_t n = a->data.size();

    for (size_t i = 0; i < n; i++)
    {
        out->data[i] = a->data[i] * b->data[i];
    }
    
    out->parents = {a, b};

    out->backward_fn = [a, b, out](){
        for (size_t i = 0; i < out->grad.size(); i++)
        {
            a->grad[i] += out->grad[i] * b->data[i];
            b->grad[i] += out->grad[i] * a->data[i];
        }
       
    };
    return out;
}

Tensor* mul_scalar(Tensor* a, float s)
{
    bool requires_grad = a->requires_grad;
    Tensor* out = new Tensor(a->shape, requires_grad);

    const size_t n = a->data.size();

    for (size_t i = 0; i < n; i++)
    {
        out->data[i] = a->data[i] * s;
    }
    out->parents = {a};
    out->backward_fn = [a, s, out](){
    for (size_t i = 0; i < out->grad.size(); i++)
    {
        a->grad[i] += out ->grad[i] * s;
    }
       
    };
    return out;
}

Tensor* matmul(Tensor* a, Tensor* b)
{
    assert(a->shape.size() == 2);
    assert(b->shape.size() == 2);
    assert(a->shape[1] == b->shape[0]);

    bool requires_grad = a->requires_grad || b->requires_grad;
    const int out_col = b->shape[1];
    const int out_row = a->shape[0];
    const int inner = a->shape[1];

    Tensor* out = new Tensor({a->shape[0], b->shape[1]}, requires_grad);

    for (int i = 0; i < out_row; i++)
    {
        for (int j = 0; j < out_col; j ++)
        {
            float C_ij = 0;
            for (int p = 0; p < inner; p++)
            {
                C_ij += a->at({i,p}) * b->at({p, j});
            }
            out->at({i, j}) = C_ij;
        }
    }

    out->parents = {a, b};
    out->backward_fn = [a, b, out](){
        int m = a->shape[0];
        int k = a->shape[1];
        int n = b->shape[1];
        for (int i = 0; i < m; i++){
            for (int p = 0; p < k; p++){
                for (int j = 0; j < n; j++){
                    a->grad_at({i,p}) += out->grad_at({i,j}) * b->at({p, j});
                }
            }
        }

        for (int p = 0; p < k; p++){
            for (int j = 0; j < n; j++){
                for (int i = 0; i < m; i++){
                    b->grad_at({p,j}) += a->at({i,p}) * out->grad_at({i,j});
                }
            }
        }
       
    };

    return out;
}


Tensor* transpose(Tensor* a)
{
    assert(a->shape.size() == 2);
    bool requires_grad = a->requires_grad;
    int out_rows = a->shape[1];
    int out_cols = a->shape[0];

    Tensor* out = new Tensor({out_rows, out_cols}, requires_grad);
    for (int i = 0; i < out_rows; i++)
    {
        for (int j = 0; j < out_cols; j ++)
        {
            out->at({i, j}) = a->at({j, i});
        }
    }
    return out;
}

Tensor* sum(Tensor* a)
{
    Tensor* out = new Tensor({1}, a->requires_grad);
    float s = 0;
    for (int i = 0; i < std::ssize(a->data); i++)
    {
        s += a->data[i];
    }
    out->data[0] = s;

    out->parents = {a};
    out->backward_fn = [a, out](){
        for (size_t i = 0; i < a->grad.size(); i++){
            a->grad[i] += out->grad[0];
        }
    };
    return out;
}