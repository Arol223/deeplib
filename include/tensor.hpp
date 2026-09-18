#pragma once
#include <vector>
#include <functional>

class Tensor {
public:
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<int> shape;
    std::vector<int> strides;
    bool requires_grad;

    std::vector<Tensor*> parents;
    std::function<void()> backward_fn;

    int size() const;
    float& at(const std::vector<int>& idx);
    const float& at(const std::vector<int>& idx) const;
    void backward();

    Tensor(std::vector<int> shape, bool requires_grad = false);

    void print() const;

    void fill(float value);
    void randomize(float lo = -1.0f, float hi = 1.0f);
    
    void reshape(const std::vector<int>& new_shape);

private:
    int flat_index(const std::vector<int>& idx) const;
    void compute_strides();
};