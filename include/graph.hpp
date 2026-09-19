#pragma once
#include "tensor.hpp"
#include <vector>

class Graph {
    public:
    Tensor* make(std::vector<int> shape, bool requires_grad = false);
    ~Graph();
    void clear();

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph() = default;

    private:
    std::vector<Tensor*> owned;
};