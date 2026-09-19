#include "graph.hpp"
#include "tensor.hpp"

Tensor* Graph::make(std::vector<int> shape, bool requires_grad)
{
    Tensor* x = new Tensor(shape, requires_grad);
    owned.push_back(x);
    return x;
}

void Graph::clear(){
    for (Tensor* t : owned){
        delete t;
    }
    owned.clear();
}

Graph::~Graph() { clear(); }