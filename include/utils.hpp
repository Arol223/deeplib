#pragma once
#include <vector>

template <typename T>
T product(const std::vector<T>& v)
{
    T result = T(1);
    for (const T& x : v)
    {
        result *=x;
    }
    return result;
}