#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

class Tensor {
public:
    std::vector<int> shape;
    std::vector<float> data;

    Tensor() = default;
    Tensor(std::vector<int> shape_, float value = 0.0f);

    std::size_t numel() const;
    float& operator[](std::size_t idx);
    const float& operator[](std::size_t idx) const;
};
