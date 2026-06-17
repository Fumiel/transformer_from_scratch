#include "tensor.hpp"

#include <numeric>

Tensor::Tensor(std::vector<int> shape_, float value) : shape(std::move(shape_)) {
    data.resize(numel(), value);
}

std::size_t Tensor::numel() const {
    if (shape.empty()) {
        return 0;
    }
    return static_cast<std::size_t>(
        std::accumulate(shape.begin(), shape.end(), 1, [](int a, int b) { return a * b; })
    );
}

float& Tensor::operator[](std::size_t idx) {
    assert(idx < data.size());
    return data[idx];
}

const float& Tensor::operator[](std::size_t idx) const {
    assert(idx < data.size());
    return data[idx];
}
