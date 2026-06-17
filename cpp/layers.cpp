#include "layers.hpp"

#include <cassert>
#include <cmath>

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    // TODO: support x [..., in_features], weight [out_features, in_features]
    // Initial smoke implementation for x [N, in_features].
    assert(x.shape.size() == 2);
    assert(weight.shape.size() == 2);

    const int n = x.shape[0];
    const int in_features = x.shape[1];
    const int out_features = weight.shape[0];
    assert(weight.shape[1] == in_features);
    assert(bias.shape.size() == 1);
    assert(bias.shape[0] == out_features);

    Tensor y({n, out_features});
    for (int i = 0; i < n; ++i) {
        for (int o = 0; o < out_features; ++o) {
            float sum = bias.data[o];
            for (int k = 0; k < in_features; ++k) {
                sum += x.data[i * in_features + k] * weight.data[o * in_features + k];
            }
            y.data[i * out_features + o] = sum;
        }
    }
    return y;
}

Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps) {
    // TODO: implement for [B, T, C].
    (void)weight;
    (void)bias;
    (void)eps;
    return x;
}

float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / 3.14159265358979323846f) * (x + 0.044715f * x * x * x)));
}

Tensor gelu_tensor(const Tensor& x) {
    Tensor y = x;
    for (float& v : y.data) {
        v = gelu(v);
    }
    return y;
}
