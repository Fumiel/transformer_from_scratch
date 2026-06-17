#pragma once

#include "tensor.hpp"

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias);
Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps = 1e-5f);
float gelu(float x);
Tensor gelu_tensor(const Tensor& x);
