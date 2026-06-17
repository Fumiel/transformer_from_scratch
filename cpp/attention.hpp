#pragma once

#include "tensor.hpp"

Tensor softmax_last_dim(const Tensor& x);
Tensor causal_self_attention(const Tensor& q, const Tensor& k, const Tensor& v, int n_head);
