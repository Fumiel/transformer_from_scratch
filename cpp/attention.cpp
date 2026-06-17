#include "attention.hpp"

#include <algorithm>
#include <cmath>

Tensor softmax_last_dim(const Tensor& x) {
    // TODO: implement numerically stable softmax for the last dimension.
    return x;
}

Tensor causal_self_attention(const Tensor& q, const Tensor& k, const Tensor& v, int n_head) {
    // TODO: implement [B, T, C] -> [B, T, C] causal multi-head attention.
    (void)k;
    (void)v;
    (void)n_head;
    return q;
}
