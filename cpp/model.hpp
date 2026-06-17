#pragma once

#include "tensor.hpp"

struct TinyGPTConfigCpp {
    int vocab_size = 128;
    int block_size = 64;
    int n_layer = 2;
    int n_head = 2;
    int n_embd = 64;
};

class TinyGPTCpp {
public:
    explicit TinyGPTCpp(TinyGPTConfigCpp config);
    Tensor forward(const std::vector<int>& input_ids) const;

private:
    TinyGPTConfigCpp config_;
};
