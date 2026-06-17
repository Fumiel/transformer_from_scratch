#include "model.hpp"

TinyGPTCpp::TinyGPTCpp(TinyGPTConfigCpp config) : config_(config) {}

Tensor TinyGPTCpp::forward(const std::vector<int>& input_ids) const {
    // TODO: implement full Transformer forward.
    // For now, return a zero logits tensor [T, vocab_size].
    Tensor logits({static_cast<int>(input_ids.size()), config_.vocab_size}, 0.0f);
    return logits;
}
