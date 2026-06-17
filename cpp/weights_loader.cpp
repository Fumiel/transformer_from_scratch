#include "tensor.hpp"

#include <string>

// TODO: implement a small JSON or binary weights loader.
// Recommended first version:
//   - export each tensor as name + shape + flat data
//   - load into std::unordered_map<std::string, Tensor>

bool load_weights_placeholder(const std::string& path) {
    (void)path;
    return false;
}
