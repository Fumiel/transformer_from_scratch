#include "model.hpp"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string prompt = "hello";
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--prompt") {
            prompt = argv[i + 1];
        }
    }

    std::vector<int> input_ids;
    input_ids.reserve(prompt.size());
    for (char ch : prompt) {
        input_ids.push_back(static_cast<unsigned char>(ch));
    }

    TinyGPTCpp model(TinyGPTConfigCpp{});
    Tensor logits = model.forward(input_ids);

    std::cout << "tiny_transformer_cpp smoke test\n";
    std::cout << "prompt: " << prompt << "\n";
    std::cout << "logits shape: [" << logits.shape[0] << ", " << logits.shape[1] << "]\n";
    return 0;
}
