#include "weight_loader.h"
#include <iostream>
int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("model file required");
        llm::WeightLoader model(argv[1], llm::Device::CPU);
        std::cout << model.weights.size() << " validated tensors, " << model.config.layers
                  << " layers\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
