#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
namespace llm {
struct Quantized {
    size_t rows, cols, group;
    std::vector<int8_t> values;
    std::vector<float> scales;
};
Quantized quantize(const std::vector<float> &weights, size_t rows, size_t cols, size_t group);
std::vector<float> dequantize(const Quantized &weights);
} // namespace llm
