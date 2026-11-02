#pragma once
#include "tensor.h"
#include <string>
#include <unordered_map>
namespace llm {
struct ModelConfig {
    uint32_t hidden, intermediate, layers, heads, kvheads, vocab, context, bos, eos;
    float epsilon, rope_base;
    void validate() const;
};
struct Weight {
    Tensor data;
    Tensor scales;
    uint32_t group = 0;
};
class WeightLoader {
  public:
    ModelConfig config{};
    std::unordered_map<std::string, Weight> weights;
    explicit WeightLoader(const std::string &path, Device device, StreamHandle stream = nullptr);
    const Weight &at(const std::string &name) const;
};
} // namespace llm
