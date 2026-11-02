#pragma once
#include "kv_cache.h"
#include "weight_loader.h"
#include <memory>
namespace llm {
class Model {
    Stream stream_;
    WeightLoader weights_;
    int blocks_, block_size_, max_blocks_, batch_;
    Tensor keys_, values_, ids_, positions_, tables_;
    Tensor x_, norm_, q_, k_, v_, attention_, projected_, gate_, up_, logits_;
    void project(const std::string &, const Tensor &, Tensor &, int rows, int in, int out);

  public:
    Model(const std::string &file, int batch, int blocks, int block_size);
    const ModelConfig &config() const {
        return weights_.config;
    }
    std::string weight_precision() const;
    std::vector<float> forward(const std::vector<int> &ids, const std::vector<int> &positions,
                               const std::vector<std::vector<int>> &tables);
};
} // namespace llm
