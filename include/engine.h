#pragma once
#include "model.h"
#include "scheduler.h"
#include <chrono>
#include <random>
#include <string>
namespace llm {
struct InferenceRequest {
    int request_id = 0;
    std::vector<int> input_tokens, output_tokens;
    int max_new_tokens = 128;
    float temperature = 0;
    int top_k = 0;
    double arrival_ms = 0;
    bool finished = false;
    std::string error;
    double ttft_ms = 0, latency_ms = 0;
};
class Engine {
    static int token_capacity(int batch) {
        if (batch <= 0 || batch > 256)
            throw std::invalid_argument("batch size must be 1..256");
        return batch * 16;
    }
    Model model_;
    int batch_, blocks_, block_size_;
    std::mt19937 random_;
    int sample(const float *, float, int);

  public:
    Engine(const std::string &file, int batch, int blocks, int block_size, unsigned seed = 42)
        : model_(file, token_capacity(batch), blocks, block_size), batch_(batch), blocks_(blocks),
          block_size_(block_size), random_(seed) {}
    const ModelConfig &config() const {
        return model_.config();
    }
    std::string weight_precision() const {
        return model_.weight_precision();
    }
    std::vector<int> generate(const std::vector<int> &prompt_tokens, int max_new_tokens,
                              float temperature, int top_k);
    void run(std::vector<InferenceRequest> &requests, Policy policy = Policy::Continuous);
    std::vector<float> logits(const std::vector<int> &tokens, size_t chunk_size = 1);
    std::pair<double, size_t> score(const std::vector<int> &tokens, size_t context);
};
} // namespace llm
