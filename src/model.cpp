#include "model.h"
#include "kernels.h"
#include <algorithm>
namespace llm {
#ifdef LLM_CPU_REFERENCE
constexpr Device model_device = Device::CPU;
#else
constexpr Device model_device = Device::CUDA;
#endif
namespace {
Tensor activation(int r, int c) {
    return Tensor({size_t(r), size_t(c)}, DType::F32, model_device);
}
} // namespace
Model::Model(const std::string &file, int batch, int blocks, int bs)
    : weights_(file, model_device, stream_), blocks_(blocks), block_size_(bs), batch_(batch) {
    if (batch <= 0 || blocks <= 0 || bs <= 0)
        throw std::invalid_argument("invalid runtime geometry");
    const auto &c = config();
    max_blocks_ = (c.context + bs - 1) / bs;
    int d = c.hidden / c.heads, kv = c.kvheads * d;
    keys_ = Tensor({c.layers, size_t(blocks), size_t(bs), c.kvheads, size_t(d)}, DType::F32,
                   model_device);
    values_ = Tensor(keys_.shape(), DType::F32, model_device);
    ids_ = Tensor({size_t(batch)}, DType::I32, model_device);
    positions_ = Tensor({size_t(batch)}, DType::I32, model_device);
    tables_ = Tensor({size_t(batch), size_t(max_blocks_)}, DType::I32, model_device);
    x_ = activation(batch, c.hidden);
    norm_ = activation(batch, c.hidden);
    q_ = activation(batch, c.hidden);
    k_ = activation(batch, kv);
    v_ = activation(batch, kv);
    attention_ = activation(batch, c.hidden);
    projected_ = activation(batch, c.hidden);
    gate_ = activation(batch, c.intermediate);
    up_ = activation(batch, c.intermediate);
    logits_ = activation(batch, c.vocab);
}
std::string Model::weight_precision() const {
    auto type = weights_.at("lm_head.weight").data.dtype();
    for (const auto &entry : weights_.weights) {
        if (entry.second.data.shape().size() == 2 && entry.first != "model.embed_tokens.weight" &&
            entry.second.data.dtype() != type)
            return "mixed";
    }
    return type == DType::F32 ? "fp32" : type == DType::F16 ? "fp16" : "int8";
}
void Model::project(const std::string &name, const Tensor &x, Tensor &y, int rows, int in,
                    int out) {
    const auto &w = weights_.at(name);
    gpu::linear(x.ptr<float>(), w.data.data(), w.scales.ptr<float>(), y.ptr<float>(), rows, out, in,
                w.data.dtype(), w.group, stream_);
}
std::vector<float> Model::forward(const std::vector<int> &ids, const std::vector<int> &pos,
                                  const std::vector<std::vector<int>> &tables) {
    int n = int(ids.size());
    const auto &c = config();
    if (!n || n > batch_ || pos.size() != ids.size() || tables.size() != ids.size())
        throw std::invalid_argument("forward batch shape");
    std::vector<int> hi(batch_, 0), hp(batch_, 0), ht(size_t(batch_) * max_blocks_, -1);
    for (int i = 0; i < n; ++i) {
        if (ids[i] < 0 || uint32_t(ids[i]) >= c.vocab || pos[i] < 0 ||
            uint32_t(pos[i]) >= c.context || tables[i].size() > size_t(max_blocks_) ||
            tables[i].size() <= size_t(pos[i] / block_size_))
            throw std::invalid_argument("forward token/position/table");
        hi[i] = ids[i];
        hp[i] = pos[i];
        for (size_t j = 0; j < tables[i].size(); ++j) {
            int b = tables[i][j];
            if (b < 0 || b >= blocks_)
                throw std::invalid_argument("physical block out of range");
            ht[i * max_blocks_ + j] = b;
        }
    }
    ids_.copy_from_host(hi.data(), ids_.bytes(), stream_);
    positions_.copy_from_host(hp.data(), positions_.bytes(), stream_);
    tables_.copy_from_host(ht.data(), tables_.bytes(), stream_);
    auto &e = weights_.at("model.embed_tokens.weight");
    gpu::embedding(e.data.data(), e.data.dtype(), ids_.ptr<int>(), x_.ptr<float>(), n, c.hidden,
                   stream_);
    for (uint32_t layer = 0; layer < c.layers; ++layer) {
        std::string p = "model.layers." + std::to_string(layer) + ".";
        auto &w = weights_.at(p + "input_layernorm.weight");
        gpu::rmsnorm(x_.ptr<float>(), w.data.ptr<float>(), norm_.ptr<float>(), n, c.hidden,
                     c.epsilon, stream_);
        project(p + "self_attn.q_proj.weight", norm_, q_, n, c.hidden, c.hidden);
        int d = c.hidden / c.heads, kv = c.kvheads * d;
        project(p + "self_attn.k_proj.weight", norm_, k_, n, c.hidden, kv);
        project(p + "self_attn.v_proj.weight", norm_, v_, n, c.hidden, kv);
        gpu::rope(q_.ptr<float>(), n, c.heads, d, d, positions_.ptr<int>(), c.rope_base, stream_);
        gpu::rope(k_.ptr<float>(), n, c.kvheads, d, d, positions_.ptr<int>(), c.rope_base, stream_);
        gpu::cache_write(k_.ptr<float>(), v_.ptr<float>(), keys_.ptr<float>(), values_.ptr<float>(),
                         tables_.ptr<int>(), positions_.ptr<int>(), n, layer, blocks_, block_size_,
                         max_blocks_, c.kvheads, d, stream_);
        gpu::paged_attention(q_.ptr<float>(), keys_.ptr<float>(), values_.ptr<float>(),
                             attention_.ptr<float>(), tables_.ptr<int>(), positions_.ptr<int>(), n,
                             layer, blocks_, block_size_, max_blocks_, c.heads, c.kvheads, d,
                             stream_);
        project(p + "self_attn.o_proj.weight", attention_, projected_, n, c.hidden, c.hidden);
        gpu::residual(x_.ptr<float>(), projected_.ptr<float>(), size_t(n) * c.hidden, stream_);
        auto &wn = weights_.at(p + "post_attention_layernorm.weight");
        gpu::rmsnorm(x_.ptr<float>(), wn.data.ptr<float>(), norm_.ptr<float>(), n, c.hidden,
                     c.epsilon, stream_);
        project(p + "mlp.gate_proj.weight", norm_, gate_, n, c.hidden, c.intermediate);
        project(p + "mlp.up_proj.weight", norm_, up_, n, c.hidden, c.intermediate);
        gpu::swiglu(gate_.ptr<float>(), up_.ptr<float>(), size_t(n) * c.intermediate, stream_);
        project(p + "mlp.down_proj.weight", gate_, projected_, n, c.intermediate, c.hidden);
        gpu::residual(x_.ptr<float>(), projected_.ptr<float>(), size_t(n) * c.hidden, stream_);
    }
    gpu::rmsnorm(x_.ptr<float>(), weights_.at("model.norm.weight").data.ptr<float>(),
                 norm_.ptr<float>(), n, c.hidden, c.epsilon, stream_);
    project("lm_head.weight", norm_, logits_, n, c.hidden, c.vocab);
    std::vector<float> out(size_t(n) * c.vocab);
#ifdef LLM_CPU_REFERENCE
    std::copy_n(logits_.ptr<float>(), out.size(), out.data());
#else
    CUDA_CHECK(cudaMemcpyAsync(out.data(), logits_.data(), out.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
#endif
    stream_.synchronize();
    return out;
}
} // namespace llm
