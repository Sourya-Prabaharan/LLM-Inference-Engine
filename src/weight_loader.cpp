#include "weight_loader.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
namespace llm {
void ModelConfig::validate() const {
    if (!hidden || !intermediate || !layers || !heads || !kvheads || !vocab || !context ||
        hidden % heads || heads % kvheads || hidden / heads > 128 || (hidden / heads) % 2 ||
        bos >= vocab || eos >= vocab || !(epsilon > 0) || !std::isfinite(epsilon) ||
        !(rope_base > 0) || !std::isfinite(rope_base))
        throw std::invalid_argument("unsupported Llama configuration");
    if (hidden > 65536 || intermediate > 262144 || layers > 256 || heads > 1024 ||
        vocab > 1000000 || context > 1048576)
        throw std::invalid_argument("model limits exceeded");
}
namespace {
void read(std::ifstream &f, void *p, size_t n) {
    if (n > size_t(std::numeric_limits<std::streamsize>::max()) ||
        !f.read(static_cast<char *>(p), std::streamsize(n)))
        throw std::runtime_error("truncated model file");
}
template <class T> T scalar(std::ifstream &f) {
    T v;
    read(f, &v, sizeof v);
    return v;
}
} // namespace
WeightLoader::WeightLoader(const std::string &path, Device device, StreamHandle stream) {
    const uint16_t endian = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian) != 1)
        throw std::runtime_error("little endian host required");
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open model: " + path);
    char magic[8];
    read(f, magic, 8);
    if (std::memcmp(magic, "LLMENG01", 8))
        throw std::runtime_error("invalid model magic");
    config.hidden = scalar<uint32_t>(f);
    config.intermediate = scalar<uint32_t>(f);
    config.layers = scalar<uint32_t>(f);
    config.heads = scalar<uint32_t>(f);
    config.kvheads = scalar<uint32_t>(f);
    config.vocab = scalar<uint32_t>(f);
    config.context = scalar<uint32_t>(f);
    config.bos = scalar<uint32_t>(f);
    config.eos = scalar<uint32_t>(f);
    config.epsilon = scalar<float>(f);
    config.rope_base = scalar<float>(f);
    config.validate();
    uint32_t count = scalar<uint32_t>(f);
    if (count != 3 + 9 * config.layers)
        throw std::runtime_error("unexpected tensor count");
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t len = scalar<uint32_t>(f);
        if (!len || len > 512)
            throw std::runtime_error("invalid name length");
        std::string name(len, '\0');
        read(f, name.data(), len);
        auto type = static_cast<DType>(scalar<uint32_t>(f));
        uint32_t rank = scalar<uint32_t>(f), group = scalar<uint32_t>(f);
        if (rank < 1 || rank > 2 || type == DType::I32)
            throw std::runtime_error("invalid weight type/rank");
        std::vector<size_t> shape;
        for (uint32_t j = 0; j < rank; ++j) {
            uint64_t d = scalar<uint64_t>(f);
            if (!d || d > 1000000)
                throw std::runtime_error("invalid weight dimension");
            shape.push_back(size_t(d));
        }
        uint64_t n = scalar<uint64_t>(f);
        size_t expected = dtype_size(type);
        for (auto d : shape) {
            if (d > SIZE_MAX / expected)
                throw std::runtime_error("weight overflow");
            expected *= d;
        }
        if (n != expected || n > uint64_t(16) * 1024 * 1024 * 1024)
            throw std::runtime_error("weight byte count mismatch");
        auto payload = f.tellg();
        f.seekg(0, std::ios::end);
        auto end = f.tellg();
        if (payload < 0 || end - payload < std::streamoff(n))
            throw std::runtime_error("truncated weights");
        f.seekg(payload);
        std::vector<uint8_t> host(n);
        read(f, host.data(), n);
        Weight w;
        w.data = Tensor(shape, type, device);
        w.data.copy_from_host(host.data(), n, stream);
#ifdef LLM_CUDA
        if (device == Device::CUDA)
            CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
        w.group = group;
        if (type == DType::I8) {
            if (rank != 2 || !group || group > shape[1])
                throw std::runtime_error("INT8 group metadata");
            size_t ng = (shape[1] + group - 1) / group;
            std::vector<float> scales(shape[0] * ng);
            read(f, scales.data(), scales.size() * sizeof(float));
            for (float s : scales)
                if (!(s > 0) || !std::isfinite(s))
                    throw std::runtime_error("invalid quantization scale");
            w.scales = Tensor({shape[0], ng}, DType::F32, device);
            w.scales.copy_from_host(scales.data(), w.scales.bytes(), stream);
#ifdef LLM_CUDA
            if (device == Device::CUDA)
                CUDA_CHECK(cudaStreamSynchronize(stream));
#endif
        } else if (group)
            throw std::runtime_error("unexpected quantization group");
        if (!weights.emplace(name, std::move(w)).second)
            throw std::runtime_error("duplicate weight: " + name);
    }
    if (f.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("trailing model data");
    auto expect = [&](const std::string &n, std::vector<size_t> shape, bool norm = false,
                      bool embedding = false) {
        const auto &w = at(n);
        if (w.data.shape() != shape)
            throw std::runtime_error("shape mismatch: " + n);
        if (norm && w.data.dtype() != DType::F32)
            throw std::runtime_error("norm must be FP32");
        if (embedding && w.data.dtype() == DType::I8)
            throw std::runtime_error("embedding cannot be INT8");
    };
    size_t h = config.hidden, k = config.kvheads * (h / config.heads), m = config.intermediate;
    expect("model.embed_tokens.weight", {config.vocab, h}, false, true);
    expect("model.norm.weight", {h}, true);
    expect("lm_head.weight", {config.vocab, h});
    for (uint32_t i = 0; i < config.layers; ++i) {
        std::string p = "model.layers." + std::to_string(i) + ".";
        expect(p + "input_layernorm.weight", {h}, true);
        expect(p + "post_attention_layernorm.weight", {h}, true);
        expect(p + "self_attn.q_proj.weight", {h, h});
        expect(p + "self_attn.k_proj.weight", {k, h});
        expect(p + "self_attn.v_proj.weight", {k, h});
        expect(p + "self_attn.o_proj.weight", {h, h});
        expect(p + "mlp.gate_proj.weight", {m, h});
        expect(p + "mlp.up_proj.weight", {m, h});
        expect(p + "mlp.down_proj.weight", {h, m});
    }
}
const Weight &WeightLoader::at(const std::string &n) const {
    auto it = weights.find(n);
    if (it == weights.end())
        throw std::runtime_error("missing weight: " + n);
    return it->second;
}
} // namespace llm
