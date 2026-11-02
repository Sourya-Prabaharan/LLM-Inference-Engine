#include "tokenizer.h"
#include <stdexcept>
namespace llm {
Tokenizer::Tokenizer(const std::string &path, int bos, int eos, int vocab) : bos_(bos), eos_(eos) {
    auto status = processor_.Load(path);
    if (!status.ok())
        throw std::runtime_error(status.ToString());
    if (processor_.GetPieceSize() != vocab || processor_.bos_id() != bos ||
        processor_.eos_id() != eos)
        throw std::runtime_error("tokenizer/model vocabulary mismatch");
}
std::vector<int> Tokenizer::encode(const std::string &text, bool bos) const {
    std::vector<int> result;
    if (bos)
        result.push_back(bos_);
    // HF's Llama legacy tokenizer recognizes registered BOS/EOS/UNK in raw strings.
    size_t begin = 0;
    while (begin < text.size()) {
        size_t next = std::string::npos;
        int special = -1;
        std::string marker;
        for (int id : {processor_.unk_id(), bos_, eos_}) {
            auto piece = processor_.IdToPiece(id);
            size_t p = text.find(piece, begin);
            if (p < next) {
                next = p;
                special = id;
                marker = piece;
            }
        }
        size_t end = next == std::string::npos ? text.size() : next;
        if (end > begin) {
            std::vector<int> chunk;
            auto s = processor_.Encode(text.substr(begin, end - begin), &chunk);
            if (!s.ok())
                throw std::runtime_error(s.ToString());
            result.insert(result.end(), chunk.begin(), chunk.end());
        }
        if (next == std::string::npos)
            break;
        result.push_back(special);
        begin = next + marker.size();
    }
    return result;
}
std::string Tokenizer::decode(const std::vector<int> &ids) const {
    std::vector<int> plain;
    for (int id : ids) {
        if (id < 0 || id >= processor_.GetPieceSize())
            throw std::invalid_argument("token out of range");
        if (id != bos_ && id != eos_)
            plain.push_back(id);
    }
    std::string out;
    auto s = processor_.Decode(plain, &out);
    if (!s.ok())
        throw std::runtime_error(s.ToString());
    return out;
}
std::string Tokenizer::chat(const std::string &user) {
    return "<|user|>\n" + user + "</s>\n<|assistant|>\n";
}
} // namespace llm
