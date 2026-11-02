#pragma once
#include <sentencepiece_processor.h>
#include <string>
#include <vector>
namespace llm {
class Tokenizer {
    sentencepiece::SentencePieceProcessor processor_;
    int bos_, eos_;

  public:
    Tokenizer(const std::string &path, int bos, int eos, int vocab);
    std::vector<int> encode(const std::string &text, bool bos = true) const;
    std::string decode(const std::vector<int> &ids) const;
    static std::string chat(const std::string &user);
};
} // namespace llm
