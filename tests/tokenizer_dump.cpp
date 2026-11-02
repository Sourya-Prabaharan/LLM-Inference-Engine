#include "tokenizer.h"
#include <iostream>
int main(int argc, char **argv) {
    try {
        if (argc < 3)
            throw std::invalid_argument("tokenizer.model text [chat]");
        llm::Tokenizer t(argv[1], 1, 2, 32000);
        std::string text = argv[2];
        auto ids = t.encode(argc > 3 ? llm::Tokenizer::chat(text) : text, argc <= 3);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i)
                std::cout << ',';
            std::cout << ids[i];
        }
        std::cout << '\n' << t.decode(ids) << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
