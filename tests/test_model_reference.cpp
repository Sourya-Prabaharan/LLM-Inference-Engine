#include "engine.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
int main(int argc, char **argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("model.bin output.bin");
        llm::Engine engine(argv[1], 3, 64, 4);
        std::vector<int> tokens = {1,  3,  5,  7,  9,  11, 13, 15, 17, 19,
                                   21, 23, 25, 27, 29, 31, 4,  6,  8};
        auto logits = engine.logits(tokens);
        for (size_t chunk : {size_t(4), size_t(16), size_t(48)}) {
            auto packed = engine.logits(tokens, chunk);
            for (size_t i = 0; i < logits.size(); ++i)
                if (std::abs(packed[i] - logits[i]) > 1e-6f)
                    throw std::runtime_error("chunked logits mismatch");
        }
        auto score = engine.score(tokens, 8);
        std::ofstream f(argv[2], std::ios::binary);
        f.write(reinterpret_cast<const char *>(logits.data()), logits.size() * sizeof(float));
        if (!f)
            throw std::runtime_error("output write failed");
        std::vector<std::vector<int>> prompts = {{1, 3}, tokens, {1, 5, 7, 11, 13}};
        std::vector<std::vector<int>> expected;
        for (auto &p : prompts)
            expected.push_back(engine.generate(p, 5, 0, 0));
        for (auto policy : {llm::Policy::Static, llm::Policy::Continuous}) {
            std::vector<llm::InferenceRequest> requests;
            for (size_t i = 0; i < prompts.size(); ++i) {
                llm::InferenceRequest r;
                r.request_id = int(i);
                r.input_tokens = prompts[i];
                r.max_new_tokens = 5;
                requests.push_back(r);
            }
            engine.run(requests, policy);
            for (size_t i = 0; i < prompts.size(); ++i)
                if (requests[i].output_tokens != expected[i] || !requests[i].error.empty())
                    throw std::runtime_error("batch parity failed");
        }
        std::cout << "CPU reference only: batched/chunked/static/continuous parity passed\n";
        for (int token : expected[0])
            std::cout << token << ' ';
        std::cout << '\n';
        std::cout << std::setprecision(17) << score.first << ' ' << score.second << '\n';
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
