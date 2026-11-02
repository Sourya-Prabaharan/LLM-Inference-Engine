#include "engine.h"
#include "tokenizer.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
int main(int argc, char **argv) {
    try {
        std::unordered_map<std::string, std::string> args;
        bool interactive = false, benchmark = false, chat = false;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help") {
                std::cout << "llm_inference --model DIR [--prompt TEXT | --tokens 1,2,3 | "
                             "--requests FILE] [--max-tokens 128] [--temperature 0] [--top-k 0] "
                             "[--precision fp16|fp32|int8] [--batch-size 8] [--kv-block-size 16] "
                             "[--kv-blocks 1024] [--policy static|continuous] [--seed 42] [--chat] "
                             "[--interactive] [--benchmark] [--logits FILE]\nRequests TSV: "
                             "arrival_ms<TAB>max_tokens<TAB>prompt\n";
                return 0;
            }
            if (key == "--interactive")
                interactive = true;
            else if (key == "--benchmark")
                benchmark = true;
            else if (key == "--chat")
                chat = true;
            else {
                if (i + 1 == argc)
                    throw std::invalid_argument("missing flag value");
                if (key != "--model" && key != "--prompt" && key != "--tokens" &&
                    key != "--requests" && key != "--max-tokens" && key != "--temperature" &&
                    key != "--top-k" && key != "--precision" && key != "--batch-size" &&
                    key != "--kv-block-size" && key != "--kv-blocks" && key != "--policy" &&
                    key != "--seed" && key != "--logits")
                    throw std::invalid_argument("unknown option: " + key);
                args[key] = argv[++i];
            }
        }
        auto get = [&](std::string k, std::string fallback) {
            auto i = args.find(k);
            return i == args.end() ? fallback : i->second;
        };
        std::string dir = get("--model", "");
        if (dir.empty())
            throw std::invalid_argument("--model required");
        if ((args.count("--tokens") + args.count("--requests") + args.count("--prompt")) > 1)
            throw std::invalid_argument("choose one of --tokens, --requests, or --prompt");
        if (interactive && (args.count("--tokens") || args.count("--requests")))
            throw std::invalid_argument("interactive requires text prompts");
        llm::Engine engine(dir + "/model.bin", std::stoi(get("--batch-size", "8")),
                           std::stoi(get("--kv-blocks", "1024")),
                           std::stoi(get("--kv-block-size", "16")),
                           std::stoul(get("--seed", "42")));
        if (args.count("--precision") && args["--precision"] != engine.weight_precision())
            throw std::invalid_argument(
                "--precision does not match actual linear weight storage; reconvert weights");
        auto &c = engine.config();
        std::unique_ptr<llm::Tokenizer> tokenizer;
        if (!args.count("--tokens"))
            tokenizer =
                std::make_unique<llm::Tokenizer>(dir + "/tokenizer.model", c.bos, c.eos, c.vocab);
        auto encode = [&](const std::string &text) {
            return tokenizer->encode(chat ? llm::Tokenizer::chat(text) : text, !chat);
        };
        std::string policy = get("--policy", "continuous");
        if (policy != "static" && policy != "continuous")
            throw std::invalid_argument("invalid policy");
        do {
            std::vector<llm::InferenceRequest> requests;
            auto make = [&](std::vector<int> ids) {
                llm::InferenceRequest r;
                r.request_id = int(requests.size());
                r.input_tokens = std::move(ids);
                r.max_new_tokens = std::stoi(get("--max-tokens", "128"));
                r.temperature = std::stof(get("--temperature", "0"));
                r.top_k = std::stoi(get("--top-k", "0"));
                return r;
            };
            if (args.count("--requests")) {
                std::ifstream file(args["--requests"]);
                if (!file)
                    throw std::runtime_error("cannot open requests file");
                std::string line;
                while (std::getline(file, line)) {
                    std::istringstream row(line);
                    std::string arrival, length, prompt;
                    if (!std::getline(row, arrival, '\t') || !std::getline(row, length, '\t') ||
                        !std::getline(row, prompt))
                        throw std::invalid_argument("invalid request TSV");
                    auto r = make(encode(prompt));
                    r.arrival_ms = std::stod(arrival);
                    r.max_new_tokens = std::stoi(length);
                    requests.push_back(std::move(r));
                }
            } else if (args.count("--tokens")) {
                std::vector<int> ids;
                std::istringstream input(args["--tokens"]);
                std::string s;
                while (std::getline(input, s, ','))
                    ids.push_back(std::stoi(s));
                requests.push_back(make(ids));
            } else {
                std::string prompt = get("--prompt", "");
                if (interactive) {
                    std::cerr << "> ";
                    if (!std::getline(std::cin, prompt))
                        break;
                }
                requests.push_back(make(encode(prompt)));
            }
            if (args.count("--logits")) {
                if (requests.size() != 1)
                    throw std::invalid_argument("logits requires one request");
                auto logits = engine.logits(requests[0].input_tokens);
                std::ofstream file(args["--logits"], std::ios::binary);
                file.write(reinterpret_cast<const char *>(logits.data()),
                           logits.size() * sizeof(float));
                if (!file)
                    throw std::runtime_error("logit export failed");
                return 0;
            }
            if (benchmark && !requests.empty()) {
                auto warm = requests[0];
                warm.max_new_tokens = 1;
                warm.arrival_ms = 0;
                std::vector<llm::InferenceRequest> w{warm};
                engine.run(w);
            }
            auto start = std::chrono::steady_clock::now();
            engine.run(requests,
                       policy == "static" ? llm::Policy::Static : llm::Policy::Continuous);
            double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            size_t total = 0;
            bool failed = false;
            if (!benchmark)
                std::cout << "Backend: CUDA | FP32 activations | Paged KV | " << policy
                          << " batching\n";
            for (auto &r : requests) {
                total += r.output_tokens.size();
                if (!r.error.empty()) {
                    failed = true;
                    std::cerr << "Request " << r.request_id << ": " << r.error << '\n';
                }
                if (!benchmark) {
                    if (tokenizer)
                        std::cout << tokenizer->decode(r.output_tokens) << '\n';
                    else {
                        for (int token : r.output_tokens)
                            std::cout << token << ' ';
                        std::cout << '\n';
                    }
                    std::cout << "Prompt tokens: " << r.input_tokens.size()
                              << " Generated tokens: " << r.output_tokens.size()
                              << " TTFT: " << r.ttft_ms << " ms Completion: " << r.latency_ms
                              << " ms\n";
                }
            }
            if (benchmark) {
                cudaDeviceProp prop{};
                int device = 0, version = 0;
                CUDA_CHECK(cudaGetDevice(&device));
                CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
                CUDA_CHECK(cudaRuntimeGetVersion(&version));
                std::cout << std::setprecision(10) << "{\"backend\":\"CUDA\",\"gpu\":\""
                          << prop.name << "\",\"cuda_runtime\":" << version << ",\"policy\":\""
                          << policy << "\",\"weight_precision\":\"" << engine.weight_precision()
                          << "\",\"activation_precision\":\"fp32\",\"kv_precision\":\"fp32\","
                             "\"batch_size\":"
                          << std::stoi(get("--batch-size", "8"))
                          << ",\"kv_block_size\":" << std::stoi(get("--kv-block-size", "16"))
                          << ",\"kv_blocks\":" << std::stoi(get("--kv-blocks", "1024"))
                          << ",\"seconds\":" << seconds << ",\"generated_tokens\":" << total
                          << ",\"tokens_per_second\":" << (seconds ? total / seconds : 0)
                          << ",\"peak_tracked_gpu_bytes\":" << llm::Tensor::peak_gpu_bytes.load()
                          << ",\"requests\":[";
                for (size_t i = 0; i < requests.size(); ++i) {
                    if (i)
                        std::cout << ',';
                    auto &r = requests[i];
                    std::cout << "{\"id\":" << r.request_id
                              << ",\"failed\":" << (r.error.empty() ? "false" : "true")
                              << ",\"ttft_ms\":" << r.ttft_ms << ",\"latency_ms\":" << r.latency_ms
                              << ",\"tokens\":" << r.output_tokens.size() << ",\"token_ids\":[";
                    for (size_t j = 0; j < r.output_tokens.size(); ++j) {
                        if (j)
                            std::cout << ',';
                        std::cout << r.output_tokens[j];
                    }
                    std::cout << "]}";
                }
                std::cout << "]}\n";
            } else
                std::cout << "Overall generated tokens/s: " << (seconds ? total / seconds : 0)
                          << " Peak tracked GPU MiB: "
                          << llm::Tensor::peak_gpu_bytes.load() / 1048576.0 << '\n';
            if (failed)
                return 2;
        } while (interactive);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
