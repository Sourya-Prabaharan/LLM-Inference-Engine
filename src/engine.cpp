#include "engine.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>
#include <unordered_map>
namespace llm {
int Engine::sample(const float *logits, float temperature, int top_k) {
    int v = config().vocab;
    for (int i = 0; i < v; ++i)
        if (!std::isfinite(logits[i]))
            throw std::runtime_error("nonfinite model logits");
    if (temperature == 0)
        return int(std::max_element(logits, logits + v) - logits);
    std::vector<int> ids(v);
    std::iota(ids.begin(), ids.end(), 0);
    int k = top_k == 0 ? v : std::min(top_k, v);
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::vector<double> weights(k);
    float m = logits[ids[0]];
    for (int i = 0; i < k; ++i)
        weights[i] = std::exp(double(logits[ids[i]] - m) / temperature);
    std::discrete_distribution<int> distribution(weights.begin(), weights.end());
    return ids[distribution(random_)];
}
std::vector<int> Engine::generate(const std::vector<int> &p, int n, float t, int k) {
    InferenceRequest r;
    r.input_tokens = p;
    r.max_new_tokens = n;
    r.temperature = t;
    r.top_k = k;
    std::vector<InferenceRequest> requests{r};
    run(requests);
    if (!requests[0].error.empty())
        throw std::runtime_error(requests[0].error);
    return requests[0].output_tokens;
}
void Engine::run(std::vector<InferenceRequest> &requests, Policy policy) {
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    auto elapsed = [&] {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    };
    Scheduler scheduler(batch_, policy);
    PagedKVCache cache(blocks_, block_size_);
    std::unordered_map<int, size_t> lookup;
    std::vector<size_t> pending;
    for (size_t i = 0; i < requests.size(); ++i) {
        auto &r = requests[i];
        if (!lookup.emplace(r.request_id, i).second)
            throw std::invalid_argument("duplicate request id");
        if (r.input_tokens.empty() || r.max_new_tokens < 0 || r.temperature < 0 ||
            !std::isfinite(r.temperature) || r.top_k < 0 || r.arrival_ms < 0 ||
            !std::isfinite(r.arrival_ms) ||
            r.input_tokens.size() + size_t(r.max_new_tokens) > config().context)
            throw std::invalid_argument("request parameters/context");
        for (int token : r.input_tokens)
            if (token < 0 || uint32_t(token) >= config().vocab)
                throw std::invalid_argument("invalid prompt token");
        r.output_tokens.clear();
        r.error.clear();
        r.finished = r.max_new_tokens == 0;
        r.ttft_ms = r.latency_ms = 0;
        if (!r.finished)
            pending.push_back(i);
    }
    std::stable_sort(pending.begin(), pending.end(), [&](size_t a, size_t b) {
        return requests[a].arrival_ms < requests[b].arrival_ms;
    });
    size_t next = 0;
    while (next < pending.size() || !scheduler.empty()) {
        double now = elapsed();
        while (next < pending.size() && requests[pending[next]].arrival_ms <= now) {
            scheduler.submit(requests[pending[next++]].request_id);
        }
        for (int id : scheduler.admit())
            cache.allocate_request(id);
        auto active = scheduler.active();
        if (active.empty()) {
            if (next < pending.size())
                std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(
                    std::min(1.0, std::max(0.0, requests[pending[next]].arrival_ms - elapsed()))));
            continue;
        }
        std::vector<int> ids, pos, row_ids;
        std::vector<std::vector<int>> tables;
        for (int id : active) {
            auto &r = requests[lookup.at(id)];
            size_t p = cache.tokens(id);
            size_t count =
                p < r.input_tokens.size() ? std::min(size_t(16), r.input_tokens.size() - p) : 1;
            try {
                cache.append_tokens(id, count);
            } catch (const std::runtime_error &e) {
                r.finished = true;
                r.error = e.what();
                r.latency_ms = elapsed() - r.arrival_ms;
                cache.release_request(id);
                scheduler.finish(id);
                continue;
            }
            for (size_t j = 0; j < count; ++j) {
                size_t position = p + j;
                ids.push_back(position < r.input_tokens.size() ? r.input_tokens[position]
                                                               : r.output_tokens.back());
                pos.push_back(int(position));
                row_ids.push_back(id);
                tables.push_back(cache.get_block_table(id));
            }
        }
        if (ids.empty())
            continue;
        auto output = model_.forward(ids, pos, tables);
        for (size_t row = 0; row < row_ids.size(); ++row) {
            int id = row_ids[row];
            auto &r = requests[lookup.at(id)];
            if (size_t(pos[row]) + 1 < r.input_tokens.size())
                continue;
            int token = sample(output.data() + row * config().vocab, r.temperature, r.top_k);
            r.output_tokens.push_back(token);
            if (r.output_tokens.size() == 1)
                r.ttft_ms = elapsed() - r.arrival_ms;
            if (uint32_t(token) == config().eos ||
                r.output_tokens.size() >= size_t(r.max_new_tokens)) {
                r.finished = true;
                r.latency_ms = elapsed() - r.arrival_ms;
                cache.release_request(id);
                scheduler.finish(id);
            }
        }
    }
}
std::vector<float> Engine::logits(const std::vector<int> &tokens, size_t chunk_size) {
    if (tokens.empty() || tokens.size() > config().context)
        throw std::invalid_argument("logit sequence length");
    if (chunk_size == 0 || chunk_size > size_t(token_capacity(batch_)))
        throw std::invalid_argument("logit chunk exceeds token batch capacity");
    PagedKVCache cache(blocks_, block_size_);
    cache.allocate_request(0);
    std::vector<float> all;
    all.reserve(tokens.size() * config().vocab);
    for (size_t p = 0; p < tokens.size(); p += chunk_size) {
        size_t count = std::min(chunk_size, tokens.size() - p);
        cache.append_tokens(0, count);
        std::vector<int> ids(tokens.begin() + p, tokens.begin() + p + count), positions(count);
        std::iota(positions.begin(), positions.end(), int(p));
        std::vector<std::vector<int>> tables(count, cache.get_block_table(0));
        auto rows = model_.forward(ids, positions, tables);
        all.insert(all.end(), rows.begin(), rows.end());
    }
    return all;
}
std::pair<double, size_t> Engine::score(const std::vector<int> &tokens, size_t context) {
    if (context < 2 || context > config().context)
        throw std::invalid_argument("invalid scoring context");
    double nll = 0;
    size_t count = 0;
    for (size_t start = 0; start < tokens.size(); start += context) {
        size_t length = std::min(context, tokens.size() - start);
        if (length < 2)
            continue;
        std::vector<int> window(tokens.begin() + start, tokens.begin() + start + length);
        auto output = logits(window, size_t(token_capacity(batch_)));
        for (size_t r = 0; r + 1 < length; ++r) {
            const float *row = output.data() + r * config().vocab;
            double maximum = *std::max_element(row, row + config().vocab), sum = 0;
            for (size_t i = 0; i < config().vocab; ++i) {
                if (!std::isfinite(row[i]))
                    throw std::runtime_error("nonfinite scoring logits");
                sum += std::exp(double(row[i]) - maximum);
            }
            nll += maximum + std::log(sum) - row[window[r + 1]];
            ++count;
        }
    }
    if (count == 0)
        throw std::invalid_argument("no tokens to score");
    return {nll, count};
}
} // namespace llm
