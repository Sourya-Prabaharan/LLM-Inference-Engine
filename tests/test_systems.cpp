#include "kv_cache.h"
#include "quantization.h"
#include "scheduler.h"
#include "tensor.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#define REQUIRE(x)                                                                                 \
    do {                                                                                           \
        if (!(x))                                                                                  \
            throw std::runtime_error("failed: " #x);                                               \
    } while (0)
template <class F> void fails(F f) {
    bool threw = false;
    try {
        f();
    } catch (const std::exception &) {
        threw = true;
    }
    REQUIRE(threw);
}
int main() {
    using namespace llm;
    Tensor t({2, 3}, DType::F32, Device::CPU);
    float a[] = {1, 2, 3, 4, 5, 6}, b[6];
    t.copy_from_host(a, sizeof a);
    Tensor u = std::move(t);
    REQUIRE(t.data() == nullptr);
    u.copy_to_host(b, sizeof b);
    for (int i = 0; i < 6; ++i)
        REQUIRE(a[i] == b[i]);
    fails([&] { u.copy_from_host(a, 4); });
    fails([] { Tensor t({0}, DType::F32, Device::CPU); });
    fails([] { Tensor t({SIZE_MAX, 2}, DType::F32, Device::CPU); });
    Workspace ws(Device::CPU);
    auto p = ws.reserve(64);
    REQUIRE(ws.reserve(8) == p);
    PagedKVCache cache(4, 16);
    cache.allocate_request(1);
    cache.allocate_request(2);
    cache.append_tokens(1, 17);
    cache.append_tokens(2, 16);
    REQUIRE(cache.free_blocks() == 1);
    std::set<int> ids(cache.get_block_table(1).begin(), cache.get_block_table(1).end());
    REQUIRE(!ids.count(cache.get_block_table(2)[0]));
    fails([&] { cache.append_tokens(1, 48); });
    REQUIRE(cache.tokens(1) == 17 && cache.free_blocks() == 1);
    cache.append_tokens(1, 15);
    REQUIRE(cache.free_blocks() == 1);
    cache.append_tokens(1, 1);
    REQUIRE(cache.free_blocks() == 0);
    cache.release_request(1);
    REQUIRE(cache.free_blocks() == 3);
    cache.release_request(2);
    REQUIRE(cache.free_blocks() == 4);
    fails([&] { cache.release_request(2); });
    for (auto policy : {Policy::Static, Policy::Continuous}) {
        Scheduler s(2, policy);
        s.submit(1);
        s.submit(2);
        s.submit(3);
        REQUIRE(s.admit().size() == 2);
        fails([&] { s.submit(1); });
        s.finish(1);
        REQUIRE(s.admit().size() == (policy == Policy::Static ? 0 : 1));
        s.finish(2);
        if (policy == Policy::Static)
            REQUIRE(s.admit().size() == 1);
        s.finish(3);
        REQUIRE(s.empty());
    }
    std::vector<float> w = {0, 0, 0, 1, -2, 3, 4};
    auto q = quantize(w, 1, 7, 3);
    auto d = dequantize(q);
    for (size_t i = 0; i < w.size(); ++i)
        REQUIRE(std::abs(w[i] - d[i]) <= q.scales[i / 3] / 2 + 1e-6f);
    fails([] { quantize({NAN}, 1, 1, 128); });
    std::cout << "PASS tensor ownership, transfers, invalid shapes, workspace reuse, KV "
                 "boundaries/isolation/OOM/reuse, scheduling policies, INT8 tails/error bounds\n";
}
