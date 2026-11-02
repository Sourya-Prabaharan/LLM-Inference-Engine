#include "kv_cache.h"
#include <limits>
#include <stdexcept>
namespace llm {
PagedKVCache::PagedKVCache(size_t n, size_t b) : block_size_(b) {
    if (!n || !b || n > size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid cache geometry");
    for (size_t i = n; i > 0; --i)
        free_.push_back(int(i - 1));
}
void PagedKVCache::allocate_request(int id) {
    if (!requests_.emplace(id, Sequence{}).second)
        throw std::invalid_argument("duplicate request");
}
void PagedKVCache::append_tokens(int id, size_t count) {
    auto &s = requests_.at(id);
    if (count > std::numeric_limits<size_t>::max() - s.tokens)
        throw std::overflow_error("token count overflow");
    size_t total = s.tokens + count;
    size_t need = total / block_size_ + (total % block_size_ != 0);
    size_t extra = need - s.blocks.size();
    if (extra > free_.size())
        throw std::runtime_error("KV cache exhausted");
    s.blocks.reserve(need);
    for (size_t i = 0; i < extra; ++i) {
        s.blocks.push_back(free_.back());
        free_.pop_back();
    }
    s.tokens = total;
}
void PagedKVCache::release_request(int id) {
    auto it = requests_.find(id);
    if (it == requests_.end())
        throw std::out_of_range("unknown request");
    for (int b : it->second.blocks)
        free_.push_back(b);
    requests_.erase(it);
}
const std::vector<int> &PagedKVCache::get_block_table(int id) const {
    return requests_.at(id).blocks;
}
size_t PagedKVCache::tokens(int id) const {
    return requests_.at(id).tokens;
}
} // namespace llm
