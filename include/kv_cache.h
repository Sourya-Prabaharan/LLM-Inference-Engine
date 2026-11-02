#pragma once
#include <cstddef>
#include <unordered_map>
#include <vector>
namespace llm {
// Host allocator; GPU storage uses [layer, physical_block, token, kv_head, dim].
// All mutations must occur after the owning execution stream completes.
class PagedKVCache {
    struct Sequence {
        size_t tokens = 0;
        std::vector<int> blocks;
    };
    size_t block_size_;
    std::vector<int> free_;
    std::unordered_map<int, Sequence> requests_;

  public:
    PagedKVCache(size_t blocks, size_t block_size);
    void allocate_request(int id);
    void append_tokens(int id, size_t count);
    void release_request(int id);
    const std::vector<int> &get_block_table(int id) const;
    size_t tokens(int id) const;
    size_t free_blocks() const {
        return free_.size();
    }
    size_t block_size() const {
        return block_size_;
    }
    size_t reserved_tokens(int id) const {
        return get_block_table(id).size() * block_size_;
    }
};
} // namespace llm
