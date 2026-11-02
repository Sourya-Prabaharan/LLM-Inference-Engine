#pragma once
#include <cstddef>
#include <deque>
#include <vector>
namespace llm {
enum class Policy { Static, Continuous };
// Admission/state only: execution is supplied by the engine as one GPU batch.
class Scheduler {
    size_t capacity_;
    Policy policy_;
    std::deque<int> waiting_;
    std::vector<int> active_;

  public:
    Scheduler(size_t capacity, Policy policy);
    void submit(int id);
    std::vector<int> admit();
    void finish(int id);
    const std::vector<int> &active() const {
        return active_;
    }
    bool empty() const {
        return waiting_.empty() && active_.empty();
    }
};
} // namespace llm
