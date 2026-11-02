#include "scheduler.h"
#include <algorithm>
#include <stdexcept>
namespace llm {
Scheduler::Scheduler(size_t n, Policy p) : capacity_(n), policy_(p) {
    if (!n)
        throw std::invalid_argument("zero batch capacity");
}
void Scheduler::submit(int id) {
    if (std::find(waiting_.begin(), waiting_.end(), id) != waiting_.end() ||
        std::find(active_.begin(), active_.end(), id) != active_.end())
        throw std::invalid_argument("duplicate request");
    waiting_.push_back(id);
}
std::vector<int> Scheduler::admit() {
    std::vector<int> added;
    if (policy_ == Policy::Static && !active_.empty())
        return added;
    while (active_.size() < capacity_ && !waiting_.empty()) {
        int id = waiting_.front();
        waiting_.pop_front();
        active_.push_back(id);
        added.push_back(id);
    }
    return added;
}
void Scheduler::finish(int id) {
    auto it = std::find(active_.begin(), active_.end(), id);
    if (it == active_.end())
        throw std::invalid_argument("request is not active");
    active_.erase(it);
}
} // namespace llm
