#pragma once
#include <stdexcept>
#include <string>
#ifdef LLM_CUDA
#include <cuda_runtime.h>
namespace llm {
inline void check_cuda(cudaError_t e, const char *file, int line) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " +
                                 cudaGetErrorString(e));
}
#define CUDA_CHECK(expr) ::llm::check_cuda((expr), __FILE__, __LINE__)
using StreamHandle = cudaStream_t;
class Stream {
    cudaStream_t stream_{};

  public:
    Stream() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }
    ~Stream() {
        if (stream_)
            cudaStreamDestroy(stream_);
    }
    Stream(const Stream &) = delete;
    Stream &operator=(const Stream &) = delete;
    operator cudaStream_t() const {
        return stream_;
    }
    void synchronize() const {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
};
} // namespace llm
#else
namespace llm {
using StreamHandle = void *;
#ifdef LLM_CPU_REFERENCE
// Test-only synchronous execution context. Never used by the CUDA executable.
class Stream {
  public:
    operator StreamHandle() const {
        return nullptr;
    }
    void synchronize() const {}
};
#endif
} // namespace llm
#endif
