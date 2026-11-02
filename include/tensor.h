#pragma once
#include "cuda_utils.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace llm {
enum class DType : uint32_t { F32 = 0, F16 = 1, I8 = 2, I32 = 3 };
enum class Device { CPU, CUDA };
size_t dtype_size(DType type);
class Tensor {
    void *data_ = nullptr;
    size_t bytes_ = 0;
    std::vector<size_t> shape_;
    DType dtype_ = DType::F32;
    Device device_ = Device::CPU;
    void release() noexcept;

  public:
    static std::atomic<size_t> gpu_bytes, peak_gpu_bytes;
    Tensor() = default;
    Tensor(std::vector<size_t> shape, DType dtype, Device device);
    ~Tensor() {
        release();
    }
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
    Tensor(Tensor &&other) noexcept;
    Tensor &operator=(Tensor &&other) noexcept;
    void *data() {
        return data_;
    }
    const void *data() const {
        return data_;
    }
    template <class T> T *ptr() {
        return static_cast<T *>(data_);
    }
    template <class T> const T *ptr() const {
        return static_cast<const T *>(data_);
    }
    size_t bytes() const {
        return bytes_;
    }
    size_t size() const {
        return bytes_ / dtype_size(dtype_);
    }
    DType dtype() const {
        return dtype_;
    }
    Device device() const {
        return device_;
    }
    const std::vector<size_t> &shape() const {
        return shape_;
    }
    void copy_from_host(const void *source, size_t bytes, StreamHandle stream = nullptr);
    void copy_to_host(void *dest, size_t bytes, StreamHandle stream = nullptr) const;
};
// Grow-only workspace: callers must synchronize before growth and before destruction.
class Workspace {
    Tensor storage_;
    Device device_;

  public:
    explicit Workspace(Device d) : device_(d) {}
    void *reserve(size_t bytes) {
        if (bytes > storage_.bytes())
            storage_ = Tensor({bytes}, DType::I8, device_);
        return storage_.data();
    }
    size_t capacity() const {
        return storage_.bytes();
    }
};
} // namespace llm
