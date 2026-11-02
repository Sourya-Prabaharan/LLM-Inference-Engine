#include "tensor.h"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
namespace llm {
std::atomic<size_t> Tensor::gpu_bytes{0}, Tensor::peak_gpu_bytes{0};
size_t dtype_size(DType t) {
    switch (t) {
    case DType::F32:
    case DType::I32:
        return 4;
    case DType::F16:
        return 2;
    case DType::I8:
        return 1;
    }
    throw std::invalid_argument("invalid dtype");
}
Tensor::Tensor(std::vector<size_t> shape, DType dtype, Device device)
    : shape_(std::move(shape)), dtype_(dtype), device_(device) {
    if (shape_.empty())
        throw std::invalid_argument("tensor rank must be positive");
    bytes_ = dtype_size(dtype);
    for (size_t d : shape_) {
        if (!d || d > std::numeric_limits<size_t>::max() / bytes_)
            throw std::invalid_argument("zero or overflowing tensor dimension");
        bytes_ *= d;
    }
    if (device == Device::CPU) {
        data_ = std::malloc(bytes_);
        if (!data_)
            throw std::bad_alloc();
    } else {
#ifdef LLM_CUDA
        CUDA_CHECK(cudaMalloc(&data_, bytes_));
        size_t now = gpu_bytes.fetch_add(bytes_) + bytes_, peak = peak_gpu_bytes.load();
        while (now > peak && !peak_gpu_bytes.compare_exchange_weak(peak, now)) {
        }
#else
        throw std::runtime_error("CUDA tensor requested in host-only build");
#endif
    }
}
void Tensor::release() noexcept {
    if (data_) {
        if (device_ == Device::CPU)
            std::free(data_);
        else {
#ifdef LLM_CUDA
            cudaFree(data_);
            gpu_bytes.fetch_sub(bytes_);
#endif
        }
    }
    data_ = nullptr;
    bytes_ = 0;
}
Tensor::Tensor(Tensor &&t) noexcept {
    *this = std::move(t);
}
Tensor &Tensor::operator=(Tensor &&t) noexcept {
    if (this != &t) {
        release();
        data_ = std::exchange(t.data_, nullptr);
        bytes_ = std::exchange(t.bytes_, 0);
        shape_ = std::move(t.shape_);
        dtype_ = t.dtype_;
        device_ = t.device_;
    }
    return *this;
}
void Tensor::copy_from_host(const void *src, size_t n, StreamHandle stream) {
    if (n != bytes_ || !src)
        throw std::invalid_argument("copy size/source mismatch");
    if (device_ == Device::CPU)
        std::memcpy(data_, src, n);
    else {
#ifdef LLM_CUDA
        CUDA_CHECK(cudaMemcpyAsync(data_, src, n, cudaMemcpyHostToDevice, stream));
#else
        (void)stream;
#endif
    }
}
void Tensor::copy_to_host(void *dst, size_t n, StreamHandle stream) const {
    if (n != bytes_ || !dst)
        throw std::invalid_argument("copy size/destination mismatch");
    if (device_ == Device::CPU)
        std::memcpy(dst, data_, n);
    else {
#ifdef LLM_CUDA
        CUDA_CHECK(cudaMemcpyAsync(dst, data_, n, cudaMemcpyDeviceToHost, stream));
#else
        (void)stream;
#endif
    }
}
} // namespace llm
