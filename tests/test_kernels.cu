#include "kernels.h"
#include "quantization.h"
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <iostream>
#include <random>
using namespace llm;
Tensor gpu_data(const std::vector<float> &x) {
    Tensor t({x.size()}, DType::F32, Device::CUDA);
    t.copy_from_host(x.data(), t.bytes());
    CUDA_CHECK(cudaDeviceSynchronize());
    return t;
}
std::vector<float> host(const Tensor &t) {
    std::vector<float> x(t.size());
    t.copy_to_host(x.data(), t.bytes());
    CUDA_CHECK(cudaDeviceSynchronize());
    return x;
}
void close(const std::vector<float> &a, const std::vector<float> &b, float tol) {
    if (a.size() != b.size())
        throw std::runtime_error("size mismatch");
    for (size_t i = 0; i < a.size(); ++i)
        if (!std::isfinite(a[i]) || std::abs(a[i] - b[i]) > tol)
            throw std::runtime_error("numerical mismatch at " + std::to_string(i) + ": " +
                                     std::to_string(a[i]) + " vs " + std::to_string(b[i]));
}
int main() {
    try {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1, 1);
        auto random = [&](int n) {
            std::vector<float> x(n);
            for (auto &v : x)
                v = dist(rng);
            return x;
        };
        for (auto shape : std::vector<std::vector<int>>{{1, 7, 19}, {17, 33, 31}, {32, 32, 32}}) {
            int m = shape[0], n = shape[1], k = shape[2];
            auto a = random(m * k), w = random(n * k);
            auto da = gpu_data(a), dw = gpu_data(w);
            Tensor out({size_t(m * n)}, DType::F32, Device::CUDA);
            std::vector<float> ref(m * n, 0);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    for (int z = 0; z < k; ++z)
                        ref[i * n + j] += a[i * k + z] * w[j * k + z];
            for (bool naive : {true, false}) {
                gpu::linear(da.ptr<float>(), dw.data(), nullptr, out.ptr<float>(), m, n, k,
                            DType::F32, 0, nullptr, naive);
                close(host(out), ref, 2e-5);
            }
            std::vector<__half> ha(a.size()), hw(w.size());
            for (size_t i = 0; i < a.size(); ++i)
                ha[i] = __float2half(a[i]);
            for (size_t i = 0; i < w.size(); ++i)
                hw[i] = __float2half(w[i]);
            Tensor dah({a.size()}, DType::F16, Device::CUDA),
                dwh({w.size()}, DType::F16, Device::CUDA);
            dah.copy_from_host(ha.data(), dah.bytes());
            dwh.copy_from_host(hw.data(), dwh.bytes());
            std::fill(ref.begin(), ref.end(), 0);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    for (int z = 0; z < k; ++z)
                        ref[i * n + j] += __half2float(ha[i * k + z]) * __half2float(hw[j * k + z]);
            gpu::linear_half(dah.data(), dwh.data(), out.ptr<float>(), m, n, k, nullptr);
            close(host(out), ref, 2e-5);
            auto quant = quantize(w, n, k, 7);
            Tensor qw({size_t(n * k)}, DType::I8, Device::CUDA);
            qw.copy_from_host(quant.values.data(), qw.bytes());
            auto scales = gpu_data(quant.scales);
            auto deq = dequantize(quant);
            std::fill(ref.begin(), ref.end(), 0);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < n; ++j)
                    for (int z = 0; z < k; ++z)
                        ref[i * n + j] += a[i * k + z] * deq[j * k + z];
            gpu::linear(da.ptr<float>(), qw.data(), scales.ptr<float>(), out.ptr<float>(), m, n, k,
                        DType::I8, 7, nullptr);
            close(host(out), ref, 2e-5);
        }
        {
            int rows = 3, d = 37;
            auto x = random(rows * d), w = random(d);
            auto dx = gpu_data(x), dw = gpu_data(w);
            Tensor out({size_t(rows * d)}, DType::F32, Device::CUDA);
            std::vector<float> ref(x.size());
            for (int r = 0; r < rows; ++r) {
                float ss = 0;
                for (int i = 0; i < d; ++i)
                    ss += x[r * d + i] * x[r * d + i];
                for (int i = 0; i < d; ++i)
                    ref[r * d + i] = x[r * d + i] / std::sqrt(ss / d + 1e-5f) * w[i];
            }
            gpu::rmsnorm(dx.ptr<float>(), dw.ptr<float>(), out.ptr<float>(), rows, d, 1e-5f,
                         nullptr);
            close(host(out), ref, 1e-5);
        }
        {
            std::vector<float> x = {1000, 1001, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
            auto dx = gpu_data(x);
            gpu::softmax(dx.ptr<float>(), 2, 3, nullptr);
            close(host(dx), {1 / (1 + std::exp(1.f)), 1 / (1 + std::exp(-1.f)), 0, 0, 0, 0}, 1e-6);
        }
        {
            int d = 16, heads = 2, rows = 3;
            auto x = random(d * heads * rows), ref = x;
            std::vector<int> pos = {0, 17, 2047};
            Tensor dp({3}, DType::I32, Device::CUDA);
            dp.copy_from_host(pos.data(), dp.bytes());
            auto dx = gpu_data(x);
            for (int r = 0; r < rows; ++r)
                for (int h = 0; h < heads; ++h)
                    for (int j = 0; j < d / 2; ++j) {
                        int a = (r * heads + h) * d + j, b = a + d / 2;
                        float angle = pos[r] * std::pow(10000.f, -2.f * j / d);
                        ref[a] = x[a] * std::cos(angle) - x[b] * std::sin(angle);
                        ref[b] = x[a] * std::sin(angle) + x[b] * std::cos(angle);
                    }
            gpu::rope(dx.ptr<float>(), rows, heads, d, d, dp.ptr<int>(), 10000, nullptr);
            close(host(dx), ref, 2e-4);
        }
        {
            auto x = random(37), u = random(37), w = random(37);
            std::vector<float> ref(37);
            auto dx = gpu_data(x), du = gpu_data(u);
            for (int i = 0; i < 37; ++i)
                ref[i] = x[i] / (1 + std::exp(-x[i])) * u[i];
            gpu::swiglu(dx.ptr<float>(), du.ptr<float>(), 37, nullptr);
            close(host(dx), ref, 1e-6);
            for (int i = 0; i < 37; ++i)
                ref[i] += u[i];
            gpu::residual(dx.ptr<float>(), du.ptr<float>(), 37, nullptr);
            close(host(dx), ref, 1e-6);
            std::vector<__half> hx(37);
            float ss = 0;
            for (int i = 0; i < 37; ++i) {
                hx[i] = __float2half(x[i]);
                float v = __half2float(hx[i]);
                ss += v * v;
            }
            for (int i = 0; i < 37; ++i)
                ref[i] = __half2float(hx[i]) / std::sqrt(ss / 37 + 1e-5f) * w[i];
            Tensor half({37}, DType::F16, Device::CUDA);
            half.copy_from_host(hx.data(), half.bytes());
            auto dw = gpu_data(w);
            gpu::rmsnorm_half(half.data(), dw.ptr<float>(), dx.ptr<float>(), 1, 37, 1e-5f, nullptr);
            close(host(dx), ref, 1e-5);
        }
        for (int length : {1, 17, 65}) {
            int h = 4, kh = 2, d = 16;
            auto q = gpu_data(random(length * h * d)), k = gpu_data(random(length * kh * d)),
                 v = gpu_data(random(length * kh * d));
            Tensor a({size_t(length * h * d)}, DType::F32, Device::CUDA),
                b({size_t(length * h * d)}, DType::F32, Device::CUDA),
                scores({size_t(length * h * length)}, DType::F32, Device::CUDA);
            gpu::attention_naive(q.ptr<float>(), k.ptr<float>(), v.ptr<float>(), a.ptr<float>(),
                                 scores.ptr<float>(), length, length, h, kh, d, 0, nullptr);
            gpu::attention_fused(q.ptr<float>(), k.ptr<float>(), v.ptr<float>(), b.ptr<float>(),
                                 length, length, h, kh, d, 0, nullptr);
            close(host(a), host(b), 2e-5);
            // Permuted physical block table, direct writes and reads, including partial blocks.
            int bs = 16, blocks = (length + bs - 1) / bs, mb = blocks;
            Tensor kc({size_t(blocks * bs * kh * d)}, DType::F32, Device::CUDA),
                vc({size_t(blocks * bs * kh * d)}, DType::F32, Device::CUDA);
            std::vector<int> table(length * mb), pos(length);
            for (int r = 0; r < length; ++r) {
                pos[r] = r;
                for (int j = 0; j < mb; ++j)
                    table[r * mb + j] = mb - 1 - j;
            }
            Tensor dt({table.size()}, DType::I32, Device::CUDA),
                dp({pos.size()}, DType::I32, Device::CUDA);
            dt.copy_from_host(table.data(), dt.bytes());
            dp.copy_from_host(pos.data(), dp.bytes());
            gpu::cache_write(k.ptr<float>(), v.ptr<float>(), kc.ptr<float>(), vc.ptr<float>(),
                             dt.ptr<int>(), dp.ptr<int>(), length, 0, blocks, bs, mb, kh, d,
                             nullptr);
            gpu::paged_attention(q.ptr<float>(), kc.ptr<float>(), vc.ptr<float>(), b.ptr<float>(),
                                 dt.ptr<int>(), dp.ptr<int>(), length, 0, blocks, bs, mb, h, kh, d,
                                 nullptr);
            close(host(a), host(b), 2e-5);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        if (Tensor::gpu_bytes.load() != 0)
            throw std::runtime_error("owned GPU allocations leaked");
        std::cout
            << "PASS CUDA GEMM, INT8, RMSNorm, softmax, RoPE, causal GQA and paged attention\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
