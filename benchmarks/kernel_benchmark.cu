#include "kernels.h"
#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
using namespace llm;
struct Events {
    cudaEvent_t a{}, b{};
    Events() {
        CUDA_CHECK(cudaEventCreate(&a));
        CUDA_CHECK(cudaEventCreate(&b));
    }
    ~Events() {
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
};
struct Blas {
    cublasHandle_t handle{};
    Blas() {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS create failed");
        if (cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS math mode failed");
    }
    ~Blas() {
        cublasDestroy(handle);
    }
};
std::vector<float> time_gpu(const std::function<void()> &fn) {
    for (int i = 0; i < 5; ++i)
        fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    Events e;
    std::vector<float> times;
    for (int i = 0; i < 30; ++i) {
        CUDA_CHECK(cudaEventRecord(e.a));
        fn();
        CUDA_CHECK(cudaEventRecord(e.b));
        CUDA_CHECK(cudaEventSynchronize(e.b));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e.a, e.b));
        times.push_back(ms);
    }
    return times;
}
Tensor input(size_t n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-.1, .1);
    std::vector<float> x(n);
    for (auto &v : x)
        v = d(rng);
    Tensor t({n}, DType::F32, Device::CUDA);
    t.copy_from_host(x.data(), t.bytes());
    CUDA_CHECK(cudaDeviceSynchronize());
    return t;
}
void report(const char *name, const std::vector<float> &t, double flops = 0) {
    auto sorted = t;
    std::sort(sorted.begin(), sorted.end());
    double avg = std::accumulate(t.begin(), t.end(), 0.0) / t.size(), variance = 0;
    for (float x : t)
        variance += (x - avg) * (x - avg);
    double median = (sorted[14] + sorted[15]) / 2;
    std::cout << '"' << name << "\":{\"mean_ms\":" << avg << ",\"median_ms\":" << median
              << ",\"variance_ms2\":" << variance / t.size()
              << ",\"tflops\":" << (flops ? flops / (median * 1e9) : 0) << ",\"raw_ms\":[";
    for (size_t i = 0; i < t.size(); ++i) {
        if (i)
            std::cout << ',';
        std::cout << t[i];
    }
    std::cout << "]}";
}
float error(const Tensor &a, const Tensor &b) {
    std::vector<float> x(a.size()), y(b.size());
    a.copy_to_host(x.data(), a.bytes());
    b.copy_to_host(y.data(), b.bytes());
    CUDA_CHECK(cudaDeviceSynchronize());
    float e = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]))
            throw std::runtime_error("nonfinite output");
        e = std::max(e, std::abs(x[i] - y[i]));
    }
    return e;
}
int main(int argc, char **argv) {
    try {
        if (argc < 2)
            throw std::invalid_argument("gemm M N K | attention sequence [decode]");
        cudaDeviceProp prop{};
        int device = 0, version = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        CUDA_CHECK(cudaRuntimeGetVersion(&version));
        std::cout << std::setprecision(9) << "{\"backend\":\"CUDA\",\"gpu\":\"" << prop.name
                  << "\",\"cuda_runtime\":" << version << ",\"dtype\":\"fp32\",";
        std::string mode = argv[1];
        if (mode == "gemm") {
            if (argc != 5)
                throw std::invalid_argument("gemm M N K");
            int m = std::stoi(argv[2]), n = std::stoi(argv[3]), k = std::stoi(argv[4]);
            if (m <= 0 || n <= 0 || k <= 0)
                throw std::invalid_argument("positive sizes required");
            auto a = input(size_t(m) * k), w = input(size_t(n) * k);
            Tensor out({size_t(m) * n}, DType::F32, Device::CUDA),
                ref({size_t(m) * n}, DType::F32, Device::CUDA);
            Blas blas;
            float alpha = 1, beta = 0;
            auto baseline = [&] {
                if (cublasSgemm(blas.handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                                w.ptr<float>(), k, a.ptr<float>(), k, &beta, ref.ptr<float>(),
                                n) != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("cuBLAS SGEMM failed");
            };
            auto naive = time_gpu([&] {
                gpu::linear(a.ptr<float>(), w.data(), nullptr, out.ptr<float>(), m, n, k,
                            DType::F32, 0, nullptr, true);
            });
            auto tiled = time_gpu([&] {
                gpu::linear(a.ptr<float>(), w.data(), nullptr, out.ptr<float>(), m, n, k,
                            DType::F32, 0, nullptr);
            });
            auto reference = time_gpu(baseline);
            std::cout << "\"M\":" << m << ",\"N\":" << n << ",\"K\":" << k
                      << ",\"max_abs_error\":" << error(out, ref) << ',';
            double flops = 2.0 * m * n * k;
            report("naive", naive, flops);
            std::cout << ',';
            report("tiled", tiled, flops);
            std::cout << ',';
            report("cublas", reference, flops);
        } else if (mode == "attention") {
            if (argc < 3)
                throw std::invalid_argument("attention sequence [decode]");
            int tokens = std::stoi(argv[2]), rows = argc > 3 ? 1 : tokens, h = 32, kh = 4, d = 64;
            if (tokens <= 0)
                throw std::invalid_argument("positive sequence required");
            auto q = input(size_t(rows) * h * d), k = input(size_t(tokens) * kh * d),
                 v = input(size_t(tokens) * kh * d);
            Tensor out({size_t(rows) * h * d}, DType::F32, Device::CUDA),
                ref({size_t(rows) * h * d}, DType::F32, Device::CUDA),
                scores({size_t(rows) * h * tokens}, DType::F32, Device::CUDA);
            int offset = tokens - rows;
            auto naive = time_gpu([&] {
                gpu::attention_naive(q.ptr<float>(), k.ptr<float>(), v.ptr<float>(),
                                     ref.ptr<float>(), scores.ptr<float>(), rows, tokens, h, kh, d,
                                     offset, nullptr);
            });
            auto fused = time_gpu([&] {
                gpu::attention_fused(q.ptr<float>(), k.ptr<float>(), v.ptr<float>(),
                                     out.ptr<float>(), rows, tokens, h, kh, d, offset, nullptr);
            });
            std::cout << "\"sequence\":" << tokens << ",\"query_rows\":" << rows
                      << ",\"score_buffer_bytes\":" << scores.bytes()
                      << ",\"max_abs_error\":" << error(out, ref) << ',';
            report("naive", naive);
            std::cout << ',';
            report("fused", fused);
        } else
            throw std::invalid_argument("unknown mode");
        std::cout << "}\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
