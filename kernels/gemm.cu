#include "kernels.h"
#include <cuda_fp16.h>
namespace llm::gpu {
template <class T>
__device__ float read_weight(const T *w, const float *, int n, int k, int K, int) {
    return float(w[n * K + k]);
}
template <>
__device__ float read_weight<__half>(const __half *w, const float *, int n, int k, int K, int) {
    return __half2float(w[n * K + k]);
}
template <>
__device__ float read_weight<int8_t>(const int8_t *w, const float *s, int n, int k, int K, int g) {
    return float(w[n * K + k]) * s[n * ((K + g - 1) / g) + k / g];
}
template <class T, bool tiled, class A = float>
__global__ void gemm(const A *a, const T *w, const float *scales, float *c, int M, int N, int K,
                     int group) {
    constexpr int B = 16;
    int row = blockIdx.y * B + threadIdx.y, col = blockIdx.x * B + threadIdx.x;
    float sum = 0;
    if constexpr (tiled) {
        __shared__ float as[B][B], bs[B][B + 1];
        for (int t = 0; t < K; t += B) {
            int ak = t + threadIdx.x, bk = t + threadIdx.x;
            int weight_row = blockIdx.x * B + threadIdx.y;
            as[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? float(a[row * K + ak]) : 0;
            // Coalesced reads along K, transpose into padded shared memory.
            bs[threadIdx.x][threadIdx.y] =
                (weight_row < N && bk < K) ? read_weight(w, scales, weight_row, bk, K, group) : 0;
            __syncthreads();
#pragma unroll
            for (int k = 0; k < B; ++k)
                sum = fmaf(as[threadIdx.y][k], bs[k][threadIdx.x], sum);
            __syncthreads();
        }
    } else {
        if (row < M && col < N)
            for (int k = 0; k < K; ++k)
                sum = fmaf(float(a[row * K + k]), read_weight(w, scales, col, k, K, group), sum);
    }
    if (row < M && col < N)
        c[row * N + col] = sum;
}
template <class T>
void launch(const float *a, const void *w, const float *s, float *c, int M, int N, int K, int g,
            StreamHandle stream, bool naive) {
    dim3 grid((N + 15) / 16, (M + 15) / 16), block(16, 16);
    if (naive)
        gemm<T, false><<<grid, block, 0, stream>>>(a, static_cast<const T *>(w), s, c, M, N, K, g);
    else
        gemm<T, true><<<grid, block, 0, stream>>>(a, static_cast<const T *>(w), s, c, M, N, K, g);
}
void linear(const float *a, const void *w, const float *s, float *c, int M, int N, int K,
            DType type, int group, StreamHandle stream, bool naive) {
    if (M <= 0 || N <= 0 || K <= 0)
        throw std::invalid_argument("linear dimensions");
    if (type == DType::F32)
        launch<float>(a, w, s, c, M, N, K, group, stream, naive);
    else if (type == DType::F16)
        launch<__half>(a, w, s, c, M, N, K, group, stream, naive);
    else if (type == DType::I8 && group > 0 && s)
        launch<int8_t>(a, w, s, c, M, N, K, group, stream, naive);
    else
        throw std::invalid_argument("linear dtype/scales");
    CUDA_CHECK(cudaGetLastError());
}
void linear_half(const void *a, const void *w, float *c, int M, int N, int K, StreamHandle stream) {
    if (M <= 0 || N <= 0 || K <= 0)
        throw std::invalid_argument("half GEMM dimensions");
    gemm<__half, true, __half><<<dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16), 0, stream>>>(
        static_cast<const __half *>(a), static_cast<const __half *>(w), nullptr, c, M, N, K, 0);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace llm::gpu
