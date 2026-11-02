#include "kernels.h"
#include <cfloat>
#include <cuda_fp16.h>
namespace llm::gpu {
__device__ float reduce(float x, float *scratch, bool maximum) {
    int t = threadIdx.x;
    scratch[t] = x;
    __syncthreads();
    for (int n = blockDim.x / 2; n; n /= 2) {
        if (t < n)
            scratch[t] = maximum ? fmaxf(scratch[t], scratch[t + n]) : scratch[t] + scratch[t + n];
        __syncthreads();
    }
    return scratch[0];
}
template <class T> __global__ void norm(const T *x, const float *w, float *y, int d, float eps) {
    __shared__ float s[256];
    int r = blockIdx.x;
    float sum = 0;
    for (int i = threadIdx.x; i < d; i += 256) {
        float v = float(x[r * d + i]);
        sum += v * v;
    }
    float scale = rsqrtf(reduce(sum, s, false) / d + eps);
    for (int i = threadIdx.x; i < d; i += 256)
        y[r * d + i] = float(x[r * d + i]) * scale * w[i];
}
void rmsnorm(const float *x, const float *w, float *y, int r, int d, float eps, StreamHandle st) {
    if (r <= 0 || d <= 0 || eps <= 0)
        throw std::invalid_argument("RMSNorm geometry");
    norm<<<r, 256, 0, st>>>(x, w, y, d, eps);
    CUDA_CHECK(cudaGetLastError());
}
void rmsnorm_half(const void *x, const float *w, float *y, int rows, int dim, float eps,
                  StreamHandle st) {
    if (rows <= 0 || dim <= 0 || eps <= 0)
        throw std::invalid_argument("half RMSNorm geometry");
    norm<<<rows, 256, 0, st>>>(static_cast<const __half *>(x), w, y, dim, eps);
    CUDA_CHECK(cudaGetLastError());
}
__global__ void rotate(float *x, int heads, int dim, int rotary, const int *pos, float base) {
    int r = blockIdx.x, h = blockIdx.y;
    for (int i = threadIdx.x; i < rotary / 2; i += blockDim.x) {
        float angle = pos[r] * powf(base, -2.0f * i / rotary), s, c;
        sincosf(angle, &s, &c);
        int a = (r * heads + h) * dim + i, b = a + rotary / 2;
        float u = x[a], v = x[b];
        x[a] = u * c - v * s;
        x[b] = u * s + v * c;
    }
}
void rope(float *x, int r, int h, int d, int rot, const int *p, float base, StreamHandle st) {
    if (r <= 0 || h <= 0 || rot <= 0 || rot > d || rot % 2 || base <= 0)
        throw std::invalid_argument("RoPE geometry");
    rotate<<<dim3(r, h), 128, 0, st>>>(x, h, d, rot, p, base);
    CUDA_CHECK(cudaGetLastError());
}
__global__ void sm(float *x, int cols) {
    __shared__ float s[256];
    int row = blockIdx.x;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += 256)
        m = fmaxf(m, x[row * cols + i]);
    m = reduce(m, s, true);
    // Every warp must load the max before shared scratch is reused for the sum.
    __syncthreads();
    float sum = 0;
    for (int i = threadIdx.x; i < cols; i += 256) {
        float v = m == -INFINITY ? 0 : expf(x[row * cols + i] - m);
        x[row * cols + i] = v;
        sum += v;
    }
    sum = reduce(sum, s, false);
    for (int i = threadIdx.x; i < cols; i += 256)
        x[row * cols + i] = sum ? x[row * cols + i] / sum : 0;
}
void softmax(float *x, int r, int c, StreamHandle st) {
    if (r <= 0 || c <= 0)
        throw std::invalid_argument("softmax geometry");
    sm<<<r, 256, 0, st>>>(x, c);
    CUDA_CHECK(cudaGetLastError());
}
__global__ void add(float *x, const float *y, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] += y[i];
}
__global__ void gated(float *g, const float *u, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = g[i];
        g[i] = v / (1 + expf(-v)) * u[i];
    }
}
void residual(float *x, const float *y, size_t n, StreamHandle s) {
    if (n)
        add<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
    CUDA_CHECK(cudaGetLastError());
}
void swiglu(float *g, const float *u, size_t n, StreamHandle s) {
    if (n)
        gated<<<(n + 255) / 256, 256, 0, s>>>(g, u, n);
    CUDA_CHECK(cudaGetLastError());
}
template <class T> __global__ void embed(const T *w, const int *ids, float *y, int rows, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * dim)
        y[i] = float(w[ids[i / dim] * dim + i % dim]);
}
void embedding(const void *w, DType t, const int *ids, float *y, int r, int d, StreamHandle s) {
    if (t == DType::F32)
        embed<<<(r * d + 255) / 256, 256, 0, s>>>(static_cast<const float *>(w), ids, y, r, d);
    else if (t == DType::F16)
        embed<<<(r * d + 255) / 256, 256, 0, s>>>(static_cast<const __half *>(w), ids, y, r, d);
    else
        throw std::invalid_argument("embedding dtype");
    CUDA_CHECK(cudaGetLastError());
}
} // namespace llm::gpu
