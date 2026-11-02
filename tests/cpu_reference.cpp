// Explicit CPU numerical oracle for testing transformer orchestration on non-CUDA hosts.
// This file is not linked into production inference or any GPU benchmark.
#include "kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
namespace llm::gpu {
namespace {
float half_value(uint16_t bits) {
    int sign = (bits & 0x8000) ? -1 : 1, exponent = (bits >> 10) & 31, fraction = bits & 1023;
    if (exponent == 0)
        return sign * std::ldexp(float(fraction), -24);
    if (exponent == 31)
        return fraction ? NAN : sign * INFINITY;
    return sign * std::ldexp(float(1024 + fraction), exponent - 25);
}
float weight(const void *w, DType type, size_t index) {
    if (type == DType::F32)
        return static_cast<const float *>(w)[index];
    if (type == DType::F16)
        return half_value(static_cast<const uint16_t *>(w)[index]);
    if (type == DType::I8)
        return static_cast<const int8_t *>(w)[index];
    throw std::invalid_argument("reference dtype");
}
} // namespace
void linear(const float *a, const void *w, const float *s, float *c, int M, int N, int K,
            DType type, int g, StreamHandle, bool) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                float v = weight(w, type, n * K + k);
                if (type == DType::I8)
                    v *= s[n * ((K + g - 1) / g) + k / g];
                sum += a[m * K + k] * v;
            }
            c[m * N + n] = sum;
        }
}
void rmsnorm(const float *x, const float *w, float *y, int rows, int d, float eps, StreamHandle) {
    for (int r = 0; r < rows; ++r) {
        float sum = 0;
        for (int j = 0; j < d; ++j)
            sum += x[r * d + j] * x[r * d + j];
        float scale = 1 / std::sqrt(sum / d + eps);
        for (int j = 0; j < d; ++j)
            y[r * d + j] = x[r * d + j] * scale * w[j];
    }
}
void rope(float *x, int rows, int heads, int d, int rot, const int *p, float base, StreamHandle) {
    for (int r = 0; r < rows; ++r)
        for (int h = 0; h < heads; ++h)
            for (int j = 0; j < rot / 2; ++j) {
                float angle = p[r] * std::pow(base, -2.f * j / rot);
                int a = (r * heads + h) * d + j, b = a + rot / 2;
                float u = x[a], v = x[b];
                x[a] = u * std::cos(angle) - v * std::sin(angle);
                x[b] = u * std::sin(angle) + v * std::cos(angle);
            }
}
void residual(float *x, const float *y, size_t n, StreamHandle) {
    for (size_t i = 0; i < n; ++i)
        x[i] += y[i];
}
void swiglu(float *g, const float *u, size_t n, StreamHandle) {
    for (size_t i = 0; i < n; ++i)
        g[i] = g[i] / (1 + std::exp(-g[i])) * u[i];
}
void embedding(const void *w, DType type, const int *ids, float *y, int rows, int d, StreamHandle) {
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < d; ++j)
            y[r * d + j] = weight(w, type, ids[r] * d + j);
}
void cache_write(const float *k, const float *v, float *kc, float *vc, const int *t, const int *p,
                 int rows, int layer, int blocks, int bs, int mb, int kh, int d, StreamHandle) {
    for (int r = 0; r < rows; ++r) {
        int block = t[r * mb + p[r] / bs];
        size_t offset = ((size_t(layer) * blocks + block) * bs + p[r] % bs) * kh * d;
        std::copy_n(k + r * kh * d, kh * d, kc + offset);
        std::copy_n(v + r * kh * d, kh * d, vc + offset);
    }
}
void paged_attention(const float *q, const float *k, const float *v, float *out, const int *t,
                     const int *p, int rows, int layer, int blocks, int bs, int mb, int h, int kh,
                     int d, StreamHandle) {
    for (int r = 0; r < rows; ++r)
        for (int head = 0; head < h; ++head) {
            std::vector<float> scores(p[r] + 1);
            float maximum = -INFINITY;
            int kvhead = head / (h / kh);
            auto offset = [&](int token) {
                int block = t[r * mb + token / bs];
                return (((size_t(layer) * blocks + block) * bs + token % bs) * kh + kvhead) * d;
            };
            for (int token = 0; token <= p[r]; ++token) {
                float dot = 0;
                for (int j = 0; j < d; ++j)
                    dot += q[(r * h + head) * d + j] * k[offset(token) + j];
                scores[token] = dot / std::sqrt(float(d));
                maximum = std::max(maximum, scores[token]);
            }
            float sum = 0;
            for (auto &score : scores) {
                score = std::exp(score - maximum);
                sum += score;
            }
            for (int j = 0; j < d; ++j) {
                float value = 0;
                for (int token = 0; token <= p[r]; ++token)
                    value += scores[token] / sum * v[offset(token) + j];
                out[(r * h + head) * d + j] = value;
            }
        }
}
} // namespace llm::gpu
