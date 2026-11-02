#include "kernels.h"
namespace llm::gpu {
__global__ void write_cache(const float *k, const float *v, float *kc, float *vc, const int *table,
                            const int *pos, int layer, int blocks, int bs, int mb, int kh, int d) {
    int r = blockIdx.x;
    int p = pos[r], block = table[r * mb + p / bs];
    size_t base = ((size_t(layer) * blocks + block) * bs + p % bs) * kh * d;
    for (int i = threadIdx.x; i < kh * d; i += blockDim.x) {
        kc[base + i] = k[r * kh * d + i];
        vc[base + i] = v[r * kh * d + i];
    }
}
void cache_write(const float *k, const float *v, float *kc, float *vc, const int *t, const int *p,
                 int r, int l, int b, int bs, int mb, int kh, int d, StreamHandle s) {
    write_cache<<<r, 256, 0, s>>>(k, v, kc, vc, t, p, l, b, bs, mb, kh, d);
    CUDA_CHECK(cudaGetLastError());
}
// One query/head per CTA, 32 keys per tile. Q and KV tiles live in shared memory.
// Running max, partition function and unnormalized output avoid an S*S score tensor.
template <bool paged>
__global__ void online(const float *q, const float *k, const float *v, float *out, const int *table,
                       const int *pos, int tokens, int layer, int blocks, int bs, int mb, int h,
                       int kh, int d, int offset) {
    extern __shared__ float mem[];
    float *sq = mem;
    float *sk = sq + d;
    float *sv = sk + 32 * d;
    float *scores = sv + 32 * d;
    __shared__ float m, l, alpha, weights[32];
    int r = blockIdx.x, head = blockIdx.y, kvhead = head / (h / kh), t = threadIdx.x;
    int end = paged ? pos[r] + 1 : min(tokens, offset + r + 1);
    if (t < d)
        sq[t] = q[(r * h + head) * d + t];
    if (t == 0) {
        m = -INFINITY;
        l = 0;
    }
    float acc = 0;
    __syncthreads();
    for (int start = 0; start < end; start += 32) {
        int count = min(32, end - start);
        for (int i = t; i < count * d; i += blockDim.x) {
            int token = start + i / d, dim = i % d;
            size_t ix;
            if constexpr (paged) {
                int block = table[r * mb + token / bs];
                ix = (((size_t(layer) * blocks + block) * bs + token % bs) * kh + kvhead) * d + dim;
            } else
                ix = (size_t(token) * kh + kvhead) * d + dim;
            sk[i] = k[ix];
            sv[i] = v[ix];
        }
        __syncthreads();
        if (t < count) {
            float dot = 0;
            for (int j = 0; j < d; ++j)
                dot = fmaf(sq[j], sk[t * d + j], dot);
            scores[t] = dot * rsqrtf(float(d));
        }
        __syncthreads();
        if (t == 0) {
            float next = m;
            for (int j = 0; j < count; ++j)
                next = fmaxf(next, scores[j]);
            alpha = expf(m - next);
            float sum = 0;
            for (int j = 0; j < count; ++j) {
                weights[j] = expf(scores[j] - next);
                sum += weights[j];
            }
            l = l * alpha + sum;
            m = next;
        }
        __syncthreads();
        if (t < d) {
            acc *= alpha;
            for (int j = 0; j < count; ++j)
                acc = fmaf(weights[j], sv[j * d + t], acc);
        }
        __syncthreads();
    }
    if (t < d)
        out[(r * h + head) * d + t] = l ? acc / l : 0;
}
void validate(int r, int h, int kh, int d) {
    if (r <= 0 || h <= 0 || kh <= 0 || h % kh || d <= 0 || d > 128)
        throw std::invalid_argument("attention requires dim <=128 and divisible GQA heads");
}
void paged_attention(const float *q, const float *k, const float *v, float *o, const int *t,
                     const int *p, int r, int l, int b, int bs, int mb, int h, int kh, int d,
                     StreamHandle s) {
    validate(r, h, kh, d);
    online<true><<<dim3(r, h), 128, (65 * d + 32) * sizeof(float), s>>>(q, k, v, o, t, p, 0, l, b,
                                                                        bs, mb, h, kh, d, 0);
    CUDA_CHECK(cudaGetLastError());
}
void attention_fused(const float *q, const float *k, const float *v, float *o, int r, int tokens,
                     int h, int kh, int d, int offset, StreamHandle s) {
    validate(r, h, kh, d);
    if (tokens <= 0 || offset < 0)
        throw std::invalid_argument("attention sequence");
    online<false><<<dim3(r, h), 128, (65 * d + 32) * sizeof(float), s>>>(
        q, k, v, o, nullptr, nullptr, tokens, 0, 0, 0, 0, h, kh, d, offset);
    CUDA_CHECK(cudaGetLastError());
}
__global__ void score(const float *q, const float *k, float *s, int rows, int tokens, int h, int kh,
                      int d, int offset) {
    int token = blockIdx.x * blockDim.x + threadIdx.x, head = blockIdx.y, row = blockIdx.z;
    if (token < tokens) {
        float dot = 0;
        for (int i = 0; i < d; ++i)
            dot = fmaf(q[(row * h + head) * d + i], k[(token * kh + head / (h / kh)) * d + i], dot);
        s[(row * h + head) * tokens + token] =
            token <= offset + row ? dot * rsqrtf(float(d)) : -INFINITY;
    }
}
__global__ void weighted(const float *s, const float *v, float *out, int tokens, int h, int kh,
                         int d) {
    int row = blockIdx.x, head = blockIdx.y;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float sum = 0;
        for (int t = 0; t < tokens; ++t)
            sum =
                fmaf(s[(row * h + head) * tokens + t], v[(t * kh + head / (h / kh)) * d + j], sum);
        out[(row * h + head) * d + j] = sum;
    }
}
void attention_naive(const float *q, const float *k, const float *v, float *o, float *scores, int r,
                     int tokens, int h, int kh, int d, int offset, StreamHandle st) {
    validate(r, h, kh, d);
    if (tokens <= 0 || offset < 0)
        throw std::invalid_argument("attention sequence");
    score<<<dim3((tokens + 127) / 128, h, r), 128, 0, st>>>(q, k, scores, r, tokens, h, kh, d,
                                                            offset);
    softmax(scores, r * h, tokens, st);
    weighted<<<dim3(r, h), 128, 0, st>>>(scores, v, o, tokens, h, kh, d);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace llm::gpu
