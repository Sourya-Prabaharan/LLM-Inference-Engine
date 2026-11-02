#pragma once
#include "tensor.h"
namespace llm::gpu {
void linear_half(const void *, const void *, float *, int M, int N, int K, StreamHandle);
// All runtime activations are FP32. Linear weights may be FP32, FP16, or grouped INT8.
// Row-major linear: [M,K] * weights[N,K]^T -> [M,N].
void linear(const float *, const void *, const float *scales, float *, int M, int N, int K, DType,
            int group, StreamHandle, bool naive = false);
void rmsnorm_half(const void *, const float *, float *, int rows, int dim, float eps, StreamHandle);
void rmsnorm(const float *, const float *, float *, int rows, int dim, float eps, StreamHandle);
void rope(float *, int rows, int heads, int dim, int rotary, const int *positions, float base,
          StreamHandle);
void softmax(float *, int rows, int cols, StreamHandle);
void residual(float *, const float *, size_t, StreamHandle);
void swiglu(float *, const float *, size_t, StreamHandle);
void embedding(const void *, DType, const int *, float *, int rows, int dim, StreamHandle);
// Cache [layer, block, block_token, KV head, dimension]. Tables [row,max_blocks].
void cache_write(const float *, const float *, float *, float *, const int *tables,
                 const int *positions, int rows, int layer, int blocks, int block_size,
                 int max_blocks, int kvheads, int dim, StreamHandle);
void paged_attention(const float *, const float *, const float *, float *, const int *tables,
                     const int *positions, int rows, int layer, int blocks, int block_size,
                     int max_blocks, int qheads, int kvheads, int dim, StreamHandle);
// Dense attention Q[rows,heads,dim], K/V[tokens,kvheads,dim]; causal q_offset.
void attention_naive(const float *, const float *, const float *, float *, float *scores, int rows,
                     int tokens, int heads, int kvheads, int dim, int q_offset, StreamHandle);
void attention_fused(const float *, const float *, const float *, float *, int rows, int tokens,
                     int heads, int kvheads, int dim, int q_offset, StreamHandle);
} // namespace llm::gpu
