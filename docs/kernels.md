# CUDA kernels

## Linear layers

Each 16×16 block computes an output tile with FP32 per-thread accumulators. Threads
load contiguous K coordinates for activations and weights; weights are transposed
into a padded shared tile to support the output-column access pattern. Every thread
participates in barriers, including tails. Out-of-range tile values are zero.

FP16 weights convert on tile load. INT8 weights multiply by the row/group scale on
tile load, without materializing a full dequantized matrix. A separate FP16-input
GEMM kernel preserves FP32 accumulation. The inference activation path remains FP32.

There is no Tensor Core path. The 16×16 tile and scalar FP32 arithmetic are unlikely
to approach Tensor Core cuBLAS throughput on large half-precision workloads. The
provided GEMM benchmark compares FP32 custom kernels against pedantic FP32 SGEMM;
it does not mix TF32 or Tensor Core baselines into that ratio. Vectorized loads,
register blocking, and Tensor Core variants require measured profiling first.

## Normalization and pointwise operations

RMSNorm uses one 256-thread CTA per row, FP32 sum of squares and reciprocal square
root. Softmax separately reduces max and sum in shared memory, and returns zero
for a fully masked row. RoPE pairs the first and second half of each head, using
absolute positions and configurable base/rotary dimension. SwiGLU fuses SiLU and
multiplication. Embedding and residual kernels handle contiguous rows.

## Online attention

One CTA owns one query/head, loads Q into shared memory, then visits 32-key K/V
tiles. For each score tile with row maximum `m_tile`:

```
m_new = max(m_old, m_tile)
a = exp(m_old - m_new)
p_j = exp(score_j - m_new)
l_new = a*l_old + sum(p_j)
o_new = a*o_old + sum(p_j*v_j)
result = o_final / l_final
```

The maximum, partition function, and output accumulator stay on chip. There is no
quadratic global score buffer. Shared storage is `(65*head_dim + 32)*4` bytes, plus
small shared scalars/weights. With dimension 128 this fits below 48 KiB.
GQA maps `kv_head = q_head/(q_heads/kv_heads)`. Dense attention uses a query position
offset; paged attention follows the physical table for each key. Partial tiles and
causal end positions are bounded explicitly.

This implementation is algorithmically fused but does not implement multi-query
CTA tiles, warp-specialized asynchronous copies, tensor-core QK/PV, or FlashAttention-2
work partitioning. Profile before claiming optimization success. The separate baseline
materializes scores, applies softmax, and multiplies by V in three kernels.

Run `compute-sanitizer` memcheck/racecheck and kernel parity tests before measuring.
No CUDA compilation or GPU validation was possible on the development Mac.
