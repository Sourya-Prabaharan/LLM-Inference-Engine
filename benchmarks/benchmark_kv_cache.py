"""Allocation accounting only; deliberately not a GPU latency benchmark."""
import argparse
from common import save
p = argparse.ArgumentParser(); p.add_argument('--lengths', default='17,31,65,127,255,511,1023'); p.add_argument('--context', type=int, default=2048); p.add_argument('--block-size', type=int, default=16); p.add_argument('--output', default='results/kv_cache.json'); a = p.parse_args()
lengths = [int(x) for x in a.lengths.split(',')]
if a.block_size <= 0 or a.context <= 0 or any(x <= 0 or x > a.context for x in lengths): p.error('invalid cache workload')
used = sum(lengths); contiguous = len(lengths)*a.context; paged = sum((x+a.block_size-1)//a.block_size*a.block_size for x in lengths)
# TinyLlama 22 layers, 4 KV heads, dim 64; K+V, FP32 cache.
bytes_per_token = 22*4*64*2*4
save(a.output, dict(kind='analytical_allocation_accounting_not_gpu_measurement', lengths=lengths, block_size=a.block_size, used_bytes=used*bytes_per_token, contiguous_assigned_bytes=contiguous*bytes_per_token, paged_assigned_bytes=paged*bytes_per_token, contiguous_waste_bytes=(contiguous-used)*bytes_per_token, paged_internal_fragmentation_bytes=(paged-used)*bytes_per_token, waste_reduction_fraction=1-(paged-used)/(contiguous-used) if contiguous>used else None, note='Free pool blocks are reusable capacity, not released CUDA memory. Pool reservation is excluded from assigned-byte comparison.'))
