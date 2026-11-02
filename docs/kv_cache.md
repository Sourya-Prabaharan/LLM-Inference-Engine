# Paged KV cache

K and V each use `[layer,physical_block,token_in_block,kv_head,head_dim]` FP32 storage.
The host allocator manages request token counts, logical block vectors, and a global
free list. Appending checks required capacity before mutation; allocation failures
leave token count and table unchanged. Completion returns all blocks for reuse.

GPU kernels receive packed `[row,max_logical_blocks]` tables and absolute positions.
Writes map the new token position directly into its physical block. Attention walks
logical token positions and loads K/V through the table; it never gathers a sequence
into a contiguous temporary. A stream synchronization after logits precedes the next
scheduler mutation, so recycled blocks cannot race a previous request's kernels.

## Accounting

For request lengths `L_i`, maximum context `C`, block size `B`, and bytes/token `T`:

* Contiguous assigned capacity: `N*C*T`.
* Paged assigned capacity: `sum(ceil(L_i/B)*B)*T`.
* Live KV data: `sum(L_i)*T`.
* Waste: assigned capacity minus live KV data.
* Waste reduction: `1 - paged_waste/contiguous_waste`, undefined for zero baseline waste.

For this runtime `T = layers*kv_heads*head_dim*2*sizeof(float)`.
The host accounting script uses TinyLlama's actual geometry and labels its results
analytical. It is not a GPU allocation benchmark. The physical GPU pool reserves
`blocks*B*T` bytes at construction; unused pool blocks remain allocated but available
to any request. Reducing assigned waste does not release the pool to the driver.

The contiguous baseline is accounting only; there is no alternate contiguous serving
backend. GPU concurrency/isolation tests must still be executed. Cache exhaustion
fails a request visibly and releases its blocks; swapping and preemption are absent.

`cache_benchmark` additionally allocates and initializes actual CUDA pools for both
schemes, recording requested bytes and driver free-memory deltas. Both pools serve
the same listed live workload; the paged pool is sized to its required blocks, with
no extra reserve. This isolates allocation waste rather than claiming that the
runtime's default fixed-size pool shrinks automatically. Its results remain
unmeasured until the GPU validation script runs on hardware.
