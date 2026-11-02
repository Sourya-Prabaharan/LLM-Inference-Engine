# Scheduling and execution

Requests specify IDs, token prompts, maximum generated length, sampling settings,
and arrival time relative to the start of `Engine::run`. Submission is a synchronous
workload API; it does not expose a thread-safe live queue or network transport.

Continuous mode admits FIFO pending requests at each generation boundary until the
active limit is reached. Static mode admits only when the previous active group is
empty. Both use identical transformer kernels and cache layouts. Completed requests
release cache blocks before the next iteration. Zero-generation requests finish
without allocating KV storage. Invalid IDs, tokens, or context lengths are rejected.

For each iteration, each request contributes up to 16 prompt tokens if prefilling,
or one token if decoding. All contributed rows are packed into one transformer
forward pass. Thus Q/K/V, output, gate/up/down, and LM-head GEMMs process multiple
rows across requests. This is not a loop invoking independent GPU inference per request.
Causal masks distinguish positions within a prompt chunk. Sampling runs only after
the final prompt token, then once per decode token.

The fixed prefill chunk prevents a long prompt from monopolizing an iteration.
The next iteration's FIFO admission avoids starvation for finite workloads. A failed
cache extension marks only that request failed; other requests proceed. There is no
admission reservation, eviction, retry, or preemption strategy. Large requests may
fail late when the pool is exhausted; consumers must inspect errors.

TTFT includes queueing from scheduled arrival to the first generated token.
Completion latency includes queueing through the final generated token. Overall
throughput divides generated tokens by wall-clock run time, including arrival gaps.
EOS may shorten requests. Static/continuous comparisons must use identical prompts,
arrival times, generation limits, precision, seeds, and pool geometry. Use greedy
sampling when testing exact output parity because sampling order consumes RNG state.
