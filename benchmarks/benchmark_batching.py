import argparse
from common import invoke, save, summary
p = argparse.ArgumentParser(); p.add_argument('--binary', default='build/llm_inference'); p.add_argument('--model', required=True); p.add_argument('--requests', required=True); p.add_argument('--runs', type=int, default=5); p.add_argument('--batch-size', type=int, default=8); p.add_argument('--output', default='results/batching.json'); a = p.parse_args()
if a.runs < 2: p.error('use at least two runs')
records = {}
for policy in ['static', 'continuous']:
    runs = [invoke([a.binary, '--model', a.model, '--requests', a.requests, '--batch-size', a.batch_size, '--policy', policy, '--chat', '--benchmark']) for _ in range(a.runs)]
    records[policy] = {'runs': runs, 'tokens_per_second': summary([r['tokens_per_second'] for r in runs]), 'completion_ms': summary([q['latency_ms'] for r in runs for q in r['requests']]), 'ttft_ms': summary([q['ttft_ms'] for r in runs for q in r['requests']])}
records['speedup'] = records['continuous']['tokens_per_second']['median'] / records['static']['tokens_per_second']['median']
save(a.output, records)
