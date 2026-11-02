import argparse
from common import invoke, save, summary
p = argparse.ArgumentParser(); p.add_argument('--binary', default='build/llm_inference'); p.add_argument('--model', required=True); p.add_argument('--prompt', default='Explain how CUDA kernels work.'); p.add_argument('--max-tokens', type=int, default=128); p.add_argument('--runs', type=int, default=5); p.add_argument('--output', default='results/end_to_end.json'); a = p.parse_args()
if a.runs < 2: p.error('use at least two runs')
runs = [invoke([a.binary, '--model', a.model, '--prompt', a.prompt, '--max-tokens', a.max_tokens, '--benchmark']) for _ in range(a.runs)]
save(a.output, dict(runs=runs, tokens_per_second=summary([r['tokens_per_second'] for r in runs])))
