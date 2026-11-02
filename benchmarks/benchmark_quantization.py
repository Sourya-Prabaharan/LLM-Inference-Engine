import argparse
from pathlib import Path
import sys
from common import invoke, save
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from evaluate_perplexity import evaluate
p = argparse.ArgumentParser(); p.add_argument('--binary', default='build/llm_inference'); p.add_argument('--fp16-model', required=True); p.add_argument('--int8-model', required=True); p.add_argument('--reference-model', required=True); p.add_argument('--max-tokens', type=int); p.add_argument('--output', default='results/quantization.json'); a = p.parse_args()
records = {}
for precision, model in [('fp16', a.fp16_model), ('int8', a.int8_model)]:
    records[precision] = dict(weight_file_bytes=(Path(model)/'model.bin').stat().st_size, perplexity=evaluate(a.binary, model, a.reference_model, max_tokens=a.max_tokens), inference_runs=[invoke([a.binary, '--model', model, '--prompt', 'Explain GPU memory.', '--max-tokens', '128', '--chat', '--benchmark']) for _ in range(5)])
records['storage_reduction_fraction'] = 1-records['int8']['weight_file_bytes']/records['fp16']['weight_file_bytes']
records['perplexity_increase'] = records['int8']['perplexity']['perplexity']-records['fp16']['perplexity']['perplexity']
save(a.output, records)
