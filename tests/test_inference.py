"""Opt-in GPU integration suite: python tests/test_inference.py --help."""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch
from transformers import AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from validate_logits import engine_logits, compare
p = argparse.ArgumentParser(); p.add_argument('--binary', default='build/llm_inference'); p.add_argument('--model', required=True); p.add_argument('--reference-model', required=True); p.add_argument('--device', default='cpu'); p.add_argument('--lengths', default='1,7,32,128,512,2048'); a = p.parse_args()
model = AutoModelForCausalLM.from_pretrained(a.reference_model, torch_dtype=torch.float32, attn_implementation='eager').to(a.device).eval()
rng = np.random.default_rng(42)
torch.backends.cuda.matmul.allow_tf32 = False
for length in [int(x) for x in a.lengths.split(',') if x]:
    model.to(a.device)
    ids = [1] + rng.integers(3, model.config.vocab_size, size=length-1).tolist()
    with torch.inference_mode(): expected = model(torch.tensor([ids], device=a.device), use_cache=False).logits[0].cpu().numpy()
    model.to('cpu')
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    actual = engine_logits(a.binary, a.model, ids, model.config.vocab_size)
    result = compare(actual, expected); print(length, result)
    if result['max_abs_error'] > 1e-3: raise AssertionError(result)
# Greedy generation uses chunked prefill, while logits export above uses incremental prefill.
ids = [1, 15043]
model.to(a.device)
with torch.inference_mode():
    expected = model.generate(torch.tensor([ids], device=a.device), max_new_tokens=8, do_sample=False)[0, len(ids):].cpu().tolist()
model.to('cpu')
if torch.cuda.is_available(): torch.cuda.empty_cache()
text = subprocess.check_output([a.binary, '--model', a.model, '--tokens', ','.join(map(str, ids)), '--max-tokens', '8'], text=True)
actual = [int(x) for x in text.splitlines()[1].split()]
if actual != expected: raise AssertionError((actual, expected))
print('PASS full-sequence reference vs incremental engine logits and greedy generation')
# Mixed prompt lengths exercise chunked prefill and concurrent cache isolation.
import json
prompts = ['Hello', 'Explain CUDA in one sentence.', 'A long prompt. ' * 20]
with tempfile.TemporaryDirectory() as tmp:
    workload = Path(tmp) / 'requests.tsv'
    workload.write_text(''.join(f'0\t8\t{x}\n' for x in prompts))
    def run(extra):
        return json.loads(subprocess.check_output([a.binary, '--model', a.model, '--chat', '--benchmark', '--max-tokens', '8']+extra, text=True))
    singles = [run(['--prompt', x])['requests'][0]['token_ids'] for x in prompts]
    for policy in ['static', 'continuous']:
        batched = run(['--requests', str(workload), '--policy', policy, '--batch-size', '2'])
        assert [r['token_ids'] for r in batched['requests']] == singles
print('PASS batched/static/continuous generation parity with independent requests')
