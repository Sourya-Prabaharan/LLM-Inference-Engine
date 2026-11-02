"""Non-overlapping context windows; score all next tokens except window boundaries."""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
import argparse
import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import subprocess
import tempfile


def evaluate(binary, model, reference_model, context=512, max_tokens=None):
    if context < 2: raise ValueError('context must be at least two')
    tokenizer = AutoTokenizer.from_pretrained(reference_model)
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    tokens = tokenizer('\n\n'.join(data['text']), add_special_tokens=False)['input_ids']
    if max_tokens is not None: tokens = tokens[:max_tokens]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'tokens.bin'
        np.asarray(tokens, dtype='<i4').tofile(path)
        result = json.loads(subprocess.check_output([
            binary, '--model', model, '--score-tokens', str(path),
            '--score-context', str(context)], text=True))
    result.update(context=context, protocol='non-overlapping windows, omit boundary transitions', max_tokens=max_tokens)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--binary', default='build/llm_inference'); p.add_argument('--model', required=True); p.add_argument('--reference-model', required=True); p.add_argument('--context', type=int, default=512); p.add_argument('--max-tokens', type=int); p.add_argument('--output', default='results/perplexity.json'); a = p.parse_args()
    result = evaluate(a.binary, a.model, a.reference_model, a.context, a.max_tokens)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
