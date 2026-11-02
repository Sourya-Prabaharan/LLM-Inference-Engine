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
from validate_logits import engine_logits


def evaluate(binary, model, reference_model, context=512, max_tokens=None):
    if context < 2: raise ValueError('context must be at least two')
    tokenizer = AutoTokenizer.from_pretrained(reference_model)
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    tokens = tokenizer('\n\n'.join(data['text']), add_special_tokens=False)['input_ids']
    if max_tokens is not None: tokens = tokens[:max_tokens]
    vocab = json.loads((Path(model) / 'config.json').read_text())['vocab_size']
    nll, count = 0., 0
    for start in range(0, len(tokens), context):
        chunk = tokens[start:start+context]
        if len(chunk) < 2: continue
        logits = engine_logits(binary, model, chunk, vocab)[:-1].astype(np.float64)
        maximum = logits.max(axis=1)
        log_partition = maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))
        nll += float((log_partition - logits[np.arange(len(chunk)-1), chunk[1:]]).sum())
        count += len(chunk)-1
    if not count: raise ValueError('no scored tokens')
    return dict(perplexity=float(np.exp(nll/count)), negative_log_likelihood=nll, scored_tokens=count, context=context, protocol='non-overlapping windows, omit boundary transitions', max_tokens=max_tokens)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--binary', default='build/llm_inference'); p.add_argument('--model', required=True); p.add_argument('--reference-model', required=True); p.add_argument('--context', type=int, default=512); p.add_argument('--max-tokens', type=int); p.add_argument('--output', default='results/perplexity.json'); a = p.parse_args()
    result = evaluate(a.binary, a.model, a.reference_model, a.context, a.max_tokens)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
