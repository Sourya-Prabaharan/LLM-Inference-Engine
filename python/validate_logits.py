import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from reference_inference import reference


def compare(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('invalid logit shape or nonfinite values')
    error = np.abs(actual - expected)
    return dict(max_abs_error=float(error.max()), mean_abs_error=float(error.mean()),
                max_relative_error=float((error / np.maximum(np.abs(expected), 1e-8)).max()),
                token_agreement=float(np.mean(actual.argmax(-1) == expected.argmax(-1))))


def engine_logits(binary, model, tokens, vocab):
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp) / 'logits.bin'
        subprocess.run([binary, '--model', model, '--tokens', ','.join(map(str, tokens)), '--logits', str(out)], check=True)
        return np.fromfile(out, dtype='<f4').reshape(len(tokens), vocab)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--reference-model', required=True); p.add_argument('--model', required=True); p.add_argument('--binary', default='build/llm_inference'); p.add_argument('--tokens', default='1,15043'); p.add_argument('--device', default='cpu'); p.add_argument('--tolerance', type=float, default=1e-3); p.add_argument('--output', default='results/logits.json'); a = p.parse_args()
    tokens = [int(x) for x in a.tokens.split(',')]
    expected = reference(a.reference_model, tokens, a.device)
    actual = engine_logits(a.binary, a.model, tokens, expected.shape[-1])
    result = compare(actual, expected); result['tolerance'] = a.tolerance
    result['passed'] = result['max_abs_error'] <= a.tolerance
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result['passed']: raise SystemExit(1)
