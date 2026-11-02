"""Test production C++ transformer orchestration with test-only CPU arithmetic."""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from convert_weights import convert

torch.manual_seed(42)
config = LlamaConfig(hidden_size=16, intermediate_size=28, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
                     max_position_embeddings=64, bos_token_id=1, eos_token_id=2,
                     attention_dropout=0., rms_norm_eps=1e-5, tie_word_embeddings=False)
config._attn_implementation = 'eager'
model = LlamaForCausalLM(config).eval()
tokens = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,4,6,8]
with torch.inference_mode():
    expected = model(torch.tensor([tokens]), use_cache=False).logits[0].numpy()
    expected_nll = 0.
    expected_count = 0
    for start in range(0, len(tokens), 8):
        window = torch.tensor([tokens[start:start+8]])
        if window.shape[1] < 2: continue
        logits = model(window, use_cache=False).logits[0, :-1].double()
        expected_nll += torch.nn.functional.cross_entropy(logits, window[0, 1:], reduction='sum').item()
        expected_count += window.shape[1]-1
    generation = model.generate(torch.tensor([[1,3]]), max_new_tokens=5, do_sample=False)[0,2:].tolist()
with tempfile.TemporaryDirectory() as root:
    src = Path(root)/'source'; model.save_pretrained(src, safe_serialization=True)
    for precision in ['fp32', 'fp16', 'int8']:
        dst = Path(root)/precision; convert(src, dst, precision, 7)
        out = Path(root)/'logits.bin'
        run = subprocess.check_output([sys.argv[1], str(dst/'model.bin'), str(out)], text=True)
        actual = np.fromfile(out, dtype='<f4').reshape(len(tokens), config.vocab_size)
        error = float(np.max(np.abs(expected-actual)))
        if precision == 'fp32':
            assert error < 1e-5, error
            assert list(map(int, run.splitlines()[1].split())) == generation
            nll, count = run.splitlines()[2].split()
            assert int(count) == expected_count
            assert abs(float(nll)-expected_nll) < 1e-5, (nll,expected_nll)
        # Quantized modes must be finite; error is reported, not assigned a relaxed parity claim.
        assert np.isfinite(actual).all()
        print(json.dumps(dict(backend='test_only_cpu_reference', precision=precision, max_abs_error=error, batched_parity=True)))
