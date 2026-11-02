"""Deterministic safetensors conversion and C++ parser integration, including corruption."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from safetensors.numpy import save_file
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from convert_weights import convert, quantize_array
LOADER = sys.argv.pop(1) if len(sys.argv) > 1 else 'build/test_weights'

class Conversion(unittest.TestCase):
    def test_roundtrip_and_corruption(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as root:
            src = Path(root) / 'source'; src.mkdir()
            config = dict(model_type='llama', hidden_size=8, intermediate_size=12,
                          num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                          vocab_size=16, max_position_embeddings=32, bos_token_id=1,
                          eos_token_id=2, rms_norm_eps=1e-5)
            (src / 'config.json').write_text(json.dumps(config))
            shapes = {'model.embed_tokens.weight': (16, 8), 'lm_head.weight': (16, 8), 'model.norm.weight': (8,)}
            for name, shape in [('input_layernorm', (8,)), ('post_attention_layernorm', (8,)),
                                ('self_attn.q_proj', (8, 8)), ('self_attn.k_proj', (4, 8)),
                                ('self_attn.v_proj', (4, 8)), ('self_attn.o_proj', (8, 8)),
                                ('mlp.gate_proj', (12, 8)), ('mlp.up_proj', (12, 8)), ('mlp.down_proj', (8, 12))]:
                shapes[f'model.layers.0.{name}.weight'] = shape
            save_file({name: rng.normal(size=shape).astype(np.float32) for name, shape in shapes.items()}, src / 'model.safetensors')
            for precision in ('fp32', 'fp16', 'int8'):
                dst = Path(root) / precision
                convert(src, dst, precision, group_size=3)
                subprocess.run([LOADER, str(dst / 'model.bin')], check=True)
                data = (dst / 'model.bin').read_bytes()
                for damaged in (b'BADMAGIC' + data[8:], data[:-3], data + b'junk'):
                    bad = dst / 'bad.bin'; bad.write_bytes(damaged)
                    result = subprocess.run([LOADER, str(bad)], capture_output=True)
                    self.assertNotEqual(result.returncode, 0)

    def test_quantization_bound(self):
        w = np.random.default_rng(7).normal(size=(5, 17)).astype(np.float32)
        w[0] = 0
        q, s = quantize_array(w, 6)
        for c in range(17):
            self.assertTrue(np.all(np.abs(w[:, c] - q[:, c] * s[:, c // 6]) <= s[:, c // 6] / 2 + 1e-6))

if __name__ == '__main__': unittest.main()
