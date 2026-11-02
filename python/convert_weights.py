"""Streaming safetensors -> documented LLMENG01 format; no inference framework required."""
import argparse
import json
from pathlib import Path
import shutil
import struct
import numpy as np
from safetensors import safe_open


def quantize_array(w, group_size):
    if w.ndim != 2 or not 0 < group_size <= w.shape[1] or not np.isfinite(w).all():
        raise ValueError('invalid grouped quantization input')
    rows, cols = w.shape
    groups = (cols + group_size - 1) // group_size
    q = np.empty(w.shape, dtype=np.int8)
    scales = np.empty((rows, groups), dtype='<f4')
    for g in range(groups):
        s = slice(g * group_size, min(cols, (g + 1) * group_size))
        x = w[:, s].astype(np.float32)
        scale = np.max(np.abs(x), axis=1) / 127
        scale[scale == 0] = 1
        scales[:, g] = scale
        # Match C++ round-half-away-from-zero.
        z = x / scale[:, None]
        q[:, s] = (np.sign(z) * np.floor(np.abs(z) + .5)).clip(-127, 127).astype(np.int8)
    return q, scales


def convert(input_model_path, output_model_path, precision='fp16', group_size=128):
    src, dst = Path(input_model_path), Path(output_model_path)
    c = json.loads((src / 'config.json').read_text())
    if c.get('model_type') != 'llama' or c.get('hidden_act', 'silu') != 'silu':
        raise ValueError('only Llama with SiLU is supported')
    if c.get('rope_scaling') or c.get('attention_bias') or c.get('mlp_bias'):
        raise ValueError('scaled RoPE and projection biases are unsupported')
    fields = [c[k] for k in ('hidden_size', 'intermediate_size', 'num_hidden_layers',
              'num_attention_heads', 'num_key_value_heads', 'vocab_size',
              'max_position_embeddings', 'bos_token_id', 'eos_token_id')]
    names = ['model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight']
    for i in range(c['num_hidden_layers']):
        names += [f'model.layers.{i}.{s}.weight' for s in
                  ('input_layernorm', 'post_attention_layernorm', 'self_attn.q_proj',
                   'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                   'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')]
    files = sorted(src.glob('*.safetensors'))
    index = {}
    for file in files:
        with safe_open(file, framework='numpy') as f:
            for name in f.keys():
                if name in index:
                    raise ValueError(f'duplicate tensor {name}')
                index[name] = file
    if c.get('tie_word_embeddings') and 'lm_head.weight' not in index:
        index['lm_head.weight'] = index['model.embed_tokens.weight']
    missing = set(names) - index.keys()
    if missing:
        raise ValueError(f'missing tensors: {missing}')
    dst.mkdir(parents=True, exist_ok=True)
    temporary = dst / 'model.bin.tmp'
    with temporary.open('wb') as out:
        out.write(b'LLMENG01')
        out.write(struct.pack('<9I2fI', *fields, c['rms_norm_eps'], c.get('rope_theta', 10000.), len(names)))
        for name in names:
            source_name = name
            with safe_open(index[name], framework='pt', device='cpu') as f:
                if name == 'lm_head.weight' and name not in f.keys():
                    source_name = 'model.embed_tokens.weight'
                tensor = f.get_tensor(source_name)
                # NumPy has no native bfloat16 dtype; promote only during offline conversion.
                import torch
                w = (tensor.float() if tensor.dtype == torch.bfloat16 else tensor).numpy()
            norm = 'norm.weight' in name
            scales = None
            if precision == 'int8' and w.ndim == 2 and name != 'model.embed_tokens.weight':
                w, scales = quantize_array(w, min(group_size, w.shape[1]))
                dtype, group = 2, min(group_size, w.shape[1])
            else:
                w = w.astype('<f4' if norm or precision == 'fp32' else '<f2')
                dtype, group = (0 if w.dtype.itemsize == 4 else 1), 0
            encoded = name.encode()
            out.write(struct.pack('<I', len(encoded)) + encoded)
            out.write(struct.pack('<3I', dtype, w.ndim, group))
            out.write(struct.pack('<' + 'Q' * w.ndim, *w.shape))
            out.write(struct.pack('<Q', w.nbytes))
            out.write(w.tobytes())
            if scales is not None:
                out.write(scales.tobytes())
    temporary.replace(dst / 'model.bin')
    for name in ('config.json', 'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json'):
        if (src / name).exists():
            shutil.copyfile(src / name, dst / name)
    metadata = {'precision': precision, 'group_size': group_size, 'weight_file_bytes': (dst / 'model.bin').stat().st_size,
                'source_safetensors_bytes': sum(f.stat().st_size for f in files)}
    (dst / 'conversion.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def quantize_model(input_model_path, output_model_path, group_size=128):
    return convert(input_model_path, output_model_path, 'int8', group_size)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input'); p.add_argument('output')
    p.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp16')
    p.add_argument('--group-size', type=int, default=128)
    a = p.parse_args()
    print(json.dumps(convert(a.input, a.output, a.precision, a.group_size), indent=2))
