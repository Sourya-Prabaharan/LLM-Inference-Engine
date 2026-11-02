"""Export Hugging Face logits using the same token IDs as the C++ runtime."""
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
import argparse
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def reference(model_path, tokens, device='cpu', dtype='fp32'):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype={'fp32': torch.float32, 'fp16': torch.float16}[dtype], attn_implementation='eager').to(device).eval()
    with torch.inference_mode():
        result = model(torch.tensor([tokens], device=device), use_cache=False).logits[0].float().cpu().numpy()
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True); p.add_argument('--prompt', default='Hello'); p.add_argument('--tokens'); p.add_argument('--device', default='cpu'); p.add_argument('--dtype', choices=['fp32', 'fp16'], default='fp32'); p.add_argument('--output', default='results/reference.npy'); a = p.parse_args()
    tokens = [int(x) for x in a.tokens.split(',')] if a.tokens else AutoTokenizer.from_pretrained(a.model).encode(a.prompt)
    np.save(a.output, reference(a.model, tokens, a.device, a.dtype))
    print(json.dumps({'tokens': tokens, 'output': a.output, 'dtype': a.dtype, 'device': a.device}))
