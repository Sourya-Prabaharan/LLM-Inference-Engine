import argparse
from huggingface_hub import snapshot_download
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='models/tinyllama-hf')
    p.add_argument('--revision', default='fe8a4ea1ffedaf415f4da2f062534de366a451e6')
    a = p.parse_args()
    snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', revision=a.revision,
                      local_dir=a.output, allow_patterns=['*.json', '*.model', '*.safetensors'])
