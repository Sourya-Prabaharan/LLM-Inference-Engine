import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
import argparse
from pathlib import Path
import subprocess
from transformers import AutoTokenizer
p = argparse.ArgumentParser(); p.add_argument('--assets', default='models/tokenizer'); p.add_argument('--binary', default='build/tokenizer_dump'); a = p.parse_args()
tokenizer = AutoTokenizer.from_pretrained(a.assets, use_fast=True)
prompts = ['', 'Hello', ' Hello', '  spaces  ', 'hello\nworld', 'café 中文 🦙', '<s>Hello</s>', 'a</s>b', ' tabs\tand\nnewlines', '<unk>test']
for text in prompts:
    output = subprocess.check_output([a.binary, str(Path(a.assets)/'tokenizer.model'), text], text=True)
    actual = [int(x) for x in output.splitlines()[0].split(',') if x]
    expected = tokenizer.encode(text)
    assert actual == expected, (repr(text), actual, expected)
for text in ['Hello', 'Explain GPU memory.', 'café 中文 🦙']:
    output = subprocess.check_output([a.binary, str(Path(a.assets)/'tokenizer.model'), text, 'chat'], text=True)
    actual = [int(x) for x in output.splitlines()[0].split(',') if x]
    expected = tokenizer.apply_chat_template([{'role':'user','content':text}], add_generation_prompt=True)
    assert actual == expected, ('chat', repr(text), actual, expected)
print(f'PASS {len(prompts)} raw tokenizer cases and 3 chat template cases')
