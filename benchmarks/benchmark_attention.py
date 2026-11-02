import argparse
from common import invoke, save
p = argparse.ArgumentParser(); p.add_argument('--binary', default='build/kernel_benchmark'); p.add_argument('--output', default='results/attention.json'); a = p.parse_args()
records = []
for length in [128, 256, 512, 1024, 2048, 4096]:
    for mode in ['prefill', 'decode']:
        r = invoke([a.binary, 'attention', length] + (['decode'] if mode == 'decode' else []))
        r['workload'] = mode
        r['speedup'] = r['naive']['median_ms'] / r['fused']['median_ms']
        records.append(r)
save(a.output, records)
