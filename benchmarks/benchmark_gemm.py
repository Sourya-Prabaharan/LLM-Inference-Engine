import argparse
from common import invoke, save
p = argparse.ArgumentParser(); p.add_argument('--binary', default='build/kernel_benchmark'); p.add_argument('--output', default='results/gemm.json'); a = p.parse_args()
records = []
for m, n, k in [(1, 2048, 2048), (16, 2048, 2048), (128, 2048, 2048), (128, 5632, 2048), (128, 2048, 5632), (17, 257, 129)]:
    r = invoke([a.binary, 'gemm', m, n, k])
    r['throughput_fraction_of_cublas'] = r['cublas']['median_ms'] / r['tiled']['median_ms']
    r['target_80_percent_met'] = r['throughput_fraction_of_cublas'] >= .8
    records.append(r)
save(a.output, records)
