"""Report measured target outcomes without substituting missing measurements."""
import argparse
import json
from pathlib import Path


def generate(directory):
    root = Path(directory)
    def read(name):
        path = root / name
        return json.loads(path.read_text()) if path.exists() else None
    lines = ['# NVIDIA validation results', '',
             'Raw JSON and hardware metadata in this directory are the source of these values.', '']
    checks = []
    logits = read('logits.json')
    if logits is not None:
        checks.append(('FP32 logits max absolute error <= 1e-3', logits['max_abs_error'], logits['max_abs_error'] <= 1e-3))
    gemm = read('gemm.json')
    if gemm:
        for r in gemm:
            checks.append((f"GEMM {r['M']}x{r['N']}x{r['K']} >= 80% cuBLAS", r['throughput_fraction_of_cublas'], r['throughput_fraction_of_cublas'] >= .8))
    attention = read('attention.json')
    if attention:
        for r in attention:
            if r['sequence'] == 2048:
                checks.append((f"Attention 2048 {r['workload']} >= 3.5x", r['speedup'], r['speedup'] >= 3.5))
    quant = read('quantization.json')
    if quant:
        checks.append(('INT8 weight-file reduction >= 47%', quant['storage_reduction_fraction'], quant['storage_reduction_fraction'] >= .47))
        checks.append(('WikiText-2 perplexity increase < 0.1', quant['perplexity_increase'], quant['perplexity_increase'] < .1))
    batching = read('batching.json')
    if batching:
        checks.append(('Continuous/static throughput >= 3x', batching['speedup'], batching['speedup'] >= 3))
    cache = read('kv_gpu.json')
    if cache is not None:
        checks.append(('Sized GPU KV pool waste reduction >= 90%', cache['waste_reduction_fraction'], cache['waste_reduction_fraction'] >= .9))
    lines += ['| Target | Actual value | Outcome |', '|---|---:|---|']
    for label, actual, passed in checks:
        lines.append(f"| {label} | {actual:.9g} | {'Met' if passed else 'Not met'} |")
    if not checks:
        lines.append('| No measured results available | — | Unverified |')
    lines += ['', 'Missing measurements remain unverified. KV GPU measurements use a pool sized to the identical live workload; analytical accounting is reported separately.',
              'An unmet performance target is reported as measured; it is never replaced with the target number.', '']
    result = '\n'.join(lines)
    (root / 'report.md').write_text(result)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory', default='results/gpu'); a = p.parse_args()
    print(generate(a.directory))
