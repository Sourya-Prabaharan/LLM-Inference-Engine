import json
from pathlib import Path
import subprocess
import statistics


def invoke(command):
    return json.loads(subprocess.check_output([str(x) for x in command], text=True))


def save(path, records):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


def summary(values):
    ordered = sorted(values)
    return dict(median=statistics.median(values), mean=statistics.mean(values),
                variance=statistics.pvariance(values), p95=ordered[min(len(ordered)-1, int(.95*len(ordered)))])
