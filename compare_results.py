import json
from pathlib import Path

print(f"{'Run ID':<32} {'C':>6} {'Accuracy':>10}")

for path in sorted(Path("artifacts").glob("*/metrics.json")):
    metrics = json.loads(path.read_text())

    # Skip older runs that did not record C.
    if "C" not in metrics:
        continue

    print(
        f"{metrics['run_id']:<32}"
        f"{metrics['C']:>6.2f}"
        f"{metrics['accuracy']:>10.3f}"
    )