import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ARTIFACTS = Path("artifacts")
REQUIRED_FILES = ("model.joblib", "metrics.json", "requirements.txt")
RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

def fail(message):
    sys.exit(f"Data check failed: {message}. Nothing was trained.")

def is_complete(run_dir):
    return run_dir.is_dir() and all(
        (run_dir / name).is_file() for name in REQUIRED_FILES
    )

def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0,
                        help="Smaller values mean stronger regularization.")
    parser.add_argument("--run-name", 
                        help="Stable output name. Retries with the same "
                            "name reuse finished output.")
    parser.add_argument("--data",
                        help="Dataset folder containing data.csv and "
                             "manifest.json. Omit to use the built-in data.")
    args = parser.parse_args()

    if args.c <= 0:
        parser.error("--c must be greater than zero")

    if args.run_name is None:
        # No name given: behave as before, with a unique random name.
        args.run_name = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-" + uuid4().hex[:8]
        )
    elif not RUN_NAME_PATTERN.fullmatch(args.run_name):
        parser.error("--run-name may use only letters, digits, '.', '_' and '-'")
    return args

def load_data(data_dir):
    if data_dir is None:
        iris = load_iris()
        info = {"data_name": "builtin-iris", "data_sha256": None,
                "data_rows": len(iris.target)}
        return iris.data, iris.target, info

    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.json"
    csv_path = data_dir / "data.csv"
    for path in (manifest_path, csv_path):
        if not path.is_file():
            fail(f"missing {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1 Did the file arrive exactly as published?
    actual = sha256_of(csv_path)
    if actual != manifest["sha256"]:
        fail(f"{csv_path} does not match the checksum in its manifest")

    # 2 Is every row usable?
    rows, labels = [],[]
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FEATURES + ["species"]:
            fail(f"unexpected columns {reader.fieldnames}")
        for line_no, record in enumerate(reader, start=2):
            try:
                values = [float(record[name]) for name in FEATURES]
            except (TypeError, ValueError):
                fail(f"non-numeric value on line {line_no}")
            if not all(math.isfinite(v) and v > 0 for v in values):
                fail(f"invalid measurement on line {line_no}")
            if record["sepcies"] not in CLASS_NAMES:
                fail(f"unknown species on line {line_no}")
            rows.append(values)
            labels.append(CLASS_NAMES.index(record["sepcies"]))

    # 3 Did every row arrive?
    if len(rows) != manifest["row"]:
        fail(f"expected {manifest['row']} rows, found {len(rows)}")

    info = {"data_name": manifest["name"], "data_sha256": actual, "data_rows": len(rows)}
    return np.array(rows), np.array(labels), info

def train_into(output_dir, args, X, y, data_info):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2, random_state=42, stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=args.c, max_iter=1000, random_state=42),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    dependencies = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True, capture_output=True, text=True,
    )
    (output_dir / "requirements.txt").write_text(dependencies.stdout, encoding="utf-8")

    joblib.dump(model, output_dir / "model.joblib")

    metrics = {
        "run_id": args.run_name,
        "C": args.c,
        **data_info,
        "test_size": 0.2,
        "model_type": "StandardScaler + LogisticRegression",
        "max_tier": 1000,
        "accuracy": float(accuracy),
        "training_samples": len(y_train),
        "test_samples": len(y_test),
        "random_seed": 42,
        "scikit_learn_version": sklearn.__version__,
        "feature_names": FEATURES,
        "class_name": CLASS_NAMES,
    }
    # Written last: its presence means the other files are already saved.
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    reloaded = joblib.load(output_dir / "model.joblib")
    assert (reloaded.predict(X_test) == predictions).all()
    print("Reload check passed: predictions match")

    return accuracy


def main():
    args = parse_args()

    # Verify the data before touching any output
    X, y, data_info = load_data(args.data)
    final_dir = ARTIFACTS / args.run_name

    if final_dir.exists():
        # if is_complete(final_dir):
        #     print(f"Run {args.run_name} is already complete; skipping traning")
        #     return
        if not is_complete(final_dir):
            sys.exit(f"{final_dir} exists but is incomplete. "
                    "Inspect it; nothing was changed.")
        previous = json.loads(
            (final_dir / "metrics.json").read_text(encoding="utf-8")
        )

        same_inputs = (previous.get("C") == args.c
                       and previous.get("data_sha256") == data_info["data_sha256"])
        if same_inputs:
            print(f"Run {args.run_name} is already complete; skipping training.")
            return
        sys.exit(f"Run name {args.run_name} was already used with different "
                 "input. Choose a new run name; nothing was changed.")

    # Write to a temporary folder first.
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    temp_dir = ARTIFACTS / f".tmp-{args.run_name}-{uuid4().hex[:8]}"
    temp_dir.mkdir()

    try:
        accuracy = train_into(temp_dir, args, X, y, data_info)

        # Publish: one rename makes the whole folder appear at once.
        try:
            temp_dir.rename(final_dir)
        except OSError:
            if is_complete(final_dir):
                print("Another attempt pulished this run first; "
                      "discarding this copy.")
                shutil.rmtree(temp_dir)
                return
            raise
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(f"Run ID: {args.run_name}")
    print(f"Data: {data_info['data_name']} ({data_info['data_rows']} rows)")
    print(f"C: {args.c}")
    print(f"Test accuracy: {accuracy: .3f}")
    print(f"Published: {final_dir}")

if __name__ == "__main__":
    main()