import argparse
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

def is_complete(run_dir):
    return run_dir.is_dir() and all(
        (run_dir / name).is_file() for name in REQUIRED_FILES
    )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0,
                        help="Smaller values mean stronger regularization.")
    parser.add_argument("--run-name", 
                        help="Stable output name. Retries with the same "
                            "name reuse finished output.")
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

def train_into(output_dir, args):
    dataset = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target,
        test_size=0.2, random_state=42, stratify=dataset.target,
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
        "test_size": 0.2,
        "model_type": "StandardScaler + LogisticRegression",
        "max_tier": 1000,
        "dataset": "iris",
        "accuracy": float(accuracy),
        "training_samples": len(y_train),
        "test_samples": len(y_test),
        "random_seed": 42,
        "scikit_learn_version": sklearn.__version__,
        "feature_names": list(dataset.feature_names),
        "class_name": list(dataset.target_names),
    }
    # Written last: its presence means the other files are already saved.
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    reloaded = joblib.load(output_dir / "model.joblib")
    assert (reloaded.predict(X_test) == predictions).all()
    print("Reload check passed: predictions match")

    return accuracy


def main():
    args = parse_args()
    final_dir = ARTIFACTS / args.run_name

    #skip finished work.
    if final_dir.exists():
        if is_complete(final_dir):
            print(f"Run {args.run_name} is already complete; skipping traning")
            return
        sys.exit(f"{final_dir} exists but is incomplete. "
                 "Inspect it; nothing was changed.")

    # Write to a temporary folder first.
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    temp_dir = ARTIFACTS / f".tmp-{args.run_name}-{uuid4().hex[:8]}"
    temp_dir.mkdir()

    try:
        accuracy = train_into(temp_dir, args)

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
    print(f"C: {args.c}")
    print(f"Test accuracy: {accuracy: .3f}")
    print(f"Published: {final_dir}")

if __name__ == "__main__":
    main()