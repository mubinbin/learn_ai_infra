#!/bin/bash
set -euo pipefail

BASE_URL=http://127.0.0.1:8000
EXPECTED_RUN_ID="${MODEL_RUN_ID:?Set MODEL_RUN_ID before running the tests.}"

# Store response bodies temporarily and remove them when finished.
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

check_status() {
    if [ "$2" != "$3" ]; then
        echo "FAIL: $1 - expected HTTP $3, got $2"
        cat "$TEMP_DIR/response.json"
        exit 1
    fi
}

check_field() {
    python3 - "$TEMP_DIR/response.json" "$1" "$2" << 'PYTHON'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

key = sys.argv[2]
expected = sys.argv[3]
actual = data.get(key)

if actual != expected:
    sys.exit(
        f"FAIL: {key} - expected {expected!r}, got {actual!r}"
    )
PYTHON
}

# "=== 1. Health Check ==="
status=$(curl --silent --show-error \
    --connect-timeout 3 --max-time 10 \
    -o "$TEMP_DIR/response.json" \
    -w '%{http_code}' \
    "$BASE_URL/health")

check_status "Health" "$status" "200"
check_field "status" "ok"
check_field "model_run_id" "$EXPECTED_RUN_ID"
echo "PASS: health and model run ID"

# "=== 2. Valid prediction ==="
status=$(curl --silent --show-error \
    --connect-timeout 3 --max-time 10 \
    -o "$TEMP_DIR/response.json" \
    -w '%{http_code}' \
    -X POST "$BASE_URL/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }')

check_status "Prediction" "$status" "200"
check_field "species" "setosa"
check_field "model_run_id" "$EXPECTED_RUN_ID"
echo "PASS: prediction and model run ID"

# "=== 3. Invalid input - expect HTTP 422 ==="
status=$(curl --silent --show-error \
    --connect-timeout 3 --max-time 10 \
    -o "$TEMP_DIR/response.json" \
    -w '%{http_code}' \
    -X POST "$BASE_URL/predict" \
    -H "Content-Type: application/json" \
    -d '{"sepal_length": -1}')

check_status "Invalid input" "$status" "422"
echo "PASS: invalid input rejected"

echo "All checks passed"