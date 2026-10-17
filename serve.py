import json
import os
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Choose a specific training run-not an implicit "latest" model.
run_id = os.environ.get("MODEL_RUN_ID")
if not run_id:
    raise RuntimeError("Set MODEL_RUN_ID before starting the server")

project_dir = Path(__file__).resolve().parent
run_dir = project_dir / "artifacts" / run_id

# Load our own trusted model and its metadata once at startup
model = joblib.load(run_dir / "model.joblib")
metadata = json.loads(
    (run_dir / "metrics.json").read_text(encoding="utf-8")
)

app = FastAPI(title="Iris Prediction API")

class FlowerMeasurements(BaseModel):
    sepal_length: float = Field(gt=0, allow_inf_nan=False)
    sepal_width: float = Field(gt=0, allow_inf_nan=False)
    petal_length: float = Field(gt=0, allow_inf_nan=False)
    petal_width: float = Field(gt=0, allow_inf_nan=False)

@app.get("/health")
def health():
    return{
        "status": "ok",
        "model_run_id": run_id
    }

@app.post("/predict")
def predict(flower: FlowerMeasurements):
    # Keep the feature order identical to training
    features = [[
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width
    ]]

    class_id = int(model.predict(features)[0])

    return {
        "species": metadata["class_names"][class_id],
        "model_run_id": run_id
    }