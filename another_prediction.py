import json
import joblib

model = joblib.load("artifacts/model.joblib")

with open("artifacts/metrics.json") as f:
    metadata = json.load(f)

# Feature order:
# sepal length, sepal width, petal length, petal width (cm)
sample = [[10, 5.5, 3.4, 2.2]]

predition = int(model.predict(sample)[0])
print("Predicted species:", metadata["class_names"][predition])