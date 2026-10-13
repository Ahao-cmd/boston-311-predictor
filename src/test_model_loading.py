import joblib
from pathlib import Path


model_files = [
    Path("models/logistic_regression_v1.joblib"),
    Path("models/random_forest_v1.joblib")
]


for model_path in model_files:

    print("\nLoading:", model_path)

    bundle = joblib.load(model_path)

    print("Model:", bundle["model_name"])
    print("Threshold:", bundle["threshold"])
    print("Features:", bundle["features"])

    pipeline = bundle["pipeline"]

    print("Classifier:", type(
        pipeline.named_steps["classifier"]
    ).__name__)

    print("Model loaded successfully!")