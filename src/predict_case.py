import joblib
import pandas as pd
from pathlib import Path


# 1. Example 311 service request

open_time = pd.Timestamp("2024-10-15 10:00:00")

request = {
    "subject": "Public Works Department",
    "reason": "Street Cleaning",
    "type": "Requests for Street Cleaning",
    "source": "Citizens Connect App",
    "neighborhood": "Dorchester",
    "open_weekday": str(open_time.dayofweek),
    "open_hour": open_time.hour
}

request_df = pd.DataFrame([request])


# 2. Saved model paths

model_paths = [
    Path("models/logistic_regression_v1.joblib"),
    Path("models/random_forest_v1.joblib")
]


# 3. Display input

print("\n=== BOSTON 311 REQUEST ===")

for key, value in request.items():
    print(f"{key}: {value}")


# 4. Load models and predict

for model_path in model_paths:

    bundle = joblib.load(model_path)

    model = bundle["pipeline"]
    threshold = bundle["threshold"]
    features = bundle["features"]

    # Use the same feature order as training
    X = request_df[features]

    # Get the model score for OVERDUE
    classes = list(model.named_steps["classifier"].classes_)
    overdue_index = classes.index(1)

    overdue_score = model.predict_proba(X)[0, overdue_index]

    # Apply the saved classification threshold
    prediction = (
        "OVERDUE"
        if overdue_score >= threshold
        else "ONTIME"
    )

    print("\n============================")
    print("Model:", bundle["model_name"])
    print("Threshold:", threshold)

    print(
        "OVERDUE score:",
        f"{overdue_score:.4f}"
    )

    print("Prediction:", prediction)
    