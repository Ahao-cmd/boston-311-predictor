import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


DATA_PATH = "data/raw/boston_311_2024_full.csv"
CUTOFF = pd.Timestamp("2024-10-01")
END_DATE = pd.Timestamp("2025-01-01")


# 1. Load relevant columns

columns = [
    "open_dt",
    "closed_dt",
    "case_status",
    "on_time",
    "subject",
    "reason",
    "type",
    "source",
    "neighborhood"
]

df = pd.read_csv(
    DATA_PATH,
    usecols=columns,
    low_memory=False
)


# 2. Convert datetime columns

for column in ["open_dt", "closed_dt"]:
    df[column] = pd.to_datetime(
        df[column],
        format="mixed",
        errors="coerce"
    )


# 3. Select cases with known final outcomes

df = df[
    (df["case_status"] == "Closed")
    & (df["on_time"].isin(["ONTIME", "OVERDUE"]))
    & (df["open_dt"].notna())
    & (df["closed_dt"].notna())
].copy()

# Remove inconsistent date records
df = df[df["closed_dt"] >= df["open_dt"]].copy()


# 4. Create target

df["target"] = (
    df["on_time"] == "OVERDUE"
).astype(int)


# 5. Feature engineering

df["open_hour"] = df["open_dt"].dt.hour

df["open_weekday"] = (
    df["open_dt"].dt.dayofweek.astype(str)
)


# 6. Chronological train/test split

train_df = df[
    (df["open_dt"] < CUTOFF)
    & (df["closed_dt"] < CUTOFF)
].copy()

test_df = df[
    (df["open_dt"] >= CUTOFF)
    & (df["open_dt"] < END_DATE)
].copy()


# 7. Choose model features

categorical_features = [
    "subject",
    "reason",
    "type",
    "source",
    "neighborhood",
    "open_weekday"
]

numerical_features = [
    "open_hour"
]

features = categorical_features + numerical_features

X_train = train_df[features]
y_train = train_df["target"]

X_test = test_df[features]
y_test = test_df["target"]


# 8. Preprocess features

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("categorical", categorical_transformer, categorical_features),
    ("numerical", numerical_transformer, numerical_features)
])


# 9. Build model

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])


# 10. Train

print("\n=== FULL DATASET TRAINING ===")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(4))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(4))

if y_train.nunique() != 2 or y_test.nunique() != 2:
    raise ValueError("Training and testing both require two target classes.")

print("\nTraining Random Forest...")

model.fit(X_train, y_train)


# 11. Predict

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Threshold selected using the validation dataset
SELECTED_THRESHOLD = 0.50

# Predictions using the selected threshold
y_pred_adjusted = (
    y_prob >= SELECTED_THRESHOLD
).astype(int)

# 12. Evaluate

majority_class = y_train.mode()[0]

baseline_accuracy = (
    y_test == majority_class
).mean()

print("\n=== MODEL EVALUATION ===")

print(
    "Majority baseline accuracy:",
    round(baseline_accuracy, 4)
)

print(
    "Random Forest accuracy:",
    round(accuracy_score(y_test, y_pred), 4)
)

print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["ONTIME", "OVERDUE"],
        zero_division=0
    )
)

print("\nConfusion matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )
)

print(
    "\nROC-AUC:",
    round(roc_auc_score(y_test, y_prob), 4)
)

# 13. Evaluate selected threshold

print("\n=== ADJUSTED THRESHOLD EVALUATION ===")

print("Selected threshold:", SELECTED_THRESHOLD)

print(
    "Adjusted accuracy:",
    round(accuracy_score(y_test, y_pred_adjusted), 4)
)

print("\nAdjusted classification report:")

print(
    classification_report(
        y_test,
        y_pred_adjusted,
        labels=[0, 1],
        target_names=["ONTIME", "OVERDUE"],
        zero_division=0
    )
)

print("\nAdjusted confusion matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred_adjusted,
        labels=[0, 1]
    )
)

# 14. Save trained model

model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "random_forest_v1.joblib"

model_bundle = {
    "pipeline": model,
    "threshold": SELECTED_THRESHOLD,
    "features": features,
    "model_name": "Random Forest",
    "target_labels": {
        0: "ONTIME",
        1: "OVERDUE"
    }
}

joblib.dump(
    model_bundle,
    model_path,
    compress=3
)

print("\nModel saved to:", model_path)