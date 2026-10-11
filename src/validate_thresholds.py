import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score
)


DATA_PATH = "data/raw/boston_311_2024_full.csv"

TRAIN_CUTOFF = pd.Timestamp("2024-07-01")
VALIDATION_CUTOFF = pd.Timestamp("2024-10-01")


# Load data
df = pd.read_csv(DATA_PATH, low_memory=False)

for column in ["open_dt", "closed_dt"]:
    df[column] = pd.to_datetime(
        df[column],
        format="mixed",
        errors="coerce"
    )

# Only use cases with confirmed final outcomes
df = df[
    (df["case_status"] == "Closed")
    & (df["on_time"].isin(["ONTIME", "OVERDUE"]))
    & (df["open_dt"].notna())
    & (df["closed_dt"].notna())
    & (df["closed_dt"] >= df["open_dt"])
].copy()

df["target"] = (df["on_time"] == "OVERDUE").astype(int)

df["open_hour"] = df["open_dt"].dt.hour
df["open_weekday"] = df["open_dt"].dt.dayofweek.astype(str)


# Train: cases resolved before July 1
train_df = df[
    (df["open_dt"] < TRAIN_CUTOFF)
    & (df["closed_dt"] < TRAIN_CUTOFF)
].copy()

# Validation: July-September cases resolved before October 1
val_df = df[
    (df["open_dt"] >= TRAIN_CUTOFF)
    & (df["open_dt"] < VALIDATION_CUTOFF)
    & (df["closed_dt"] < VALIDATION_CUTOFF)
].copy()


categorical_features = [
    "subject",
    "reason",
    "type",
    "source",
    "neighborhood",
    "open_weekday"
]

numerical_features = ["open_hour"]

features = categorical_features + numerical_features

X_train = train_df[features]
y_train = train_df["target"]

X_val = val_df[features]
y_val = val_df["target"]


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

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    ))
])


print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

model.fit(X_train, y_train)

# Probabilities for OVERDUE = 1
probabilities = model.predict_proba(X_val)[:, 1]

print("\nValidation ROC-AUC:",
      round(roc_auc_score(y_val, probabilities), 4))

print("\n=== THRESHOLD COMPARISON ===")
print("Threshold | Precision | Recall | F2 Score")

for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f2 = fbeta_score(
        y_val, predictions, beta=2, zero_division=0
    )

    print(
        f"{threshold:.2f}      | "
        f"{precision:.3f}     | "
        f"{recall:.3f}  | "
        f"{f2:.3f}"
    )