import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# 1. Load prepared data

df = pd.read_csv(
    "data/processed/boston_311_prepared.csv"
)


# 2. Select features

categorical_features = [
    "subject",
    "reason",
    "type",
    "source",
    "neighborhood"
]

numerical_features = [
    "open_hour",
    "open_weekday",
    "open_month"
]

features = categorical_features + numerical_features

X = df[features]
y = df["target"]


# 3. Chronological train/test split

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# 4. Preprocess categorical features

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


# 5. Preprocess numerical features

numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


preprocessor = ColumnTransformer([
    ("categorical", categorical_transformer, categorical_features),
    ("numerical", numerical_transformer, numerical_features)
])


# 6. Build model

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# 7. Train model

print("Training Logistic Regression...")

model.fit(X_train, y_train)


# 8. Make predictions

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# 9. Evaluate results

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

majority_class = y_train.mode()[0]

baseline_accuracy = (
    y_test == majority_class
).mean()

print(
    "\nMajority-class baseline accuracy:",
    round(baseline_accuracy, 4)
)

print(
    "Logistic Regression accuracy:",
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
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

if y_test.nunique() == 2:
    print(
        "\nROC-AUC:",
        round(roc_auc_score(y_test, y_prob), 4)
    )