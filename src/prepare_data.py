import pandas as pd


DATA_PATH = "data/raw/boston_311_2024_sample.csv"


df = pd.read_csv(DATA_PATH)


# -----------------------------------
# 1. Keep only closed cases
# -----------------------------------

df = df[df["case_status"] == "Closed"].copy()


# -----------------------------------
# 2. Convert target
# -----------------------------------

df["target"] = df["on_time"].map({
    "ONTIME": 0,
    "OVERDUE": 1
})


# -----------------------------------
# 3. Convert date columns
# -----------------------------------

df["open_dt"] = pd.to_datetime(
    df["open_dt"],
    format="mixed",
    errors="coerce"
)

df["sla_target_dt"] = pd.to_datetime(
    df["sla_target_dt"],
    format="mixed",
    errors="coerce"
)


# -----------------------------------
# 4. Create time features
# -----------------------------------

df["open_hour"] = df["open_dt"].dt.hour
df["open_weekday"] = df["open_dt"].dt.dayofweek
df["open_month"] = df["open_dt"].dt.month


# -----------------------------------
# 5. SLA duration
# -----------------------------------

df["sla_hours"] = (
    df["sla_target_dt"] - df["open_dt"]
).dt.total_seconds() / 3600


# -----------------------------------
# 6. Select model features
# -----------------------------------

features = [
    "subject",
    "reason",
    "type",
    "department",
    "queue",
    "source",
    "neighborhood",
    "open_hour",
    "open_weekday",
    "open_month",
    "sla_hours",
]

# Sort by request creation time before saving
df = df.sort_values("open_dt")

model_df = df[features + ["target"]].copy()
from pathlib import Path

output_path = Path("data/processed/boston_311_prepared.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

model_df.to_csv(output_path, index=False)

print("\nPrepared data saved to:", output_path)


# -----------------------------------
# 7. Display summary
# -----------------------------------

print("\nPrepared dataset shape:")
print(model_df.shape)

print("\nTarget distribution:")
print(model_df["target"].value_counts())

print("\nTarget percentages:")
print(
    model_df["target"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nMissing values:")
print(model_df.isnull().sum())

print("\nFirst 5 rows:")
print(model_df.head())