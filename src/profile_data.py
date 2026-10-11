import pandas as pd


DATA_PATH = "data/raw/boston_311_2024_sample.csv"

df = pd.read_csv(DATA_PATH)


print("\n=== ON_TIME VALUES ===")
print(df["on_time"].value_counts(dropna=False))


print("\n=== ON_TIME PERCENTAGES ===")
print(
    df["on_time"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)


columns_to_check = [
    "subject",
    "reason",
    "type",
    "department",
    "queue",
    "source",
    "neighborhood",
    "case_status",
]

print("\n=== UNIQUE VALUES ===")

for column in columns_to_check:
    print(f"{column}: {df[column].nunique()} unique values")


print("\n=== TOP 10 SUBJECTS ===")
print(df["subject"].value_counts().head(10))


print("\n=== TOP 10 REASONS ===")
print(df["reason"].value_counts().head(10))


print("\n=== TOP 10 TYPES ===")
print(df["type"].value_counts().head(10))


print("\n=== SOURCES ===")
print(df["source"].value_counts())


print("\n=== CASE STATUS ===")
print(df["case_status"].value_counts())