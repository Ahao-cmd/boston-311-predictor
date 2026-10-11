import pandas as pd


DATA_PATH = "data/raw/boston_311_2024_sample.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print(column)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False))

# Check the time coverage of the sample

dates = pd.to_datetime(
    df["open_dt"],
    format="mixed",
    errors="coerce"
)

print("\n=== DATE RANGE ===")
print("Earliest request:", dates.min())
print("Latest request:", dates.max())

print("\n=== REQUESTS PER MONTH ===")
print(
    dates.dt.to_period("M")
    .value_counts()
    .sort_index()
)