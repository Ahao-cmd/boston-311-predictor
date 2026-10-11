import pandas as pd


DATA_PATH = "data/raw/boston_311_2024_full.csv"

# Load only the columns needed for this inspection
df = pd.read_csv(
    DATA_PATH,
    usecols=[
        "open_dt",
        "closed_dt",
        "case_status",
        "on_time"
    ],
    low_memory=False
)

print("\n=== FULL DATASET ===")
print("Total records:", len(df))


# 1. Parse dates
df["open_dt"] = pd.to_datetime(
    df["open_dt"],
    format="mixed",
    errors="coerce"
)

df["closed_dt"] = pd.to_datetime(
    df["closed_dt"],
    format="mixed",
    errors="coerce"
)


# 2. Check status distribution
print("\n=== CASE STATUS ===")
print(df["case_status"].value_counts(dropna=False))


# 3. Check ONTIME / OVERDUE distribution
print("\n=== ON_TIME VALUES ===")
print(df["on_time"].value_counts(dropna=False))


# 4. Compare case status with on_time
print("\n=== STATUS VS ON_TIME ===")
print(
    pd.crosstab(
        df["case_status"],
        df["on_time"],
        dropna=False
    )
)


# 5. Select historical closed cases with valid labels
closed = df[
    (df["case_status"] == "Closed")
    & (df["on_time"].isin(["ONTIME", "OVERDUE"]))
    & (df["open_dt"].notna())
].copy()

print("\n=== VALID CLOSED CASES ===")
print("Records:", len(closed))


# 6. Monthly overdue percentages
closed["month"] = closed["open_dt"].dt.to_period("M")

monthly = closed.groupby("month")["on_time"].agg(
    total="size",
    overdue=lambda values: (values == "OVERDUE").sum()
)

monthly["overdue_pct"] = (
    monthly["overdue"] / monthly["total"] * 100
).round(2)

print("\n=== MONTHLY OVERDUE RATES ===")
print(monthly)


# 7. Check which training labels were available
#    before October 1, 2024

cutoff = pd.Timestamp("2024-10-01")

early_cases = closed[
    closed["open_dt"] < cutoff
].copy()

available_by_cutoff = early_cases[
    early_cases["closed_dt"].notna()
    & (early_cases["closed_dt"] < cutoff)
]

print("\n=== TRAINING CUTOFF CHECK ===")
print("Cases opened before Oct 1:", len(early_cases))
print(
    "Cases also closed before Oct 1:",
    len(available_by_cutoff)
)
print(
    "Cases not yet confirmed closed by cutoff:",
    len(early_cases) - len(available_by_cutoff)
)