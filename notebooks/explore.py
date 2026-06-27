import pandas as pd

columns = [
    "timestamp", "can_id", "dlc",
    "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
    "flag"
]

df = pd.read_csv("data/DoS_dataset.csv", header=None, names=columns)

print("Raw shape:", df.shape)

# Step 1 — Drop rows with missing values
df = df.dropna()
print("After dropping nulls:", df.shape)

# Step 2 — Convert hex string columns to integers
hex_cols = ["can_id", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]
for col in hex_cols:
    df[col] = df[col].apply(lambda x: int(x, 16))

# Step 3 — Convert flag to binary (R=0 normal, T=1 attack)
df["flag"] = df["flag"].map({"R": 0, "T": 1})

print("\nCleaned data types:")
print(df.dtypes)

print("\nCleaned head:")
print(df.head())

print("\nFlag counts after cleaning:")
print(df["flag"].value_counts())

# Save cleaned data so we don't repeat this every time
df.to_csv("data/dos_clean.csv", index=False)
print("\nSaved cleaned data to data/dos_clean.csv")