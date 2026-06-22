"""
fix_ticks_csv.py
----------------
Your ticks.csv was exported from pandas with positional column headers
(0, 1, 2, ... 41) instead of named ones. This script renames them to the
names your Rust TickCsvRow struct expects via serde aliases.

CSV column layout (44 cols total, 0-indexed):
  col 0  — row index (drop it)
  col 1  — timestamp (unix ms)
  col 2  — datetime string (redundant, drop)
  col 3  — bid_price1
  col 4  — bid_size1
  col 5  — bid_price2
  col 6  — bid_size2
  ... (10 bid levels total, cols 3-22)
  col 23 — ask_price1
  col 24 — ask_size1
  ... (10 ask levels total, cols 23-42)
  col 43 — (sometimes a trailing column, drop if present)

Usage:
    python fix_ticks_csv.py data/ticks.csv data/ticks_fixed.csv
    # Then rename: move data/ticks_fixed.csv data/ticks.csv
"""

import pandas as pd
import sys

input_path  = sys.argv[1] if len(sys.argv) > 1 else "data/ticks.csv"
output_path = sys.argv[2] if len(sys.argv) > 2 else "data/ticks_fixed.csv"

print(f"Reading {input_path} ...")
df = pd.read_csv(input_path, header=0, index_col=0)  # drop the unnamed row-index col

print(f"Shape before rename: {df.shape}")
print(f"Current columns: {list(df.columns[:6])} ...")

# Build new column names
# col[0] = unix_ms timestamp, col[1] = human datetime (redundant)
# then 10 bid levels (price, size), then 10 ask levels (price, size)
new_cols = ["timestamp", "datetime_str"]

for i in range(1, 11):
    new_cols.append(f"bid_price{i}")
    new_cols.append(f"bid_size{i}")

for i in range(1, 11):
    new_cols.append(f"ask_price{i}")
    new_cols.append(f"ask_size{i}")

# Trim to actual number of columns (safety in case trailing col)
new_cols = new_cols[: len(df.columns)]
df.columns = new_cols

# Drop the redundant human-readable datetime (we keep unix ms as timestamp)
df = df.drop(columns=["datetime_str"], errors="ignore")

print(f"New columns: {list(df.columns[:8])} ...")
print(f"Shape after rename: {df.shape}")
print(f"Sample row:\n{df.iloc[0]}")

df.to_csv(output_path, index=False)
print(f"\nDone! Fixed CSV written to: {output_path}")
print("Now run:  move data\\ticks_fixed.csv data\\ticks.csv   (Windows)")
print("  or:     mv data/ticks_fixed.csv data/ticks.csv       (Linux/Mac)")