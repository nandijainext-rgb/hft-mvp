#!/usr/bin/env python3
"""
Export feature rows from data/ticks.csv for the Rust signal engine.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "ticks.csv"
OUTPUT_PATH = BASE_DIR / "data" / "features.csv"

print(f"Loading {INPUT_PATH}...")
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df):,} rows. Columns: {list(df.columns[:6])}")

df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)
df["symbol"] = "BTCUSDT"

df["mid_price"] = (df["bid_price1"] + df["ask_price1"]) / 2
df["spread"] = (df["ask_price1"] - df["bid_price1"]).clip(lower=0)

bid_size_cols = [f"bid_size{i}" for i in range(1, 11)]
ask_size_cols = [f"ask_size{i}" for i in range(1, 11)]
df["bid_volume"] = df[bid_size_cols].sum(axis=1)
df["ask_volume"] = df[ask_size_cols].sum(axis=1)
df["total_liquidity"] = df["bid_volume"] + df["ask_volume"]

df["order_book_imbalance"] = (
    (df["bid_volume"] - df["ask_volume"]) / (df["total_liquidity"] + 1e-9)
).clip(-1, 1)
df["volume_imbalance"] = (
    df["bid_volume"] - df["ask_volume"]
) / (df["total_liquidity"] + 1e-9)
df["liquidity_ratio"] = (df["bid_volume"] / (df["ask_volume"] + 1e-9)).clip(0.01, 100)

returns = df["mid_price"].pct_change()
df["rolling_volatility"] = returns.rolling(50).std().fillna(0)
df["momentum"] = df["mid_price"].pct_change(20).fillna(0)
df["trade_intensity"] = np.where(returns.notna(), 1.0, 0.0)
df["trade_intensity"] = df["trade_intensity"].rolling(20).sum().fillna(0)

output_cols = [
    "timestamp",
    "symbol",
    "spread",
    "mid_price",
    "order_book_imbalance",
    "rolling_volatility",
    "momentum",
    "liquidity_ratio",
    "volume_imbalance",
    "trade_intensity",
    "bid_volume",
    "ask_volume",
    "total_liquidity",
]

df_out = df[output_cols].iloc[50:].dropna().reset_index(drop=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(OUTPUT_PATH, index=False)
print(f"Done - exported {len(df_out):,} rows -> {OUTPUT_PATH}")
print(df_out.head())
