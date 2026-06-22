# HFT MVP — Kaggle LOB Dataset Setup

## Dataset

Download from Kaggle:
https://www.kaggle.com/datasets/siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data

Place the CSV file at:
```
hft/data/ticks.csv
```

## What Changed from Original Code

| File | Change |
|------|--------|
| `src/market_data/tick.rs` | **Rewrote `TickCsvRow`** — now maps all 40 LOB columns (10 bid levels × price+size, 10 ask levels × price+size). Timestamp is now `i64` milliseconds (Unix epoch), not a string. `Tick` struct now holds `bids: Vec<Level>` and `asks: Vec<Level>` instead of flat `bid_price`/`ask_price` fields. |
| `src/market_data/handler.rs` | Updated `debug!` log to use `tick.bid_price()` / `tick.ask_price()` methods. Otherwise unchanged. |
| `src/main.rs` | Removed unused module declarations (`mod api`, `mod db`, etc.) that would fail to compile without the stub files. Log output updated to show `bid_depth`/`ask_depth`. |
| `backend/cargo.toml` | Stripped DB/HTTP dependencies (`sqlx`, `redis`, `reqwest`, `actix-web`) for the MVP build. Add them back when you implement those phases. |
| `backend/.env` | Simplified — removed Postgres/Redis/inference vars not needed for MVP. |
| `backend/Docker_compose.yml` | MVP compose — only the Rust backend service, no Postgres/Redis/ML inference. |

## CSV Column Format Expected

The Kaggle dataset uses this column layout:

```
timestamp, bid_price1, ask_price1, bid_size1, ask_size1,
           bid_price2, ask_price2, bid_size2, ask_size2,
           ...
           bid_price10, ask_price10, bid_size10, ask_size10
```

- `timestamp` — milliseconds since Unix epoch (e.g. `1688169600123`)
- `bid_priceN` — best N-th bid price (bid_price1 = best bid)
- `ask_priceN` — best N-th ask price (ask_price1 = best ask)
- `bid_sizeN` / `ask_sizeN` — quantity at that level

## Run Locally (without Docker)

```bash
cd hft

# Copy your .env
cp backend/.env .env

# Put the Kaggle CSV here
mkdir -p data
cp ~/Downloads/BTCUSDT_LOB.csv data/ticks.csv

# Build and run
cargo build --manifest-path backend/cargo.toml
TICK_DATA_PATH=data/ticks.csv cargo run --manifest-path backend/cargo.toml
```

## Run via Docker

```bash
cd hft/backend
docker compose -f Docker_compose.yml up --build
```

## Next Phases (not in MVP)

- Add `sqlx` back + migrations to persist ticks to Postgres
- Add `actix-web` API + WebSocket endpoint to stream ticks to a frontend
- Add `redis` for real-time orderbook cache
- Add Python ML inference service (DeepLOB model)
