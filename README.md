# HFT AI Signal Engine

This project is a Rust + ONNX + Redis MVP for generating AI-assisted high-frequency trading signals. It accepts market feature data, runs an ONNX model, applies a confidence rule, stores the result, and exposes the signal through a REST API and a simple browser dashboard.

It returns one of three signals:

- `BUY`: the model expects upward movement with enough confidence.
- `SELL`: the model expects downward movement with enough confidence.
- `HOLD`: the model is unsure or the confidence threshold was not met.

This project does not place trades. It is a signal engine and dashboard only.

## Install Links

Install these before running the full project:

| Tool | Why It Is Needed | Link |
|------|------------------|------|
| Git | Clone or manage the repository | https://git-scm.com/downloads |
| Rust | Build and run the backend | https://www.rust-lang.org/tools/install |
| Microsoft C++ Build Tools | Required by Rust on Windows MSVC | https://visualstudio.microsoft.com/visual-cpp-build-tools/ |
| Python 3 | Export/create model artifacts and features | https://www.python.org/downloads/ |
| Docker Desktop | Easiest way to run Redis locally | https://www.docker.com/products/docker-desktop/ |
| Redis | Signal storage backend, if not using Docker | https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/ |
| ONNX | Model format used by the inference engine | https://onnx.ai/ |

Useful documentation:

- Actix Web: https://actix.rs/
- Redis Docker image: https://hub.docker.com/_/redis
- ONNX Runtime Rust crate: https://crates.io/crates/ort
- scikit-learn: https://scikit-learn.org/stable/install.html
- skl2onnx: https://onnx.ai/sklearn-onnx/

## Project Layout

```text
hft_mvp_updated/
  src/
    main.rs                         # Starts backend, API, model, Redis, and frontend serving
    api/signal_handlers.rs          # /health, /predict, /signal, /signal/history, /model/info
    signal_engine/                  # ONNX loading, inference, signal generation, prediction store
    features/                       # Feature formulas, rolling window, feature vectors
    market_data/                    # CSV/tick/order book adapters
    orderbook/                      # Order book levels and metrics
    redis/redis_client.rs           # Redis read/write layer
  frontend/
    index.html                      # Dashboard UI
    script.js                       # Calls backend at http://localhost:8080
    style.css                       # Dashboard styling
  ml/models/
    best_model.onnx                 # Bundled ONNX model artifact
    scaler.json                     # Bundled scaler artifact
  create_test_model.py              # Creates a minimal test model
  export_features.py                # Converts tick CSV into model feature rows
  export_scaler.py                  # Exports sklearn scaler/model artifacts
```

## Pipeline Overview

The project is organized as a trading-signal pipeline:

```text
Raw market data
      |
      v
Tick parser and LOB adapter
      |
      v
Order book reconstruction
      |
      v
Feature engineering
      |
      v
ONNX model inference
      |
      v
Signal rules: BUY / SELL / HOLD
      |
      v
Redis + in-memory prediction store
      |
      v
REST API + frontend dashboard
```

### Phase 1 - Market Data

Raw tick/order-book rows are loaded from CSV and converted into typed Rust structures.

Main files:

| File | Purpose |
|------|---------|
| `src/market_data/tick.rs` | Defines a market tick. |
| `src/market_data/lob_adapter.rs` | Converts level-2 CSV columns into order-book data. |
| `src/market_data/handler.rs` | Loads, sorts, and serves market rows. |

### Phase 2 - Order Book

Bid and ask levels are organized into an order book so the system can calculate market shape.

Main files:

| File | Purpose |
|------|---------|
| `src/orderbook/level.rs` | One price/quantity level. |
| `src/orderbook/order_book.rs` | Bid/ask book structure. |
| `src/orderbook/metrics.rs` | Spread, depth, imbalance, and related metrics. |

### Phase 3 - Feature Engineering

Order-book state is transformed into model-ready numeric features.

Features include:

- spread
- mid price
- order book imbalance
- rolling volatility
- momentum
- liquidity ratio
- volume imbalance
- trade intensity
- bid volume
- ask volume
- total liquidity

Main files:

| File | Purpose |
|------|---------|
| `src/features/calculators.rs` | Small reusable feature formulas. |
| `src/features/rolling_window.rs` | Rolling buffer used for volatility and momentum. |
| `src/features/feature_vector.rs` | Feature vector shape used by earlier feature APIs. |
| `src/features/feature_engine.rs` | Combines raw book data into features. |
| `src/signal_engine/inference.rs` | Defines the API `FeatureVector` consumed by `/predict`. |

### Phase 4 - Model Training and Export

Training happens in Python. The trained model and scaler are exported into artifacts the Rust backend can load.

Expected runtime artifacts:

```text
ml/models/best_model.onnx
ml/models/scaler.json
```

Optional artifacts supported by the loader:

```text
ml/models/feature_columns.json
ml/models/label_encoder.json
ml/models/training_metadata.json
ml/models/model_metrics.json
```

If optional files are missing, the backend uses default feature order and label mapping.

Main files:

| File | Purpose |
|------|---------|
| `ml/notebook/hft_phase4_v2.ipynb` | Training/export notebook. |
| `export_features.py` | Builds `data/features.csv` from `data/ticks.csv`. |
| `export_scaler.py` | Exports sklearn scaler/model into JSON/ONNX. |
| `create_test_model.py` | Creates a minimal local ONNX model for testing. |

### Phase 5 - Signal Engine and Dashboard

This is the runnable backend. It loads ONNX artifacts, connects to Redis, starts Actix Web, serves the frontend, and exposes prediction APIs.

Main files:

| File | Purpose |
|------|---------|
| `src/main.rs` | App startup, model loading, Redis connection, routes, frontend serving. |
| `src/signal_engine/onnx_loader.rs` | Loads ONNX session, scaler, feature columns, label encoder. |
| `src/signal_engine/inference.rs` | Validates features, scales input, runs ONNX inference. |
| `src/signal_engine/signal_generator.rs` | Converts inference output into final trading signal. |
| `src/signal_engine/prediction.rs` | Keeps recent predictions in memory. |
| `src/api/signal_handlers.rs` | REST API handlers. |
| `src/redis/redis_client.rs` | Stores latest signal and signal history in Redis. |
| `frontend/` | Browser dashboard served by the backend. |

## First-Time Setup

Open PowerShell in the project folder:

```powershell
cd C:\Users\ADMIN\Downloads\hft_mvp_updated
```

### 1. Check Rust

```powershell
rustc --version
cargo --version
```

If these commands fail, install Rust from:

```text
https://www.rust-lang.org/tools/install
```

### 2. Start Redis

Using Docker:

```powershell
docker run -d --name redis-hft -p 6379:6379 redis:7-alpine
```

If the container already exists:

```powershell
docker start redis-hft
```

Check Redis is reachable:

```powershell
docker exec -it redis-hft redis-cli ping
```

Expected output:

```text
PONG
```

### 3. Confirm Model Artifacts

The backend defaults to:

```text
ml/models
```

Confirm the files exist:

```powershell
Get-ChildItem ml\models
```

You should see at least:

```text
best_model.onnx
scaler.json
```

If you want to use a different model folder:

```powershell
$env:ML_DIR = "path/to/your/models"
cargo run
```

### 4. Optional Python Setup

Only needed if you want to generate features or export/create model artifacts:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas onnx scikit-learn skl2onnx
```

Generate `data/features.csv` from `data/ticks.csv`:

```powershell
python export_features.py
```

Create a minimal test model:

```powershell
python create_test_model.py
```

Note: `create_test_model.py` currently writes to `models/`. Either run with:

```powershell
$env:ML_DIR = "models"
cargo run
```

or copy the generated files into `ml/models`.

## Run The Project

Start Redis first, then run:

```powershell
cargo run
```

For a release build:

```powershell
cargo run --release
```

The backend listens at:

```text
http://localhost:8080
```

The dashboard is served by the backend:

```text
http://localhost:8080/
```

The frontend JavaScript calls:

```text
http://localhost:8080/health
http://localhost:8080/predict
http://localhost:8080/signal/history/BTCUSDT?limit=20
```

If you open `frontend/index.html` separately through VS Code Live Server, the frontend still calls `http://localhost:8080`, so the Rust backend must be running on port `8080`.

## API Endpoints

### Health

```text
GET http://localhost:8080/health
```

Returns backend status, model version, prediction count, and symbol count.

### Predict

```text
POST http://localhost:8080/predict
```

Example body:

```json
{
  "timestamp": 1718000000000,
  "symbol": "BTCUSDT",
  "spread": 1.0,
  "mid_price": 30000.0,
  "order_book_imbalance": 0.1,
  "rolling_volatility": 0.002,
  "momentum": 0.001,
  "liquidity_ratio": 1.2,
  "volume_imbalance": 0.05,
  "trade_intensity": 80.0,
  "bid_volume": 1200.0,
  "ask_volume": 1000.0,
  "total_liquidity": 2200.0
}
```

Example PowerShell request:

```powershell
$body = @{
  timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
  symbol = "BTCUSDT"
  spread = 1.0
  mid_price = 30000.0
  order_book_imbalance = 0.1
  rolling_volatility = 0.002
  momentum = 0.001
  liquidity_ratio = 1.2
  volume_imbalance = 0.05
  trade_intensity = 80.0
  bid_volume = 1200.0
  ask_volume = 1000.0
  total_liquidity = 2200.0
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -ContentType "application/json" `
  -Body $body `
  "http://localhost:8080/predict"
```

Example response:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1718000000000,
  "symbol": "BTCUSDT",
  "signal": "HOLD",
  "confidence": 0.6,
  "prob_sell": 0.2,
  "prob_hold": 0.6,
  "prob_buy": 0.2,
  "model_version": "v1782781833",
  "inference_ms": 2.4,
  "is_actionable": false
}
```

### Latest Signal

```text
GET http://localhost:8080/signal/BTCUSDT
```

Returns the latest signal stored for `BTCUSDT`.

### Signal History

```text
GET http://localhost:8080/signal/history/BTCUSDT?limit=20
```

Returns recent predictions for `BTCUSDT`, newest first.

### Model Info

```text
GET http://localhost:8080/model/info
```

Returns loaded metadata, feature order, and active model version.

## How Signals Are Decided

The model returns probabilities for `SELL`, `HOLD`, and `BUY`.

The app only emits an actionable `BUY` or `SELL` when:

```text
confidence > 0.70
```

Decision table:

| Model Output | Confidence | Final Signal |
|--------------|------------|--------------|
| BUY | Greater than 0.70 | BUY |
| SELL | Greater than 0.70 | SELL |
| BUY | 0.70 or lower | HOLD |
| SELL | 0.70 or lower | HOLD |
| HOLD | Any confidence | HOLD |

This makes the engine conservative. When the model is unsure, the API returns `HOLD`.

## Frontend Dashboard

The dashboard shows:

- backend, Redis, and model status
- simulated live market values
- top bid/ask table
- generated feature vector values
- AI signal and probabilities
- recent prediction history

The frontend currently generates a live-looking `BTCUSDT` feature vector in the browser and posts it to `/predict` once per second. The backend performs real ONNX inference and stores the resulting signal.

Open:

```text
http://localhost:8080/
```

If the dashboard says it cannot reach the backend:

1. Confirm this works in the browser:

   ```text
   http://localhost:8080/health
   ```

2. Confirm port `8080` is not occupied by an old server:

   ```powershell
   netstat -ano | findstr :8080
   ```

3. Stop the old process if needed:

   ```powershell
   Stop-Process -Id <PID> -Force
   ```

4. Restart:

   ```powershell
   cargo run
   ```

5. Hard refresh the browser:

   ```text
   Ctrl + F5
   ```

## Exporting A Real Model

If your training pipeline produced sklearn artifacts, export them into the runtime model folder:

```powershell
python export_scaler.py `
  --scaler-pkl path\to\scaler.pkl `
  --scaler-json ml\models\scaler.json `
  --model-pkl path\to\best_model.pkl `
  --onnx-path ml\models\best_model.onnx `
  --n-features 12
```

Then run:

```powershell
cargo run --release
```

## Tests

Run:

```powershell
cargo test
```

Run compile checks:

```powershell
cargo check
```

Some Redis integration tests may be marked ignored because they require a running Redis instance.

## Common Problems

### Redis connection failed

Start Redis:

```powershell
docker start redis-hft
```

Or recreate it:

```powershell
docker run -d --name redis-hft -p 6379:6379 redis:7-alpine
```

### Model file missing

The backend looks in `ml/models` by default.

Check:

```powershell
Get-ChildItem ml\models
```

If your model is in another folder:

```powershell
$env:ML_DIR = "models"
cargo run
```

### Port 8080 already in use

Find the process:

```powershell
netstat -ano | findstr :8080
```

Stop it:

```powershell
Stop-Process -Id <PID> -Force
```

Or run on another port:

```powershell
$env:BIND_ADDR = "127.0.0.1:8081"
cargo run
```

Then open:

```text
http://localhost:8081/
```

### Frontend cannot reach backend

Make sure `frontend/script.js` points to:

```js
const API_BASE_URL = "http://localhost:8080";
```

Then confirm:

```text
http://localhost:8080/health
```

If that works, hard refresh the browser with `Ctrl + F5`.

## Important Note

This project is for software development, local testing, and demonstration. It is not financial advice and should not be connected to real trading without risk controls, monitoring, compliance review, and a tested execution layer.
