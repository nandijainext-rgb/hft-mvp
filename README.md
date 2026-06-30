# HFT AI Signal Engine

This project is a small service that looks at market features, asks an AI model what the next trading signal should be, and returns one of three answers:

- `BUY`: the model is confident price may move up.
- `SELL`: the model is confident price may move down.
- `HOLD`: the model is not confident enough, so do nothing.

Think of it as a decision helper. It does not place trades by itself. It only creates signals that another system or a person can review.

## What This App Does

1. Loads an AI model from `models/best_model.onnx`.
2. Loads a scaler from `models/scaler.json` so incoming numbers are prepared the same way as training data.
3. Connects to Redis, which is a fast temporary storage service.
4. Starts a web server on port `8080`.
5. Accepts market feature data through `/predict`.
6. Returns a trading signal and saves the latest signal.

## Phase-by-Phase Structure

The project is built in phases. Each phase adds one layer to the trading system.

```text
Phase 1: Read market data
        |
        v
Phase 2: Build the order book
        |
        v
Phase 3: Create useful features
        |
        v
Phase 4: Train/export the AI model
        |
        v
Phase 5: Run the AI signal engine
        |
        v
Phase 6: Execution engine, future work
```

### Phase 1 - Market Data

This phase reads raw market data from a CSV file and turns it into clean Rust objects that the rest of the system can understand.

```text
CSV file
   |
   v
Tick parser
   |
   v
Clean Tick data
```

Main files:

| File | What It Does |
|------|--------------|
| `src/market_data/tick.rs` | Defines one market update, called a `Tick`. |
| `src/market_data/lob_adapter.rs` | Converts CSV order-book columns into useful market data. |
| `src/market_data/handler.rs` | Loads and sorts the market data. |

Plain-English result: the app can read market rows such as prices, sizes, bids, and asks.

### Phase 2 - Order Book

This phase organizes market data into an order book. An order book is the live list of people willing to buy and sell at different prices.

```text
Clean Tick data
   |
   v
OrderBook
   |
   v
Best bid, best ask, spread, depth
```

Main files:

| File | What It Does |
|------|--------------|
| `src/orderbook/level.rs` | Represents one price level in the book. |
| `src/orderbook/order_book.rs` | Stores bid and ask levels. |
| `src/orderbook/metrics.rs` | Calculates order-book measurements. |

Plain-English result: the app can understand the current market shape, not just one raw row.

### Phase 3 - Feature Engineering

This phase turns order-book data into numbers that the AI model can use.

Examples of features:

- spread
- mid price
- order book imbalance
- rolling volatility
- momentum
- liquidity ratio
- trade intensity

```text
OrderBook data
   |
   v
FeatureEngine
   |
   v
FeatureVector
```

Main files:

| File | What It Does |
|------|--------------|
| `src/features/calculators.rs` | Small formulas for market measurements. |
| `src/features/rolling_window.rs` | Keeps recent values so moving calculations can be made. |
| `src/features/feature_vector.rs` | Defines the final list of numbers sent to the model. |
| `src/features/feature_engine.rs` | Combines all calculations into one feature engine. |

Plain-English result: the app creates the exact input numbers the AI model expects.

### Phase 4 - Model Training and Export

This phase happens mostly outside the Rust app. A model is trained in Python, then exported into files the Rust app can load.

```text
Training data
   |
   v
Python model training
   |
   v
best_model.onnx + scaler.json
```

Main files in this repo:

| File | What It Does |
|------|--------------|
| `export_scaler.py` | Converts a trained Python scaler/model into Rust-friendly files. |
| `create_test_model.py` | Creates a small test model when the real model is not ready. |
| `models/best_model.onnx` | The AI model file used by Phase 5. |
| `models/scaler.json` | Tells Phase 5 how to prepare input numbers before prediction. |

Plain-English result: the trained model becomes portable and can run inside the Rust backend.

### Phase 5 - AI Signal Engine

This is the current main app. It loads the model, receives feature data, asks the AI for a prediction, and returns a trading signal.

```text
FeatureVector
   |
   v
InferenceEngine
   |
   v
BUY, SELL, or HOLD
   |
   v
Redis + API response
```

Main files:

| File | What It Does |
|------|--------------|
| `src/main.rs` | Starts Phase 5 and opens the web API. |
| `src/signal_engine/onnx_loader.rs` | Loads the AI model and scaler. |
| `src/signal_engine/inference.rs` | Runs the model prediction. |
| `src/signal_engine/signal_generator.rs` | Applies the confidence rule and creates the final signal. |
| `src/signal_engine/prediction.rs` | Keeps recent predictions in memory. |
| `src/api/signal_handlers.rs` | Provides `/health`, `/predict`, and signal lookup URLs. |
| `src/redis/redis_client.rs` | Saves and reads signals from Redis. |

Plain-English result: the app can answer, "Based on these market numbers, should we buy, sell, or hold?"

### Phase 6 - Execution Engine

This phase is future work. It would read the signals from Phase 5 and decide whether to place real or simulated orders.

```text
TradingSignal
   |
   v
Risk checks
   |
   v
Order decision
   |
   v
Broker/exchange or simulator
```

Expected future pieces:

| Part | What It Would Do |
|------|------------------|
| Risk checks | Stop unsafe trades before they happen. |
| Position tracking | Know what the system already owns. |
| Order execution | Send approved orders to a broker, exchange, or simulator. |
| Monitoring | Show whether trades and signals are behaving correctly. |

Plain-English result: this would be the layer that acts on signals. It is not part of the current app yet.

## Main Files

| File | Plain-English Purpose |
|------|------------------------|
| `src/main.rs` | Starts the whole app, loads the model, connects Redis, and opens the API. |
| `src/signal_engine/inference.rs` | Sends market features into the AI model and reads the model's answer. |
| `src/signal_engine/signal_generator.rs` | Turns the model answer into `BUY`, `SELL`, or `HOLD`. |
| `src/signal_engine/prediction.rs` | Keeps recent signals in memory while the app is running. |
| `src/redis/redis_client.rs` | Saves and reads the latest signal from Redis. |
| `src/api/signal_handlers.rs` | Defines the web URLs such as `/health`, `/predict`, and `/signal/AAPL`. |
| `create_test_model.py` | Creates a fake test model so the app can be tested without the real trained model. |
| `export_scaler.py` | Converts Phase 4 training files into files this Rust app can use. |

## What You Need Installed

- Rust, to run the backend.
- Python, only if you need to create or export model files.
- Redis, because the app stores the latest signal there.
- Docker Desktop is the easiest way to run Redis on Windows.

## First-Time Setup

### 1. Start Redis

Open PowerShell and run:

```powershell
docker run -d --name redis-hft -p 6379:6379 redis:7-alpine
```

If Docker says the name already exists, Redis may already be created. You can start it with:

```powershell
docker start redis-hft
```

### 2. Create a Test Model

Use this when you do not yet have the real trained model from Phase 4.

```powershell
pip install scikit-learn skl2onnx onnx numpy
python create_test_model.py
```

This creates:

- `models/best_model.onnx`
- `models/scaler.json`

These files let the app start and let tests run.

### 3. Run the App

```powershell
cargo run
```

For a faster production-style run:

```powershell
cargo run --release
```

When it starts successfully, the app listens at:

```text
http://localhost:8080
```

## Using the App

### Check If It Is Running

Open this in a browser:

```text
http://localhost:8080/health
```

You should see a response showing whether the app and Redis are healthy.

### Ask for a Prediction

Send market feature data to:

```text
POST http://localhost:8080/predict
```

Example request:

```json
{
  "timestamp": 1718000000000,
  "symbol": "AAPL",
  "spread": 0.01,
  "mid_price": 150.0,
  "order_book_imbalance": 0.2,
  "rolling_volatility": 0.05,
  "momentum": 0.03,
  "liquidity_ratio": 1.5,
  "volume_imbalance": 0.1,
  "trade_intensity": 120.0,
  "bid_volume": 1000.0,
  "ask_volume": 900.0,
  "total_liquidity": 1900.0
}
```

Example response:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": 1718000000000,
  "symbol": "AAPL",
  "signal": "BUY",
  "confidence": 0.87,
  "prob_sell": 0.05,
  "prob_hold": 0.08,
  "prob_buy": 0.87,
  "model_version": "v1718000000",
  "inference_ms": 2.4,
  "is_actionable": true
}
```

### Read the Latest Signal

```text
GET http://localhost:8080/signal/AAPL
```

This returns the latest stored signal for `AAPL`.

### Read Signal History

```text
GET http://localhost:8080/signal/history/AAPL?limit=20
```

This returns recent signals for `AAPL`, newest first.

## How Signals Are Decided

The model returns probabilities for `SELL`, `HOLD`, and `BUY`.

The app only allows a real `BUY` or `SELL` when confidence is above `0.70`.

| Model Result | Confidence | Final Signal |
|--------------|------------|--------------|
| BUY | More than 0.70 | BUY |
| SELL | More than 0.70 | SELL |
| BUY or SELL | 0.70 or lower | HOLD |
| HOLD | Any confidence | HOLD |

This makes the app conservative. When the model is unsure, it chooses `HOLD`.

## Using a Real Trained Model

If Phase 4 already produced trained files, use:

```powershell
pip install scikit-learn skl2onnx onnx numpy
python export_scaler.py `
  --scaler-pkl ../phase4/ml/models/scaler.pkl `
  --scaler-json ./models/scaler.json `
  --model-pkl ../phase4/ml/models/best_model.pkl `
  --onnx-path ./models/best_model.onnx
```

After this, run:

```powershell
cargo run --release
```

## Running Tests

To check that the code is working:

```powershell
cargo test
```

Current expected result:

```text
70 passed, 0 failed, 1 ignored
```

The ignored test is a Redis integration test that is only meant to run when Redis is available and you explicitly ask for it.

## Common Problems

### Redis is not running

Start it:

```powershell
docker start redis-hft
```

Or recreate it:

```powershell
docker run -d --name redis-hft -p 6379:6379 redis:7-alpine
```

### Model file is missing

Create a test model:

```powershell
python create_test_model.py
```

Then confirm these files exist:

```text
models/best_model.onnx
models/scaler.json
```

### Port 8080 is already in use

Run the app on another port:

```powershell
$env:BIND_ADDR = "0.0.0.0:8081"
cargo run
```

Then open:

```text
http://localhost:8081/health
```

## Simple Flow

```text
Market numbers go in
        |
        v
AI model checks them
        |
        v
App chooses BUY, SELL, or HOLD
        |
        v
Signal is saved in Redis and returned by the API
```

## Important Note

This project is for software development and testing. It is not financial advice, and it should not be connected to real trading without proper risk controls, monitoring, and review.
