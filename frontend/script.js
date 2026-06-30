/* =========================================================
   AI HIGH FREQUENCY TRADING DASHBOARD - SCRIPT
   ========================================================= */

const API_BASE_URL = "http://localhost:8080";

const SYMBOL = "BTCUSDT";
const REFRESH_INTERVAL_MS = 1000;

const ENDPOINTS = {
  health: `${API_BASE_URL}/health`,
  predict: `${API_BASE_URL}/predict`,
  signalHistory: `${API_BASE_URL}/signal/history/${SYMBOL}?limit=20`,
};

let lastMidPrice = 30000;
let previousMidPrice = 30000;
let latestFeatures = null;

function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function formatNumber(value, decimals = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toFixed(decimals);
}

function formatTime(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleTimeString();
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Request failed (${response.status}) for ${url}: ${body}`);
  }
  return response.json();
}

function setStatusPill(elementId, isOnline) {
  const pill = document.getElementById(elementId);
  if (!pill) return;
  const dot = pill.querySelector(".dot");
  dot.classList.remove("dot-loading", "dot-online", "dot-offline");
  dot.classList.add(isOnline ? "dot-online" : "dot-offline");
}

function setConnectionBanner(connected, message) {
  const bar = document.getElementById("connectionBar");
  const text = document.getElementById("connectionText");
  const spinner = document.getElementById("globalSpinner");
  if (!bar || !text || !spinner) return;
  text.textContent = message;
  spinner.style.display = connected ? "none" : "inline-block";
  bar.style.borderBottomColor = connected ? "var(--buy)" : "var(--sell)";
}

function stampRefreshTime() {
  setText("refreshStamp", `Last refresh: ${new Date().toLocaleTimeString()}`);
}

function tickClock() {
  setText("liveClock", new Date().toLocaleTimeString());
}

function generateMarketSnapshot() {
  previousMidPrice = lastMidPrice;
  const movement = (Math.random() - 0.5) * 8;
  lastMidPrice = Math.max(1, lastMidPrice + movement);

  const spread = 0.5 + Math.random() * 1.5;
  const bidPrice = lastMidPrice - spread / 2;
  const askPrice = lastMidPrice + spread / 2;
  const bidSize = 1 + Math.random() * 9;
  const askSize = 1 + Math.random() * 9;
  const tradeSize = 0.1 + Math.random() * 2;
  const bidVolume = bidSize * 100;
  const askVolume = askSize * 100;
  const totalLiquidity = bidVolume + askVolume;

  const bids = Array.from({ length: 10 }, (_, idx) => ({
    price: bidPrice - idx * spread,
    quantity: bidSize * (1 + Math.random() * 0.6),
  }));

  const asks = Array.from({ length: 10 }, (_, idx) => ({
    price: askPrice + idx * spread,
    quantity: askSize * (1 + Math.random() * 0.6),
  }));

  const orderBookImbalance =
    totalLiquidity === 0 ? 0 : (bidVolume - askVolume) / totalLiquidity;
  const volumeImbalance =
    bidSize + askSize === 0 ? 0 : (bidSize - askSize) / (bidSize + askSize);
  const momentum =
    previousMidPrice === 0 ? 0 : (lastMidPrice - previousMidPrice) / previousMidPrice;

  latestFeatures = {
    timestamp: Date.now(),
    symbol: SYMBOL,
    spread,
    mid_price: lastMidPrice,
    order_book_imbalance: orderBookImbalance,
    rolling_volatility: Math.abs(momentum) + Math.random() * 0.002,
    momentum,
    liquidity_ratio: askVolume === 0 ? 0 : bidVolume / askVolume,
    volume_imbalance: volumeImbalance,
    trade_intensity: 20 + Math.random() * 150,
    bid_volume: bidVolume,
    ask_volume: askVolume,
    total_liquidity: totalLiquidity,
  };

  return {
    timestamp: latestFeatures.timestamp,
    symbol: SYMBOL,
    bid_price: bidPrice,
    ask_price: askPrice,
    last_price: lastMidPrice,
    bid_size: bidSize,
    ask_size: askSize,
    trade_size: tradeSize,
    bids,
    asks,
    best_bid: bidPrice,
    best_ask: askPrice,
    spread,
    mid_price: lastMidPrice,
    imbalance: orderBookImbalance,
  };
}

async function loadHealth() {
  try {
    const data = await fetchJSON(ENDPOINTS.health);
    const healthy = data.status === "ok";

    setStatusPill("backendStatus", true);
    setStatusPill("redisStatus", healthy);
    setStatusPill("modelStatus", Boolean(data.model_version));
    setText("s-modelVersion", data.model_version ?? "--");
    setConnectionBanner(true, "Connected to backend");
    return true;
  } catch (err) {
    setStatusPill("backendStatus", false);
    setStatusPill("redisStatus", false);
    setStatusPill("modelStatus", false);
    setConnectionBanner(false, "Unable to reach backend - retrying...");
    console.error("loadHealth error:", err);
    return false;
  }
}

function renderMarketData(data) {
  setText("m-timestamp", formatTime(data.timestamp));
  setText("m-symbol", data.symbol ?? "--");
  setText("m-bidPrice", formatNumber(data.bid_price));
  setText("m-askPrice", formatNumber(data.ask_price));
  setText("m-lastPrice", formatNumber(data.last_price));
  setText("m-bidSize", formatNumber(data.bid_size, 4));
  setText("m-askSize", formatNumber(data.ask_size, 4));
  setText("m-tradeSize", formatNumber(data.trade_size, 4));
  setText("marketSymbolBadge", data.symbol ?? "--");
}

function renderOrderBookSide(tbodyId, levels) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;

  if (!levels || levels.length === 0) {
    tbody.innerHTML = `<tr><td colspan="2" class="empty-row">No data</td></tr>`;
    return;
  }

  tbody.innerHTML = levels
    .slice(0, 10)
    .map((level) => {
      const price = formatNumber(level.price);
      const qty = formatNumber(level.quantity, 4);
      return `<tr><td>${price}</td><td>${qty}</td></tr>`;
    })
    .join("");
}

function renderOrderBook(data) {
  renderOrderBookSide("bidsTableBody", data.bids);
  renderOrderBookSide("asksTableBody", data.asks);
  setText("ob-bestBid", formatNumber(data.best_bid));
  setText("ob-bestAsk", formatNumber(data.best_ask));
  setText("ob-spread", formatNumber(data.spread));
  setText("ob-midPrice", formatNumber(data.mid_price));
  setText("ob-imbalance", formatNumber(data.imbalance, 4));
}

function renderFeatures(data) {
  setText("f-spread", formatNumber(data.spread));
  setText("f-momentum", formatNumber(data.momentum, 4));
  setText("f-volatility", formatNumber(data.rolling_volatility, 4));
  setText("f-liquidityRatio", formatNumber(data.liquidity_ratio, 4));
  setText("f-volumeImbalance", formatNumber(data.volume_imbalance, 4));
  setText("f-tradeIntensity", formatNumber(data.trade_intensity, 4));
  setText("f-bidVolume", formatNumber(data.bid_volume, 4));
  setText("f-askVolume", formatNumber(data.ask_volume, 4));
  setText("f-totalLiquidity", formatNumber(data.total_liquidity, 4));
  setText("f-obImbalance", formatNumber(data.order_book_imbalance, 4));
}

function applySignalStyling(signal) {
  const card = document.getElementById("signalCard");
  if (!card) return;
  card.classList.remove("signal-buy", "signal-sell", "signal-hold");

  const normalized = String(signal || "").toUpperCase();
  if (normalized === "BUY") card.classList.add("signal-buy");
  else if (normalized === "SELL") card.classList.add("signal-sell");
  else card.classList.add("signal-hold");
}

function updateProbaBar(key, value) {
  const pct = value === null || value === undefined || Number.isNaN(Number(value))
    ? 0
    : Math.round(Number(value) * 100);
  const fill = document.getElementById(`proba-${key}`);
  const label = document.getElementById(`proba-${key}-pct`);
  if (fill) fill.style.width = `${pct}%`;
  if (label) label.textContent = `${pct}%`;
}

function renderSignal(data) {
  const signal = data.signal ?? "HOLD";
  setText("signalText", String(signal).toUpperCase());
  applySignalStyling(signal);

  const confidencePct = formatNumber((data.confidence ?? 0) * 100, 1);
  setText("signalConfidenceText", `Confidence: ${confidencePct}%`);
  setText("s-confidence", `${confidencePct}%`);
  setText("s-modelVersion", data.model_version ?? "--");
  setText("s-inferenceTime", `${formatNumber(data.inference_ms, 2)} ms`);

  updateProbaBar("buy", data.prob_buy);
  updateProbaBar("sell", data.prob_sell);
  updateProbaBar("hold", data.prob_hold);
}

async function runPrediction() {
  if (!latestFeatures) return;

  const signal = await fetchJSON(ENDPOINTS.predict, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(latestFeatures),
  });

  renderSignal(signal);
}

function predictionClass(prediction) {
  const normalized = String(prediction || "").toUpperCase();
  if (normalized === "BUY") return "pred-buy";
  if (normalized === "SELL") return "pred-sell";
  return "pred-hold";
}

async function loadSignalHistory() {
  try {
    const data = await fetchJSON(ENDPOINTS.signalHistory);
    const tbody = document.getElementById("historyTableBody");
    if (!tbody) return;

    const records = Array.isArray(data.records) ? data.records : [];

    if (records.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" class="empty-row">No predictions yet</td></tr>`;
      return;
    }

    tbody.innerHTML = records
      .slice(0, 20)
      .map((record) => {
        const time = formatTime(record.timestamp ?? record.stored_at);
        const symbol = record.symbol ?? "--";
        const prediction = record.signal ?? "--";
        const confidence = formatNumber((record.confidence ?? 0) * 100, 1);
        return `<tr><td>${time}</td><td>${symbol}</td><td class="${predictionClass(prediction)}">${prediction}</td><td>${confidence}%</td></tr>`;
      })
      .join("");
  } catch (err) {
    const tbody = document.getElementById("historyTableBody");
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="4" class="empty-row">No predictions yet</td></tr>`;
    }
    console.error("loadSignalHistory error:", err);
  }
}

async function refreshAll() {
  const connected = await loadHealth();
  const market = generateMarketSnapshot();
  renderMarketData(market);
  renderOrderBook(market);
  renderFeatures(latestFeatures);

  if (connected) {
    try {
      await runPrediction();
      await loadSignalHistory();
    } catch (err) {
      setConnectionBanner(false, "Backend reached, but prediction failed");
      console.error("prediction error:", err);
    }
  }

  stampRefreshTime();
}

function startDashboard() {
  tickClock();
  setInterval(tickClock, 1000);
  refreshAll();
  setInterval(refreshAll, REFRESH_INTERVAL_MS);
}

document.addEventListener("DOMContentLoaded", startDashboard);
