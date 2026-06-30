use std::collections::VecDeque;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use tracing::debug;

use super::{
    calculators::{
        liquidity_ratio, mid_price, mid_price_return, momentum,
        total_liquidity, trade_intensity_from_count, volume_imbalance,
    },
    feature_vector::FeatureVector,
    rolling_window::RollingWindow,
};

/// Snapshot of order book state consumed by the feature engine.
///
/// This mirrors the Phase 2 `BookMetrics` and `OrderBookSnapshot` output,
/// but uses plain `f64` so we avoid Decimal arithmetic in the hot path.
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub best_bid: f64,
    pub best_ask: f64,
    pub bid_volume: f64,
    pub ask_volume: f64,
    /// Pre-computed OBI from Phase 2 (pass-through)
    pub order_book_imbalance: f64,
}

// ── Window sizes ──────────────────────────────────────────────────────────────

const VOLATILITY_WINDOW: usize = 50; // snapshots for rolling volatility
const MOMENTUM_WINDOW: usize = 20;   // snapshots for momentum look-back
const MID_PRICE_HISTORY: usize = 51; // need 50 returns → 51 mid prices

// ── Trade intensity ───────────────────────────────────────────────────────────

const INTENSITY_WINDOW_SECS: u64 = 1;

/// Tracks per-second update counts using a sliding window of `Instant`s.
struct IntensityTracker {
    /// Timestamps of recent updates (oldest first).
    timestamps: VecDeque<Instant>,
    window: Duration,
}

impl IntensityTracker {
    fn new() -> Self {
        Self {
            timestamps: VecDeque::with_capacity(4096),
            window: Duration::from_secs(INTENSITY_WINDOW_SECS),
        }
    }

    /// Record a new update and return the current updates-per-second.
    fn record_and_compute(&mut self) -> f64 {
        let now = Instant::now();
        self.timestamps.push_back(now);
        // Evict entries older than the window
        let cutoff = now - self.window;
        while self.timestamps.front().map_or(false, |&t| t < cutoff) {
            self.timestamps.pop_front();
        }
        trade_intensity_from_count(self.timestamps.len())
    }
}

// ── Per-symbol feature engine ─────────────────────────────────────────────────

/// Maintains all rolling state for a single symbol and emits a `FeatureVector`
/// on each call to `update()`.
pub struct SymbolFeatureEngine {
    symbol: String,

    /// Rolling window of mid-price RETURNS (for volatility).
    returns_window: RollingWindow,

    /// Sliding buffer of raw mid prices (for momentum look-back).
    mid_prices: RollingWindow,

    /// Intensity tracker for updates-per-second.
    intensity: IntensityTracker,

    /// Previous mid price (for return calculation).
    prev_mid: Option<f64>,

    /// Total snapshots processed.
    tick_count: u64,
}

impl SymbolFeatureEngine {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            returns_window: RollingWindow::new(VOLATILITY_WINDOW),
            mid_prices: RollingWindow::new(MID_PRICE_HISTORY),
            intensity: IntensityTracker::new(),
            prev_mid: None,
            tick_count: 0,
        }
    }

    /// Process a new order book snapshot and return a fully populated
    /// `FeatureVector`. Always returns `Some` — rolling features will be
    /// `None` inside the vector until their windows are warm.
    pub fn update(&mut self, snap: &OrderBookSnapshot) -> FeatureVector {
        self.tick_count += 1;

        // ── Level-1 microstructure ────────────────────────────────────────────
        let mid = mid_price(snap.best_bid, snap.best_ask);
        let spread = (snap.best_ask - snap.best_bid).max(0.0);

        // ── Rolling returns (for volatility) ──────────────────────────────────
        if let Some(prev) = self.prev_mid {
            if let Some(ret) = mid_price_return(mid, prev) {
                self.returns_window.push(ret);
            }
        }

        // ── Mid-price history (for momentum) ─────────────────────────────────
        self.mid_prices.push(mid);

        self.prev_mid = Some(mid);

        // ── Volatility = std_dev of last 50 returns ───────────────────────────
        let rolling_volatility = self.returns_window.std_dev();

        // ── Momentum = (mid_now - mid_20_ago) / mid_20_ago ───────────────────
        // mid_prices has capacity 51, so oldest = 51 ticks ago when full.
        // We want the price from exactly 20 ticks ago: index = len - 21
        // (index 0 is oldest, index len-1 is latest = mid_now, we skip it)
        let momentum_val: Option<f64> = {
            let len = self.mid_prices.len();
            if len >= MOMENTUM_WINDOW + 1 {
                // position of mid 20 ticks ago from the perspective of the window
                let idx = len - 1 - MOMENTUM_WINDOW;
                self.mid_prices.get(idx).and_then(|old| momentum(mid, old))
            } else {
                None
            }
        };

        // ── Volume features ───────────────────────────────────────────────────
        let bid_vol = snap.bid_volume;
        let ask_vol = snap.ask_volume;
        let vol_imbalance = volume_imbalance(bid_vol, ask_vol);
        let liq_ratio = liquidity_ratio(bid_vol, ask_vol);
        let total_liq = total_liquidity(bid_vol, ask_vol);

        // ── Trade intensity ───────────────────────────────────────────────────
        let intensity = self.intensity.record_and_compute();

        // ── OBI (pass-through from Phase 2) ───────────────────────────────────
        let obi = snap.order_book_imbalance;

        debug!(
            symbol = %self.symbol,
            tick   = self.tick_count,
            mid    = mid,
            spread = spread,
            obi    = obi,
            vol    = ?rolling_volatility,
            mom    = ?momentum_val,
            "Feature vector computed"
        );

        FeatureVector {
            timestamp: snap.timestamp,
            symbol: self.symbol.clone(),
            spread,
            mid_price: mid,
            order_book_imbalance: obi,
            rolling_volatility,
            momentum: momentum_val,
            volume_imbalance: vol_imbalance,
            liquidity_ratio: liq_ratio,
            trade_intensity: intensity,
            bid_volume: bid_vol,
            ask_volume: ask_vol,
            total_liquidity: total_liq,
        }
    }

    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

// ── Multi-symbol registry ─────────────────────────────────────────────────────

/// Thread-safe wrapper for `SymbolFeatureEngine` instances.
/// One engine per symbol, created on first sight.
pub struct FeatureEngineRegistry {
    engines: std::collections::HashMap<String, SymbolFeatureEngine>,
}

impl FeatureEngineRegistry {
    pub fn new() -> Self {
        Self {
            engines: std::collections::HashMap::new(),
        }
    }

    /// Process a snapshot for its symbol, auto-creating the engine if needed.
    pub fn process(&mut self, snap: &OrderBookSnapshot) -> FeatureVector {
        self.engines
            .entry(snap.symbol.clone())
            .or_insert_with(|| SymbolFeatureEngine::new(&snap.symbol))
            .update(snap)
    }

    /// Get a read-only reference to a symbol's engine.
    pub fn get(&self, symbol: &str) -> Option<&SymbolFeatureEngine> {
        self.engines.get(symbol)
    }

    pub fn symbols(&self) -> impl Iterator<Item = &String> {
        self.engines.keys()
    }
}

impl Default for FeatureEngineRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::calculators::order_book_imbalance;
    use chrono::Utc;

    fn snap(bid: f64, ask: f64, bid_vol: f64, ask_vol: f64) -> OrderBookSnapshot {
        let obi = order_book_imbalance(bid_vol, ask_vol).unwrap_or(0.0);
        OrderBookSnapshot {
            timestamp: Utc::now(),
            symbol: "BTCUSDT".into(),
            best_bid: bid,
            best_ask: ask,
            bid_volume: bid_vol,
            ask_volume: ask_vol,
            order_book_imbalance: obi,
        }
    }

    #[test]
    fn single_tick_no_rolling_features() {
        let mut eng = SymbolFeatureEngine::new("BTCUSDT");
        let fv = eng.update(&snap(30000.0, 30001.0, 2.0, 1.0));
        assert_eq!(fv.spread, 1.0);
        assert!((fv.mid_price - 30000.5).abs() < 1e-10);
        assert!(fv.rolling_volatility.is_none());
        assert!(fv.momentum.is_none());
    }

    #[test]
    fn volatility_available_after_50_ticks() {
        let mut eng = SymbolFeatureEngine::new("BTCUSDT");
        for i in 0..51 {
            let price = 30000.0 + i as f64 * 0.5;
            eng.update(&snap(price, price + 1.0, 2.0, 1.0));
        }
        let last = eng.update(&snap(30025.0, 30026.0, 2.0, 1.0));
        assert!(last.rolling_volatility.is_some(), "volatility should be available");
        assert!(last.rolling_volatility.unwrap() >= 0.0);
    }

    #[test]
    fn momentum_available_after_20_ticks() {
        let mut eng = SymbolFeatureEngine::new("BTCUSDT");
        // Push 20 ticks at 30000, then one tick at 30010
        for _ in 0..20 {
            eng.update(&snap(30000.0, 30001.0, 2.0, 1.0));
        }
        let last = eng.update(&snap(30010.0, 30011.0, 2.0, 1.0));
        let mom = last.momentum.expect("momentum should be available after 20 ticks");
        // (30010.5 - 30000.5) / 30000.5 ≈ 0.000333
        assert!(mom > 0.0, "price rose, momentum should be positive: {mom}");
    }

    #[test]
    fn volume_imbalance_bid_heavy() {
        let mut eng = SymbolFeatureEngine::new("BTCUSDT");
        let fv = eng.update(&snap(30000.0, 30001.0, 3.0, 1.0));
        // (3 - 1) / (3 + 1) = 0.5
        assert!((fv.volume_imbalance - 0.5).abs() < 1e-10);
    }

    #[test]
    fn liquidity_ratio_correct() {
        let mut eng = SymbolFeatureEngine::new("BTCUSDT");
        let fv = eng.update(&snap(30000.0, 30001.0, 4.0, 2.0));
        assert!((fv.liquidity_ratio - 2.0).abs() < 1e-10);
    }

    #[test]
    fn total_liquidity_is_sum() {
        let mut eng = SymbolFeatureEngine::new("BTCUSDT");
        let fv = eng.update(&snap(30000.0, 30001.0, 3.0, 2.0));
        assert!((fv.total_liquidity - 5.0).abs() < 1e-10);
    }

    #[test]
    fn trade_intensity_increments() {
        let mut eng = SymbolFeatureEngine::new("BTCUSDT");
        let fv1 = eng.update(&snap(30000.0, 30001.0, 1.0, 1.0));
        let fv2 = eng.update(&snap(30000.0, 30001.0, 1.0, 1.0));
        // Both within 1 second, so count ≥ 1
        assert!(fv1.trade_intensity >= 1.0);
        assert!(fv2.trade_intensity >= 2.0);
    }

    #[test]
    fn registry_creates_per_symbol_engines() {
        let mut reg = FeatureEngineRegistry::new();
        let s1 = OrderBookSnapshot {
            symbol: "BTCUSDT".into(),
            ..snap(30000.0, 30001.0, 1.0, 1.0)
        };
        let s2 = OrderBookSnapshot {
            symbol: "ETHUSDT".into(),
            ..snap(2000.0, 2001.0, 5.0, 3.0)
        };
        reg.process(&s1);
        reg.process(&s2);
        assert!(reg.get("BTCUSDT").is_some());
        assert!(reg.get("ETHUSDT").is_some());
        assert_eq!(reg.get("BTCUSDT").unwrap().tick_count(), 1);
    }
}
