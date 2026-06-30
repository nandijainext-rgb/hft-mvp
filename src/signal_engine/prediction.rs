// src/signal_engine/prediction.rs
//
// PredictionStore: thread-safe ring-buffer that holds the last N predictions
// per symbol and provides retrieval for the REST API.

use std::collections::HashMap;
use std::collections::VecDeque;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::signal_generator::TradingSignal;

// ─────────────────────────────────────────────────────────────────────────────
// PredictionRecord — a stored prediction with extra metadata
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRecord {
    #[serde(flatten)]
    pub signal: TradingSignal,
    pub stored_at: i64,   // Unix ms when the record was stored
}

// ─────────────────────────────────────────────────────────────────────────────
// PredictionStore
// ─────────────────────────────────────────────────────────────────────────────

pub struct PredictionStore {
    /// Map from symbol → deque of recent predictions (oldest first)
    store: RwLock<HashMap<String, VecDeque<PredictionRecord>>>,
    capacity: usize,
}

impl PredictionStore {
    /// Create a new store with `capacity` predictions per symbol.
    pub fn new(capacity: usize) -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
            capacity,
        }
    }

    /// Push a new `TradingSignal` into the store for its symbol.
    pub fn push(&self, signal: TradingSignal) {
        let record = PredictionRecord {
            stored_at: chrono::Utc::now().timestamp_millis(),
            signal,
        };
        let symbol = record.signal.symbol.clone();

        let mut guard = self.store.write();
        let deque = guard
            .entry(symbol)
            .or_insert_with(|| VecDeque::with_capacity(self.capacity));

        if deque.len() >= self.capacity {
            deque.pop_front();
        }
        deque.push_back(record);
    }

    /// Retrieve the latest prediction for a symbol.
    pub fn latest(&self, symbol: &str) -> Option<PredictionRecord> {
        self.store
            .read()
            .get(symbol)
            .and_then(|d| d.back().cloned())
    }

    /// Retrieve the last `limit` predictions for a symbol (newest first).
    pub fn history(&self, symbol: &str, limit: usize) -> Vec<PredictionRecord> {
        self.store
            .read()
            .get(symbol)
            .map(|d| {
                d.iter()
                    .rev()
                    .take(limit)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Total predictions across all symbols.
    pub fn total_count(&self) -> usize {
        self.store.read().values().map(|d| d.len()).sum()
    }

    /// Number of known symbols.
    pub fn symbol_count(&self) -> usize {
        self.store.read().len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal_engine::signal_generator::SignalClass;

    fn make_signal(symbol: &str, signal: SignalClass, confidence: f32) -> TradingSignal {
        TradingSignal {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
            symbol: symbol.to_string(),
            signal,
            confidence,
            prob_sell: 0.1,
            prob_hold: 0.1,
            prob_buy: 0.8,
            model_version: "v1".to_string(),
            inference_ms: 1.5,
            is_actionable: confidence > 0.70,
        }
    }

    #[test]
    fn test_push_and_latest() {
        let store = PredictionStore::new(10);
        store.push(make_signal("AAPL", SignalClass::Buy, 0.85));
        let rec = store.latest("AAPL").unwrap();
        assert_eq!(rec.signal.symbol, "AAPL");
    }

    #[test]
    fn test_latest_returns_newest() {
        let store = PredictionStore::new(10);
        store.push(make_signal("AAPL", SignalClass::Buy, 0.85));
        store.push(make_signal("AAPL", SignalClass::Sell, 0.75));
        let rec = store.latest("AAPL").unwrap();
        assert!(matches!(rec.signal.signal, SignalClass::Sell));
    }

    #[test]
    fn test_capacity_eviction() {
        let store = PredictionStore::new(3);
        for i in 0..5 {
            let class = if i % 2 == 0 { SignalClass::Buy } else { SignalClass::Sell };
            store.push(make_signal("TSLA", class, 0.80));
        }
        let history = store.history("TSLA", 100);
        assert_eq!(history.len(), 3, "Should cap at capacity");
    }

    #[test]
    fn test_history_order_newest_first() {
        let store = PredictionStore::new(100);
        for i in 0..5u32 {
            let _ = i;
            store.push(make_signal("MSFT", SignalClass::Hold, 0.50));
        }
        // push a distinctive last item
        store.push(make_signal("MSFT", SignalClass::Buy, 0.90));
        let history = store.history("MSFT", 6);
        // Newest first
        assert!(matches!(history[0].signal.signal, SignalClass::Buy));
    }

    #[test]
    fn test_unknown_symbol_returns_none() {
        let store = PredictionStore::new(10);
        assert!(store.latest("UNKNOWN").is_none());
        assert!(store.history("UNKNOWN", 10).is_empty());
    }

    #[test]
    fn test_multiple_symbols_independent() {
        let store = PredictionStore::new(10);
        store.push(make_signal("AAPL", SignalClass::Buy,  0.90));
        store.push(make_signal("TSLA", SignalClass::Sell, 0.80));

        assert!(matches!(
            store.latest("AAPL").unwrap().signal.signal,
            SignalClass::Buy
        ));
        assert!(matches!(
            store.latest("TSLA").unwrap().signal.signal,
            SignalClass::Sell
        ));
    }

    #[test]
    fn test_total_count() {
        let store = PredictionStore::new(100);
        store.push(make_signal("A", SignalClass::Buy, 0.80));
        store.push(make_signal("B", SignalClass::Sell, 0.75));
        store.push(make_signal("A", SignalClass::Hold, 0.50));
        assert_eq!(store.total_count(), 3);
        assert_eq!(store.symbol_count(), 2);
    }
}