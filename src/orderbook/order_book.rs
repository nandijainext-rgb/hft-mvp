use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

use crate::market_data::tick::Tick;
use super::level::PriceLevel;
use super::metrics::BookMetrics;

/// How many levels to retain per side.
const MAX_LEVELS: usize = 10;

/// A two-sided limit order book maintained from live ticks.
///
/// Bids  → `BTreeMap<Decimal, Decimal>` keyed by price ascending;
///          iteration is reversed to get best-bid-first order.
/// Asks  → `BTreeMap<Decimal, Decimal>` keyed by price ascending;
///          iteration is forward (lowest ask first).
///
/// BTreeMap gives O(log n) insert/lookup and O(n) iteration.
/// With MAX_LEVELS = 10, n is tiny — all ops are effectively O(1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    /// bid_price → quantity  (ascending key, read in reverse)
    bids: BTreeMap<Decimal, Decimal>,
    /// ask_price → quantity  (ascending key, read forward)
    asks: BTreeMap<Decimal, Decimal>,
    pub last_updated: DateTime<Utc>,
}

impl OrderBook {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            last_updated: Utc::now(),
        }
    }

    // ── Core update ───────────────────────────────────────────────────────────

    /// Consume a tick and update both sides of the book.
    ///
    /// Steps per side:
    ///   1. Upsert the level (replace quantity if price exists, insert if not).
    ///   2. If the map now has > MAX_LEVELS entries, drop the *worst* level.
    ///      Worst bid  = lowest price  (furthest from best).
    ///      Worst ask  = highest price (furthest from best).
    pub fn update(&mut self, tick: &Tick) {
        // ── Bid side ──────────────────────────────────────────────────────────
        self.bids.insert(tick.bid_price, tick.bid_size);
        if self.bids.len() > MAX_LEVELS {
            // remove lowest (worst) bid
            if let Some(worst) = self.bids.keys().next().copied() {
                self.bids.remove(&worst);
            }
        }

        // ── Ask side ──────────────────────────────────────────────────────────
        self.asks.insert(tick.ask_price, tick.ask_size);
        if self.asks.len() > MAX_LEVELS {
            // remove highest (worst) ask
            if let Some(worst) = self.asks.keys().next_back().copied() {
                self.asks.remove(&worst);
            }
        }

        self.last_updated = tick.timestamp;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Best (highest) bid price, or None if book is empty.
    #[inline]
    pub fn best_bid(&self) -> Option<Decimal> {
        self.bids.keys().next_back().copied()
    }

    /// Best (lowest) ask price, or None if book is empty.
    #[inline]
    pub fn best_ask(&self) -> Option<Decimal> {
        self.asks.keys().next().copied()
    }

    /// Bid-ask spread. Returns None if either side is empty.
    #[inline]
    pub fn spread(&self) -> Option<Decimal> {
        Some(self.best_ask()? - self.best_bid()?)
    }

    /// Mid price = (best_bid + best_ask) / 2. None if either side empty.
    #[inline]
    pub fn mid_price(&self) -> Option<Decimal> {
        Some((self.best_bid()? + self.best_ask()?) / Decimal::from(2))
    }

    /// Order-book imbalance in [-1, +1].
    ///
    /// OBI = (ΣBidVol - ΣAskVol) / (ΣBidVol + ΣAskVol)
    /// Returns None if total volume is zero (empty book).
    pub fn order_book_imbalance(&self) -> Option<Decimal> {
        let bid_vol = self.bid_volume();
        let ask_vol = self.ask_volume();
        let total = bid_vol + ask_vol;
        if total.is_zero() {
            return None;
        }
        Some((bid_vol - ask_vol) / total)
    }

    /// Sum of all bid quantities across retained levels.
    pub fn bid_volume(&self) -> Decimal {
        self.bids.values().copied().sum()
    }

    /// Sum of all ask quantities across retained levels.
    pub fn ask_volume(&self) -> Decimal {
        self.asks.values().copied().sum()
    }

    /// Total liquidity = bid_volume + ask_volume.
    #[inline]
    pub fn total_liquidity(&self) -> Decimal {
        self.bid_volume() + self.ask_volume()
    }

    // ── Level snapshots ───────────────────────────────────────────────────────

    /// Top N bid levels, best first (highest price first).
    pub fn top_n_bids(&self, n: usize) -> Vec<PriceLevel> {
        self.bids
            .iter()
            .rev()
            .take(n)
            .map(|(&price, &quantity)| PriceLevel::new(price, quantity))
            .collect()
    }

    /// Top N ask levels, best first (lowest price first).
    pub fn top_n_asks(&self, n: usize) -> Vec<PriceLevel> {
        self.asks
            .iter()
            .take(n)
            .map(|(&price, &quantity)| PriceLevel::new(price, quantity))
            .collect()
    }

    // ── Metric bundle ─────────────────────────────────────────────────────────

    /// Compute all microstructure metrics in one pass (no repeated iteration).
    pub fn compute_metrics(&self) -> Option<BookMetrics> {
        let best_bid = self.best_bid()?;
        let best_ask = self.best_ask()?;
        let bid_volume = self.bid_volume();
        let ask_volume = self.ask_volume();
        let total = bid_volume + ask_volume;

        let obi = if total.is_zero() {
            Decimal::ZERO
        } else {
            (bid_volume - ask_volume) / total
        };

        Some(BookMetrics {
            bid_volume,
            ask_volume,
            total_liquidity: total,
            obi,
            spread: best_ask - best_bid,
            mid_price: (best_bid + best_ask) / Decimal::from(2),
            best_bid,
            best_ask,
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_data::tick::Tick;
    use chrono::Utc;
    use rust_decimal_macros::dec;
    use uuid::Uuid;

    fn make_tick(bid: Decimal, bid_sz: Decimal, ask: Decimal, ask_sz: Decimal) -> Tick {
        Tick {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            symbol: "BTCUSDT".into(),
            bid_price: bid,
            bid_size: bid_sz,
            ask_price: ask,
            ask_size: ask_sz,
            last_trade_price: (bid + ask) / dec!(2),
            last_trade_size: dec!(0.1),
        }
    }

    fn book_with_one_tick() -> OrderBook {
        let mut ob = OrderBook::new("BTCUSDT");
        ob.update(&make_tick(dec!(30000), dec!(1.0), dec!(30001), dec!(0.5)));
        ob
    }

    // ── basic accessors ───────────────────────────────────────────────────────

    #[test]
    fn test_best_bid_after_update() {
        let ob = book_with_one_tick();
        assert_eq!(ob.best_bid(), Some(dec!(30000)));
    }

    #[test]
    fn test_best_ask_after_update() {
        let ob = book_with_one_tick();
        assert_eq!(ob.best_ask(), Some(dec!(30001)));
    }

    #[test]
    fn test_spread() {
        let ob = book_with_one_tick();
        assert_eq!(ob.spread(), Some(dec!(1)));
    }

    #[test]
    fn test_mid_price() {
        let ob = book_with_one_tick();
        assert_eq!(ob.mid_price(), Some(dec!(30000.5)));
    }

    // ── OBI ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_obi_equal_sides() {
        let mut ob = OrderBook::new("BTCUSDT");
        ob.update(&make_tick(dec!(30000), dec!(1.0), dec!(30001), dec!(1.0)));
        assert_eq!(ob.order_book_imbalance(), Some(dec!(0)));
    }

    #[test]
    fn test_obi_all_bid() {
        // bid vol >> ask vol → OBI approaches +1
        let mut ob = OrderBook::new("BTCUSDT");
        ob.update(&make_tick(dec!(30000), dec!(10.0), dec!(30001), dec!(0.0001)));
        let obi = ob.order_book_imbalance().unwrap();
        assert!(obi > dec!(0.99), "Expected OBI > 0.99, got {obi}");
    }

    #[test]
    fn test_obi_all_ask() {
        let mut ob = OrderBook::new("BTCUSDT");
        ob.update(&make_tick(dec!(30000), dec!(0.0001), dec!(30001), dec!(10.0)));
        let obi = ob.order_book_imbalance().unwrap();
        assert!(obi < dec!(-0.99), "Expected OBI < -0.99, got {obi}");
    }

    // ── level capping ─────────────────────────────────────────────────────────

    #[test]
    fn test_bids_capped_at_10() {
        let mut ob = OrderBook::new("BTCUSDT");
        for i in 0u32..15 {
            ob.update(&make_tick(
                Decimal::from(30000 + i),
                dec!(1.0),
                Decimal::from(30050 + i),
                dec!(1.0),
            ));
        }
        assert_eq!(ob.top_n_bids(20).len(), 10);
    }

    #[test]
    fn test_asks_capped_at_10() {
        let mut ob = OrderBook::new("BTCUSDT");
        for i in 0u32..15 {
            ob.update(&make_tick(
                Decimal::from(29900 + i),
                dec!(1.0),
                Decimal::from(30001 + i),
                dec!(1.0),
            ));
        }
        assert_eq!(ob.top_n_asks(20).len(), 10);
    }

    // ── level ordering ────────────────────────────────────────────────────────

    #[test]
    fn test_bids_descending() {
        let mut ob = OrderBook::new("BTCUSDT");
        for &p in &[dec!(29998), dec!(29999), dec!(30000)] {
            ob.update(&make_tick(p, dec!(1.0), p + dec!(1), dec!(1.0)));
        }
        let bids = ob.top_n_bids(3);
        assert!(bids[0].price > bids[1].price);
        assert!(bids[1].price > bids[2].price);
    }

    #[test]
    fn test_asks_ascending() {
        let mut ob = OrderBook::new("BTCUSDT");
        for &p in &[dec!(30001), dec!(30002), dec!(30003)] {
            ob.update(&make_tick(p - dec!(1), dec!(1.0), p, dec!(1.0)));
        }
        let asks = ob.top_n_asks(3);
        assert!(asks[0].price < asks[1].price);
        assert!(asks[1].price < asks[2].price);
    }

    // ── upsert (same price → replace qty) ────────────────────────────────────

    #[test]
    fn test_same_price_replaces_quantity() {
        let mut ob = OrderBook::new("BTCUSDT");
        ob.update(&make_tick(dec!(30000), dec!(1.0), dec!(30001), dec!(0.5)));
        ob.update(&make_tick(dec!(30000), dec!(5.0), dec!(30001), dec!(2.0)));
        // Only one level per price
        assert_eq!(ob.top_n_bids(5).len(), 1);
        assert_eq!(ob.top_n_bids(1)[0].quantity, dec!(5.0));
    }

    // ── empty book guards ─────────────────────────────────────────────────────

    #[test]
    fn test_empty_book_returns_none() {
        let ob = OrderBook::new("BTCUSDT");
        assert!(ob.best_bid().is_none());
        assert!(ob.best_ask().is_none());
        assert!(ob.spread().is_none());
        assert!(ob.mid_price().is_none());
        assert!(ob.order_book_imbalance().is_none());
    }

    // ── compute_metrics ───────────────────────────────────────────────────────

    #[test]
    fn test_compute_metrics_consistent() {
        let ob = book_with_one_tick();
        let m = ob.compute_metrics().unwrap();
        assert_eq!(m.best_bid, ob.best_bid().unwrap());
        assert_eq!(m.best_ask, ob.best_ask().unwrap());
        assert_eq!(m.spread, ob.spread().unwrap());
        assert_eq!(m.mid_price, ob.mid_price().unwrap());
        assert_eq!(m.bid_volume, ob.bid_volume());
        assert_eq!(m.ask_volume, ob.ask_volume());
        assert_eq!(m.total_liquidity, ob.total_liquidity());
        assert_eq!(m.obi, ob.order_book_imbalance().unwrap());
    }
}