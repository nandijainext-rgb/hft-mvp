/// Kaggle BTCUSDT Perpetual LOB Dataset Adapter
/// Dataset: https://www.kaggle.com/datasets/siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data
///
/// The raw Kaggle file has 10 price+size levels per side (40 columns).
/// This module:
///   • Deserialises all 10 levels into LobSnapshot
///   • Exposes to_tick() to convert level-1 into the Tick type
///   • Exputes OBI (Order Book Imbalance) across N levels — used by Phase 3
///
/// The MarketDataHandler in handler.rs still reads a flat CSV with
/// columns:  timestamp, bid_price, bid_size, ask_price, ask_size …
/// If your Kaggle file uses bid_price1 / bid_qty1 / … naming instead,
/// tick.rs already has serde aliases that handle it automatically.
///
/// Use this module directly when you need full depth (Phase 2 onwards).

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::Deserialize;
use uuid::Uuid;

use super::tick::{parse_timestamp, Tick};

// ─── 10-level LOB row ─────────────────────────────────────────────────────────
// Accepts the most common naming variants from Kaggle / exchange exports.
// Add more `alias` entries if your file uses different names.

#[derive(Debug, Deserialize)]
pub struct LobRow {
    #[serde(alias = "time", alias = "Time", alias = "Timestamp", alias = "ts")]
    pub timestamp: String,

    // ── Bid levels (best = 1) ─────────────────────────────────────────────────
    #[serde(alias = "bp1", alias = "BidPrice1", alias = "bid_price1")]
    pub bid_price1: f64,
    #[serde(alias = "bq1", alias = "BidQty1", alias = "bid_qty1", alias = "bid_size1")]
    pub bid_size1: f64,

    #[serde(alias = "bp2", alias = "BidPrice2", alias = "bid_price2", default)]
    pub bid_price2: Option<f64>,
    #[serde(alias = "bq2", alias = "BidQty2", alias = "bid_qty2", alias = "bid_size2", default)]
    pub bid_size2: Option<f64>,

    #[serde(alias = "bp3", alias = "BidPrice3", alias = "bid_price3", default)]
    pub bid_price3: Option<f64>,
    #[serde(alias = "bq3", alias = "BidQty3", alias = "bid_qty3", alias = "bid_size3", default)]
    pub bid_size3: Option<f64>,

    #[serde(alias = "bp4", alias = "BidPrice4", alias = "bid_price4", default)]
    pub bid_price4: Option<f64>,
    #[serde(alias = "bq4", alias = "BidQty4", alias = "bid_qty4", alias = "bid_size4", default)]
    pub bid_size4: Option<f64>,

    #[serde(alias = "bp5", alias = "BidPrice5", alias = "bid_price5", default)]
    pub bid_price5: Option<f64>,
    #[serde(alias = "bq5", alias = "BidQty5", alias = "bid_qty5", alias = "bid_size5", default)]
    pub bid_size5: Option<f64>,

    #[serde(alias = "bp6", alias = "BidPrice6", alias = "bid_price6", default)]
    pub bid_price6: Option<f64>,
    #[serde(alias = "bq6", alias = "BidQty6", alias = "bid_qty6", alias = "bid_size6", default)]
    pub bid_size6: Option<f64>,

    #[serde(alias = "bp7", alias = "BidPrice7", alias = "bid_price7", default)]
    pub bid_price7: Option<f64>,
    #[serde(alias = "bq7", alias = "BidQty7", alias = "bid_qty7", alias = "bid_size7", default)]
    pub bid_size7: Option<f64>,

    #[serde(alias = "bp8", alias = "BidPrice8", alias = "bid_price8", default)]
    pub bid_price8: Option<f64>,
    #[serde(alias = "bq8", alias = "BidQty8", alias = "bid_qty8", alias = "bid_size8", default)]
    pub bid_size8: Option<f64>,

    #[serde(alias = "bp9", alias = "BidPrice9", alias = "bid_price9", default)]
    pub bid_price9: Option<f64>,
    #[serde(alias = "bq9", alias = "BidQty9", alias = "bid_qty9", alias = "bid_size9", default)]
    pub bid_size9: Option<f64>,

    #[serde(alias = "bp10", alias = "BidPrice10", alias = "bid_price10", default)]
    pub bid_price10: Option<f64>,
    #[serde(alias = "bq10", alias = "BidQty10", alias = "bid_qty10", alias = "bid_size10", default)]
    pub bid_size10: Option<f64>,

    // ── Ask levels (best = 1) ─────────────────────────────────────────────────
    #[serde(alias = "ap1", alias = "AskPrice1", alias = "ask_price1")]
    pub ask_price1: f64,
    #[serde(alias = "aq1", alias = "AskQty1", alias = "ask_qty1", alias = "ask_size1")]
    pub ask_size1: f64,

    #[serde(alias = "ap2", alias = "AskPrice2", alias = "ask_price2", default)]
    pub ask_price2: Option<f64>,
    #[serde(alias = "aq2", alias = "AskQty2", alias = "ask_qty2", alias = "ask_size2", default)]
    pub ask_size2: Option<f64>,

    #[serde(alias = "ap3", alias = "AskPrice3", alias = "ask_price3", default)]
    pub ask_price3: Option<f64>,
    #[serde(alias = "aq3", alias = "AskQty3", alias = "ask_qty3", alias = "ask_size3", default)]
    pub ask_size3: Option<f64>,

    #[serde(alias = "ap4", alias = "AskPrice4", alias = "ask_price4", default)]
    pub ask_price4: Option<f64>,
    #[serde(alias = "aq4", alias = "AskQty4", alias = "ask_qty4", alias = "ask_size4", default)]
    pub ask_size4: Option<f64>,

    #[serde(alias = "ap5", alias = "AskPrice5", alias = "ask_price5", default)]
    pub ask_price5: Option<f64>,
    #[serde(alias = "aq5", alias = "AskQty5", alias = "ask_qty5", alias = "ask_size5", default)]
    pub ask_size5: Option<f64>,

    #[serde(alias = "ap6", alias = "AskPrice6", alias = "ask_price6", default)]
    pub ask_price6: Option<f64>,
    #[serde(alias = "aq6", alias = "AskQty6", alias = "ask_qty6", alias = "ask_size6", default)]
    pub ask_size6: Option<f64>,

    #[serde(alias = "ap7", alias = "AskPrice7", alias = "ask_price7", default)]
    pub ask_price7: Option<f64>,
    #[serde(alias = "aq7", alias = "AskQty7", alias = "ask_qty7", alias = "ask_size7", default)]
    pub ask_size7: Option<f64>,

    #[serde(alias = "ap8", alias = "AskPrice8", alias = "ask_price8", default)]
    pub ask_price8: Option<f64>,
    #[serde(alias = "aq8", alias = "AskQty8", alias = "ask_qty8", alias = "ask_size8", default)]
    pub ask_size8: Option<f64>,

    #[serde(alias = "ap9", alias = "AskPrice9", alias = "ask_price9", default)]
    pub ask_price9: Option<f64>,
    #[serde(alias = "aq9", alias = "AskQty9", alias = "ask_qty9", alias = "ask_size9", default)]
    pub ask_size9: Option<f64>,

    #[serde(alias = "ap10", alias = "AskPrice10", alias = "ask_price10", default)]
    pub ask_price10: Option<f64>,
    #[serde(alias = "aq10", alias = "AskQty10", alias = "ask_qty10", alias = "ask_size10", default)]
    pub ask_size10: Option<f64>,

    // ── Optional trade columns ────────────────────────────────────────────────
    #[serde(
        alias = "LastPrice", alias = "last_price", alias = "trade_price",
        alias = "close", alias = "price",
        default
    )]
    pub last_trade_price: Option<f64>,

    #[serde(
        alias = "LastQty", alias = "last_qty", alias = "trade_qty",
        alias = "trade_size", alias = "volume",
        default
    )]
    pub last_trade_size: Option<f64>,
}

// ─── Full order book snapshot ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LobSnapshot {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    /// Bid levels (price, size), index 0 = best bid
    pub bids: Vec<(Decimal, Decimal)>,
    /// Ask levels (price, size), index 0 = best ask
    pub asks: Vec<(Decimal, Decimal)>,
    pub last_trade_price: Decimal,
    pub last_trade_size: Decimal,
}

impl LobSnapshot {
    pub fn best_bid(&self) -> Option<(Decimal, Decimal)> {
        self.bids.first().copied()
    }

    pub fn best_ask(&self) -> Option<(Decimal, Decimal)> {
        self.asks.first().copied()
    }

    pub fn mid_price(&self) -> Option<Decimal> {
        let (b, _) = self.best_bid()?;
        let (a, _) = self.best_ask()?;
        Some((b + a) / Decimal::from(2))
    }

    pub fn spread(&self) -> Option<Decimal> {
        let (b, _) = self.best_bid()?;
        let (a, _) = self.best_ask()?;
        Some(a - b)
    }

    /// Order Book Imbalance across the top `levels` levels.
    /// OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    /// Range: [-1, +1]; positive = bid-heavy (bullish pressure)
    pub fn obi(&self, levels: usize) -> Decimal {
        let n = levels.min(self.bids.len()).min(self.asks.len());
        if n == 0 {
            return Decimal::ZERO;
        }
        let bid_vol: Decimal = self.bids[..n].iter().map(|(_, s)| *s).sum();
        let ask_vol: Decimal = self.asks[..n].iter().map(|(_, s)| *s).sum();
        let total = bid_vol + ask_vol;
        if total == Decimal::ZERO {
            return Decimal::ZERO;
        }
        (bid_vol - ask_vol) / total
    }

    /// Convert to Tick (level-1 only) for MarketDataHandler compatibility
    pub fn to_tick(&self) -> Option<Tick> {
        let (bid_price, bid_size) = self.best_bid()?;
        let (ask_price, ask_size) = self.best_ask()?;
        Some(Tick {
            id: Uuid::new_v4(),
            timestamp: self.timestamp,
            symbol: self.symbol.clone(),
            bid_price,
            bid_size,
            ask_price,
            ask_size,
            last_trade_price: self.last_trade_price,
            last_trade_size: self.last_trade_size,
        })
    }
}

// ─── LobRow → LobSnapshot conversion ─────────────────────────────────────────

impl LobRow {
    pub fn into_snapshot(self, symbol: &str) -> anyhow::Result<LobSnapshot> {
        let timestamp = parse_timestamp(&self.timestamp)?;

        let d = |v: f64| -> anyhow::Result<Decimal> {
            Decimal::try_from(v).map_err(|e| anyhow::anyhow!("Decimal conversion: {}", e))
        };

        let mut bids: Vec<(Decimal, Decimal)> = Vec::with_capacity(10);
        let mut asks: Vec<(Decimal, Decimal)> = Vec::with_capacity(10);

        // Level 1 — always required
        bids.push((d(self.bid_price1)?, d(self.bid_size1)?));
        asks.push((d(self.ask_price1)?, d(self.ask_size1)?));

        // Levels 2-10 — optional
        macro_rules! push_level {
            ($bp:expr, $bq:expr, $ap:expr, $aq:expr) => {
                if let (Some(bp), Some(bq), Some(ap), Some(aq)) =
                    ($bp, $bq, $ap, $aq)
                {
                    bids.push((d(bp)?, d(bq)?));
                    asks.push((d(ap)?, d(aq)?));
                }
            };
        }

        push_level!(self.bid_price2, self.bid_size2, self.ask_price2, self.ask_size2);
        push_level!(self.bid_price3, self.bid_size3, self.ask_price3, self.ask_size3);
        push_level!(self.bid_price4, self.bid_size4, self.ask_price4, self.ask_size4);
        push_level!(self.bid_price5, self.bid_size5, self.ask_price5, self.ask_size5);
        push_level!(self.bid_price6, self.bid_size6, self.ask_price6, self.ask_size6);
        push_level!(self.bid_price7, self.bid_size7, self.ask_price7, self.ask_size7);
        push_level!(self.bid_price8, self.bid_size8, self.ask_price8, self.ask_size8);
        push_level!(self.bid_price9, self.bid_size9, self.ask_price9, self.ask_size9);
        push_level!(self.bid_price10, self.bid_size10, self.ask_price10, self.ask_size10);

        // Validate: no crossed book
        if let (Some((b1, _)), Some((a1, _))) = (bids.first(), asks.first()) {
            anyhow::ensure!(
                b1 < a1,
                "Crossed book at {}: bid={} >= ask={}",
                timestamp, b1, a1
            );
        }

        // Derive last trade from mid if not in CSV
        let mid = (d(self.bid_price1)? + d(self.ask_price1)?) / Decimal::from(2);
        let last_trade_price = self
            .last_trade_price
            .filter(|&v| v > 0.0)
            .map(|v| d(v))
            .transpose()?
            .unwrap_or(mid);

        let last_trade_size = self
            .last_trade_size
            .filter(|&v| v > 0.0)
            .map(|v| d(v))
            .transpose()?
            .unwrap_or(Decimal::ONE);

        Ok(LobSnapshot {
            timestamp,
            symbol: symbol.to_string(),
            bids,
            asks,
            last_trade_price,
            last_trade_size,
        })
    }
}

// ─── CSV inspection helper ────────────────────────────────────────────────────

/// Call this once on startup (RUST_LOG=info) to see your exact column names.
#[allow(dead_code)]
pub fn inspect_csv_headers(path: &str) -> anyhow::Result<()> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;
    let headers = rdr.headers()?.clone();
    println!("=== {} columns in '{}' ===", headers.len(), path);
    for (i, h) in headers.iter().enumerate() {
        println!("  [{:02}] {}", i, h);
    }
    if let Some(row) = rdr.records().next() {
        let row = row?;
        println!("\n=== First data row ===");
        for (h, v) in headers.iter().zip(row.iter()) {
            println!("  {:20} = {}", h, v);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn make_snapshot(bid_vols: &[f64], ask_vols: &[f64]) -> LobSnapshot {
        let bids = bid_vols
            .iter()
            .enumerate()
            .map(|(i, &v)| (dec!(30000) - Decimal::from(i as i64), Decimal::try_from(v).unwrap()))
            .collect();
        let asks = ask_vols
            .iter()
            .enumerate()
            .map(|(i, &v)| (dec!(30001) + Decimal::from(i as i64), Decimal::try_from(v).unwrap()))
            .collect();
        LobSnapshot {
            timestamp: Utc::now(),
            symbol: "BTCUSDT".into(),
            bids,
            asks,
            last_trade_price: dec!(30000.5),
            last_trade_size: dec!(0.1),
        }
    }

    #[test]
    fn test_obi_balanced() {
        let snap = make_snapshot(&[1.0, 1.0], &[1.0, 1.0]);
        assert_eq!(snap.obi(2), dec!(0));
    }

    #[test]
    fn test_obi_bid_heavy() {
        let snap = make_snapshot(&[3.0], &[1.0]);
        // (3 - 1) / (3 + 1) = 0.5
        assert_eq!(snap.obi(1), dec!(0.5));
    }

    #[test]
    fn test_obi_ask_heavy() {
        let snap = make_snapshot(&[1.0], &[3.0]);
        // (1 - 3) / (1 + 3) = -0.5
        assert_eq!(snap.obi(1), dec!(-0.5));
    }

    #[test]
    fn test_obi_partial_levels() {
        // Only use top 1 level even though snapshot has 2
        let snap = make_snapshot(&[3.0, 10.0], &[1.0, 10.0]);
        assert_eq!(snap.obi(1), dec!(0.5)); // deep levels ignored
    }

    #[test]
    fn test_to_tick() {
        let snap = make_snapshot(&[1.5], &[0.8]);
        let tick = snap.to_tick().unwrap();
        assert_eq!(tick.bid_price, dec!(30000));
        assert_eq!(tick.ask_price, dec!(30001));
        assert!(tick.is_valid());
    }

    #[test]
    fn test_spread() {
        let snap = make_snapshot(&[1.0], &[1.0]);
        assert_eq!(snap.spread(), Some(dec!(1)));
    }
}