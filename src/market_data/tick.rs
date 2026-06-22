use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Core market tick — symbol-agnostic, works for equities and crypto.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tick {
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub bid_price: Decimal,
    pub bid_size: Decimal,
    pub ask_price: Decimal,
    pub ask_size: Decimal,
    pub last_trade_price: Decimal,
    pub last_trade_size: Decimal,
}

impl Tick {
    pub fn spread(&self) -> Decimal {
        self.ask_price - self.bid_price
    }

    pub fn mid_price(&self) -> Decimal {
        (self.bid_price + self.ask_price) / Decimal::from(2)
    }

    pub fn is_valid(&self) -> bool {
        self.bid_price > Decimal::ZERO
            && self.ask_price > Decimal::ZERO
            && self.bid_price < self.ask_price
            && self.bid_size > Decimal::ZERO
            && self.ask_size > Decimal::ZERO
            && self.last_trade_price > Decimal::ZERO
            && self.last_trade_size > Decimal::ZERO
    }
}

// ─── CSV Deserialization ──────────────────────────────────────────────────────
//
// This struct uses serde `alias` so it accepts MULTIPLE column naming schemes:
//
//   Scheme A — our generated CSV / standard naming:
//     timestamp, symbol, bid_price, bid_size, ask_price, ask_size,
//     last_trade_price, last_trade_size
//
//   Scheme B — Kaggle BTCUSDT LOB dataset (level-1 columns):
//     timestamp, bid_price1, bid_qty1, ask_price1, ask_qty1  (no symbol column)
//
//   Scheme C — Bybit / Binance export naming:
//     time/Time, bp1/BidPrice1, bq1/BidQty1, ap1/AskPrice1, aq1/AskQty1
//
// HOW TO ADD YOUR OWN COLUMN NAMES:
//   Just add another `alias = "your_column_name"` to the relevant field below.

#[derive(Debug, Deserialize)]
pub struct TickCsvRow {
    // ── Timestamp ─────────────────────────────────────────────────────────────
    // Accepted: "timestamp", "time", "Time", "Timestamp", "ts", "date_time"
    // Accepted values: RFC3339, "YYYY-MM-DD HH:MM:SS.mmm", Unix ms (int), Unix ns (int)
    #[serde(alias = "time", alias = "Time", alias = "Timestamp", alias = "ts", alias = "date_time")]
    pub timestamp: String,

    // ── Symbol ────────────────────────────────────────────────────────────────
    // OPTIONAL — Kaggle LOB files often omit this; defaults to "BTCUSDT"
    #[serde(
        alias = "Symbol", alias = "sym", alias = "pair", alias = "instrument",
        default = "default_symbol"
    )]
    pub symbol: String,

    // ── Bid Price (level 1) ───────────────────────────────────────────────────
    #[serde(alias = "bid_price1", alias = "BidPrice1", alias = "bp1", alias = "best_bid_price")]
    pub bid_price: f64,

    // ── Bid Size / Qty (level 1) ──────────────────────────────────────────────
    #[serde(
        alias = "bid_size1", alias = "bid_qty1", alias = "BidQty1",
        alias = "bq1", alias = "bid_vol1", alias = "best_bid_qty"
    )]
    pub bid_size: f64,

    // ── Ask Price (level 1) ───────────────────────────────────────────────────
    #[serde(alias = "ask_price1", alias = "AskPrice1", alias = "ap1", alias = "best_ask_price")]
    pub ask_price: f64,

    // ── Ask Size / Qty (level 1) ──────────────────────────────────────────────
    #[serde(
        alias = "ask_size1", alias = "ask_qty1", alias = "AskQty1",
        alias = "aq1", alias = "ask_vol1", alias = "best_ask_qty"
    )]
    pub ask_size: f64,

    // ── Last Trade Price ──────────────────────────────────────────────────────
    // OPTIONAL — many LOB snapshots don't include this; defaults to mid price
    #[serde(
        alias = "LastPrice", alias = "last_price", alias = "trade_price",
        alias = "close", alias = "price",
        default
    )]
    pub last_trade_price: Option<f64>,

    // ── Last Trade Size ───────────────────────────────────────────────────────
    // OPTIONAL — defaults to 0.0 (will be set to 1.0 in try_from to pass validation)
    #[serde(
        alias = "LastQty", alias = "last_qty", alias = "trade_qty",
        alias = "trade_size", alias = "volume",
        default
    )]
    pub last_trade_size: Option<f64>,
}

fn default_symbol() -> String {
    "BTCUSDT".to_string()
}

impl TryFrom<TickCsvRow> for Tick {
    type Error = anyhow::Error;

    fn try_from(row: TickCsvRow) -> anyhow::Result<Self> {
        let timestamp = parse_timestamp(&row.timestamp)?;

        let bid_price = Decimal::try_from(row.bid_price)?;
        let ask_price = Decimal::try_from(row.ask_price)?;

        // Derive last_trade_price from mid if not present in CSV
        let mid = (bid_price + ask_price) / Decimal::from(2);
        let last_trade_price = row
            .last_trade_price
            .filter(|&v| v > 0.0)
            .map(Decimal::try_from)
            .transpose()?
            .unwrap_or(mid);

        // Default trade size to 1 if not present (prevents validation failure)
        let last_trade_size = row
            .last_trade_size
            .filter(|&v| v > 0.0)
            .map(Decimal::try_from)
            .transpose()?
            .unwrap_or(Decimal::ONE);

        let tick = Tick {
            id: Uuid::new_v4(),
            timestamp,
            symbol: row.symbol,
            bid_price,
            bid_size: Decimal::try_from(row.bid_size)?,
            ask_price,
            ask_size: Decimal::try_from(row.ask_size)?,
            last_trade_price,
            last_trade_size,
        };

        anyhow::ensure!(tick.is_valid(), "Tick failed validation: {:?}", tick);
        Ok(tick)
    }
}

// ─── Timestamp parser — handles all common formats ───────────────────────────

pub fn parse_timestamp(raw: &str) -> anyhow::Result<DateTime<Utc>> {
    use chrono::TimeZone;
    let raw = raw.trim();

    // RFC 3339 / ISO 8601  e.g. "2023-07-01T09:30:00.000Z"
    if let Ok(dt) = DateTime::parse_from_rfc3339(raw) {
        return Ok(dt.with_timezone(&Utc));
    }

    // Space-separated datetime  e.g. "2023-07-01 09:30:00.123"
    if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S%.f") {
        return Ok(ndt.and_utc());
    }

    // Date-only  e.g. "2023-07-01"
    if let Ok(nd) = chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
        return Ok(nd.and_hms_opt(0, 0, 0).unwrap().and_utc());
    }

    // Numeric timestamp (Unix epoch)
    if let Ok(n) = raw.parse::<i64>() {
        return if n > 1_000_000_000_000_000 {
            // nanoseconds  (> 10^15)
            Utc.timestamp_opt(n / 1_000_000_000, (n % 1_000_000_000) as u32)
                .single()
                .ok_or_else(|| anyhow::anyhow!("Invalid Unix ns: {}", n))
        } else if n > 1_000_000_000_000 {
            // milliseconds  (> 10^12)
            Utc.timestamp_millis_opt(n)
                .single()
                .ok_or_else(|| anyhow::anyhow!("Invalid Unix ms: {}", n))
        } else {
            // seconds
            Utc.timestamp_opt(n, 0)
                .single()
                .ok_or_else(|| anyhow::anyhow!("Invalid Unix s: {}", n))
        };
    }

    // Floating-point Unix timestamp  e.g. "1688201400.123"
    if let Ok(f) = raw.parse::<f64>() {
        let secs = f as i64;
        let nanos = ((f - secs as f64) * 1e9) as u32;
        return Utc
            .timestamp_opt(secs, nanos)
            .single()
            .ok_or_else(|| anyhow::anyhow!("Invalid float timestamp: {}", f));
    }

    anyhow::bail!(
        "Cannot parse timestamp '{}'. \
         Supported formats: RFC3339, 'YYYY-MM-DD HH:MM:SS[.fff]', \
         Unix seconds/milliseconds/nanoseconds (integer or float).",
        raw
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn sample_tick() -> Tick {
        Tick {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            symbol: "BTCUSDT".into(),
            bid_price: dec!(30000.00),
            bid_size: dec!(1.5),
            ask_price: dec!(30001.00),
            ask_size: dec!(0.8),
            last_trade_price: dec!(30000.50),
            last_trade_size: dec!(0.2),
        }
    }

    #[test]
    fn test_spread() {
        let t = sample_tick();
        assert_eq!(t.spread(), dec!(1.00));
    }

    #[test]
    fn test_mid_price() {
        let t = sample_tick();
        assert_eq!(t.mid_price(), dec!(30000.5));
    }

    #[test]
    fn test_valid_tick() {
        assert!(sample_tick().is_valid());
    }

    #[test]
    fn test_invalid_crossed_book() {
        let mut t = sample_tick();
        t.bid_price = dec!(30002.00); // bid > ask
        assert!(!t.is_valid());
    }

    // ── Timestamp parsing ──────────────────────────────────────────────────

    #[test]
    fn test_parse_rfc3339() {
        let dt = parse_timestamp("2023-07-01T09:30:00Z").unwrap();
        assert_eq!(dt.year(), 2023);
    }

    #[test]
    fn test_parse_space_datetime() {
        let dt = parse_timestamp("2024-01-15 09:30:00.123").unwrap();
        assert_eq!(dt.year(), 2024);
    }

    #[test]
    fn test_parse_unix_ms() {
        let dt = parse_timestamp("1673778600000").unwrap();
        assert_eq!(dt.year(), 2023);
    }

    #[test]
    fn test_parse_unix_ns() {
        let dt = parse_timestamp("1673778600000000000").unwrap();
        assert_eq!(dt.year(), 2023);
    }

    // ── CSV row conversion ─────────────────────────────────────────────────

    #[test]
    fn test_standard_columns_convert() {
        let row = TickCsvRow {
            timestamp: "2024-01-15 09:30:00.000".into(),
            symbol: "BTCUSDT".into(),
            bid_price: 30000.0,
            bid_size: 1.5,
            ask_price: 30001.0,
            ask_size: 0.8,
            last_trade_price: Some(30000.5),
            last_trade_size: Some(0.2),
        };
        let tick = Tick::try_from(row).unwrap();
        assert_eq!(tick.symbol, "BTCUSDT");
        assert!(tick.is_valid());
    }

    #[test]
    fn test_missing_last_trade_defaults_to_mid() {
        let row = TickCsvRow {
            timestamp: "2024-01-15 09:30:00.000".into(),
            symbol: "BTCUSDT".into(),
            bid_price: 30000.0,
            bid_size: 1.5,
            ask_price: 30002.0,
            ask_size: 0.8,
            last_trade_price: None,  // not in Kaggle LOB data
            last_trade_size: None,
        };
        let tick = Tick::try_from(row).unwrap();
        assert_eq!(tick.last_trade_price, dec!(30001.0)); // mid = (30000+30002)/2
        assert!(tick.is_valid());
    }

    #[test]
    fn test_missing_symbol_defaults_to_btcusdt() {
        // Simulates what happens when symbol column is absent
        // (serde uses default_symbol())
        let row = TickCsvRow {
            timestamp: "2024-01-15 09:30:00.000".into(),
            symbol: default_symbol(),
            bid_price: 30000.0,
            bid_size: 1.5,
            ask_price: 30001.0,
            ask_size: 0.8,
            last_trade_price: None,
            last_trade_size: None,
        };
        let tick = Tick::try_from(row).unwrap();
        assert_eq!(tick.symbol, "BTCUSDT");
    }
}