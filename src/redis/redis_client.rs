use anyhow::Result;
use redis::{aio::ConnectionManager, AsyncCommands, Client};
use tracing::{debug, warn};

use crate::features::FeatureVector;

/// Max history entries kept per symbol in a Redis List.
const HISTORY_MAX_LEN: isize = 100;

/// Async Redis client that stores and retrieves feature vectors.
///
/// Key schema:
///   `features:{symbol}`         → Hash  (latest vector fields)
///   `features:{symbol}:history` → List  (last 100 serialised JSON vectors)
#[derive(Clone)]
pub struct RedisClient {
    conn: ConnectionManager,
}

impl RedisClient {
    pub async fn new(url: &str) -> Result<Self> {
        let client = Client::open(url)?;
        let conn = ConnectionManager::new(client).await?;
        Ok(Self { conn })
    }

    // ── Write ─────────────────────────────────────────────────────────────────

    /// Store a feature vector as a Redis Hash (latest) and append to history list.
    pub async fn store_features(&mut self, fv: &FeatureVector) -> Result<()> {
        let hash_key = format!("features:{}", fv.symbol);
        let list_key = format!("features:{}:history", fv.symbol);

        // ── Hash: latest values (HSET multi-field) ────────────────────────────
        let ts = fv.timestamp.timestamp_millis().to_string();
        let fields: Vec<(&str, String)> = vec![
            ("timestamp",            ts.clone()),
            ("spread",               fv.spread.to_string()),
            ("mid_price",            fv.mid_price.to_string()),
            ("obi",                  fv.order_book_imbalance.to_string()),
            ("volatility",           fv.rolling_volatility.map(|v| v.to_string()).unwrap_or_default()),
            ("momentum",             fv.momentum.map(|v| v.to_string()).unwrap_or_default()),
            ("volume_imbalance",     fv.volume_imbalance.to_string()),
            ("liquidity_ratio",      fv.liquidity_ratio.to_string()),
            ("trade_intensity",      fv.trade_intensity.to_string()),
            ("bid_volume",           fv.bid_volume.to_string()),
            ("ask_volume",           fv.ask_volume.to_string()),
            ("total_liquidity",      fv.total_liquidity.to_string()),
        ];

        let _: () = self.conn.hset_multiple(&hash_key, &fields).await?;

        // ── List: history (LPUSH + LTRIM to cap at 100) ───────────────────────
        let json = serde_json::to_string(fv)?;
        let _: () = self.conn.lpush(&list_key, &json).await?;
        let _: () = self.conn.ltrim(&list_key, 0, HISTORY_MAX_LEN - 1).await?;

        debug!(key = %hash_key, "Feature vector stored");
        Ok(())
    }

    // ── Read ──────────────────────────────────────────────────────────────────

    /// Retrieve the latest feature vector for a symbol (from the Hash).
    /// Returns `None` if the symbol has never been seen.
    pub async fn get_latest(&mut self, symbol: &str) -> Result<Option<LatestFeatures>> {
        let key = format!("features:{symbol}");

        let exists: bool = self.conn.exists(&key).await?;
        if !exists {
            return Ok(None);
        }

        // Fetch all fields in one HGETALL
        let raw: Vec<(String, String)> = self.conn.hgetall(&key).await?;
        if raw.is_empty() {
            return Ok(None);
        }

        let mut lf = LatestFeatures::default();
        for (field, value) in raw {
            match field.as_str() {
                "timestamp"        => lf.timestamp        = value,
                "spread"           => lf.spread           = value,
                "mid_price"        => lf.mid_price        = value,
                "obi"              => lf.obi              = value,
                "volatility"       => lf.volatility       = value,
                "momentum"         => lf.momentum         = value,
                "volume_imbalance" => lf.volume_imbalance = value,
                "liquidity_ratio"  => lf.liquidity_ratio  = value,
                "trade_intensity"  => lf.trade_intensity  = value,
                "bid_volume"       => lf.bid_volume       = value,
                "ask_volume"       => lf.ask_volume       = value,
                "total_liquidity"  => lf.total_liquidity  = value,
                _ => {}
            }
        }

        Ok(Some(lf))
    }

    /// Retrieve last N feature vectors for a symbol (from the history List).
    /// Returns newest-first. Returns `[]` if symbol unknown.
    pub async fn get_history(&mut self, symbol: &str, n: isize) -> Result<Vec<FeatureVector>> {
        let key = format!("features:{symbol}:history");
        let raw: Vec<String> = self.conn.lrange(&key, 0, n - 1).await?;

        let mut out = Vec::with_capacity(raw.len());
        for s in raw {
            match serde_json::from_str::<FeatureVector>(&s) {
                Ok(fv) => out.push(fv),
                Err(e) => warn!(error = %e, "Skipping malformed history entry"),
            }
        }

        Ok(out)
    }

    /// Compute aggregate stats across the last N history entries.
    pub async fn get_stats(&mut self, symbol: &str) -> Result<FeatureStats> {
        let history = self.get_history(symbol, HISTORY_MAX_LEN).await?;
        Ok(FeatureStats::compute(&history))
    }

    /// Ping Redis. Returns true if reachable.
    pub async fn ping(&mut self) -> bool {
        let r: redis::RedisResult<String> = redis::cmd("PING")
            .query_async(&mut self.conn)
            .await;
        match r {
            Ok(s) if s == "PONG" => true,
            Ok(s) => { warn!(resp = %s, "Unexpected PING response"); false }
            Err(e) => { warn!(error = %e, "Redis PING failed"); false }
        }
    }
}

// ── Response types ────────────────────────────────────────────────────────────

/// Latest feature values as raw strings (as stored in Redis Hash).
#[derive(Debug, Default, serde::Serialize)]
pub struct LatestFeatures {
    pub timestamp:        String,
    pub spread:           String,
    pub mid_price:        String,
    pub obi:              String,
    pub volatility:       String,
    pub momentum:         String,
    pub volume_imbalance: String,
    pub liquidity_ratio:  String,
    pub trade_intensity:  String,
    pub bid_volume:       String,
    pub ask_volume:       String,
    pub total_liquidity:  String,
}

/// Aggregate statistics computed over the history window.
#[derive(Debug, serde::Serialize)]
pub struct FeatureStats {
    pub mean_volatility: f64,
    pub mean_momentum:   f64,
    pub mean_obi:        f64,
    pub avg_spread:      f64,
    pub sample_count:    usize,
}

impl FeatureStats {
    pub fn compute(history: &[FeatureVector]) -> Self {
        let n = history.len();
        if n == 0 {
            return Self {
                mean_volatility: 0.0,
                mean_momentum:   0.0,
                mean_obi:        0.0,
                avg_spread:      0.0,
                sample_count:    0,
            };
        }

        let mut sum_vol = 0.0f64;
        let mut vol_count = 0usize;
        let mut sum_mom = 0.0f64;
        let mut mom_count = 0usize;
        let mut sum_obi = 0.0f64;
        let mut sum_spread = 0.0f64;

        for fv in history {
            if let Some(v) = fv.rolling_volatility {
                sum_vol += v;
                vol_count += 1;
            }
            if let Some(m) = fv.momentum {
                sum_mom += m;
                mom_count += 1;
            }
            sum_obi += fv.order_book_imbalance;
            sum_spread += fv.spread;
        }

        Self {
            mean_volatility: if vol_count > 0 { sum_vol / vol_count as f64 } else { 0.0 },
            mean_momentum:   if mom_count > 0 { sum_mom / mom_count as f64 } else { 0.0 },
            mean_obi:        sum_obi / n as f64,
            avg_spread:      sum_spread / n as f64,
            sample_count:    n,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn sample_fv(symbol: &str, spread: f64, obi: f64) -> FeatureVector {
        FeatureVector {
            timestamp: Utc::now(),
            symbol: symbol.into(),
            spread,
            mid_price: 30000.0,
            order_book_imbalance: obi,
            rolling_volatility: Some(0.001),
            momentum: Some(0.0005),
            volume_imbalance: 0.1,
            liquidity_ratio: 1.5,
            trade_intensity: 10.0,
            bid_volume: 3.0,
            ask_volume: 2.0,
            total_liquidity: 5.0,
        }
    }

    #[test]
    fn feature_stats_compute_empty() {
        let stats = FeatureStats::compute(&[]);
        assert_eq!(stats.sample_count, 0);
        assert_eq!(stats.mean_volatility, 0.0);
    }

    #[test]
    fn feature_stats_compute_values() {
        let history = vec![
            sample_fv("BTCUSDT", 1.0, 0.2),
            sample_fv("BTCUSDT", 2.0, 0.4),
            sample_fv("BTCUSDT", 3.0, 0.6),
        ];
        let stats = FeatureStats::compute(&history);
        assert_eq!(stats.sample_count, 3);
        assert!((stats.avg_spread - 2.0).abs() < 1e-10);
        assert!((stats.mean_obi - 0.4).abs() < 1e-10);
    }

    #[test]
    fn feature_stats_skips_none_rolling() {
        let mut fvs = vec![
            sample_fv("BTCUSDT", 1.0, 0.0),
            sample_fv("BTCUSDT", 1.0, 0.0),
        ];
        fvs[0].rolling_volatility = None;
        fvs[1].rolling_volatility = Some(0.01);
        let stats = FeatureStats::compute(&fvs);
        // Only one valid volatility entry
        assert!((stats.mean_volatility - 0.01).abs() < 1e-10);
    }

    /// Integration test — requires a running Redis instance.
    /// Run with: REDIS_URL=redis://127.0.0.1:6379 cargo test test_redis -- --ignored
    #[tokio::test]
    #[ignore]
    async fn test_store_and_retrieve() {
        let url = std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".into());
        let mut client = RedisClient::new(&url).await.expect("connect");
        assert!(client.ping().await);

        let fv = sample_fv("BTCUSDT", 1.0, 0.2);
        client.store_features(&fv).await.expect("store");

        let latest = client.get_latest("BTCUSDT").await.expect("get latest");
        assert!(latest.is_some());
        let lf = latest.unwrap();
        assert_eq!(lf.spread, "1");

        let history = client.get_history("BTCUSDT", 10).await.expect("history");
        assert!(!history.is_empty());
    }
}