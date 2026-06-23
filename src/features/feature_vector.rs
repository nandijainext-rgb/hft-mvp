use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// ML-ready feature vector produced for every incoming order book snapshot.
///
/// All features are `f64` so they can be fed directly into:
///   - XGBoost (via xgboost-rs or a subprocess call)
///   - Linfa (logistic regression, random forest)
///   - ONNX Runtime (once a model is exported)
///   - scikit-learn via a REST bridge
///
/// `None` fields indicate the feature could not yet be computed (insufficient
/// history in the rolling window). Consumers should handle missing values via
/// imputation or by waiting until all windows are warm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    // ── Identity ──────────────────────────────────────────────────────────────
    pub timestamp: DateTime<Utc>,
    pub symbol: String,

    // ── Level-1 microstructure (always available) ─────────────────────────────
    /// Bid-ask spread = best_ask - best_bid
    pub spread: f64,
    /// Mid price = (best_bid + best_ask) / 2
    pub mid_price: f64,
    /// Order Book Imbalance ∈ [-1, +1]
    pub order_book_imbalance: f64,

    // ── Rolling features (require window to be warm) ──────────────────────────
    /// Std-dev of mid-price returns over last 50 snapshots. None until warm.
    pub rolling_volatility: Option<f64>,
    /// (mid_now - mid_20_ticks_ago) / mid_20_ticks_ago. None until ≥20 ticks.
    pub momentum: Option<f64>,

    // ── Volume / liquidity ────────────────────────────────────────────────────
    /// (bid_vol - ask_vol) / (bid_vol + ask_vol)
    pub volume_imbalance: f64,
    /// bid_vol / ask_vol  (guarded against division by zero)
    pub liquidity_ratio: f64,
    /// updates per second (rolling 1-second window)
    pub trade_intensity: f64,
    /// Raw bid volume summed across retained levels
    pub bid_volume: f64,
    /// Raw ask volume summed across retained levels
    pub ask_volume: f64,
    /// bid_volume + ask_volume
    pub total_liquidity: f64,
}

impl FeatureVector {
    /// Convert to a flat `Vec<f64>` for model inference.
    ///
    /// Missing values (`None`) are substituted with `0.0`.
    /// The order here is the canonical feature ordering for Phase 4.
    pub fn to_array(&self) -> Vec<f64> {
        vec![
            self.spread,
            self.mid_price,
            self.order_book_imbalance,
            self.rolling_volatility.unwrap_or(0.0),
            self.momentum.unwrap_or(0.0),
            self.volume_imbalance,
            self.liquidity_ratio,
            self.trade_intensity,
            self.bid_volume,
            self.ask_volume,
            self.total_liquidity,
        ]
    }

    /// Feature names in the same order as `to_array()`. Useful for XGBoost
    /// `DMatrix::set_feature_names()` and logging.
    pub fn feature_names() -> &'static [&'static str] {
        &[
            "spread",
            "mid_price",
            "order_book_imbalance",
            "rolling_volatility",
            "momentum",
            "volume_imbalance",
            "liquidity_ratio",
            "trade_intensity",
            "bid_volume",
            "ask_volume",
            "total_liquidity",
        ]
    }

    /// Number of features (matches `to_array().len()`).
    pub const FEATURE_DIM: usize = 11;
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn sample_fv() -> FeatureVector {
        FeatureVector {
            timestamp: Utc::now(),
            symbol: "BTCUSDT".into(),
            spread: 1.0,
            mid_price: 30000.5,
            order_book_imbalance: 0.2,
            rolling_volatility: Some(0.001),
            momentum: Some(0.0005),
            volume_imbalance: 0.1,
            liquidity_ratio: 1.5,
            trade_intensity: 12.3,
            bid_volume: 3.0,
            ask_volume: 2.0,
            total_liquidity: 5.0,
        }
    }

    #[test]
    fn to_array_has_correct_dim() {
        let fv = sample_fv();
        assert_eq!(fv.to_array().len(), FeatureVector::FEATURE_DIM);
    }

    #[test]
    fn to_array_matches_feature_names_len() {
        assert_eq!(
            FeatureVector::feature_names().len(),
            FeatureVector::FEATURE_DIM
        );
    }

    #[test]
    fn none_fields_default_to_zero() {
        let mut fv = sample_fv();
        fv.rolling_volatility = None;
        fv.momentum = None;
        let arr = fv.to_array();
        // rolling_volatility is index 3, momentum is index 4
        assert_eq!(arr[3], 0.0);
        assert_eq!(arr[4], 0.0);
    }

    #[test]
    fn serialises_to_json() {
        let fv = sample_fv();
        let json = serde_json::to_string(&fv).unwrap();
        assert!(json.contains("\"symbol\":\"BTCUSDT\""));
        assert!(json.contains("\"spread\":1.0"));
    }
}