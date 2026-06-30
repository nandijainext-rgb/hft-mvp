// src/signal_engine/signal_generator.rs
//
// Applies business rules on top of raw inference to produce a TradingSignal:
//   • Only emit BUY/SELL when confidence > CONFIDENCE_THRESHOLD
//   • Otherwise → HOLD

use serde::{Deserialize, Serialize};
use tracing::debug;
use uuid::Uuid;

use super::inference::{FeatureVector, InferenceResult};

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

pub const CONFIDENCE_THRESHOLD: f32 = 0.70;

// ─────────────────────────────────────────────────────────────────────────────
// SignalClass
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum SignalClass {
    Sell,
    Hold,
    Buy,
}

impl SignalClass {
    /// Map ONNX label integer to enum.
    pub fn from_label(label: i64) -> anyhow::Result<Self> {
        match label {
            0 => Ok(Self::Sell),
            1 => Ok(Self::Hold),
            2 => Ok(Self::Buy),
            other => anyhow::bail!("Unknown model label: {}", other),
        }
    }

    pub fn from_name(name: &str) -> anyhow::Result<Self> {
        match name.trim().to_ascii_uppercase().as_str() {
            "SELL" => Ok(Self::Sell),
            "HOLD" => Ok(Self::Hold),
            "BUY" => Ok(Self::Buy),
            other => anyhow::bail!("Unknown model label name: {}", other),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sell => "SELL",
            Self::Hold => "HOLD",
            Self::Buy  => "BUY",
        }
    }
}

impl std::fmt::Display for SignalClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TradingSignal
// ─────────────────────────────────────────────────────────────────────────────

/// Final tradable signal emitted by the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingSignal {
    pub id: String,
    pub timestamp: i64,          // Unix ms
    pub symbol: String,
    pub signal: SignalClass,
    pub confidence: f32,
    pub prob_sell: f32,
    pub prob_hold: f32,
    pub prob_buy: f32,
    pub model_version: String,
    pub inference_ms: f64,
    /// Whether the signal passed the confidence threshold
    pub is_actionable: bool,
}

impl TradingSignal {
    /// Apply confidence-filter business rules.
    ///
    /// Rules:
    ///   - BUY  is emitted only when predicted_class == BUY  && confidence > threshold
    ///   - SELL is emitted only when predicted_class == SELL && confidence > threshold
    ///   - All other cases → HOLD  (never filtered by confidence)
    pub fn from_inference(fv: &FeatureVector, result: &InferenceResult) -> Self {
        let is_actionable = result.confidence > CONFIDENCE_THRESHOLD
            && result.predicted_class != SignalClass::Hold;

        let signal = if is_actionable {
            result.predicted_class
        } else {
            SignalClass::Hold
        };

        debug!(
            symbol     = %fv.symbol,
            raw_class  = ?result.predicted_class,
            confidence = result.confidence,
            emitted    = ?signal,
            "Signal decision"
        );

        Self {
            id: Uuid::new_v4().to_string(),
            timestamp: fv.timestamp,
            symbol: fv.symbol.clone(),
            signal,
            confidence: result.confidence,
            prob_sell: result.prob_sell,
            prob_hold: result.prob_hold,
            prob_buy: result.prob_buy,
            model_version: result.model_version.clone(),
            inference_ms: result.inference_ms,
            is_actionable,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use crate::signal_engine::inference::FeatureVector;

    fn make_fv(symbol: &str) -> FeatureVector {
        FeatureVector {
            timestamp: Utc::now().timestamp_millis(),
            symbol: symbol.to_string(),
            spread: 0.01,
            mid_price: 100.0,
            order_book_imbalance: 0.1,
            rolling_volatility: 0.02,
            momentum: 0.01,
            liquidity_ratio: 1.0,
            volume_imbalance: 0.05,
            trade_intensity: 80.0,
            bid_volume: 500.0,
            ask_volume: 480.0,
            total_liquidity: 980.0,
        }
    }

    fn make_result(class: SignalClass, confidence: f32) -> InferenceResult {
        let (ps, ph, pb) = match class {
            SignalClass::Sell => (confidence, (1.0 - confidence) / 2.0, (1.0 - confidence) / 2.0),
            SignalClass::Hold => ((1.0 - confidence) / 2.0, confidence, (1.0 - confidence) / 2.0),
            SignalClass::Buy  => ((1.0 - confidence) / 2.0, (1.0 - confidence) / 2.0, confidence),
        };
        InferenceResult {
            predicted_class: class,
            confidence,
            prob_sell: ps,
            prob_hold: ph,
            prob_buy: pb,
            inference_ms: 1.2,
            model_version: "v1".to_string(),
        }
    }

    #[test]
    fn test_buy_high_confidence_emits_buy() {
        let signal = TradingSignal::from_inference(
            &make_fv("AAPL"),
            &make_result(SignalClass::Buy, 0.85),
        );
        assert_eq!(signal.signal, SignalClass::Buy);
        assert!(signal.is_actionable);
    }

    #[test]
    fn test_buy_low_confidence_emits_hold() {
        let signal = TradingSignal::from_inference(
            &make_fv("AAPL"),
            &make_result(SignalClass::Buy, 0.60),
        );
        assert_eq!(signal.signal, SignalClass::Hold);
        assert!(!signal.is_actionable);
    }

    #[test]
    fn test_sell_high_confidence_emits_sell() {
        let signal = TradingSignal::from_inference(
            &make_fv("TSLA"),
            &make_result(SignalClass::Sell, 0.80),
        );
        assert_eq!(signal.signal, SignalClass::Sell);
        assert!(signal.is_actionable);
    }

    #[test]
    fn test_sell_low_confidence_emits_hold() {
        let signal = TradingSignal::from_inference(
            &make_fv("TSLA"),
            &make_result(SignalClass::Sell, 0.65),
        );
        assert_eq!(signal.signal, SignalClass::Hold);
        assert!(!signal.is_actionable);
    }

    #[test]
    fn test_hold_always_hold() {
        for conf in [0.40f32, 0.70, 0.90] {
            let signal = TradingSignal::from_inference(
                &make_fv("MSFT"),
                &make_result(SignalClass::Hold, conf),
            );
            assert_eq!(signal.signal, SignalClass::Hold);
        }
    }

    #[test]
    fn test_boundary_confidence_threshold() {
        // Exactly at threshold — must be HOLD (strict >)
        let signal = TradingSignal::from_inference(
            &make_fv("GOOG"),
            &make_result(SignalClass::Buy, CONFIDENCE_THRESHOLD),
        );
        assert_eq!(signal.signal, SignalClass::Hold);

        // Just above — must be BUY
        let signal = TradingSignal::from_inference(
            &make_fv("GOOG"),
            &make_result(SignalClass::Buy, CONFIDENCE_THRESHOLD + 0.001),
        );
        assert_eq!(signal.signal, SignalClass::Buy);
    }

    #[test]
    fn test_signal_has_unique_ids() {
        let fv = make_fv("AMZN");
        let res = make_result(SignalClass::Buy, 0.90);
        let s1 = TradingSignal::from_inference(&fv, &res);
        let s2 = TradingSignal::from_inference(&fv, &res);
        assert_ne!(s1.id, s2.id);
    }
}
