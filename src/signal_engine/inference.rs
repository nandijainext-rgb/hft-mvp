// src/signal_engine/inference.rs  (INTEGRATION UPDATE)
//
// Changes from the original Phase 5 version:
//   1. InferenceEngine now also owns a FeatureColumns and a LabelEncoder,
//      loaded once at startup alongside the scaler and ONNX session.
//   2. FeatureVector::to_named_map() replaces the old fixed-order
//      to_raw_slice() — features are matched by NAME, then reordered by
//      FeatureColumns::validate_and_order(), so a field-order change in
//      either Phase 3 or Phase 4 can never silently corrupt predictions.
//   3. Output label parsing now goes through LabelEncoder::label_to_name()
//      and returns a SignalClass via from_name() instead of the old
//      hardcoded from_label() integer match.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use ort::inputs;
use ort::session::{Session, SessionOutputs};
use ort::value::Tensor;
use parking_lot::Mutex;
use tracing::debug;

use super::onnx_loader::{load_session, FeatureColumns, LabelEncoder, ScalerParams};
use super::signal_generator::SignalClass;

// ─────────────────────────────────────────────────────────────────────────────
// FeatureVector — now produces a name→value map instead of a fixed-order array
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct FeatureVector {
    pub timestamp: i64,
    pub symbol: String,
    pub spread: f64,
    pub mid_price: f64,
    pub order_book_imbalance: f64,
    pub rolling_volatility: f64,
    pub momentum: f64,
    pub liquidity_ratio: f64,
    pub volume_imbalance: f64,
    pub trade_intensity: f64,
    pub bid_volume: f64,
    pub ask_volume: f64,
    pub total_liquidity: f64,
}

impl FeatureVector {
    /// Map field names to values. `FeatureColumns::validate_and_order` uses
    /// this to build the correctly-ordered tensor input, validating against
    /// `feature_columns.json` rather than trusting struct field order.
    pub fn to_named_map(&self) -> HashMap<String, f64> {
        let mut m = HashMap::with_capacity(11);
        m.insert("spread".to_string(), self.spread);
        m.insert("mid_price".to_string(), self.mid_price);
        m.insert("order_book_imbalance".to_string(), self.order_book_imbalance);
        m.insert("rolling_volatility".to_string(), self.rolling_volatility);
        m.insert("momentum".to_string(), self.momentum);
        m.insert("liquidity_ratio".to_string(), self.liquidity_ratio);
        m.insert("volume_imbalance".to_string(), self.volume_imbalance);
        m.insert("trade_intensity".to_string(), self.trade_intensity);
        m.insert("bid_volume".to_string(), self.bid_volume);
        m.insert("ask_volume".to_string(), self.ask_volume);
        m.insert("total_liquidity".to_string(), self.total_liquidity);
        m.insert("price_change".to_string(), 0.0);
        m
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceResult {
    pub predicted_class: SignalClass,
    pub confidence: f32,
    pub prob_sell: f32,
    pub prob_hold: f32,
    pub prob_buy: f32,
    pub inference_ms: f64,
    pub model_version: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// InferenceEngine
// ─────────────────────────────────────────────────────────────────────────────

pub struct InferenceEngine {
    session: Mutex<Session>,
    scaler: ScalerParams,
    feature_columns: FeatureColumns,
    label_encoder: LabelEncoder,
    pub model_version: String,
}

impl InferenceEngine {
    /// Load model + scaler + feature_columns + label_encoder, all once, at startup.
    ///
    /// `ml_dir` should point at the directory containing all Phase 4 exports:
    ///   best_model.onnx, scaler.json, feature_columns.json, label_encoder.json
    pub fn new<P: AsRef<Path>>(ml_dir: P) -> Result<Self> {
        let ml_dir = ml_dir.as_ref();

        let model_path = ml_dir.join("best_model.onnx");
        let scaler_path = ml_dir.join("scaler.json");
        let columns_path = ml_dir.join("feature_columns.json");
        let labels_path = ml_dir.join("label_encoder.json");

        let session = load_session(&model_path)?;
        let scaler = ScalerParams::load(&scaler_path)?;
        let feature_columns = FeatureColumns::load(&columns_path)?;
        let label_encoder = LabelEncoder::load(&labels_path)?;

        // Cross-check: scaler and feature_columns must agree on order.
        // A mismatch here means Phase 4's export cell produced inconsistent
        // artifacts — fail fast at startup rather than silently mispredicting.
        if scaler.feature_order != feature_columns.feature_order {
            anyhow::bail!(
                "scaler.json and feature_columns.json disagree on feature order!\n  \
                 scaler.json:           {:?}\n  feature_columns.json:  {:?}\n\
                 Re-run the Phase 4 export cell — both files must be written from \
                 the same MODEL_FEATURES list.",
                scaler.feature_order,
                feature_columns.feature_order
            );
        }

        let model_version = Self::read_model_version(ml_dir)
            .unwrap_or_else(|| Self::compute_fallback_version(&model_path));

        tracing::info!("InferenceEngine ready – model_version={}", model_version);

        Ok(Self {
            session: Mutex::new(session),
            scaler,
            feature_columns,
            label_encoder,
            model_version,
        })
    }

    pub fn run(&self, fv: &FeatureVector) -> Result<InferenceResult> {
        let t0 = Instant::now();

        // 1. Validate + reorder by NAME against feature_columns.json
        //    (catches silent reordering bugs between Phase 3/4/5)
        let named = fv.to_named_map();
        let ordered_raw = self.feature_columns.validate_and_order(&named)?;

        // 2. Normalise
        let normalised = self.scaler.transform(&ordered_raw)?;
        let n_features = normalised.len();

        // 3. Build [1, n_features] f32 tensor
        let tensor = Tensor::<f32>::from_array(([1usize, n_features], normalised))
            .context("Building input tensor")?;

        // 4. Run inference
        let mut session = self.session.lock();
        let outputs = session
            .run(inputs!["float_input" => tensor])
            .context("ONNX session.run()")?;

        // 5. Parse outputs via LabelEncoder (no hardcoded 0/1/2 mapping)
        let (prob_sell, prob_hold, prob_buy, label) = Self::parse_outputs(&outputs)?;
        let class_name = self.label_encoder.label_to_name(label)?;
        let predicted_class = SignalClass::from_name(&class_name)?;

        let confidence = match predicted_class {
            SignalClass::Sell => prob_sell,
            SignalClass::Hold => prob_hold,
            SignalClass::Buy => prob_buy,
        };

        let inference_ms = t0.elapsed().as_secs_f64() * 1000.0;

        debug!(
            symbol = %fv.symbol,
            signal = ?predicted_class,
            confidence,
            inference_ms,
            "Inference complete"
        );

        Ok(InferenceResult {
            predicted_class,
            confidence,
            prob_sell,
            prob_hold,
            prob_buy,
            inference_ms,
            model_version: self.model_version.clone(),
        })
    }

    /// Expose loaded metadata for the /model/info endpoint.
    pub fn feature_order(&self) -> &[String] {
        &self.feature_columns.feature_order
    }

    fn parse_outputs(outputs: &SessionOutputs) -> Result<(f32, f32, f32, i64)> {
        let (_, probs_tensor) = outputs[1]
            .try_extract_tensor::<f32>()
            .context("Extracting probability tensor")?;
        let probs: Vec<f32> = probs_tensor.iter().cloned().collect();

        if probs.len() < 3 {
            anyhow::bail!("Expected ≥3 probability values, got {}", probs.len());
        }

        let (_, label_tensor) = outputs[0]
            .try_extract_tensor::<i64>()
            .context("Extracting label tensor")?;
        let label = *label_tensor.iter().next().context("Label tensor was empty")?;

        Ok((probs[0], probs[1], probs[2], label))
    }

    /// Read model_version out of training_metadata.json if present.
    fn read_model_version(ml_dir: &Path) -> Option<String> {
        let path = ml_dir.join("training_metadata.json");
        let content = std::fs::read_to_string(path).ok()?;
        let json: serde_json::Value = serde_json::from_str(&content).ok()?;
        json.get("model_version")?.as_str().map(String::from)
    }

    fn compute_fallback_version(model_path: &Path) -> String {
        use std::time::UNIX_EPOCH;
        std::fs::metadata(model_path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| format!("v{}", d.as_secs()))
            .unwrap_or_else(|| "v0".to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fv() -> FeatureVector {
        FeatureVector {
            timestamp: 1_700_000_000_000,
            symbol: "AAPL".to_string(),
            spread: 0.01,
            mid_price: 150.0,
            order_book_imbalance: 0.2,
            rolling_volatility: 0.05,
            momentum: 0.03,
            liquidity_ratio: 1.5,
            volume_imbalance: 0.1,
            trade_intensity: 120.0,
            bid_volume: 1000.0,
            ask_volume: 900.0,
            total_liquidity: 1900.0,
        }
    }

    #[test]
    fn test_to_named_map_has_all_fields() {
        let fv = sample_fv();
        let map = fv.to_named_map();
        assert_eq!(map.len(), 12);
        assert_eq!(map["spread"], 0.01);
        assert_eq!(map["total_liquidity"], 1900.0);
    }

    #[test]
    fn test_engine_loads_from_directory_when_artifacts_present() {
        let ml_dir = "models";
        if !std::path::Path::new(ml_dir).join("best_model.onnx").exists() {
            println!("Skipping: no model artifacts in ./models");
            return;
        }
        let engine = InferenceEngine::new(ml_dir);
        assert!(engine.is_ok(), "Engine should load: {:?}", engine.err());
    }

    #[test]
    fn test_full_inference_when_artifacts_present() {
        let ml_dir = "models";
        if !std::path::Path::new(ml_dir).join("best_model.onnx").exists() {
            println!("Skipping: no model artifacts in ./models");
            return;
        }
        let engine = InferenceEngine::new(ml_dir).unwrap();
        let fv = sample_fv();
        let result = engine.run(&fv);
        assert!(result.is_ok(), "Inference failed: {:?}", result.err());
        let r = result.unwrap();
        let prob_sum = r.prob_sell + r.prob_hold + r.prob_buy;
        assert!((prob_sum - 1.0).abs() < 0.01, "Probs don't sum to 1: {}", prob_sum);
        assert!(r.inference_ms < 50.0);
    }
}
