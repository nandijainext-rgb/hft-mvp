use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalerParams {
    pub scaler_type: String,
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
    #[serde(default)]
    pub variance: Vec<f64>,
    pub number_of_features: usize,
    pub feature_order: Vec<String>,
}

impl ScalerParams {
    pub fn transform(&self, raw: &[f64]) -> Result<Vec<f32>> {
        if raw.len() != self.mean.len() {
            anyhow::bail!(
                "Feature length mismatch: scaler expects {}, got {}",
                self.mean.len(),
                raw.len()
            );
        }

        Ok(raw
            .iter()
            .zip(self.mean.iter())
            .zip(self.scale.iter())
            .map(|((x, m), s)| {
                if *s == 0.0 {
                    0.0
                } else {
                    ((x - m) / s) as f32
                }
            })
            .collect())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            warn!("scaler.json not found at {:?}; using identity scaler", path);
            return Ok(Self::identity(default_feature_order()));
        }

        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Reading scaler from {:?}", path))?;
        let json: serde_json::Value =
            serde_json::from_str(&content).context("Parsing scaler.json")?;

        let mean = read_f64_vec(&json, "mean")
            .or_else(|| read_f64_vec(&json, "mean_"))
            .context("scaler.json missing mean/mean_")?;
        let scale = read_f64_vec(&json, "scale")
            .or_else(|| read_f64_vec(&json, "scale_"))
            .context("scaler.json missing scale/scale_")?;
        let feature_order =
            read_string_vec(&json, "feature_order").unwrap_or_else(|| default_feature_order_for_len(mean.len()));
        let number_of_features = json
            .get("number_of_features")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(feature_order.len());

        let scaler = Self {
            scaler_type: json
                .get("scaler_type")
                .and_then(|v| v.as_str())
                .unwrap_or("standard")
                .to_string(),
            mean,
            scale,
            variance: read_f64_vec(&json, "variance").unwrap_or_default(),
            number_of_features,
            feature_order,
        };

        if scaler.mean.len() != scaler.number_of_features
            || scaler.scale.len() != scaler.number_of_features
            || scaler.feature_order.len() != scaler.number_of_features
        {
            anyhow::bail!(
                "scaler.json is internally inconsistent: number_of_features={}, mean.len()={}, scale.len()={}, feature_order.len()={}",
                scaler.number_of_features,
                scaler.mean.len(),
                scaler.scale.len(),
                scaler.feature_order.len()
            );
        }

        info!(
            "Loaded {} scaler ({} features): {:?}",
            scaler.scaler_type, scaler.number_of_features, scaler.feature_order
        );
        Ok(scaler)
    }

    pub fn identity(feature_order: Vec<String>) -> Self {
        let n = feature_order.len();
        Self {
            scaler_type: "identity".to_string(),
            mean: vec![0.0; n],
            scale: vec![1.0; n],
            variance: vec![1.0; n],
            number_of_features: n,
            feature_order,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureColumns {
    pub feature_order: Vec<String>,
}

impl FeatureColumns {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            warn!(
                "feature_columns.json not found at {:?}; using default feature order",
                path
            );
            return Ok(Self {
                feature_order: default_feature_order(),
            });
        }

        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Reading feature_columns.json from {:?}", path))?;
        let cols: FeatureColumns =
            serde_json::from_str(&content).context("Parsing feature_columns.json")?;
        info!(
            "Loaded feature order ({} features): {:?}",
            cols.feature_order.len(),
            cols.feature_order
        );
        Ok(cols)
    }

    pub fn validate_and_order(&self, named: &HashMap<String, f64>) -> Result<Vec<f64>> {
        let mut missing = Vec::new();
        let mut ordered = Vec::with_capacity(self.feature_order.len());

        for name in &self.feature_order {
            match named.get(name) {
                Some(v) => ordered.push(*v),
                None if name == "price_change" || name.starts_with("unused_feature_") => ordered.push(0.0),
                None => missing.push(name.clone()),
            }
        }

        if !missing.is_empty() {
            anyhow::bail!(
                "Feature vector is missing required fields (per feature_columns.json): {:?}",
                missing
            );
        }

        let extra: Vec<&String> = named
            .keys()
            .filter(|k| !self.feature_order.contains(k))
            .collect();
        if !extra.is_empty() {
            warn!("Feature vector has unexpected extra fields, ignoring: {:?}", extra);
        }

        Ok(ordered)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelEncoder {
    #[serde(flatten)]
    pub mapping: HashMap<String, String>,
}

impl LabelEncoder {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            warn!(
                "label_encoder.json not found at {:?}; using default 0=SELL, 1=HOLD, 2=BUY mapping",
                path
            );
            return Ok(Self::default_mapping());
        }

        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Reading label_encoder.json from {:?}", path))?;
        let encoder: LabelEncoder =
            serde_json::from_str(&content).context("Parsing label_encoder.json")?;
        info!("Loaded label encoder: {:?}", encoder.mapping);
        Ok(encoder)
    }

    pub fn default_mapping() -> Self {
        let mut mapping = HashMap::new();
        mapping.insert("0".to_string(), "SELL".to_string());
        mapping.insert("1".to_string(), "HOLD".to_string());
        mapping.insert("2".to_string(), "BUY".to_string());
        Self { mapping }
    }

    pub fn label_to_name(&self, label: i64) -> Result<String> {
        self.mapping
            .get(&label.to_string())
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Unknown model label: {} (not in label_encoder.json)", label))
    }
}

pub fn load_session<P: AsRef<Path>>(model_path: P) -> Result<Session> {
    let model_path = model_path.as_ref();

    if !model_path.exists() {
        anyhow::bail!("ONNX model not found at {:?}", model_path);
    }

    info!("Initialising ONNX Runtime session from {:?}", model_path);

    let session = Session::builder()
        .map_err(|e| anyhow::anyhow!("Creating ONNX session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|e| anyhow::anyhow!("Setting ONNX graph optimization: {e}"))?
        .with_intra_threads(4)
        .map_err(|e| anyhow::anyhow!("Setting ONNX intra-op threads: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow::anyhow!("Loading ONNX model from {:?}: {e}", model_path))?;

    for (i, input) in session.inputs().iter().enumerate() {
        info!("  Model input  [{}]: {:?}  type={:?}", i, input.name(), input.dtype());
    }
    for (i, output) in session.outputs().iter().enumerate() {
        info!("  Model output [{}]: {:?}  type={:?}", i, output.name(), output.dtype());
    }

    Ok(session)
}

fn read_f64_vec(json: &serde_json::Value, key: &str) -> Option<Vec<f64>> {
    json.get(key)?
        .as_array()?
        .iter()
        .map(|v| v.as_f64())
        .collect()
}

fn read_string_vec(json: &serde_json::Value, key: &str) -> Option<Vec<String>> {
    json.get(key)?
        .as_array()?
        .iter()
        .map(|v| v.as_str().map(str::to_string))
        .collect()
}

fn default_feature_order_for_len(len: usize) -> Vec<String> {
    let mut order = default_feature_order();
    while order.len() < len {
        order.push(format!("unused_feature_{}", order.len()));
    }
    order.truncate(len);
    order
}

fn default_feature_order() -> Vec<String> {
    [
        "spread",
        "mid_price",
        "order_book_imbalance",
        "rolling_volatility",
        "momentum",
        "liquidity_ratio",
        "volume_imbalance",
        "trade_intensity",
        "bid_volume",
        "ask_volume",
        "total_liquidity",
        "price_change",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaler_transform_matches_standard_formula() {
        let scaler = ScalerParams {
            scaler_type: "standard".into(),
            mean: vec![100.0, 50.0],
            scale: vec![10.0, 5.0],
            variance: vec![100.0, 25.0],
            number_of_features: 2,
            feature_order: vec!["a".into(), "b".into()],
        };
        let out = scaler.transform(&[110.0, 55.0]).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_scaler_wrong_length_errors() {
        let scaler = ScalerParams::identity(vec!["a".into(), "b".into(), "c".into()]);
        assert!(scaler.transform(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_feature_columns_validate_and_order_happy_path() {
        let cols = FeatureColumns {
            feature_order: vec!["spread".into(), "mid_price".into(), "momentum".into()],
        };
        let mut named = HashMap::new();
        named.insert("momentum".to_string(), 0.03);
        named.insert("spread".to_string(), 0.01);
        named.insert("mid_price".to_string(), 150.0);

        let ordered = cols.validate_and_order(&named).unwrap();
        assert_eq!(ordered, vec![0.01, 150.0, 0.03]);
    }

    #[test]
    fn test_feature_columns_missing_field_errors() {
        let cols = FeatureColumns {
            feature_order: vec!["spread".into(), "mid_price".into()],
        };
        let mut named = HashMap::new();
        named.insert("spread".to_string(), 0.01);
        let result = cols.validate_and_order(&named);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mid_price"));
    }

    #[test]
    fn test_feature_columns_extra_field_is_ignored_not_error() {
        let cols = FeatureColumns {
            feature_order: vec!["spread".into()],
        };
        let mut named = HashMap::new();
        named.insert("spread".to_string(), 0.01);
        named.insert("unexpected_field".to_string(), 999.0);
        let result = cols.validate_and_order(&named);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), vec![0.01]);
    }

    #[test]
    fn test_label_encoder_default_mapping() {
        let enc = LabelEncoder::default_mapping();
        assert_eq!(enc.label_to_name(0).unwrap(), "SELL");
        assert_eq!(enc.label_to_name(1).unwrap(), "HOLD");
        assert_eq!(enc.label_to_name(2).unwrap(), "BUY");
        assert!(enc.label_to_name(99).is_err());
    }

    #[test]
    fn test_label_encoder_loads_from_json_string() {
        let json = r#"{"0":"SELL","1":"HOLD","2":"BUY"}"#;
        let enc: LabelEncoder = serde_json::from_str(json).unwrap();
        assert_eq!(enc.label_to_name(2).unwrap(), "BUY");
    }
}
