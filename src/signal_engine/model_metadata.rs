use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub training_metadata: serde_json::Value,
    pub model_metrics: serde_json::Value,
}

impl ModelMetadata {
    pub fn load<P: AsRef<Path>>(ml_dir: P) -> Self {
        let ml_dir = ml_dir.as_ref();
        Self {
            training_metadata: read_json_or_empty(ml_dir.join("training_metadata.json")),
            model_metrics: read_json_or_empty(ml_dir.join("model_metrics.json")),
        }
    }
}

fn read_json_or_empty(path: impl AsRef<Path>) -> serde_json::Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|content| serde_json::from_str(&content).ok())
        .unwrap_or_else(|| serde_json::json!({}))
}
