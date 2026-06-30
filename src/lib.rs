// src/lib.rs  (NEW FILE)
//
// Re-exports the modules that main.rs already declares, so integration
// tests (tests/parity_test.rs) can `use hft_signal_engine::signal_engine::...`
// without duplicating module declarations.
//
// main.rs keeps its own `mod signal_engine; mod redis; mod api;` lines for
// the binary target — this file just mirrors them for the [lib] target.
// Both point at the same files on disk; Cargo compiles each target separately.

pub mod api;
pub mod features;
pub mod redis;
pub mod signal_engine;

use std::sync::Arc;

use redis::redis_client::RedisClient;
use signal_engine::inference::InferenceEngine;
use signal_engine::model_metadata::ModelMetadata;
use signal_engine::prediction::PredictionStore;

/// Shared application state injected into every Actix handler.
pub struct AppState {
    pub inference_engine: Arc<InferenceEngine>,
    pub prediction_store: Arc<PredictionStore>,
    pub redis_client: Arc<RedisClient>,
    pub model_metadata: Arc<ModelMetadata>,
}
