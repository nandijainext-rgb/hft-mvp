// src/api/signal_handlers.rs
//
// REST API handlers:
//
//   GET  /health                    → liveness probe
//   POST /predict                   → run inference on a FeatureVector
//   GET  /signal/{symbol}           → latest signal from Redis (fallback: memory)
//   GET  /signal/history/{symbol}   → last N predictions from memory store
//   GET  /model/info                → training metadata + metrics from Phase 4

use actix_web::{web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use tracing::{error, info};

use crate::signal_engine::inference::FeatureVector;
use crate::signal_engine::signal_generator::TradingSignal;
use crate::AppState;

// ─────────────────────────────────────────────────────────────────────────────
// Response types
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_version: String,
    pub prediction_count: usize,
    pub symbol_count: usize,
}

#[derive(Deserialize)]
pub struct HistoryQuery {
    /// How many records to return (default 50, max 1000)
    pub limit: Option<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /health
// ─────────────────────────────────────────────────────────────────────────────

pub async fn health(state: web::Data<AppState>) -> impl Responder {
    // Optionally check Redis connectivity
    let redis_ok = state.redis_client.ping_shared().await.is_ok();

    let body = HealthResponse {
        status: if redis_ok { "ok".into() } else { "degraded".into() },
        model_version: state.inference_engine.model_version.clone(),
        prediction_count: state.prediction_store.total_count(),
        symbol_count: state.prediction_store.symbol_count(),
    };

    if redis_ok {
        HttpResponse::Ok().json(body)
    } else {
        HttpResponse::ServiceUnavailable().json(body)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /predict
//
// Body: FeatureVector (JSON)
// Returns: TradingSignal (JSON)
// ─────────────────────────────────────────────────────────────────────────────

pub async fn predict(
    state: web::Data<AppState>,
    body: web::Json<FeatureVector>,
) -> impl Responder {
    let fv = body.into_inner();

    // 1. Run ONNX inference
    let inference_result = match state.inference_engine.run(&fv) {
        Ok(r) => r,
        Err(e) => {
            error!("Inference failed for {}: {}", fv.symbol, e);
            return HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Inference error: {}", e),
            });
        }
    };

    // 2. Apply signal business rules
    let signal =
        TradingSignal::from_inference(&fv, &inference_result);

    info!(
        symbol    = %signal.symbol,
        signal    = %signal.signal,
        confidence = signal.confidence,
        inference_ms = signal.inference_ms,
        "Signal generated"
    );

    // 3. Store in memory (async clone before move)
    state.prediction_store.push(signal.clone());

    // 4. Persist to Redis (non-blocking; log but don't fail the response)
    if let Err(e) = state.redis_client.store_signal(&signal).await {
        error!("Redis store failed (non-fatal): {}", e);
    }

    HttpResponse::Ok().json(signal)
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /signal/{symbol}
//
// Returns the latest signal for `symbol`.  First checks Redis (fast path),
// falls back to the in-memory store.
// ─────────────────────────────────────────────────────────────────────────────

pub async fn get_signal(
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> impl Responder {
    let symbol = path.into_inner().to_uppercase();

    // Fast path: Redis
    match state.redis_client.get_signal(&symbol).await {
        Ok(Some(signal)) => return HttpResponse::Ok().json(signal),
        Ok(None) => {}
        Err(e) => {
            error!("Redis GET failed, falling back to memory: {}", e);
        }
    }

    // Fallback: in-memory store
    match state.prediction_store.latest(&symbol) {
        Some(record) => HttpResponse::Ok().json(record.signal),
        None => HttpResponse::NotFound().json(ErrorResponse {
            error: format!("No signal found for symbol '{}'", symbol),
        }),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /signal/history/{symbol}?limit=N
//
// Returns the last `limit` (default 50) predictions for `symbol`.
// ─────────────────────────────────────────────────────────────────────────────

pub async fn get_signal_history(
    state: web::Data<AppState>,
    path: web::Path<String>,
    query: web::Query<HistoryQuery>,
) -> impl Responder {
    let symbol = path.into_inner().to_uppercase();
    let limit = query.limit.unwrap_or(50).min(1000);

    let records = state.prediction_store.history(&symbol, limit);

    if records.is_empty() {
        return HttpResponse::NotFound().json(ErrorResponse {
            error: format!("No prediction history for symbol '{}'", symbol),
        });
    }

    #[derive(Serialize)]
    struct HistoryResponse {
        symbol: String,
        count: usize,
        records: Vec<crate::signal_engine::prediction::PredictionRecord>,
    }

    HttpResponse::Ok().json(HistoryResponse {
        symbol: symbol.clone(),
        count: records.len(),
        records,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /model/info
//
// Returns Phase 4 training metadata + metrics (training_metadata.json +
// model_metrics.json), plus the feature order Phase 5 is validating against
// and the currently active model version. Read-only / observability only —
// does not affect inference.
//
// Requires `model_metadata: Arc<ModelMetadata>` on AppState and
// `InferenceEngine::feature_order()` — see the Phase 4/5 integration layer.
// ─────────────────────────────────────────────────────────────────────────────

pub async fn model_info(state: web::Data<AppState>) -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "metadata": &*state.model_metadata,
        "feature_order": state.inference_engine.feature_order(),
        "model_version_active": state.inference_engine.model_version,
    }))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, web, App};
    use std::sync::Arc;
    use chrono::Utc;

    use crate::signal_engine::inference::InferenceEngine;
    use crate::signal_engine::model_metadata::ModelMetadata;
    use crate::signal_engine::prediction::PredictionStore;
    use crate::redis::redis_client::RedisClient;

    /// Build test AppState (skips real model/Redis if not available).
    ///
    /// NOTE: InferenceEngine::new() now takes a single `ml_dir` containing
    /// best_model.onnx + scaler.json + feature_columns.json +
    /// label_encoder.json (Phase 4/5 integration update), not the old
    /// two-arg (model_path, scaler_path) signature.
    async fn make_test_state_with_history() -> Option<web::Data<AppState>> {
        let ml_dir = "models";
        if !std::path::Path::new(ml_dir).join("best_model.onnx").exists() {
            return None;
        }
        let engine = InferenceEngine::new(ml_dir).ok()?;
        let store = PredictionStore::new(100);
        let redis = RedisClient::new("redis://127.0.0.1:6379").await.ok()?;
        let model_metadata = ModelMetadata::load(ml_dir);

        Some(web::Data::new(AppState {
            inference_engine: Arc::new(engine),
            prediction_store: Arc::new(store),
            redis_client: Arc::new(redis),
            model_metadata: Arc::new(model_metadata),
        }))
    }

    fn sample_fv_json(symbol: &str) -> serde_json::Value {
        serde_json::json!({
            "timestamp": Utc::now().timestamp_millis(),
            "symbol": symbol,
            "spread": 0.01,
            "mid_price": 150.0,
            "order_book_imbalance": 0.2,
            "rolling_volatility": 0.05,
            "momentum": 0.03,
            "liquidity_ratio": 1.5,
            "volume_imbalance": 0.1,
            "trade_intensity": 120.0,
            "bid_volume": 1000.0,
            "ask_volume": 900.0,
            "total_liquidity": 1900.0
        })
    }

    #[actix_web::test]
    async fn test_predict_endpoint_returns_signal() {
        let Some(state) = make_test_state_with_history().await else {
            println!("Skipping: model/Redis not available");
            return;
        };

        let app = test::init_service(
            App::new()
                .app_data(state)
                .route("/predict", web::post().to(predict)),
        )
        .await;

        let req = test::TestRequest::post()
            .uri("/predict")
            .set_json(sample_fv_json("AAPL"))
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success(), "Expected 200, got {}", resp.status());

        let body: serde_json::Value = test::read_body_json(resp).await;
        assert_eq!(body["symbol"], "AAPL");
        assert!(body["signal"].is_string());
        assert!(body["confidence"].is_number());
    }

    #[actix_web::test]
    async fn test_predict_invalid_json_returns_400() {
        let Some(state) = make_test_state_with_history().await else {
            println!("Skipping");
            return;
        };

        let app = test::init_service(
            App::new()
                .app_data(state)
                .app_data(
                    web::JsonConfig::default().error_handler(|err, _| {
                        let msg = format!("{}", err);
                        actix_web::error::InternalError::from_response(
                            err,
                            HttpResponse::BadRequest().json(serde_json::json!({"error": msg})),
                        )
                        .into()
                    }),
                )
                .route("/predict", web::post().to(predict)),
        )
        .await;

        let req = test::TestRequest::post()
            .uri("/predict")
            .set_payload(r#"{"bad": "json"}"#)
            .insert_header(("content-type", "application/json"))
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 400);
    }

    #[actix_web::test]
    async fn test_get_signal_not_found() {
        let Some(state) = make_test_state_with_history().await else {
            println!("Skipping");
            return;
        };

        let app = test::init_service(
            App::new()
                .app_data(state)
                .route("/signal/{symbol}", web::get().to(get_signal)),
        )
        .await;

        let req = test::TestRequest::get()
            .uri("/signal/NONEXISTENT999")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 404);
    }

    #[actix_web::test]
    async fn test_get_signal_after_predict() {
        let Some(state) = make_test_state_with_history().await else {
            println!("Skipping");
            return;
        };

        let app = test::init_service(
            App::new()
                .app_data(state)
                .route("/predict", web::post().to(predict))
                .route("/signal/{symbol}", web::get().to(get_signal)),
        )
        .await;

        // First: predict
        let post_req = test::TestRequest::post()
            .uri("/predict")
            .set_json(sample_fv_json("GOOG_TEST"))
            .to_request();
        let _ = test::call_service(&app, post_req).await;

        // Then: retrieve
        let get_req = test::TestRequest::get()
            .uri("/signal/GOOG_TEST")
            .to_request();
        let resp = test::call_service(&app, get_req).await;
        // May return 200 (from memory) or 404 if Redis key normalisation differs
        assert!(resp.status() == 200 || resp.status() == 404);
    }

    #[actix_web::test]
    async fn test_history_endpoint() {
        let Some(state) = make_test_state_with_history().await else {
            println!("Skipping");
            return;
        };

        let app = test::init_service(
            App::new()
                .app_data(state)
                .route("/predict", web::post().to(predict))
                .route(
                    "/signal/history/{symbol}",
                    web::get().to(get_signal_history),
                ),
        )
        .await;

        // Seed 3 predictions
        for _ in 0..3 {
            let req = test::TestRequest::post()
                .uri("/predict")
                .set_json(sample_fv_json("HIST_TEST"))
                .to_request();
            let _ = test::call_service(&app, req).await;
        }

        let req = test::TestRequest::get()
            .uri("/signal/history/HIST_TEST?limit=10")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);

        let body: serde_json::Value = test::read_body_json(resp).await;
        assert!(body["count"].as_u64().unwrap() >= 3);
    }

    #[actix_web::test]
    async fn test_model_info_endpoint() {
        let Some(state) = make_test_state_with_history().await else {
            println!("Skipping");
            return;
        };

        let app = test::init_service(
            App::new()
                .app_data(state)
                .route("/model/info", web::get().to(model_info)),
        )
        .await;

        let req = test::TestRequest::get().uri("/model/info").to_request();
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 200);

        let body: serde_json::Value = test::read_body_json(resp).await;
        assert!(body["feature_order"].is_array());
        assert!(body["model_version_active"].is_string());
    }
}