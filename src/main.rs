// src/main.rs
// Phase 5 - HFT AI Signal Engine
// Loads the ONNX model once, runs inference, publishes signals to Redis, and
// serves the REST API.

mod api;
mod features;
mod redis;
mod signal_engine;

use std::path::PathBuf;
use std::sync::Arc;

use actix_web::{
    http::Method,
    middleware::DefaultHeaders,
    web, App, HttpResponse, HttpServer, Responder,
};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use api::signal_handlers;
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenv::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(true)
        .with_thread_ids(true)
        .init();

    info!("HFT Signal Engine - Phase 5 starting");

    let ml_dir = std::env::var("ML_DIR").unwrap_or_else(|_| default_project_path("ml/models"));
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());

    info!("Loading ONNX model artifacts from: {}", ml_dir);
    let inference_engine = Arc::new(
        InferenceEngine::new(&ml_dir).map_err(|e| {
            error!("Failed to load ONNX model: {}", e);
            e
        })?,
    );
    info!("ONNX model loaded successfully");
    let model_metadata = Arc::new(ModelMetadata::load(&ml_dir));

    let prediction_store = Arc::new(PredictionStore::new(1000));

    info!("Connecting to Redis: {}", redis_url);
    let redis_client = Arc::new(RedisClient::new(&redis_url).await.map_err(|e| {
        error!("Failed to connect to Redis: {}", e);
        e
    })?);
    info!("Redis connected");

    let state = web::Data::new(AppState {
        inference_engine,
        prediction_store,
        redis_client,
        model_metadata,
    });

    info!("Starting HTTP server on {}", bind_addr);
    HttpServer::new(move || {
        App::new()
            .wrap(
                DefaultHeaders::new()
                    .add(("Access-Control-Allow-Origin", "*"))
                    .add(("Access-Control-Allow-Methods", "GET, POST, OPTIONS"))
                    .add(("Access-Control-Allow-Headers", "Content-Type")),
            )
            .app_data(state.clone())
            .app_data(web::JsonConfig::default().error_handler(|err, req| {
                let msg = format!("JSON parse error: {}", err);
                tracing::warn!("{} - path: {}", msg, req.path());
                actix_web::error::InternalError::from_response(
                    err,
                    actix_web::HttpResponse::BadRequest()
                        .json(serde_json::json!({ "error": msg })),
                )
                .into()
            }))
            .route("/health", web::get().to(signal_handlers::health))
            .route("/predict", web::post().to(signal_handlers::predict))
            .route("/signal/{symbol}", web::get().to(signal_handlers::get_signal))
            .route(
                "/signal/history/{symbol}",
                web::get().to(signal_handlers::get_signal_history),
            )
            .route("/model/info", web::get().to(signal_handlers::model_info))
            .route("/", web::get().to(frontend_index))
            .route("/index.html", web::get().to(frontend_index))
            .route("/script.js", web::get().to(frontend_script))
            .route("/style.css", web::get().to(frontend_style))
            .route(
                "/{tail:.*}",
                web::method(Method::OPTIONS).to(cors_preflight),
            )
    })
    .bind(&bind_addr)?
    .run()
    .await?;

    Ok(())
}

async fn cors_preflight() -> impl Responder {
    HttpResponse::NoContent().finish()
}

async fn frontend_index() -> impl Responder {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(include_str!("../frontend/index.html"))
}

async fn frontend_script() -> impl Responder {
    HttpResponse::Ok()
        .content_type("application/javascript; charset=utf-8")
        .body(include_str!("../frontend/script.js"))
}

async fn frontend_style() -> impl Responder {
    HttpResponse::Ok()
        .content_type("text/css; charset=utf-8")
        .body(include_str!("../frontend/style.css"))
}

fn default_project_path(relative_path: &str) -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(relative_path)
        .to_string_lossy()
        .into_owned()
}
