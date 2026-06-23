use actix_web::{get, web, HttpResponse, Responder};
use parking_lot::Mutex;
use serde::Serialize;

use crate::redis::RedisClient;

/// Shared Redis client injected via Actix `web::Data`.
///
/// `Mutex<RedisClient>` because `ConnectionManager` is not `Sync`.
/// `parking_lot::Mutex` is chosen over `std::sync::Mutex` for lower overhead
/// under light contention (API calls are infrequent vs. tick throughput).
pub type SharedRedis = web::Data<Mutex<RedisClient>>;

// ── Error shape ───────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

fn not_found(symbol: &str) -> HttpResponse {
    HttpResponse::NotFound().json(ErrorResponse {
        error: format!("No features found for symbol '{symbol}'"),
    })
}

// ── GET /features/{symbol} ────────────────────────────────────────────────────

/// Returns the latest feature vector for `{symbol}`.
///
/// Response: `LatestFeatures` (all fields as strings matching Redis Hash)
/// or 404 if the symbol has never been seen.
#[get("/features/{symbol}")]
pub async fn get_latest_features(
    symbol: web::Path<String>,
    redis: SharedRedis,
) -> impl Responder {
    let symbol = symbol.into_inner().to_uppercase();
    let result = {
        let mut client = redis.lock();
        client.get_latest(&symbol).await
    };

    match result {
        Ok(Some(lf)) => HttpResponse::Ok().json(lf),
        Ok(None) => not_found(&symbol),
        Err(e) => HttpResponse::InternalServerError().json(ErrorResponse {
            error: format!("Redis error: {e}"),
        }),
    }
}

// ── GET /features/{symbol}/history ───────────────────────────────────────────

/// Returns the last 100 feature vectors for `{symbol}`, newest-first.
///
/// Response: JSON array of `FeatureVector` objects.
#[get("/features/{symbol}/history")]
pub async fn get_feature_history(
    symbol: web::Path<String>,
    redis: SharedRedis,
) -> impl Responder {
    let symbol = symbol.into_inner().to_uppercase();
    let result = {
        let mut client = redis.lock();
        client.get_history(&symbol, 100).await
    };

    match result {
        Ok(history) => {
            if history.is_empty() {
                not_found(&symbol)
            } else {
                HttpResponse::Ok().json(history)
            }
        }
        Err(e) => HttpResponse::InternalServerError().json(ErrorResponse {
            error: format!("Redis error: {e}"),
        }),
    }
}

// ── GET /features/{symbol}/stats ─────────────────────────────────────────────

/// Returns aggregate statistics over the last 100 feature vectors.
///
/// Response:
/// ```json
/// {
///   "mean_volatility": 0.0012,
///   "mean_momentum":   0.00045,
///   "mean_obi":        0.15,
///   "avg_spread":      1.5,
///   "sample_count":    87
/// }
/// ```
#[get("/features/{symbol}/stats")]
pub async fn get_feature_stats(
    symbol: web::Path<String>,
    redis: SharedRedis,
) -> impl Responder {
    let symbol = symbol.into_inner().to_uppercase();
    let result = {
        let mut client = redis.lock();
        client.get_stats(&symbol).await
    };

    match result {
        Ok(stats) => {
            if stats.sample_count == 0 {
                not_found(&symbol)
            } else {
                HttpResponse::Ok().json(stats)
            }
        }
        Err(e) => HttpResponse::InternalServerError().json(ErrorResponse {
            error: format!("Redis error: {e}"),
        }),
    }
}

// ── GET /health ───────────────────────────────────────────────────────────────

#[get("/health")]
pub async fn health() -> impl Responder {
    HttpResponse::Ok().body("OK")
}