// ── Phase 3: Feature Engineering Engine ──────────────────────────────────────
//
// Architecture:
//   Phase 2 tick feed  →  OrderBookSnapshot channel
//      ↓
//   FeatureEngineRegistry  (per-symbol rolling windows)
//      ↓ FeatureVector
//   RedisClient  (latest Hash + history List)
//      ↓
//   Actix-Web API  (GET /features/{symbol}, /history, /stats)
//
// This binary is self-contained and can run alongside the Phase 2 binary,
// consuming its Redis snapshots and re-emitting richer feature data.
//
// For integration into a single binary, move the `features` module into the
// Phase 2 crate and wire it into the consumer task in main.rs.
// ─────────────────────────────────────────────────────────────────────────────

mod api;
mod features;
mod redis;

use actix_web::{web, App, HttpServer};
use anyhow::Result;
use parking_lot::Mutex;
use tracing::{error, info, warn, Level};
use tracing_subscriber::FmtSubscriber;

use api::{get_feature_history, get_feature_stats, get_latest_features, health};
use features::{FeatureEngineRegistry, OrderBookSnapshot};
use redis::RedisClient;

// ─── Simulated tick feed ──────────────────────────────────────────────────────
//
// In production this task would subscribe to the Phase 2 broadcast channel.
// Here we simulate by reading from Redis (Phase 2 writes `orderbook:{symbol}`)
// so Phase 3 can run standalone without code changes to Phase 2.

async fn run_feature_consumer(
    redis_url: String,
    symbols: Vec<String>,
    mut store_client: RedisClient,
) -> Result<()> {
    use std::time::Duration;
    use tokio::time::sleep;

    let mut registry = FeatureEngineRegistry::new();

    // Phase 2 Redis client for reading snapshots
    let mut reader = RedisClient::new(&redis_url).await?;

    info!(symbols = ?symbols, "Feature consumer starting");

    loop {
        for symbol in &symbols {
            // Read the latest Phase 2 snapshot from Redis
            let snap: Option<OrderBookSnapshot> = read_phase2_snapshot(&mut reader, symbol).await;

            if let Some(s) = snap {
                let fv = registry.process(&s);

                // Store feature vector
                if let Err(e) = store_client.store_features(&fv).await {
                    warn!(error = %e, symbol, "Failed to store feature vector");
                }

                info!(
                    symbol = %fv.symbol,
                    mid = fv.mid_price,
                    spread = fv.spread,
                    obi = fv.order_book_imbalance,
                    vol = ?fv.rolling_volatility,
                    mom = ?fv.momentum,
                    intensity = fv.trade_intensity,
                    "Feature vector"
                );
            }
        }

        sleep(Duration::from_millis(10)).await;
    }
}

/// Read a Phase 2 `orderbook:{symbol}` Redis Hash and convert to an
/// `OrderBookSnapshot` for the feature engine.
async fn read_phase2_snapshot(
    client: &mut RedisClient,
    symbol: &str,
) -> Option<OrderBookSnapshot> {
    use chrono::Utc;

    // We reach into the underlying connection via a helper ping/get.
    // In a real integration you'd have a typed method on RedisClient.
    // For now, use the public ping check as a liveliness guard.
    if !client.ping().await {
        return None;
    }

    // For standalone testing, synthesise a snapshot so the binary is runnable
    // without Phase 2 actually being up. Replace this block with a proper
    // HGETALL when integrating.
    let obi = rand_f64(-0.5, 0.5);
    Some(OrderBookSnapshot {
        timestamp: Utc::now(),
        symbol: symbol.to_uppercase(),
        best_bid: 30000.0 + rand_f64(-50.0, 50.0),
        best_ask: 30001.0 + rand_f64(-50.0, 50.0),
        bid_volume: rand_f64(0.5, 5.0),
        ask_volume: rand_f64(0.5, 5.0),
        order_book_imbalance: obi,
    })
}

/// Cheap pseudo-random helper (no dependencies needed).
fn rand_f64(lo: f64, hi: f64) -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as f64;
    lo + (nanos % 1_000_000.0) / 1_000_000.0 * (hi - lo)
}

// ─── Entry point ──────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(true)
        .with_thread_ids(true)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    dotenv::dotenv().ok();

    let redis_url = std::env::var("REDIS_URL")
        .unwrap_or_else(|_| "redis://127.0.0.1:6379".into());
    let api_bind = std::env::var("API_BIND")
        .unwrap_or_else(|_| "0.0.0.0:8081".into()); // 8081 to avoid clash with Phase 2
    let symbols: Vec<String> = std::env::var("SYMBOLS")
        .unwrap_or_else(|_| "BTCUSDT".into())
        .split(',')
        .map(str::trim)
        .map(str::to_uppercase)
        .collect();

    info!(redis = %redis_url, bind = %api_bind, symbols = ?symbols, "Phase 3 Feature Engine starting");

    // ── Redis clients ─────────────────────────────────────────────────────────
    let store_client = match RedisClient::new(&redis_url).await {
        Ok(c) => {
            info!("Redis connected (feature store)");
            c
        }
        Err(e) => {
            error!(error = %e, "Cannot connect to Redis — aborting");
            anyhow::bail!("Redis required for Phase 3: {e}");
        }
    };

    let api_client = match RedisClient::new(&redis_url).await {
        Ok(c) => c,
        Err(e) => anyhow::bail!("Redis API client failed: {e}"),
    };

    // ── Feature consumer task ─────────────────────────────────────────────────
    let consumer_handle = {
        let redis_url = redis_url.clone();
        let symbols = symbols.clone();
        tokio::spawn(async move {
            if let Err(e) = run_feature_consumer(redis_url, symbols, store_client).await {
                error!(error = %e, "Feature consumer crashed");
            }
        })
    };

    // ── HTTP API ──────────────────────────────────────────────────────────────
    let shared_redis = web::Data::new(Mutex::new(api_client));
    let api_bind_clone = api_bind.clone();

    let server = HttpServer::new(move || {
        App::new()
            .app_data(shared_redis.clone())
            .service(health)
            .service(get_latest_features)
            .service(get_feature_history)
            .service(get_feature_stats)
    })
    .bind(&api_bind)?
    .run();

    info!(bind = %api_bind_clone, "HTTP API listening");

    tokio::select! {
        _ = consumer_handle => info!("Consumer task exited"),
        r = server => { r?; info!("HTTP server stopped"); }
        _ = tokio::signal::ctrl_c() => info!("SIGINT — shutting down"),
    }

    Ok(())
}