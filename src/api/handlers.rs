use actix_web::{get, web, HttpResponse, Responder};
use parking_lot::RwLock;
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;

use crate::orderbook::OrderBook;

/// Shared state injected into Actix-web via `web::Data`.
///
/// `Arc<RwLock<HashMap<…>>>` allows many concurrent readers (tick consumer +
/// HTTP handlers) with low contention — `parking_lot::RwLock` is
/// significantly faster than `std::sync::RwLock` under low-conflict workloads.
pub type BookRegistry = Arc<RwLock<HashMap<String, OrderBook>>>;

// ─── Response shapes ─────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct OrderBookResponse {
    pub best_bid:  String,
    pub best_ask:  String,
    pub spread:    String,
    pub mid_price: String,
    pub obi:       String,
    pub bids: Vec<LevelDto>,
    pub asks: Vec<LevelDto>,
}

#[derive(Serialize)]
pub struct LevelDto {
    pub price:    String,
    pub quantity: String,
}

#[derive(Serialize)]
pub struct MetricsResponse {
    pub bid_volume:      String,
    pub ask_volume:      String,
    pub total_liquidity: String,
    pub obi:             String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// ─── GET /orderbook/{symbol} ─────────────────────────────────────────────────

/// Returns the current best bid/ask, spread, mid-price, OBI and top-10 levels.
#[get("/orderbook/{symbol}")]
pub async fn get_orderbook(
    symbol: web::Path<String>,
    data: web::Data<BookRegistry>,
) -> impl Responder {
    let symbol = symbol.into_inner().to_uppercase();
    let registry = data.read();

    let Some(book) = registry.get(&symbol) else {
        return HttpResponse::NotFound().json(ErrorResponse {
            error: format!("No order book found for symbol '{symbol}'"),
        });
    };

    let Some(m) = book.compute_metrics() else {
        return HttpResponse::ServiceUnavailable().json(ErrorResponse {
            error: "Order book is empty (no ticks received yet)".into(),
        });
    };

    let bids = book
        .top_n_bids(10)
        .into_iter()
        .map(|l| LevelDto {
            price:    l.price.to_string(),
            quantity: l.quantity.to_string(),
        })
        .collect();

    let asks = book
        .top_n_asks(10)
        .into_iter()
        .map(|l| LevelDto {
            price:    l.price.to_string(),
            quantity: l.quantity.to_string(),
        })
        .collect();

    HttpResponse::Ok().json(OrderBookResponse {
        best_bid:  m.best_bid.to_string(),
        best_ask:  m.best_ask.to_string(),
        spread:    m.spread.to_string(),
        mid_price: m.mid_price.to_string(),
        obi:       m.obi.to_string(),
        bids,
        asks,
    })
}

// ─── GET /metrics/{symbol} ───────────────────────────────────────────────────

/// Returns volume and liquidity metrics for the given symbol.
#[get("/metrics/{symbol}")]
pub async fn get_metrics(
    symbol: web::Path<String>,
    data: web::Data<BookRegistry>,
) -> impl Responder {
    let symbol = symbol.into_inner().to_uppercase();
    let registry = data.read();

    let Some(book) = registry.get(&symbol) else {
        return HttpResponse::NotFound().json(ErrorResponse {
            error: format!("No order book found for symbol '{symbol}'"),
        });
    };

    let Some(m) = book.compute_metrics() else {
        return HttpResponse::ServiceUnavailable().json(ErrorResponse {
            error: "Order book is empty (no ticks received yet)".into(),
        });
    };

    HttpResponse::Ok().json(MetricsResponse {
        bid_volume:      m.bid_volume.to_string(),
        ask_volume:      m.ask_volume.to_string(),
        total_liquidity: m.total_liquidity.to_string(),
        obi:             m.obi.to_string(),
    })
}

// ─── GET /health ─────────────────────────────────────────────────────────────

/// Simple health-check used by Docker / load balancer.
#[get("/health")]
pub async fn health() -> impl Responder {
    HttpResponse::Ok().body("OK")
}