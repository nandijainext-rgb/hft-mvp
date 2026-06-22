// ── Active modules (Phase 1) ──────────────────────────────────────────────────
mod market_data;

// ── Future phases — uncomment as each phase is implemented ───────────────────
// mod orderbook;   // Phase 2
// mod features;    // Phase 3
// mod signal;      // Phase 5
// mod risk;        // Phase 6
// mod execution;   // Phase 7
// mod pnl;         // Phase 8
// mod db;          // Phase 9
// mod api;         // Phase 10

use anyhow::Result;
use market_data::{MarketDataConfig, MarketDataHandler};
use tracing::{error, info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // ── Logging ──────────────────────────────────────────────────────────────
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::DEBUG)
        .with_target(true)
        .with_thread_ids(true)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    dotenv::dotenv().ok();

    // Show working directory so CSV path issues are immediately visible
    let cwd = std::env::current_dir().unwrap_or_default();
    info!("HFT Simulation Engine starting — Phase 1 (Market Data)");
    info!(cwd = %cwd.display(), "Working directory");

    // ── Market Data config ────────────────────────────────────────────────────
    let tick_path = std::env::var("TICK_DATA_PATH")
        .unwrap_or_else(|_| "data/ticks.csv".into());
    let speed: f64 = std::env::var("PLAYBACK_SPEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100.0);

    info!(path = %tick_path, speed, "Market data config");

    let md_config = MarketDataConfig {
        csv_path: std::path::PathBuf::from(&tick_path),
        speed_multiplier: speed,
        loop_playback: true,
        min_tick_delay_us: 1_000,
    };

    let (handler, mut rx) = MarketDataHandler::new(md_config);

    // ── Spawn feed ────────────────────────────────────────────────────────────
    let feed_handle = tokio::spawn(async move {
        if let Err(e) = handler.run().await {
            error!(error = %e, "MarketDataHandler crashed");
        }
    });

    // ── Consume ticks (Phase 1: log every 100th tick) ─────────────────────────
    let consumer_handle = tokio::spawn(async move {
        let mut count = 0u64;
        loop {
            match rx.recv().await {
                Ok(tick) => {
                    count += 1;
                    if count == 1 || count % 100 == 0 {
                        info!(
                            count,
                            symbol = %tick.symbol,
                            bid    = %tick.bid_price,
                            ask    = %tick.ask_price,
                            mid    = %tick.mid_price(),
                            spread = %tick.spread(),
                            "Tick #{count}"
                        );
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(dropped = n, "Consumer lagged — ticks dropped");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    info!("Tick channel closed — consumer exiting");
                    break;
                }
            }
        }
    });

    // ── Wait for shutdown ─────────────────────────────────────────────────────
    tokio::select! {
        _ = feed_handle     => info!("Feed handle exited"),
        _ = consumer_handle => info!("Consumer handle exited"),
        _ = tokio::signal::ctrl_c() => info!("SIGINT — shutting down"),
    }

    info!("HFT Simulation Engine stopped");
    Ok(())
}