use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Derived market-microstructure metrics for a single snapshot.
///
/// All fields are pre-computed by `OrderBook::compute_metrics()` so
/// callers never have to hold a lock while iterating levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookMetrics {
    pub bid_volume: Decimal,
    pub ask_volume: Decimal,
    pub total_liquidity: Decimal,
    /// Order-book imbalance in [-1, +1].
    /// +1 = all volume on bid side, -1 = all on ask side.
    pub obi: Decimal,
    pub spread: Decimal,
    pub mid_price: Decimal,
    pub best_bid: Decimal,
    pub best_ask: Decimal,
}