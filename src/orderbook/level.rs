use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// A single price level in the order book.
///
/// Kept as a plain struct — no heap allocation, no Box.
/// Cloning is O(1) (two Decimal = two i128 + scale byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: Decimal,
    pub quantity: Decimal,
}

impl PriceLevel {
    #[inline]
    pub fn new(price: Decimal, quantity: Decimal) -> Self {
        Self { price, quantity }
    }

    #[inline]
    pub fn notional(&self) -> Decimal {
        self.price * self.quantity
    }
}