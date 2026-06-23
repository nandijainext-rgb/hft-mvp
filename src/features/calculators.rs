/// Pure feature calculation functions.
///
/// Each function takes the minimal inputs it needs and returns `Option<f64>`
/// to signal "not enough data yet" rather than panicking or returning NaN.
///
/// All are `#[inline]` because they are called in the hot path
/// (once per tick, potentially 100k+/sec).

// ── Feature 1: Order Book Imbalance ──────────────────────────────────────────

/// OBI = (bid_vol - ask_vol) / (bid_vol + ask_vol)  ∈ [-1, +1]
///
/// +1 = all volume on bid side (bullish pressure)
/// -1 = all volume on ask side (selling pressure)
/// 0  = balanced book
#[inline]
pub fn order_book_imbalance(bid_vol: f64, ask_vol: f64) -> Option<f64> {
    let total = bid_vol + ask_vol;
    if total == 0.0 {
        return None;
    }
    Some((bid_vol - ask_vol) / total)
}

// ── Feature 2: Spread ─────────────────────────────────────────────────────────

/// spread = best_ask - best_bid
///
/// Returns None if the result would be negative (crossed book guard).
#[inline]
pub fn spread(best_bid: f64, best_ask: f64) -> Option<f64> {
    let s = best_ask - best_bid;
    if s < 0.0 { None } else { Some(s) }
}

// ── Feature 3: Mid Price ──────────────────────────────────────────────────────

/// mid = (best_bid + best_ask) / 2
#[inline]
pub fn mid_price(best_bid: f64, best_ask: f64) -> f64 {
    (best_bid + best_ask) / 2.0
}

// ── Feature 4: Rolling Volatility ────────────────────────────────────────────

/// Compute the next mid-price return and return it (caller feeds to RollingWindow).
///
/// return_t = (mid_t - mid_{t-1}) / mid_{t-1}
///
/// Returns None if `prev_mid` is zero (avoids division by zero).
#[inline]
pub fn mid_price_return(mid: f64, prev_mid: f64) -> Option<f64> {
    if prev_mid == 0.0 {
        return None;
    }
    Some((mid - prev_mid) / prev_mid)
}

// ── Feature 5: Momentum ───────────────────────────────────────────────────────

/// momentum = (mid_now - mid_n_ticks_ago) / mid_n_ticks_ago
///
/// Returns None if `mid_n_ticks_ago` is zero or unavailable.
#[inline]
pub fn momentum(mid_now: f64, mid_n_ticks_ago: f64) -> Option<f64> {
    if mid_n_ticks_ago == 0.0 {
        return None;
    }
    Some((mid_now - mid_n_ticks_ago) / mid_n_ticks_ago)
}

// ── Feature 6: Liquidity Ratio ────────────────────────────────────────────────

/// liquidity_ratio = bid_vol / ask_vol
///
/// Returns 0.0 when ask_vol is zero to avoid division-by-zero.
/// A ratio > 1 means more bid depth than ask depth.
#[inline]
pub fn liquidity_ratio(bid_vol: f64, ask_vol: f64) -> f64 {
    if ask_vol == 0.0 {
        return 0.0;
    }
    bid_vol / ask_vol
}

// ── Feature 7: Total Liquidity ────────────────────────────────────────────────

/// total_liquidity = bid_vol + ask_vol
#[inline]
pub fn total_liquidity(bid_vol: f64, ask_vol: f64) -> f64 {
    bid_vol + ask_vol
}

// ── Feature 8: Volume Imbalance ───────────────────────────────────────────────

/// volume_imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
///
/// Equivalent to OBI but computed from level-1 volumes only.
/// Returns 0.0 when total is zero.
#[inline]
pub fn volume_imbalance(bid_vol: f64, ask_vol: f64) -> f64 {
    order_book_imbalance(bid_vol, ask_vol).unwrap_or(0.0)
}

// ── Feature 9: Trade Intensity ────────────────────────────────────────────────

/// trade_intensity = count of updates within the last 1 second
///
/// The count is maintained externally by FeatureEngine.
/// This function is here to document the formula and allow unit tests.
#[inline]
pub fn trade_intensity_from_count(updates_in_window: usize) -> f64 {
    updates_in_window as f64
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── OBI ───────────────────────────────────────────────────────────────────

    #[test]
    fn obi_balanced() {
        assert_eq!(order_book_imbalance(1.0, 1.0), Some(0.0));
    }

    #[test]
    fn obi_bid_heavy() {
        // (3 - 1) / (3 + 1) = 0.5
        assert!((order_book_imbalance(3.0, 1.0).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn obi_ask_heavy() {
        // (1 - 3) / (1 + 3) = -0.5
        assert!((order_book_imbalance(1.0, 3.0).unwrap() - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn obi_zero_total_returns_none() {
        assert_eq!(order_book_imbalance(0.0, 0.0), None);
    }

    #[test]
    fn obi_bounds() {
        // Must be in [-1, +1]
        let v = order_book_imbalance(999.0, 0.001).unwrap();
        assert!(v > 0.0 && v <= 1.0, "got {v}");
    }

    // ── Spread ────────────────────────────────────────────────────────────────

    #[test]
    fn spread_normal() {
        assert_eq!(spread(30000.0, 30001.0), Some(1.0));
    }

    #[test]
    fn spread_crossed_book_is_none() {
        assert_eq!(spread(30002.0, 30001.0), None);
    }

    #[test]
    fn spread_zero_is_ok() {
        assert_eq!(spread(30000.0, 30000.0), Some(0.0));
    }

    // ── Mid Price ─────────────────────────────────────────────────────────────

    #[test]
    fn mid_price_correct() {
        assert!((mid_price(30000.0, 30002.0) - 30001.0).abs() < 1e-10);
    }

    // ── Returns ───────────────────────────────────────────────────────────────

    #[test]
    fn mid_price_return_correct() {
        let r = mid_price_return(30001.0, 30000.0).unwrap();
        // (30001 - 30000) / 30000 ≈ 0.0000333
        assert!((r - 1.0 / 30000.0).abs() < 1e-12);
    }

    #[test]
    fn mid_price_return_zero_prev_is_none() {
        assert!(mid_price_return(100.0, 0.0).is_none());
    }

    // ── Momentum ──────────────────────────────────────────────────────────────

    #[test]
    fn momentum_positive() {
        // Price rose from 100 to 105 over 20 ticks
        let m = momentum(105.0, 100.0).unwrap();
        assert!((m - 0.05).abs() < 1e-10);
    }

    #[test]
    fn momentum_negative() {
        let m = momentum(95.0, 100.0).unwrap();
        assert!((m - (-0.05)).abs() < 1e-10);
    }

    #[test]
    fn momentum_flat() {
        assert_eq!(momentum(100.0, 100.0), Some(0.0));
    }

    #[test]
    fn momentum_zero_denom_is_none() {
        assert!(momentum(100.0, 0.0).is_none());
    }

    // ── Liquidity Ratio ───────────────────────────────────────────────────────

    #[test]
    fn liquidity_ratio_normal() {
        assert!((liquidity_ratio(2.0, 1.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn liquidity_ratio_zero_ask() {
        // Should not panic; returns 0.0
        assert_eq!(liquidity_ratio(5.0, 0.0), 0.0);
    }

    // ── Volume Imbalance ──────────────────────────────────────────────────────

    #[test]
    fn volume_imbalance_zero_total() {
        assert_eq!(volume_imbalance(0.0, 0.0), 0.0);
    }

    #[test]
    fn volume_imbalance_matches_obi() {
        let vi = volume_imbalance(3.0, 1.0);
        let obi = order_book_imbalance(3.0, 1.0).unwrap();
        assert!((vi - obi).abs() < 1e-10);
    }

    // ── Trade Intensity ───────────────────────────────────────────────────────

    #[test]
    fn trade_intensity_is_count_as_f64() {
        assert_eq!(trade_intensity_from_count(42), 42.0);
        assert_eq!(trade_intensity_from_count(0), 0.0);
    }
}