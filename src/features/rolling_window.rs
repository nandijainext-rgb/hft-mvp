use std::collections::VecDeque;

/// Generic fixed-capacity rolling window backed by a `VecDeque`.
///
/// All operations are O(1) except `variance()` / `std_dev()` which are O(n),
/// but n is bounded by `capacity` (20, 50, or 100) so this is effectively O(1)
/// in practice for our use case.
///
/// No heap re-allocation after initialisation (VecDeque pre-allocates to capacity).
#[derive(Debug, Clone)]
pub struct RollingWindow {
    buf: VecDeque<f64>,
    capacity: usize,
    /// Running sum for O(1) mean — maintained on every push/evict.
    sum: f64,
}

impl RollingWindow {
    /// Create a new window with the given fixed capacity.
    #[inline]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RollingWindow capacity must be > 0");
        Self {
            buf: VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
        }
    }

    // ── Mutation ──────────────────────────────────────────────────────────────

    /// Push a new value. If the window is full, the oldest value is evicted.
    #[inline]
    pub fn push(&mut self, value: f64) {
        if self.buf.len() == self.capacity {
            // Evict oldest
            if let Some(old) = self.buf.pop_front() {
                self.sum -= old;
            }
        }
        self.buf.push_back(value);
        self.sum += value;
    }

    // ── State ─────────────────────────────────────────────────────────────────

    /// Number of values currently in the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// True if the window contains no values.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// True if the window has been filled to capacity at least once.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.buf.len() == self.capacity
    }

    /// The configured maximum capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Most recently pushed value, or None if empty.
    #[inline]
    pub fn latest(&self) -> Option<f64> {
        self.buf.back().copied()
    }

    /// Oldest value still in the window, or None if empty.
    #[inline]
    pub fn oldest(&self) -> Option<f64> {
        self.buf.front().copied()
    }

    /// Arithmetic mean of all values in the window. O(1).
    pub fn mean(&self) -> Option<f64> {
        if self.buf.is_empty() {
            return None;
        }
        Some(self.sum / self.buf.len() as f64)
    }

    /// Population variance. O(n) — but n ≤ capacity (small constant).
    pub fn variance(&self) -> Option<f64> {
        let n = self.buf.len();
        if n < 2 {
            return None;
        }
        let mean = self.sum / n as f64;
        let var = self.buf.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        Some(var)
    }

    /// Population standard deviation. O(n).
    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }

    /// Minimum value in the window. O(n).
    pub fn min(&self) -> Option<f64> {
        self.buf.iter().copied().reduce(f64::min)
    }

    /// Maximum value in the window. O(n).
    pub fn max(&self) -> Option<f64> {
        self.buf.iter().copied().reduce(f64::max)
    }

    /// Value at index 0 (oldest end). Useful for momentum calculation.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<f64> {
        self.buf.get(idx).copied()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_len() {
        let mut w = RollingWindow::new(5);
        assert!(w.is_empty());
        w.push(1.0);
        w.push(2.0);
        assert_eq!(w.len(), 2);
        assert!(!w.is_full());
    }

    #[test]
    fn eviction_keeps_capacity() {
        let mut w = RollingWindow::new(3);
        for i in 1..=6 {
            w.push(i as f64);
        }
        assert_eq!(w.len(), 3);
        // Should contain 4, 5, 6
        assert_eq!(w.oldest(), Some(4.0));
        assert_eq!(w.latest(), Some(6.0));
    }

    #[test]
    fn mean_is_correct() {
        let mut w = RollingWindow::new(4);
        for v in [2.0, 4.0, 6.0, 8.0] {
            w.push(v);
        }
        assert!((w.mean().unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn mean_updates_on_eviction() {
        let mut w = RollingWindow::new(3);
        w.push(1.0);
        w.push(2.0);
        w.push(3.0); // mean = 2
        w.push(4.0); // evicts 1, mean = (2+3+4)/3
        let expected = (2.0 + 3.0 + 4.0) / 3.0;
        assert!((w.mean().unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn variance_and_std_dev() {
        let mut w = RollingWindow::new(4);
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            w.push(v);
        }
        // Window is [5, 5, 7, 9] after eviction (capacity 4)
        // mean = 6.5, var = ((5-6.5)^2 + (5-6.5)^2 + (7-6.5)^2 + (9-6.5)^2) / 4
        let mean = 6.5_f64;
        let expected_var = ((5.0 - mean).powi(2)
            + (5.0 - mean).powi(2)
            + (7.0 - mean).powi(2)
            + (9.0 - mean).powi(2))
            / 4.0;
        let got_var = w.variance().unwrap();
        assert!((got_var - expected_var).abs() < 1e-10, "var mismatch: {got_var}");
        assert!((w.std_dev().unwrap() - expected_var.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn min_and_max() {
        let mut w = RollingWindow::new(5);
        for v in [3.0, 1.0, 4.0, 1.0, 5.0] {
            w.push(v);
        }
        assert_eq!(w.min(), Some(1.0));
        assert_eq!(w.max(), Some(5.0));
    }

    #[test]
    fn empty_returns_none() {
        let w = RollingWindow::new(10);
        assert!(w.mean().is_none());
        assert!(w.variance().is_none());
        assert!(w.std_dev().is_none());
        assert!(w.min().is_none());
        assert!(w.max().is_none());
        assert!(w.latest().is_none());
    }

    #[test]
    fn single_element_variance_is_none() {
        let mut w = RollingWindow::new(5);
        w.push(42.0);
        // Need at least 2 values for variance
        assert!(w.variance().is_none());
    }

    #[test]
    fn get_by_index() {
        let mut w = RollingWindow::new(3);
        w.push(10.0);
        w.push(20.0);
        w.push(30.0);
        assert_eq!(w.get(0), Some(10.0));
        assert_eq!(w.get(2), Some(30.0));
        assert_eq!(w.get(5), None);
    }
}