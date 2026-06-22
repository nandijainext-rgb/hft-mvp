pub mod handler;
pub mod lob_adapter;
pub mod tick;

// Public API for this module — only export what is used right now.
// Tick will be re-exported here when Phase 2+ modules consume it.
pub use handler::{MarketDataConfig, MarketDataHandler};