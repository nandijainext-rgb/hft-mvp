use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::broadcast;
use tokio::time::sleep;
use tracing::{debug, info, warn}; // `error` removed — not used here

use super::tick::{Tick, TickCsvRow};

const CHANNEL_CAPACITY: usize = 1_024;

pub type TickSender = broadcast::Sender<Arc<Tick>>;
pub type TickReceiver = broadcast::Receiver<Arc<Tick>>;

#[derive(Debug, Clone)]
pub struct MarketDataConfig {
    pub csv_path: PathBuf,
    pub speed_multiplier: f64,
    pub loop_playback: bool,
    pub min_tick_delay_us: u64,
}

impl Default for MarketDataConfig {
    fn default() -> Self {
        Self {
            csv_path: PathBuf::from("data/ticks.csv"),
            speed_multiplier: 100.0,
            loop_playback: true,
            min_tick_delay_us: 1_000,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct PlaybackStats {
    pub ticks_published: u64,
    pub ticks_skipped: u64,
    pub loop_count: u64,
}

pub struct MarketDataHandler {
    config: MarketDataConfig,
    sender: TickSender,
    stats: Arc<parking_lot::RwLock<PlaybackStats>>,
}

impl MarketDataHandler {
    pub fn new(config: MarketDataConfig) -> (Self, TickReceiver) {
        let (sender, receiver) = broadcast::channel(CHANNEL_CAPACITY);
        let handler = Self {
            config,
            sender,
            stats: Arc::new(parking_lot::RwLock::new(PlaybackStats::default())),
        };
        (handler, receiver)
    }

    /// Subscribe a new consumer to the tick stream
    #[allow(dead_code)]
    pub fn subscribe(&self) -> TickReceiver {
        self.sender.subscribe()
    }

    /// Clone the sender for use in other components (used from Phase 2 onward)
    #[allow(dead_code)]
    pub fn sender(&self) -> TickSender {
        self.sender.clone()
    }

    /// Read a snapshot of current playback stats
    #[allow(dead_code)]
    pub fn stats(&self) -> PlaybackStats {
        self.stats.read().clone()
    }

    pub async fn run(self) -> Result<()> {
        info!(
            path = %self.config.csv_path.display(),
            speed = self.config.speed_multiplier,
            looping = self.config.loop_playback,
            "MarketDataHandler starting"
        );

        let resolved = resolve_csv_path(&self.config.csv_path);
        info!(resolved = %resolved.display(), "Resolved CSV path");

        if !resolved.exists() {
            anyhow::bail!(
                "CSV file not found at '{}'. \
                 Run cargo from your project root (the folder that contains /data/ticks.csv), \
                 or set TICK_DATA_PATH to the absolute path of your CSV.",
                resolved.display()
            );
        }

        let config = MarketDataConfig {
            csv_path: resolved,
            ..self.config
        };

        loop {
            let ticks = load_csv(&config.csv_path)
                .await
                .with_context(|| {
                    format!("Failed to load CSV from '{}'", config.csv_path.display())
                })?;

            if ticks.is_empty() {
                warn!(
                    "CSV produced zero valid ticks. \
                     Check that column names match and run with RUST_LOG=debug to see the headers."
                );
                sleep(Duration::from_secs(5)).await;
                continue;
            }

            info!(count = ticks.len(), "Loaded tick file, beginning playback");
            play_ticks(&self.sender, &self.stats, &ticks, &config).await?;

            {
                let mut stats = self.stats.write();
                stats.loop_count += 1;
            }

            if !config.loop_playback {
                info!("Playback complete. MarketDataHandler exiting.");
                break;
            }

            info!("Loop complete — restarting from beginning");
        }

        Ok(())
    }
}

/// Try the given path; if relative and not found, probe next to the binary too.
fn resolve_csv_path(path: &PathBuf) -> PathBuf {
    if path.is_absolute() {
        return path.clone();
    }
    if path.exists() {
        return path.clone();
    }
    if let Ok(exe) = std::env::current_exe() {
        let candidate = exe
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join(path);
        if candidate.exists() {
            return candidate;
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let candidate = cwd.join(path);
        if candidate.exists() {
            return candidate;
        }
    }
    path.clone()
}

async fn load_csv(path: &PathBuf) -> Result<Vec<Tick>> {
    let path = path.clone();
    tokio::task::spawn_blocking(move || {
        info!(path = %path.display(), "Opening CSV");

        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All)
            .from_path(&path)
            .with_context(|| format!("Cannot open '{}'", path.display()))?;

        // Print headers at DEBUG level so you can see the exact column names
        let headers = rdr.headers()?.clone();
        info!("CSV headers detected: {:?}", headers.iter().collect::<Vec<_>>());

        let mut ticks: Vec<Tick> = Vec::new();
        let mut row_num = 0usize;
        let mut skipped = 0usize;

        for result in rdr.deserialize::<TickCsvRow>() {
            row_num += 1;
            match result {
                Ok(row) => match Tick::try_from(row) {
                    Ok(tick) => ticks.push(tick),
                    Err(e) => {
                        skipped += 1;
                        if skipped <= 5 {
                            warn!(row = row_num, error = %e, "Skipping invalid tick");
                        }
                    }
                },
                Err(e) => {
                    skipped += 1;
                    if skipped <= 5 {
                        warn!(
                            row = row_num,
                            error = %e,
                            "CSV parse error — column name mismatch? Check headers above."
                        );
                    }
                }
            }
        }

        if skipped > 0 {
            warn!(total_skipped = skipped, total_rows = row_num, "Some rows were skipped");
        }

        ticks.sort_by_key(|t| t.timestamp);
        info!(valid = ticks.len(), skipped, "CSV load complete");
        Ok(ticks)
    })
    .await?
}

async fn play_ticks(
    sender: &TickSender,
    stats: &Arc<parking_lot::RwLock<PlaybackStats>>,
    ticks: &[Tick],
    config: &MarketDataConfig,
) -> Result<()> {
    let min_delay = Duration::from_micros(config.min_tick_delay_us);

    for window in ticks.windows(2) {
        let current = &window[0];
        let next = &window[1];

        publish(sender, stats, current);

        let delta_ns = next
            .timestamp
            .signed_duration_since(current.timestamp)
            .num_nanoseconds()
            .unwrap_or(0)
            .max(0) as f64;

        let scaled_ns = (delta_ns / config.speed_multiplier) as u64;
        let delay = Duration::from_nanos(scaled_ns).max(min_delay);

        debug!(
            symbol = %current.symbol,
            ts     = %current.timestamp,
            // FIX: bid_price and ask_price are FIELDS, not methods — no ()
            bid    = %current.bid_price,
            ask    = %current.ask_price,
            mid    = %current.mid_price(),   // mid_price() IS a method
            delay_ms = delay.as_millis(),
        );

        sleep(delay).await;
    }

    if let Some(last) = ticks.last() {
        publish(sender, stats, last);
    }

    Ok(())
}

fn publish(
    sender: &TickSender,
    stats: &Arc<parking_lot::RwLock<PlaybackStats>>,
    tick: &Tick,
) {
    let arc_tick = Arc::new(tick.clone());
    match sender.send(arc_tick) {
        Ok(_) => {
            stats.write().ticks_published += 1;
        }
        Err(_) => {
            // No active receivers yet — normal during startup
            stats.write().ticks_skipped += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    const SAMPLE_CSV: &str =
        "timestamp,symbol,bid_price,bid_size,ask_price,ask_size,last_trade_price,last_trade_size\n\
         2024-01-15 09:30:00.000,BTCUSDT,30000.00,1.5,30001.00,0.8,30000.50,0.2\n\
         2024-01-15 09:30:00.100,BTCUSDT,30001.00,2.0,30002.00,1.2,30001.50,0.3\n\
         2024-01-15 09:30:00.200,BTCUSDT,30002.00,0.9,30003.00,1.8,30002.50,0.1\n\
         2024-01-15 09:30:00.300,BTCUSDT,30001.50,1.1,30002.50,0.7,30002.00,0.4\n\
         2024-01-15 09:30:00.400,BTCUSDT,30003.00,0.6,30004.00,0.9,30003.50,0.5\n";

    fn temp_csv(content: &str) -> (NamedTempFile, PathBuf) {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        let path = f.path().to_path_buf();
        (f, path)
    }

    fn fast_config(path: PathBuf) -> MarketDataConfig {
        MarketDataConfig {
            csv_path: path,
            loop_playback: false,
            speed_multiplier: 1_000_000.0,
            min_tick_delay_us: 0,
        }
    }

    async fn drain(mut rx: TickReceiver) -> Vec<Tick> {
        let mut out = Vec::new();
        while let Ok(tick) =
            tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .unwrap_or(Err(broadcast::error::RecvError::Closed))
        {
            out.push((*tick).clone());
        }
        out
    }

    #[tokio::test]
    async fn test_loads_correct_tick_count() {
        let (_f, path) = temp_csv(SAMPLE_CSV);
        let (handler, rx) = MarketDataHandler::new(fast_config(path));
        tokio::spawn(async move { handler.run().await });
        assert_eq!(drain(rx).await.len(), 5);
    }

    #[tokio::test]
    async fn test_ticks_are_chronological() {
        let (_f, path) = temp_csv(SAMPLE_CSV);
        let (handler, rx) = MarketDataHandler::new(fast_config(path));
        tokio::spawn(async move { handler.run().await });
        let ticks = drain(rx).await;
        for w in ticks.windows(2) {
            assert!(w[0].timestamp <= w[1].timestamp);
        }
    }

    #[tokio::test]
    async fn test_multiple_subscribers() {
        let (_f, path) = temp_csv(SAMPLE_CSV);
        let (handler, rx1) = MarketDataHandler::new(fast_config(path));
        let rx2 = handler.subscribe();
        tokio::spawn(async move { handler.run().await });
        let (c1, c2) = tokio::join!(drain(rx1), drain(rx2));
        assert_eq!(c1.len(), 5);
        assert_eq!(c2.len(), 5);
    }

    #[tokio::test]
    async fn test_missing_file_returns_clear_error() {
        let config = MarketDataConfig {
            csv_path: PathBuf::from("C:\\nonexistent\\path\\ticks.csv"),
            loop_playback: false,
            speed_multiplier: 1_000_000.0,
            min_tick_delay_us: 0,
        };
        let (handler, _rx) = MarketDataHandler::new(config);
        let result = handler.run().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }
}