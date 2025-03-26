use anyhow::anyhow;
use reqwest::Client;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;
use tracing::trace;

type ProgressCallback = Arc<dyn Fn(u64, u64) + Send + Sync>;

pub async fn download_plugin(
  url: &str,
  plugin_dir: &Path,
  file_name: &str,
  cancel_token: Option<CancellationToken>,
  progress_callback: Option<ProgressCallback>,
  callback_debounce: Option<Duration>,
) -> Result<PathBuf, anyhow::Error> {
  let client = Client::new();
  let response = client.get(url).send().await?;

  if !response.status().is_success() {
    return Err(anyhow!("Failed to download file: {}", response.status()));
  }
  // Debounce settings
  let debounce_duration = callback_debounce.unwrap_or_else(|| Duration::from_millis(500));
  let mut last_update = Instant::now()
    .checked_sub(debounce_duration)
    .unwrap_or(Instant::now());

  let total_size = response
    .content_length()
    .ok_or(anyhow!("Failed to get content length"))?;

  // Create paths for the partial and final files
  let partial_path = plugin_dir.join(format!("{}.part", file_name));
  let final_path = plugin_dir.join(file_name);
  let mut part_file = File::create(&partial_path).await?;
  let mut stream = response.bytes_stream();
  let mut downloaded: u64 = 0;

  while let Some(chunk) = stream.next().await {
    if let Some(cancel_token) = &cancel_token {
      if cancel_token.is_cancelled() {
        trace!("Download canceled");
        fs::remove_file(&partial_path).await?;
        return Err(anyhow!("Download canceled"));
      }
    }

    let bytes = chunk?;
    part_file.write_all(&bytes).await?;
    downloaded += bytes.len() as u64;

    // Call the progress callback
    if let Some(progress_callback) = &progress_callback {
      let now = Instant::now();
      if now.duration_since(last_update) >= debounce_duration {
        progress_callback(downloaded, total_size);
        last_update = now;
      }
    }
  }

  // Ensure all data is written to disk
  part_file.sync_all().await?;

  // Move the temporary file to the final destination
  fs::rename(&partial_path, &final_path).await?;
  trace!("Plugin downloaded to {:?}", final_path);
  Ok(final_path)
}
