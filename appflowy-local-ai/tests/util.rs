use anyhow::Result;
use appflowy_local_ai::embedding_plugin::{EmbeddingPlugin, EmbeddingPluginConfig};
use appflowy_local_ai::ollama_plugin::{OllamaAIPlugin, OllamaPluginConfig};
use appflowy_plugin::error::PluginError;
use appflowy_plugin::manager::PluginManager;

use serde_json::{json, Value};
use simsimd::SpatialSimilarity;
use std::f64;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tracing_subscriber::fmt::Subscriber;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

pub struct LocalAITest {
  config: LocalAIConfiguration,
  pub ollama_plugin: OllamaAIPlugin,
  pub embedding_plugin: EmbeddingPlugin,
}

impl LocalAITest {
  pub fn new() -> Result<Self> {
    let config = LocalAIConfiguration::new()?;
    let sidecar = Arc::new(PluginManager::new());
    let ollama_plugin = OllamaAIPlugin::new(sidecar.clone());
    let embedding_plugin = EmbeddingPlugin::new(sidecar);
    Ok(Self {
      config,
      ollama_plugin,
      embedding_plugin,
    })
  }

  pub async fn init_chat_plugin(&self) {
    let mut config = OllamaPluginConfig::new(
      self.config.ollama_plugin_exe.clone(),
      self.config.chat_model_name.clone(),
      self.config.embedding_model_name.clone(),
      Some(self.config.ollama_server_url.clone()),
    )
    .unwrap();

    let persist_dir = tempfile::tempdir().unwrap().path().to_path_buf();
    config.set_rag_enabled(&persist_dir).unwrap();

    self.ollama_plugin.init_chat_plugin(config).await.unwrap();
  }

  pub async fn init_embedding_plugin(&self) {
    let temp_dir = tempfile::tempdir().unwrap().path().to_path_buf();
    let config = EmbeddingPluginConfig::new(
      self.config.embedding_plugin_exe.clone(),
      self.config.embedding_model_name.clone(),
      Some(temp_dir),
    )
    .unwrap();
    self
      .embedding_plugin
      .init_embedding_plugin(config)
      .await
      .unwrap();
  }

  pub async fn send_chat_message(&self, chat_id: &str, message: &str) -> String {
    self
      .ollama_plugin
      .ask_question(chat_id, message)
      .await
      .unwrap()
  }

  pub async fn stream_chat_message(
    &self,
    chat_id: &str,
    message: &str,
  ) -> ReceiverStream<Result<Value, PluginError>> {
    self
      .ollama_plugin
      .stream_question(chat_id, message, json!({}))
      .await
      .unwrap()
  }

  pub async fn generate_embedding(&self, message: &str) -> Vec<Vec<f64>> {
    self
      .embedding_plugin
      .generate_embedding(message)
      .await
      .unwrap()
  }

  pub async fn calculate_similarity(&self, input: &str, expected: &str) -> f64 {
    let left = self
      .embedding_plugin
      .generate_embedding(input)
      .await
      .unwrap();
    let right = self
      .embedding_plugin
      .generate_embedding(expected)
      .await
      .unwrap();

    let actual_embedding_flat = flatten_vec(left);
    let expected_embedding_flat = flatten_vec(right);
    let distance = f64::cosine(&actual_embedding_flat, &expected_embedding_flat)
      .expect("Vectors must be of the same length");

    distance.cos()
  }
}

// Function to flatten Vec<Vec<f64>> into Vec<f64>
fn flatten_vec(vec: Vec<Vec<f64>>) -> Vec<f64> {
  vec.into_iter().flatten().collect()
}

pub struct LocalAIConfiguration {
  ollama_server_url: String,
  ollama_plugin_exe: PathBuf,
  embedding_plugin_exe: PathBuf,
  chat_model_name: String,
  embedding_model_name: String,
}

impl LocalAIConfiguration {
  pub fn new() -> Result<Self> {
    dotenv::dotenv().ok();
    setup_log();

    // load from .env
    let ollama_server_url = dotenv::var("OLLAMA_SERVER_URL")?;
    let ollama_plugin_exe = PathBuf::from(dotenv::var("OLLAMA_PLUGIN_EXE_PATH")?);
    let chat_model_name = dotenv::var("OLLAMA_CHAT_MODEL_NAME")?;
    let embedding_plugin_exe = PathBuf::from(dotenv::var("OLLAMA_EMBEDDING_EXE_PATH")?);
    let embedding_model_name = dotenv::var("OLLAMA_EMBEDDING_MODEL_NAME")?;

    Ok(Self {
      ollama_server_url,
      ollama_plugin_exe,
      chat_model_name,
      embedding_plugin_exe,
      embedding_model_name,
    })
  }
}

pub fn setup_log() {
  static START: Once = Once::new();
  START.call_once(|| {
    let level = "trace";
    let mut filters = vec![];
    filters.push(format!("appflowy_plugin={}", level));
    filters.push(format!("appflowy_local_ai={}", level));
    std::env::set_var("RUST_LOG", filters.join(","));

    let subscriber = Subscriber::builder()
      .with_env_filter(EnvFilter::from_default_env())
      .with_line_number(true)
      .with_ansi(true)
      .finish();
    subscriber.try_init().unwrap();
  });
}

pub fn get_asset_path(name: &str) -> PathBuf {
  let file = format!("tests/asset/{name}");
  let absolute_path = std::env::current_dir().unwrap().join(Path::new(&file));
  absolute_path
}

pub async fn collect_bytes_stream(
  mut stream: ReceiverStream<Result<Value, PluginError>>,
) -> String {
  let mut list = vec![];
  while let Some(s) = stream.next().await {
    if let Value::Object(mut map) = s.unwrap() {
      let s = map.remove("1").unwrap().as_str().unwrap().to_string();
      list.push(s);
    }
  }

  list.join("")
}
