use af_local_ai::ollama_plugin::{OllamaAIPlugin, OllamaPluginConfig};
use af_plugin::error::PluginError;
use af_plugin::manager::PluginManager;
use anyhow::Result;

use bytes::Bytes;
use serde_json::{json, Value};
use simsimd::SpatialSimilarity;
use std::f64;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tracing::trace;
use tracing_subscriber::fmt::Subscriber;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

pub struct LocalAITest {
  config: LocalAIConfiguration,
  pub ollama_plugin: OllamaAIPlugin,
}

impl LocalAITest {
  pub fn new() -> Result<Self> {
    dotenv::dotenv().ok();
    setup_log();

    let config = LocalAIConfiguration::new()?;
    let sidecar = Arc::new(PluginManager::new());
    let ollama_plugin = OllamaAIPlugin::new(sidecar.clone());
    Ok(Self {
      config,
      ollama_plugin,
    })
  }

  pub async fn init_chat_plugin(&self) {
    let mut config = OllamaPluginConfig::new(
      self.config.ollama_plugin_exe.clone(),
      self.config.ollama_plugin_command.clone(),
      self.config.chat_model_name.clone(),
      self.config.embedding_model_name.clone(),
      Some(self.config.ollama_server_url.clone()),
    )
    .unwrap();

    let persist_dir = tempfile::tempdir().unwrap().path().to_path_buf();
    config.set_rag_enabled(&persist_dir).unwrap();
    config.set_log_level("debug".to_string());

    self.ollama_plugin.init_plugin(config).await.unwrap();
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
    format: Option<serde_json::Value>,
  ) -> ReceiverStream<Result<Value, PluginError>> {
    self
      .ollama_plugin
      .stream_question(chat_id, message, format, json!({}))
      .await
      .unwrap()
  }

  pub async fn generate_embedding(&self, message: &str) -> Vec<Vec<f64>> {
    self
      .ollama_plugin
      .generate_embedding(message)
      .await
      .unwrap()
  }

  async fn get_flat_embedding(&self, text: &str) -> Vec<f64> {
    let embedding = self.ollama_plugin.generate_embedding(text).await.unwrap();
    flatten_vec(embedding)
  }

  pub async fn calculate_similarity(&self, input: &str, expected: &str) -> f64 {
    // Generate flattened embeddings for both inputs.
    let mut left_vec = self.get_flat_embedding(input).await;
    let mut right_vec = self.get_flat_embedding(expected).await;

    // Ensure both vectors have the same length by truncating the longer one.
    if left_vec.len() != right_vec.len() {
      let min_len = std::cmp::min(left_vec.len(), right_vec.len());
      left_vec.truncate(min_len);
      right_vec.truncate(min_len);
    }

    // Compute the cosine distance (or angle) and then return the cosine similarity.
    let angle = f64::cosine(&left_vec, &right_vec).expect("Vectors must be of the same length");
    angle.cos()
  }
}

// Function to flatten Vec<Vec<f64>> into Vec<f64>
fn flatten_vec(vec: Vec<Vec<f64>>) -> Vec<f64> {
  vec.into_iter().flatten().collect()
}

pub struct LocalAIConfiguration {
  ollama_server_url: String,
  ollama_plugin_exe: PathBuf,
  ollama_plugin_command: String,
  #[allow(dead_code)]
  embedding_plugin_exe: PathBuf,
  chat_model_name: String,
  embedding_model_name: String,
}

impl LocalAIConfiguration {
  pub fn new() -> Result<Self> {
    // load from .env
    let ollama_server_url = dotenv::var("OLLAMA_SERVER_URL")?;
    let ollama_plugin_exe =
      PathBuf::from(dotenv::var("OLLAMA_PLUGIN_EXE_PATH").unwrap_or_default());
    let chat_model_name = dotenv::var("OLLAMA_CHAT_MODEL_NAME")?;
    let embedding_plugin_exe =
      PathBuf::from(dotenv::var("OLLAMA_EMBEDDING_EXE_PATH").unwrap_or_default());
    let embedding_model_name = dotenv::var("OLLAMA_EMBEDDING_MODEL_NAME")?;

    trace!("Ollama server url: {}", ollama_server_url);
    trace!("Ollama plugin exe: {:?}", ollama_plugin_exe);

    Ok(Self {
      ollama_server_url,
      ollama_plugin_exe,
      chat_model_name,
      embedding_plugin_exe,
      embedding_model_name,
      ollama_plugin_command: "af_ollama_plugin".to_string(),
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
  mut stream: ReceiverStream<Result<Bytes, PluginError>>,
) -> String {
  let mut list = vec![];
  while let Some(s) = stream.next().await {
    list.push(String::from_utf8(s.unwrap().to_vec()).unwrap());
  }

  list.join("")
}

pub async fn collect_json_stream(mut stream: ReceiverStream<Result<Value, PluginError>>) -> String {
  let mut list = Vec::new();
  while let Some(item) = stream.next().await {
    // Try to extract the string from the JSON object.
    // On any error or if the structure is not as expected, use an empty string.
    let s = item
        .ok() // Converts Result<Value, PluginError> to Option<Value>, discarding any error.
        .and_then(|value| {
          if let Value::Object(mut map) = value {
            map.remove("1")
                .and_then(|v| v.as_str().map(String::from))
          } else {
            None
          }
        })
        .unwrap_or_default();
    list.push(s);
  }
  list.join("")
}

pub async fn collect_completion_stream(
  mut stream: ReceiverStream<Result<Value, PluginError>>,
) -> (String, String) {
  let mut answer = Vec::new();
  let mut comment = Vec::new();
  while let Some(item) = stream.next().await {
    if let Ok(Value::Object(mut map)) = item {
      if let Some(v) = map.remove("1") {
        if let Some(s) = v.as_str() {
          answer.push(s.to_string());
        }
      }
      if let Some(v) = map.remove("4") {
        if let Some(s) = v.as_str() {
          comment.push(s.to_string());
        }
      }
    }
  }
  (answer.join(""), comment.join(""))
}
