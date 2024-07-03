use anyhow::Result;
use bytes::Bytes;
use appflowy_plugin::manager::SidecarManager;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Once;
use tokio_stream::wrappers::ReceiverStream;

use simsimd::SpatialSimilarity;
use std::f64;
use tracing_subscriber::fmt::Subscriber;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;
use appflowy_local_ai_chat::chat_plugin::ChatPluginOperation;
use appflowy_local_ai_chat::embedding_plugin::EmbeddingPluginOperation;
use appflowy_plugin::core::plugin::{PluginId, PluginInfo};
use appflowy_plugin::error::SidecarError;

pub struct LocalAITest {
  config: LocalAIConfiguration,
  manager: SidecarManager,
}

impl LocalAITest {
  pub fn new() -> Result<Self> {
    let config = LocalAIConfiguration::new()?;
    let manager = SidecarManager::new();

    Ok(Self { config, manager })
  }
  pub async fn init_chat_plugin(&self) -> PluginId {
    let info = PluginInfo {
      name: "chat".to_string(),
      exec_path: self.config.chat_bin_path.clone(),
    };
    let plugin_id = self.manager.create_plugin(info).await.unwrap();
    self
      .manager
      .init_plugin(
        plugin_id,
        json!({
            "absolute_chat_model_path":self.config.chat_model_absolute_path(),
        }),
      )
      .unwrap();

    plugin_id
  }

  pub async fn init_embedding_plugin(&self) -> PluginId {
    let info = PluginInfo {
      name: "embedding".to_string(),
      exec_path: self.config.embedding_bin_path.clone(),
    };
    let plugin_id = self.manager.create_plugin(info).await.unwrap();
    let embedding_model_path = self.config.embedding_model_absolute_path();
    self
      .manager
      .init_plugin(
        plugin_id,
        json!({
            "absolute_model_path":embedding_model_path,
        }),
      )
      .unwrap();
    plugin_id
  }

  pub async fn send_chat_message(
    &self,
    chat_id: &str,
    plugin_id: PluginId,
    message: &str,
  ) -> String {
    let plugin = self.manager.get_plugin(plugin_id).await.unwrap();
    let operation = ChatPluginOperation::new(plugin);
    operation.send_message(chat_id, message).await.unwrap()
  }

  pub async fn stream_chat_message(
    &self,
    chat_id: &str,
    plugin_id: PluginId,
    message: &str,
  ) -> ReceiverStream<Result<Bytes, SidecarError>> {
    let plugin = self.manager.get_plugin(plugin_id).await.unwrap();
    let operation = ChatPluginOperation::new(plugin);
    operation.stream_message(chat_id, message).await.unwrap()
  }

  pub async fn related_question(
    &self,
    chat_id: &str,
    plugin_id: PluginId,
  ) -> Vec<serde_json::Value> {
    let plugin = self.manager.get_plugin(plugin_id).await.unwrap();
    let operation = ChatPluginOperation::new(plugin);
    operation.get_related_questions(chat_id).await.unwrap()
  }

  pub async fn calculate_similarity(
    &self,
    plugin_id: PluginId,
    message1: &str,
    message2: &str,
  ) -> f64 {
    let plugin = self.manager.get_plugin(plugin_id).await.unwrap();
    let operation = EmbeddingPluginOperation::new(plugin);
    let left = operation.get_embeddings(message1).await.unwrap();
    let right = operation.get_embeddings(message2).await.unwrap();

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
  model_dir: String,
  chat_bin_path: PathBuf,
  chat_model_name: String,
  embedding_bin_path: PathBuf,
  embedding_model_name: String,
}

impl LocalAIConfiguration {
  pub fn new() -> Result<Self> {
    dotenv::dotenv().ok();
    setup_log();

    // load from .env
    let model_dir = dotenv::var("LOCAL_AI_MODEL_DIR")?;
    let chat_bin_path = PathBuf::from(dotenv::var("CHAT_BIN_PATH")?);
    let chat_model_name = dotenv::var("LOCAL_AI_CHAT_MODEL_NAME")?;

    let embedding_bin_path = PathBuf::from(dotenv::var("EMBEDDING_BIN_PATH")?);
    let embedding_model_name = dotenv::var("LOCAL_AI_EMBEDDING_MODEL_NAME")?;

    Ok(Self {
      model_dir,
      chat_bin_path,
      chat_model_name,
      embedding_bin_path,
      embedding_model_name,
    })
  }

  pub fn chat_model_absolute_path(&self) -> String {
    format!("{}/{}", self.model_dir, self.chat_model_name)
  }

  pub fn embedding_model_absolute_path(&self) -> String {
    format!("{}/{}", self.model_dir, self.embedding_model_name)
  }
}

pub fn setup_log() {
  static START: Once = Once::new();
  START.call_once(|| {
    let level = "trace";
    let mut filters = vec![];
    filters.push(format!("appflowy_plugin={}", level));
    filters.push(format!("appflowy_local_ai_chat={}", level));
    std::env::set_var("RUST_LOG", filters.join(","));

    let subscriber = Subscriber::builder()
      .with_env_filter(EnvFilter::from_default_env())
      .with_line_number(true)
      .with_ansi(true)
      .finish();
    subscriber.try_init().unwrap();
  });
}
