use crate::embedding_ops::EmbeddingPluginOperation;
use std::collections::HashMap;

use af_plugin::core::plugin::{
  Plugin, PluginConfig, RunningState, RunningStateReceiver, RunningStateSender,
};
use af_plugin::error::PluginError;
use af_plugin::manager::PluginManager;
use anyhow::anyhow;
use anyhow::Result;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::timeout;
use tokio_stream::wrappers::WatchStream;
use tokio_stream::StreamExt;
use tracing::{info, trace};

pub struct EmbeddingPlugin {
  plugin_manager: Arc<PluginManager>,
  plugin_config: RwLock<Option<EmbeddingPluginConfig>>,
  running_state: RunningStateSender,
  #[allow(dead_code)]
  // keep at least one receiver that make sure the sender can receive value
  running_state_rx: RunningStateReceiver,
}

impl EmbeddingPlugin {
  pub fn new(plugin_manager: Arc<PluginManager>) -> Self {
    let (running_state, rx) = tokio::sync::watch::channel(RunningState::ReadyToConnect);
    Self {
      plugin_manager,
      plugin_config: Default::default(),
      running_state: Arc::new(running_state),
      running_state_rx: rx,
    }
  }

  pub async fn init_embedding_plugin(
    &self,
    config: EmbeddingPluginConfig,
  ) -> Result<(), PluginError> {
    if let Some(existing_config) = self.plugin_config.read().await.as_ref() {
      trace!(
        "[Embedding Plugin] existing config: {:?}, new config:{:?}",
        existing_config,
        config
      );
    }

    let info = PluginConfig {
      name: "embedding".to_string(),
      exec_path: config.executable_path.clone(),
      exec_command: "".to_string(),
    };
    let plugin_id = self
      .plugin_manager
      .create_plugin(info, self.running_state.clone())
      .await?;

    let mut params = json!({});
    if let Some(persist_directory) = config.persist_directory {
      params["persist_directory"] = json!(persist_directory);
    }
    params["model_name"] = json!(config.model_name);

    let plugin = self.plugin_manager.init_plugin(plugin_id, params).await?;
    info!("[Embedding Plugin] {} setup success", plugin);
    Ok(())
  }

  pub fn subscribe_running_state(&self) -> WatchStream<RunningState> {
    WatchStream::new(self.running_state.subscribe())
  }

  pub async fn generate_embedding(&self, text: &str) -> Result<Vec<Vec<f64>>, PluginError> {
    trace!("[Embedding Plugin] generate embedding for text: {}", text);
    self.wait_plugin_ready().await?;
    let plugin = self.get_embedding_plugin().await?;
    let operation = EmbeddingPluginOperation::new(plugin);
    let embeddings = operation.gen_embeddings(text).await?;
    Ok(embeddings)
  }

  pub async fn index(
    &self,
    text: &str,
    metadata: HashMap<String, Value>,
  ) -> Result<(), PluginError> {
    trace!("[Embedding Plugin] generate embedding for text: {}", text);
    self.wait_plugin_ready().await?;
    let plugin = self.get_embedding_plugin().await?;
    let operation = EmbeddingPluginOperation::new(plugin);
    operation.embed_text(text, metadata).await?;
    Ok(())
  }

  pub async fn similarity_search(
    &self,
    query: &str,
    filter: HashMap<String, Value>,
  ) -> Result<Vec<String>, PluginError> {
    trace!("[Embedding Plugin] similarity search for query: {}", query);
    self.wait_plugin_ready().await?;
    let plugin = self.get_embedding_plugin().await?;
    let operation = EmbeddingPluginOperation::new(plugin);
    let result = operation.similarity_search(query, filter).await?;
    Ok(result)
  }

  async fn get_embedding_plugin(&self) -> Result<Weak<Plugin>> {
    let plugin_id = self
      .running_state
      .borrow()
      .plugin_id()
      .ok_or_else(|| anyhow!("Embedding plugin is not initialized yet"))?;

    let plugin = self.plugin_manager.get_plugin(plugin_id).await?;
    Ok(plugin)
  }

  async fn wait_plugin_ready(&self) -> Result<()> {
    let is_loading = self.running_state.borrow().is_loading();
    if !is_loading {
      return Ok(());
    }
    info!("[Embedding Plugin] wait for plugin to be ready");
    let mut rx = self.subscribe_running_state();
    let timeout_duration = Duration::from_secs(30);
    let result = timeout(timeout_duration, async {
      while let Some(state) = rx.next().await {
        if state.is_running() {
          break;
        }
      }
    })
    .await;

    match result {
      Ok(_) => {
        trace!("[Embedding Plugin] is ready");
        Ok(())
      },
      Err(_) => Err(anyhow!("Timeout while waiting for chat plugin to be ready")),
    }
  }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EmbeddingPluginConfig {
  pub executable_path: PathBuf,
  pub model_name: String,
  pub persist_directory: Option<PathBuf>,
}

impl EmbeddingPluginConfig {
  pub fn new<T: Into<PathBuf>>(
    bin_path: T,
    model_name: String,
    storage_path: Option<PathBuf>,
  ) -> Result<Self> {
    let executable_path = bin_path.into();
    if !executable_path.exists() {
      return Err(anyhow!(
        "Embedding binary path does not exist: {:?}",
        executable_path
      ));
    }
    if !executable_path.is_file() {
      return Err(anyhow!(
        "Embedding binary path is not a file: {:?}",
        executable_path
      ));
    }

    // Check if local_model_dir exists and is a directory

    Ok(Self {
      executable_path,
      model_name,
      persist_directory: storage_path,
    })
  }
}
