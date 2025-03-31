use crate::ai_ops::{AIPluginOperation, LocalAITranslateRowData, LocalAITranslateRowResponse};
use af_plugin::core::plugin::{
  Plugin, PluginConfig, PluginId, RunningState, RunningStateReceiver, RunningStateSender,
};
use af_plugin::error::PluginError;
use af_plugin::manager::PluginManager;
use anyhow::{anyhow, Result};

use crate::embedding_ops::EmbeddingPluginOperation;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::PathBuf;

use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::io;
use tokio::sync::RwLock;
use tokio::time::timeout;
use tokio_stream::wrappers::{ReceiverStream, WatchStream};
use tokio_stream::StreamExt;
use tracing::{error, info, instrument, trace};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct PluginInfo {
  pub version: String,
}

pub struct OllamaAIPlugin {
  plugin_manager: Arc<PluginManager>,
  plugin_config: RwLock<Option<OllamaPluginConfig>>,
  running_state: RunningStateSender,
  #[allow(dead_code)]
  // keep at least one receiver that make sure the sender can receive value
  running_state_rx: RunningStateReceiver,
  init_lock: tokio::sync::Mutex<()>,
  plugin_id: tokio::sync::Mutex<Option<PluginId>>,
  plugin_info: tokio::sync::RwLock<Option<PluginInfo>>,
}

impl OllamaAIPlugin {
  pub fn new(plugin_manager: Arc<PluginManager>) -> Self {
    let (running_state, rx) = tokio::sync::watch::channel(RunningState::ReadyToConnect);
    Self {
      plugin_manager,
      plugin_config: Default::default(),
      running_state: Arc::new(running_state),
      running_state_rx: rx,
      init_lock: tokio::sync::Mutex::new(()),
      plugin_id: Default::default(),
      plugin_info: Default::default(),
    }
  }

  pub async fn plugin_info(&self) -> Result<PluginInfo, PluginError> {
    let plugin_info = self.plugin_info.read().await.clone();
    match plugin_info {
      None => {
        self.wait_until_plugin_ready().await?;
        let plugin = self.get_ai_plugin().await?;
        let operation = AIPluginOperation::new(plugin);
        let info = operation.plugin_info().await?;
        self.plugin_info.write().await.replace(info.clone());

        Ok(info)
      },
      Some(plugin_info) => Ok(plugin_info),
    }
  }

  /// Creates a new chat session.
  ///
  /// # Arguments
  ///
  /// * `chat_id` - A string slice containing the unique identifier for the chat session.
  ///
  /// # Returns
  ///
  /// A `Result<()>` indicating success or failure.
  pub async fn create_chat(&self, chat_id: &str) -> Result<(), PluginError> {
    trace!("[AI Plugin] create chat: {}", chat_id);
    self.wait_until_plugin_ready().await?;

    let plugin = self.get_ai_plugin().await?;
    let operation = AIPluginOperation::new(plugin);
    operation.create_chat(chat_id).await?;
    Ok(())
  }

  /// Closes an existing chat session.
  ///
  /// # Arguments
  ///
  /// * `chat_id` - A string slice containing the unique identifier for the chat session to close.
  ///
  /// # Returns
  ///
  /// A `Result<()>` indicating success or failure.
  pub async fn close_chat(&self, chat_id: &str) -> Result<()> {
    trace!("[AI Plugin] close chat: {}", chat_id);
    let plugin = self.get_ai_plugin().await?;
    let operation = AIPluginOperation::new(plugin);
    operation.close_chat(chat_id).await?;
    Ok(())
  }

  pub fn subscribe_running_state(&self) -> WatchStream<RunningState> {
    WatchStream::new(self.running_state.subscribe())
  }

  pub fn get_plugin_running_state(&self) -> RunningState {
    self.running_state.borrow().clone()
  }

  /// Asks a question and returns a stream of responses.
  ///
  /// # Arguments
  ///
  /// * `chat_id` - A string slice containing the unique identifier for the chat session.
  /// * `message` - A string slice containing the question or message to send.
  ///
  /// # Returns
  ///
  /// A `Result<ReceiverStream<anyhow::Result<Bytes, SidecarError>>>` containing a stream of responses.
  pub async fn stream_question(
    &self,
    chat_id: &str,
    message: &str,
    format: Option<serde_json::Value>,
    metadata: serde_json::Value,
  ) -> Result<ReceiverStream<anyhow::Result<Value, PluginError>>, PluginError> {
    trace!("[AI Plugin] ask question: {}", message);
    self.wait_until_plugin_ready().await?;
    let plugin = self.get_ai_plugin().await?;
    let operation = AIPluginOperation::new(plugin);
    let stream = operation
      .stream_message_v2(chat_id, message, format, metadata)
      .await?;
    Ok(stream)
  }

  pub async fn get_related_question(&self, chat_id: &str) -> Result<Vec<String>, PluginError> {
    self.wait_until_plugin_ready().await?;
    let plugin = self.get_ai_plugin().await?;
    let operation = AIPluginOperation::new(plugin);
    let values = operation.get_related_questions(chat_id).await?;
    Ok(values)
  }

  pub async fn embed_file(
    &self,
    chat_id: &str,
    file_path: PathBuf,
    metadata: Option<HashMap<String, serde_json::Value>>,
  ) -> Result<(), PluginError> {
    if !file_path.exists() {
      return Err(PluginError::Io(io::Error::new(
        io::ErrorKind::NotFound,
        "file not found",
      )));
    }

    let file_path_str = file_path
      .to_str()
      .ok_or(PluginError::Io(io::Error::new(
        io::ErrorKind::NotFound,
        "file path invalid",
      )))?
      .to_string();

    self.wait_until_plugin_ready().await?;
    let plugin = self.get_ai_plugin().await?;
    let operation = AIPluginOperation::new(plugin);
    operation
      .embed_file(chat_id, file_path_str, metadata)
      .await?;
    Ok(())
  }

  /// Generates a complete answer for a given message.
  ///
  /// # Arguments
  ///
  /// * `chat_id` - A string slice containing the unique identifier for the chat session.
  /// * `message` - A string slice containing the message to generate an answer for.
  ///
  /// # Returns
  ///
  /// A `Result<String>` containing the generated answer.
  pub async fn ask_question(&self, chat_id: &str, message: &str) -> Result<String, PluginError> {
    self.wait_until_plugin_ready().await?;
    let plugin = self.get_ai_plugin().await?;
    let operation = AIPluginOperation::new(plugin);
    let answer = operation.send_message(chat_id, message, true).await?;
    Ok(answer)
  }

  #[instrument(skip_all, err)]
  pub async fn destroy_plugin(&self) -> Result<()> {
    let plugin_id = self.plugin_id.lock().await.take();
    if let Some(plugin_id) = plugin_id {
      info!("[AI Plugin]: destroy plugin: {:?}", plugin_id);

      if let Err(err) = self.plugin_manager.remove_plugin(plugin_id).await {
        error!("remove plugin failed: {:?}", err);
      }
    }
    Ok(())
  }

  pub async fn complete_text_v2(
    &self,
    message: &str,
    complete_type: u8,
    format: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
  ) -> Result<ReceiverStream<anyhow::Result<Value, PluginError>>, PluginError> {
    trace!(
      "[AI Plugin] complete text v2: {}, completion_type: {:?}, format: {:?}, metadata: {:?}",
      message,
      complete_type,
      format,
      metadata
    );
    self.wait_until_plugin_ready().await?;
    let plugin = self.get_ai_plugin().await?;
    let operation = AIPluginOperation::new(plugin);
    let stream = operation
      .complete_text_v2(message, complete_type, format, metadata)
      .await?;
    Ok(stream)
  }

  pub async fn summary_database_row(
    &self,
    row: HashMap<String, String>,
  ) -> Result<String, PluginError> {
    trace!("[AI Plugin] summary database row: {:?}", row);
    self.wait_until_plugin_ready().await?;
    let plugin = self.get_ai_plugin().await?;
    let operation = AIPluginOperation::new(plugin);
    let text = operation.summary_row(row).await?;
    Ok(text)
  }

  pub async fn translate_database_row(
    &self,
    row: LocalAITranslateRowData,
  ) -> Result<LocalAITranslateRowResponse, PluginError> {
    trace!("[AI Plugin] summary database row: {:?}", row);
    self.wait_until_plugin_ready().await?;
    let plugin = self.get_ai_plugin().await?;
    let operation = AIPluginOperation::new(plugin);
    let resp = operation.translate_row(row).await?;
    Ok(resp)
  }

  pub async fn init_plugin(&self, config: OllamaPluginConfig) -> Result<(), PluginError> {
    // Try to acquire the initialization lock without waiting.
    match self.init_lock.try_lock() {
      Ok(_guard) => {
        // We have the lock and can proceed with initialization.
        trace!("[AI Plugin] Creating chat plugin with config: {:?}", config);
        let plugin_config = PluginConfig {
          name: "af_ollama_plugin".to_string(),
          exec_path: config.executable_path.clone(),
          exec_command: config.executable_command.clone(),
        };

        if let Err(err) = self.destroy_plugin().await {
          error!("[AI Plugin] Failed to destroy plugin: {:?}", err);
        }

        let plugin_id = self
          .plugin_manager
          .create_plugin(plugin_config, self.running_state.clone())
          .await?;
        *self.plugin_id.lock().await = Some(plugin_id);

        // Set up plugin parameters.
        let mut params = json!({});
        params["verbose"] = json!(config.verbose);
        params["server_url"] = json!(config.server_url);
        params["model_name"] = json!(config.chat_model_name);

        if let Some(persist_directory) = config.persist_directory.clone() {
          params["vectorstore_config"] = json!({
            "model_name": config.embedding_model_name,
            "persist_directory": persist_directory,
          });
        }

        info!(
          "[AI Plugin] Setting up chat plugin: {:?}, params: {:?}",
          plugin_id, params
        );
        let plugin = self.plugin_manager.init_plugin(plugin_id, params).await?;
        info!("[AI Plugin] {} setup success", plugin);
        self.plugin_config.write().await.replace(config);

        let mut rx = plugin.subscribe_running_state();
        let weak_plugin = Arc::downgrade(&plugin);
        let timeout_duration = Duration::from_secs(30);
        let _ = timeout(timeout_duration, async {
          while let Some(state) = rx.next().await {
            if state.is_running() {
              let operation = AIPluginOperation::new(weak_plugin);
              if let Ok(info) = operation.plugin_info().await {
                info!("[AI Plugin] using plugin version: {}", info.version);
              }
              break;
            }
          }
        })
        .await;

        Ok(())
      },
      Err(_) => {
        // Lock is already held – an initialization is in progress.
        trace!("[AI Plugin] Initialization already in progress, returning immediately");
        Ok(())
      },
    }
  }

  pub async fn generate_embedding(&self, text: &str) -> Result<Vec<Vec<f64>>, PluginError> {
    trace!("[AI Plugin] generate embedding for text: {}", text);
    self.wait_until_plugin_ready().await?;
    let plugin = self.get_ai_plugin().await?;
    let operation = EmbeddingPluginOperation::new(plugin);
    let embeddings = operation.gen_embeddings(text).await?;
    Ok(embeddings)
  }

  pub async fn embed_text(
    &self,
    text: &str,
    metadata: HashMap<String, Value>,
  ) -> Result<(), PluginError> {
    trace!("[AI Plugin] generate embedding for text: {}", text);
    self.wait_until_plugin_ready().await?;
    let plugin = self.get_ai_plugin().await?;
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
    self.wait_until_plugin_ready().await?;
    let plugin = self.get_ai_plugin().await?;
    let operation = EmbeddingPluginOperation::new(plugin);
    let result = operation.similarity_search(query, filter).await?;
    Ok(result)
  }

  /// Waits for the plugin to be ready.
  ///
  /// The wait_plugin_ready method is an asynchronous function designed to ensure that the chat
  /// plugin is in a ready state before allowing further operations. This is crucial for maintaining
  /// the correct sequence of operations and preventing errors that could occur if operations are
  /// attempted on an unready plugin.
  ///
  /// # Returns
  ///
  /// A `Result<()>` indicating success or failure.
  async fn wait_until_plugin_ready(&self) -> Result<()> {
    let is_loading = self.running_state.borrow().is_loading();
    if !is_loading {
      return Ok(());
    }
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
      Ok(_) => Ok(()),
      Err(_) => Err(anyhow!("Timeout while waiting for chat plugin to be ready")),
    }
  }

  /// Retrieves the chat plugin.
  ///
  /// # Returns
  ///
  /// A `Result<Weak<Plugin>>` containing a weak reference to the plugin.
  pub async fn get_ai_plugin(&self) -> Result<Weak<Plugin>, PluginError> {
    let plugin_id = self
      .plugin_id
      .lock()
      .await
      .as_ref()
      .cloned()
      .ok_or_else(|| PluginError::Internal(anyhow!("chat plugin not initialized")))?;

    let plugin = self.plugin_manager.get_plugin(plugin_id).await?;
    Ok(plugin)
  }
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct OllamaPluginConfig {
  pub executable_path: PathBuf,
  pub executable_command: String,
  pub chat_model_name: String,
  pub embedding_model_name: String,
  pub server_url: String,
  pub persist_directory: Option<PathBuf>,
  pub verbose: bool,
  pub log_level: String,
}

impl OllamaPluginConfig {
  pub fn new(
    executable_path: PathBuf,
    executable_command: String,
    chat_model_name: String,
    embedding_model_name: String,
    server_url: Option<String>,
  ) -> Result<Self> {
    Ok(Self {
      executable_path,
      executable_command,
      chat_model_name,
      embedding_model_name,
      persist_directory: None,
      server_url: server_url.unwrap_or("http://localhost:11434".to_string()),
      verbose: false,
      log_level: "info".to_string(),
    })
  }
  pub fn with_verbose(mut self, verbose: bool) -> Self {
    self.verbose = verbose;
    self
  }

  pub fn set_log_level(&mut self, log_level: String) {
    self.log_level = log_level;
  }
  pub fn set_rag_enabled(&mut self, persist_directory: &PathBuf) -> Result<()> {
    if !persist_directory.exists() {
      std::fs::create_dir_all(persist_directory)?;
    }

    self.persist_directory = Some(persist_directory.clone());
    Ok(())
  }
}
