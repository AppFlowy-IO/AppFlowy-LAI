use crate::ai_ops::{
  AIPluginOperation, CompleteTextType, LocalAITranslateRowData, LocalAITranslateRowResponse,
};
use anyhow::{anyhow, Result};
use appflowy_plugin::core::plugin::{
  Plugin, PluginInfo, RunningState, RunningStateReceiver, RunningStateSender,
};
use appflowy_plugin::error::PluginError;
use appflowy_plugin::manager::PluginManager;
use appflowy_plugin::util::get_operating_system;
use bytes::Bytes;

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

pub struct OllamaAIPlugin {
  plugin_manager: Arc<PluginManager>,
  plugin_config: RwLock<Option<OllamaPluginConfig>>,
  running_state: RunningStateSender,
  #[allow(dead_code)]
  // keep at least one receiver that make sure the sender can receive value
  running_state_rx: RunningStateReceiver,
}

impl OllamaAIPlugin {
  pub fn new(plugin_manager: Arc<PluginManager>) -> Self {
    let (running_state, rx) = tokio::sync::watch::channel(RunningState::Initialization);
    Self {
      plugin_manager,
      plugin_config: Default::default(),
      running_state: Arc::new(running_state),
      running_state_rx: rx,
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
    metadata: serde_json::Value,
  ) -> Result<ReceiverStream<anyhow::Result<Value, PluginError>>, PluginError> {
    trace!("[AI Plugin] ask question: {}", message);
    self.wait_until_plugin_ready().await?;
    let plugin = self.get_ai_plugin().await?;
    let operation = AIPluginOperation::new(plugin);
    let stream = operation
      .stream_message_v2(chat_id, message, metadata)
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

  pub async fn index_file(
    &self,
    chat_id: &str,
    file_path: Option<PathBuf>,
    file_content: Option<String>,
    metadata: Option<HashMap<String, serde_json::Value>>,
  ) -> Result<(), PluginError> {
    let mut file_path_str = None;
    if let Some(file_path) = file_path {
      if !file_path.exists() {
        return Err(PluginError::Io(io::Error::new(
          io::ErrorKind::NotFound,
          "file not found",
        )));
      }

      file_path_str = Some(
        file_path
          .to_str()
          .ok_or(PluginError::Io(io::Error::new(
            io::ErrorKind::NotFound,
            "file path invalid",
          )))?
          .to_string(),
      );
    }

    self.wait_until_plugin_ready().await?;
    let plugin = self.get_ai_plugin().await?;
    let operation = AIPluginOperation::new(plugin);

    operation
      .index_file(chat_id, file_path_str, file_content, metadata)
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
  pub async fn destroy_chat_plugin(&self) -> Result<()> {
    let plugin_id = self.running_state.borrow().plugin_id();
    if let Some(plugin_id) = plugin_id {
      if let Err(err) = self.plugin_manager.remove_plugin(plugin_id).await {
        error!("remove plugin failed: {:?}", err);
      }
    }

    Ok(())
  }

  pub async fn complete_text<T: Into<CompleteTextType> + Debug>(
    &self,
    message: &str,
    complete_type: T,
  ) -> Result<ReceiverStream<anyhow::Result<Bytes, PluginError>>, PluginError> {
    trace!("[AI Plugin]  complete text: {}", message);
    self.wait_until_plugin_ready().await?;
    let plugin = self.get_ai_plugin().await?;
    let operation = AIPluginOperation::new(plugin);
    let stream = operation.complete_text(message, complete_type).await?;
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

  #[instrument(skip_all, err)]
  pub async fn init_chat_plugin(&self, config: OllamaPluginConfig) -> Result<()> {
    let state = self.running_state.borrow().clone();
    if state.is_ready() {
      if let Some(existing_config) = self.plugin_config.read().await.as_ref() {
        trace!(
          "[AI Plugin] existing config: {:?}, new config:{:?}",
          existing_config,
          config
        );
      }
    }

    let _system = get_operating_system();
    // Initialize chat plugin if the config is different
    // If the OLLAMA_PLUGIN_EXE_PATH is different, remove the old plugin
    if let Err(err) = self.destroy_chat_plugin().await {
      error!("[AI Plugin] failed to destroy plugin: {:?}", err);
    }

    // create new plugin
    trace!("[AI Plugin] create chat plugin: {:?}", config);
    let plugin_info = PluginInfo {
      name: "ollama_ai_plugin".to_string(),
      exec_path: config.executable_path.clone(),
    };
    let plugin_id = self
      .plugin_manager
      .create_plugin(plugin_info, self.running_state.clone())
      .await?;

    // init plugin
    trace!("[AI Plugin] init chat plugin model: {:?}", plugin_id);
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
      "[AI Plugin] setup chat plugin: {:?}, params: {:?}",
      plugin_id, params
    );
    let plugin = self.plugin_manager.init_plugin(plugin_id, params).await?;
    info!("[AI Plugin] {} setup success", plugin);
    self.plugin_config.write().await.replace(config);
    Ok(())
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
    info!("[AI Plugin] wait for chat plugin to be ready");
    let mut rx = self.subscribe_running_state();
    let timeout_duration = Duration::from_secs(30);
    let result = timeout(timeout_duration, async {
      while let Some(state) = rx.next().await {
        if state.is_ready() {
          break;
        }
      }
    })
    .await;

    match result {
      Ok(_) => {
        trace!("[AI Plugin] is ready");
        Ok(())
      },
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
      .running_state
      .borrow()
      .plugin_id()
      .ok_or_else(|| PluginError::Internal(anyhow!("chat plugin not initialized")))?;
    let plugin = self.plugin_manager.get_plugin(plugin_id).await?;
    Ok(plugin)
  }
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct OllamaPluginConfig {
  pub executable_path: PathBuf,
  pub chat_model_name: String,
  pub embedding_model_name: String,
  pub server_url: String,
  pub persist_directory: Option<PathBuf>,
  pub verbose: bool,
}

impl OllamaPluginConfig {
  pub fn new(
    executable_path: PathBuf,
    chat_model_name: String,
    embedding_model_name: String,
    server_url: Option<String>,
  ) -> Result<Self> {
    if !executable_path.exists() {
      return Err(anyhow!(
        "executable path does not exist: {:?}",
        executable_path
      ));
    }
    Ok(Self {
      executable_path,
      chat_model_name,
      embedding_model_name,
      persist_directory: None,
      server_url: server_url.unwrap_or("http://localhost:11434".to_string()),
      verbose: false,
    })
  }
  pub fn with_verbose(mut self, verbose: bool) -> Self {
    self.verbose = verbose;
    self
  }
  pub fn set_rag_enabled(&mut self, persist_directory: &PathBuf) -> Result<()> {
    if !persist_directory.exists() {
      std::fs::create_dir_all(persist_directory)?;
    }

    self.persist_directory = Some(persist_directory.clone());
    Ok(())
  }
}
