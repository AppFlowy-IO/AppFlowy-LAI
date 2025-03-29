use crate::ollama_plugin::PluginInfo;
use af_plugin::core::parser::{EmptyResponseParser, ResponseParser};
use af_plugin::core::plugin::Plugin;
use af_plugin::error::{PluginError, RemoteError};
use anyhow::anyhow;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Weak;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{instrument, trace};

pub struct AIPluginOperation {
  plugin: Weak<Plugin>,
}

impl AIPluginOperation {
  pub fn new(plugin: Weak<Plugin>) -> Self {
    AIPluginOperation { plugin }
  }

  fn get_plugin(&self) -> Result<std::sync::Arc<Plugin>, PluginError> {
    self
      .plugin
      .upgrade()
      .ok_or_else(|| PluginError::Internal(anyhow!("Plugin is dropped")))
  }

  async fn send_request<T: ResponseParser>(
    &self,
    method: &str,
    params: JsonValue,
  ) -> Result<T::ValueType, PluginError> {
    let plugin = self.get_plugin()?;
    let request = json!({ "method": method, "params": params });
    plugin.async_request::<T>("handle", &request).await
  }

  pub async fn plugin_info(&self) -> Result<PluginInfo, PluginError> {
    let value = self
      .send_request::<DataJsonParser>("system_info", json!({}))
      .await?;
    let info = serde_json::from_value::<PluginInfo>(value)
      .map_err(|err| PluginError::Internal(err.into()))?;

    Ok(info)
  }

  pub async fn create_chat(&self, chat_id: &str) -> Result<(), PluginError> {
    self
      .send_request::<EmptyResponseParser>("create_chat", json!({ "chat_id": chat_id, "top_k": 2}))
      .await
  }

  pub async fn close_chat(&self, chat_id: &str) -> Result<(), PluginError> {
    self
      .send_request::<EmptyResponseParser>("close_chat", json!({ "chat_id": chat_id }))
      .await
  }

  pub async fn send_message(
    &self,
    chat_id: &str,
    message: &str,
    _rag_enabled: bool,
  ) -> Result<String, PluginError> {
    self
      .send_request::<ChatResponseParser>(
        "answer",
        json!({ "chat_id": chat_id, "content": message }),
      )
      .await
  }

  #[instrument(level = "debug", skip(self), err)]
  pub async fn stream_message(
    &self,
    chat_id: &str,
    message: &str,
    metadata: serde_json::Value,
  ) -> Result<ReceiverStream<Result<Bytes, PluginError>>, PluginError> {
    let plugin = self.get_plugin()?;
    let params = json!({
        "chat_id": chat_id,
        "method": "stream_answer",
        "params": { "content": message, "metadata": metadata }
    });
    plugin.stream_request::<ChatStreamResponseParser>("handle", &params)
  }
  #[instrument(level = "debug", skip(self), err)]
  pub async fn stream_message_v2(
    &self,
    chat_id: &str,
    message: &str,
    format: Option<serde_json::Value>,
    metadata: serde_json::Value,
  ) -> Result<ReceiverStream<Result<serde_json::Value, PluginError>>, PluginError> {
    let plugin = self.get_plugin()?;

    // Build the inner params as a map.
    let mut inner_params = serde_json::Map::new();
    inner_params.insert("chat_id".to_string(), json!(chat_id));
    inner_params.insert("data".to_string(), json!({ "content": message }));
    inner_params.insert("metadata".to_string(), metadata);
    if let Some(fmt) = format {
      inner_params.insert("format".to_string(), fmt);
    }

    let params = json!({
        "method": "stream_answer_v2",
        "params": serde_json::Value::Object(inner_params)
    });

    plugin.stream_request::<JsonStringToJsonObject>("handle", &params)
  }

  pub async fn get_related_questions(&self, chat_id: &str) -> Result<Vec<String>, PluginError> {
    self
      .send_request::<ChatRelatedQuestionsResponseParser>(
        "related_question",
        json!({ "chat_id": chat_id }),
      )
      .await
  }

  #[instrument(level = "debug", skip_all, err)]
  pub async fn embed_file(
    &self,
    chat_id: &str,
    file_path: String,
    metadata: Option<HashMap<String, serde_json::Value>>,
  ) -> Result<(), PluginError> {
    let mut metadata = metadata.unwrap_or_default();
    metadata.insert("chat_id".to_string(), json!(chat_id));
    let params = json!({ "metadata": metadata, "file_path": json!(file_path) });
    trace!("[AI Plugin] indexing file: {:?}", params);
    self
      .send_request::<EmptyResponseParser>("embed_file", params)
      .await
  }

  #[instrument(level = "debug", skip(self), err)]
  pub async fn complete_text(
    &self,
    message: &str,
    complete_type: u8,
    format: Option<serde_json::Value>,
  ) -> Result<ReceiverStream<Result<Bytes, PluginError>>, PluginError> {
    let plugin = self.get_plugin()?;
    let mut inner_params = serde_json::Map::new();
    inner_params.insert("text".to_string(), json!(message));
    inner_params.insert("completion_type".to_string(), json!(complete_type));
    if let Some(fmt) = format {
      inner_params.insert("format".to_string(), fmt);
    }

    let params = json!({
        "method": "complete_text",
        "params": serde_json::Value::Object(inner_params)
    });

    plugin.stream_request::<ChatStreamResponseParser>("handle", &params)
  }
  #[instrument(level = "debug", skip_all, err)]
  pub async fn complete_text_v2(
    &self,
    message: &str,
    complete_type: u8,
    format: Option<Value>,
    metadata: Option<Value>,
  ) -> Result<ReceiverStream<Result<Value, PluginError>>, PluginError> {
    let plugin = self.get_plugin()?;

    let mut inner_params = serde_json::Map::new();
    inner_params.insert("text".to_string(), json!(message));
    inner_params.insert("completion_type".to_string(), json!(complete_type));
    if let Some(fmt) = format {
      inner_params.insert("format".to_string(), fmt);
    }

    if let Some(metadata) = metadata {
      inner_params.insert("metadata".to_string(), metadata);
    }

    let params = json!({
        "method": "complete_text_v2",
        "params": Value::Object(inner_params)
    });

    plugin.stream_request::<JsonStringToJsonObject>("handle", &params)
  }

  #[instrument(level = "debug", skip(self), err)]
  pub async fn summary_row(&self, row: HashMap<String, String>) -> Result<String, PluginError> {
    self
      .send_request::<DatabaseSummaryResponseParser>("database_summary", json!(row))
      .await
  }

  #[instrument(level = "debug", skip(self), err)]
  pub async fn translate_row(
    &self,
    data: LocalAITranslateRowData,
  ) -> Result<LocalAITranslateRowResponse, PluginError> {
    self
      .send_request::<DatabaseTranslateResponseParser>("database_translate", json!(data))
      .await
  }
}

#[derive(Clone, Debug, Serialize)]
pub struct LocalAITranslateRowData {
  pub cells: Vec<LocalAITranslateItem>,
  pub language: String,
  pub include_header: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct LocalAITranslateItem {
  pub title: String,
  pub content: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LocalAITranslateRowResponse {
  pub items: Vec<HashMap<String, String>>,
}

pub struct ChatResponseParser;
impl ResponseParser for ChatResponseParser {
  type ValueType = String;

  fn parse_json(json: JsonValue) -> Result<Self::ValueType, RemoteError> {
    json
      .get("data")
      .and_then(|data| data.as_str())
      .map(String::from)
      .ok_or(RemoteError::ParseResponse(json))
  }
}

pub struct DataJsonParser;
impl ResponseParser for DataJsonParser {
  type ValueType = Value;

  fn parse_json(json: JsonValue) -> Result<Self::ValueType, RemoteError> {
    json
      .get("data")
      .cloned()
      .ok_or(RemoteError::ParseResponse(json))
  }
}

pub struct ChatStreamResponseParser;
impl ResponseParser for ChatStreamResponseParser {
  type ValueType = Bytes;

  fn parse_json(json: JsonValue) -> Result<Self::ValueType, RemoteError> {
    json
      .as_str()
      .map(|message| Bytes::from(message.to_string()))
      .ok_or(RemoteError::ParseResponse(json))
  }
}

pub struct JsonStringToJsonObject;
impl ResponseParser for JsonStringToJsonObject {
  type ValueType = serde_json::Value;

  fn parse_json(json: JsonValue) -> Result<Self::ValueType, RemoteError> {
    json
      .as_str()
      .and_then(|s| serde_json::from_str(s).ok())
      .ok_or(RemoteError::ParseResponse(json))
  }
}

pub struct ChatRelatedQuestionsResponseParser;
impl ResponseParser for ChatRelatedQuestionsResponseParser {
  type ValueType = Vec<String>;

  fn parse_json(json: JsonValue) -> Result<Self::ValueType, RemoteError> {
    json
      .get("data")
      .and_then(|data| data.as_array())
      .map(|array| {
        array
          .iter()
          .flat_map(|item| {
            item
              .get("content")
              .map(|s| s.as_str().map(|s| s.to_string()))?
          })
          .collect()
      })
      .ok_or(RemoteError::ParseResponse(json))
  }
}

#[derive(Debug, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum CompleteTextType {
  ImproveWriting = 1,
  SpellingAndGrammar = 2,
  MakeShorter = 3,
  MakeLonger = 4,
  ContinueWriting = 5,
  Explain = 6,
  AskAI = 7,
  Custom = 8,
}

impl From<u8> for CompleteTextType {
  fn from(value: u8) -> Self {
    match value {
      1 => CompleteTextType::ImproveWriting,
      2 => CompleteTextType::SpellingAndGrammar,
      3 => CompleteTextType::MakeShorter,
      4 => CompleteTextType::MakeLonger,
      5 => CompleteTextType::ContinueWriting,
      6 => CompleteTextType::Explain,
      7 => CompleteTextType::AskAI,
      8 => CompleteTextType::Custom,
      _ => CompleteTextType::AskAI,
    }
  }
}

pub struct DatabaseSummaryResponseParser;
impl ResponseParser for DatabaseSummaryResponseParser {
  type ValueType = String;

  fn parse_json(json: JsonValue) -> Result<Self::ValueType, RemoteError> {
    json
      .get("data")
      .and_then(|data| data.as_str())
      .map(|s| s.to_string())
      .ok_or(RemoteError::ParseResponse(json))
  }
}

pub struct DatabaseTranslateResponseParser;
impl ResponseParser for DatabaseTranslateResponseParser {
  type ValueType = LocalAITranslateRowResponse;

  fn parse_json(json: JsonValue) -> Result<Self::ValueType, RemoteError> {
    json
      .get("data")
      .and_then(|data| LocalAITranslateRowResponse::deserialize(data).ok())
      .ok_or(RemoteError::ParseResponse(json))
  }
}

// async fn collect_answer(
//   mut stream: QuestionStream,
//   stop_when_num_of_char: Option<usize>,
// ) -> String {
//   let mut answer = String::new();
//   let mut num_of_char: usize = 0;
//   while let Some(value) = stream.next().await {
//     num_of_char += match value.unwrap() {
//       QuestionStreamValue::Answer { value } => {
//         answer.push_str(&value);
//         value.len()
//       },
//       QuestionStreamValue::Metadata { .. } => 0,
//       QuestionStreamValue::KeepAlive => 0,
//     };
//     if let Some(stop_when_num_of_char) = stop_when_num_of_char {
//       if num_of_char >= stop_when_num_of_char {
//         break;
//       }
//     }
//   }
//   answer
// }
