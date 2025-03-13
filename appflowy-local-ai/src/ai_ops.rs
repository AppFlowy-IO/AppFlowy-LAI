use anyhow::anyhow;
use appflowy_plugin::core::parser::{DefaultResponseParser, ResponseParser};
use appflowy_plugin::core::plugin::Plugin;
use appflowy_plugin::error::{PluginError, RemoteError};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value as JsonValue;
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

  pub async fn create_chat(&self, chat_id: &str) -> Result<(), PluginError> {
    self
      .send_request::<DefaultResponseParser>(
        "create_chat",
        json!({ "chat_id": chat_id, "top_k": 2}),
      )
      .await
  }

  pub async fn close_chat(&self, chat_id: &str) -> Result<(), PluginError> {
    self
      .send_request::<DefaultResponseParser>("close_chat", json!({ "chat_id": chat_id }))
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
    let params = json!({
        "method": "stream_answer_v2",
        "params": {"chat_id": chat_id, "data": {"content": message}, "metadata": metadata, "format": format}
    });
    plugin.stream_request::<ChatStreamResponseV2Parser>("handle", &params)
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
    file_path: Option<String>,
    file_content: Option<String>,
    metadata: Option<HashMap<String, serde_json::Value>>,
  ) -> Result<(), PluginError> {
    if file_path.is_none() && file_content.is_none() {
      return Err(PluginError::Internal(anyhow!(
        "file_path or content must be provided"
      )));
    }

    let mut metadata = metadata.unwrap_or_default();
    metadata.insert("chat_id".to_string(), json!(chat_id));
    let mut params = json!({ "metadatas": [metadata] });

    if let Some(file_path) = file_path {
      params["file_path"] = json!(file_path);
    }

    if let Some(content) = file_content {
      params["file_content"] = json!(content);
    }

    trace!("[AI Plugin] indexing file: {:?}", params);
    self
      .send_request::<DefaultResponseParser>(
        "embed_file",
        json!({ "chat_id": chat_id, "params": params }),
      )
      .await
  }

  #[instrument(level = "debug", skip(self), err)]
  pub async fn complete_text<T: Into<CompleteTextType> + Debug>(
    &self,
    message: &str,
    complete_type: T,
    format: Option<serde_json::Value>,
  ) -> Result<ReceiverStream<Result<Bytes, PluginError>>, PluginError> {
    let plugin = self.get_plugin()?;
    let complete_type = complete_type.into() as u8;
    let params = json!({
        "method": "complete_text",
        "params": { "text": message, "type": complete_type, "format": format }
    });
    plugin.stream_request::<ChatStreamResponseParser>("handle", &params)
  }

  #[instrument(level = "debug", skip(self), err)]
  pub async fn summary_row(&self, row: HashMap<String, String>) -> Result<String, PluginError> {
    let params = json!({"params": row });
    self
      .send_request::<DatabaseSummaryResponseParser>("database_summary", params)
      .await
  }

  #[instrument(level = "debug", skip(self), err)]
  pub async fn translate_row(
    &self,
    data: LocalAITranslateRowData,
  ) -> Result<LocalAITranslateRowResponse, PluginError> {
    let params = json!({"params": data });
    self
      .send_request::<DatabaseTranslateResponseParser>("database_translate", params)
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

pub struct ChatStreamResponseV2Parser;
impl ResponseParser for ChatStreamResponseV2Parser {
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
  AskAI = 5,
}

impl From<u8> for CompleteTextType {
  fn from(value: u8) -> Self {
    match value {
      1 => CompleteTextType::ImproveWriting,
      2 => CompleteTextType::SpellingAndGrammar,
      3 => CompleteTextType::MakeShorter,
      4 => CompleteTextType::MakeLonger,
      5 => CompleteTextType::AskAI,
      _ => CompleteTextType::ImproveWriting,
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
