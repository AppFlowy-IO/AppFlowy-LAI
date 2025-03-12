use anyhow::anyhow;
use appflowy_plugin::core::parser::{DefaultResponseParser, ResponseParser};
use appflowy_plugin::core::plugin::Plugin;
use appflowy_plugin::error::{PluginError, RemoteError};
use serde_json::Value as JsonValue;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Weak;

pub struct EmbeddingPluginOperation {
  plugin: Weak<Plugin>,
}

impl EmbeddingPluginOperation {
  pub fn new(plugin: Weak<Plugin>) -> Self {
    EmbeddingPluginOperation { plugin }
  }

  pub async fn embed_documents(&self, message: &str) -> Result<Vec<Vec<f64>>, PluginError> {
    let plugin = self
      .plugin
      .upgrade()
      .ok_or(PluginError::Internal(anyhow!("Plugin is dropped")))?;
    let params = json!({"method": "embed_documents", "params": {"input": message }});
    plugin
      .async_request::<EmbeddingResponseParse>("handle", &params)
      .await
  }

  pub async fn embed_text(
    &self,
    message: &str,
    metadata: HashMap<String, Value>,
  ) -> Result<(), PluginError> {
    let plugin = self
      .plugin
      .upgrade()
      .ok_or(PluginError::Internal(anyhow!("Plugin is dropped")))?;
    let metadata = json!(metadata);
    let params =
      json!({"method": "embed_text", "params": {"input": message, "metadata": metadata }});
    plugin
      .async_request::<DefaultResponseParser>("handle", &params)
      .await
  }

  pub async fn similarity_search(
    &self,
    query: &str,
    filter: HashMap<String, Value>,
  ) -> Result<Vec<String>, PluginError> {
    let plugin = self
      .plugin
      .upgrade()
      .ok_or(PluginError::Internal(anyhow!("Plugin is dropped")))?;
    let params =
      json!({"method": "similarity_search", "params": {"query": query, "filter": filter }});
    plugin
      .async_request::<SimilaritySearchResponseParse>("handle", &params)
      .await
  }
}

pub struct SimilaritySearchResponseParse;
impl ResponseParser for SimilaritySearchResponseParse {
  type ValueType = Vec<String>;

  fn parse_json(json: JsonValue) -> Result<Self::ValueType, RemoteError> {
    if json.is_object() {
      if let Some(data) = json.get("data") {
        if let Some(array) = data.as_array() {
          let mut result = Vec::new();
          for item in array {
            if let Some(value) = item.as_str() {
              result.push(value.to_string());
            } else {
              return Err(RemoteError::ParseResponse(json));
            }
          }
          return Ok(result);
        }
      }
    }
    Err(RemoteError::ParseResponse(json))
  }
}

pub struct EmbeddingResponseParse;
impl ResponseParser for EmbeddingResponseParse {
  type ValueType = Vec<Vec<f64>>;

  fn parse_json(json: JsonValue) -> Result<Self::ValueType, RemoteError> {
    if json.is_object() {
      if let Some(embeddings) = json.get("data") {
        if let Some(array) = embeddings.as_array() {
          let mut result = Vec::new();
          for item in array {
            if let Some(inner_array) = item.as_array() {
              let mut inner_result = Vec::new();
              for num in inner_array {
                if let Some(value) = num.as_f64() {
                  inner_result.push(value);
                } else {
                  return Err(RemoteError::ParseResponse(json));
                }
              }
              result.push(inner_result);
            } else {
              return Err(RemoteError::ParseResponse(json));
            }
          }
          return Ok(result);
        }
      }
    }
    Err(RemoteError::ParseResponse(json))
  }
}
