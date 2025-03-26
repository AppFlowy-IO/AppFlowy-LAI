use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
pub struct ToolsList {
  pub tools: Vec<Tool>,
}

#[derive(Debug, Deserialize)]
pub struct Tool {
  pub name: String,
  pub description: String,
  #[serde(rename = "inputSchema")]
  pub input_schema: InputSchema,
}

#[derive(Debug, Deserialize)]
pub struct InputSchema {
  #[serde(rename = "type")]
  pub schema_type: String,
  pub properties: HashMap<String, Property>,
  #[serde(default)]
  pub description: Option<String>,
  pub required: Option<Vec<String>>,
  pub title: String,
  #[serde(rename = "$defs", default)]
  pub defs: Option<HashMap<String, InputSchema>>,
}

#[derive(Debug, Deserialize)]
pub struct Property {
  #[serde(default)]
  pub default: Option<Value>,
  #[serde(default)]
  pub items: Option<Box<Property>>,
  #[serde(rename = "$ref", default)]
  pub reference: Option<String>,
  #[serde(default)]
  pub description: Option<String>,
  pub title: Option<String>,
  #[serde(rename = "type")]
  pub property_type: Option<String>,
}
