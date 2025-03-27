use crate::entities::ToolsList;
use anyhow::Result;
use mcp_daemon::protocol::RequestOptions;
use mcp_daemon::transport::{ClientStdioTransport, Transport};
use mcp_daemon::types::Implementation;
use mcp_daemon::Client;
use serde_json::{json, Value};
use std::time::Duration;
use tracing::{error, info};

#[derive(Debug, Clone)]
pub struct MCPServerConfig {
  pub server_cmd: String,
  pub args: Vec<String>,
}

// https://modelcontextprotocol.io/docs/tools/inspector
// https://modelcontextprotocol.io/docs/concepts/tools
#[derive(Clone)]
pub struct MCPClient {
  pub client: Client<ClientStdioTransport>,
  pub transport: ClientStdioTransport,
  pub server_config: MCPServerConfig,
}

impl MCPClient {
  pub async fn new_stdio(config: MCPServerConfig) -> Result<Self> {
    info!(
      "Connecting to running server with command: {} {}",
      config.server_cmd,
      config.args.join(" ")
    );
    let args_str: Vec<&str> = config.args.iter().map(String::as_str).collect();
    let transport = ClientStdioTransport::new(&config.server_cmd, &args_str)?;
    let client = Client::builder(transport.clone()).build();
    Ok(MCPClient {
      client,
      transport,
      server_config: config,
    })
  }

  pub async fn initialize(&self) -> Result<()> {
    self.transport.open().await?;

    let cloned_client = self.client.clone();
    tokio::spawn(async move {
      if let Err(err) = cloned_client.start().await {
        error!("Error starting client: {}", err);
      }
    });

    let implementation = Implementation {
      name: "mcp-client".to_string(),
      version: "0.0.1".to_string(),
    };
    self.client.initialize(implementation).await?;
    Ok(())
  }

  pub async fn ping(&self) -> Result<Value> {
    let resp = self
      .client
      .request("ping", None, Default::default())
      .await?;
    Ok(resp)
  }

  pub async fn list_tools(&self) -> Result<ToolsList> {
    let resp = self
      .client
      .request("tools/list", None, Default::default())
      .await?;
    dbg!(&resp);

    let tools = serde_json::from_value::<ToolsList>(resp)?;
    Ok(tools)
  }

  /// Send a tools/call request to MCP server with parameters
  pub async fn call_tool(
    &self,
    name: &str,
    arguments: Option<Value>,
    timeout: Option<Duration>,
  ) -> Result<Value> {
    let timeout = timeout.unwrap_or_else(|| Duration::from_secs(5));
    let resp = self
      .client
      .request(
        "tools/call",
        Some(json!({
          "name": name,
          "arguments": arguments
        })),
        RequestOptions::default().timeout(timeout),
      )
      .await?;
    Ok(resp)
  }

  pub async fn stop(&mut self) -> Result<()> {
    self.transport.close().await?;
    Ok(())
  }
}
