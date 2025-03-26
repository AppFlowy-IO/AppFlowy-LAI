use af_mcp::client::{MCPClient, MCPServerConfig};

#[tokio::test]
async fn connect_to_server() {
  // Load environment variables from a .env file, if available
  dotenv::dotenv().ok();

  // Get the path to the MCP server executable from the environment
  let command = dotenv::var("MCP_SERVER_EXE_PATH").unwrap_or_default();
  if command.is_empty() {
    panic!("MCP_SERVER_EXE_PATH environment variable is not set");
  }

  let config = MCPServerConfig {
    server_cmd: command,
    args: vec![".".to_string()],
  };

  let client = MCPClient::new_stdio(config)
    .await
    .expect("Failed to create MCPClient");

  // Initialize the client
  client.initialize().await.expect("Initialization failed");

  // Send a ping to the server and print the response
  let resp = client.ping().await.expect("Ping failed");
  println!("Ping response: {}", resp);

  // List the tools using the updated list_tools() method
  let tools_list = client.list_tools().await.expect("Listing tools failed");
  dbg!(&tools_list);

  // Ensure the tools vector is not empty
  assert!(
    !tools_list.tools.is_empty(),
    "Tools array should not be empty"
  );

  // Collect the tool names from the ToolsList struct
  let tool_names: Vec<&str> = tools_list
    .tools
    .iter()
    .map(|tool| tool.name.as_str())
    .collect();

  // Expected tool names based on the JSON input
  let expected_tools = vec![
    "read_file",
    "read_multiple_files",
    "write_file",
    "edit_file",
    "create_directory",
    "list_directory",
    "directory_tree",
    "move_file",
    "search_files",
    "get_file_info",
    "list_allowed_dirs",
  ];

  for expected in expected_tools {
    assert!(
      tool_names.contains(&expected),
      "Expected tool '{}' is missing",
      expected
    );
  }
}
