//
#[tokio::test]
async fn load_aws_chat_bin_test() {
  setup_log();
  let plugin_manager = PluginManager::new();
  let llm_chat = OllamaAIPlugin::new(Arc::new(plugin_manager));

  let chat_bin = OLLAMA_PLUGIN_EXE_PATH().await;
  // clear_extended_attributes(&chat_bin).await.unwrap();

  let mut chat_config = OllamaPluginConfig::new(chat_bin, chat_model()).unwrap();
  chat_config = chat_config.with_device("gpu");
  llm_chat.init_chat_plugin(chat_config).await.unwrap();

  let chat_id = uuid::Uuid::new_v4().to_string();
  let resp = llm_chat
    .ask_question(&chat_id, "what is banana?")
    .await
    .unwrap();
  assert!(!resp.is_empty());
  eprintln!("response: {:?}", resp);
}

async fn OLLAMA_PLUGIN_EXE_PATH() -> PathBuf {
  let url = "https://appflowy-local-ai.s3.amazonaws.com/macos-latest/AppFlowyAI_release.zip?AWSAccessKeyId=AKIAVQA4ULIFKSXHI6PI&Signature=p8evDjdypl58nbGK8qJ%2F1l0Zs%2FU%3D&Expires=1721044152";
  let temp_dir = temp_dir().join("download_plugin");
  if !temp_dir.exists() {
    std::fs::create_dir(&temp_dir).unwrap();
  }
  let path = download_plugin(url, &temp_dir, "AppFlowyAI.zip", None, None, None)
    .await
    .unwrap();
  println!("Downloaded plugin to {:?}", path);

  zip_extract(&path, &temp_dir).unwrap();
  temp_dir.join("appflowy_ai_plugin")
}

fn chat_model() -> PathBuf {
  let model_dir = PathBuf::from(dotenv::var("OLLAMA_SERVER_URL").unwrap());
  let chat_model = dotenv::var("OLLAMA_CHAT_MODEL_NAME").unwrap();
  model_dir.join(chat_model)
}
