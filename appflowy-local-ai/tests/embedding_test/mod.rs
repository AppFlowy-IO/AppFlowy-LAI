use crate::util::LocalAITest;
use serde_json::json;
use std::collections::HashMap;

#[tokio::test]
async fn ci_generate_embedding_test() {
  let test = LocalAITest::new().unwrap();
  test.init_embedding_plugin().await;

  let id = uuid::Uuid::new_v4().to_string();
  let mut metadata = HashMap::new();
  metadata.insert("id".to_string(), json!(id));

  test.embedding_manager.index("AppFlowy is an AI collaborative workspace where you achieve more without losing control of your data", metadata.clone()).await.unwrap();
  let resp = test
    .embedding_manager
    .similarity_search("AppFlowy", metadata)
    .await
    .unwrap();
  eprintln!("embedding response: {:?}", resp);
}
