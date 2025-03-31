use crate::util::{collect_completion_stream, collect_json_stream, get_asset_path, LocalAITest};

use std::collections::HashMap;

use af_local_ai::ai_ops::{CompleteTextType, LocalAITranslateItem, LocalAITranslateRowData};

use serde_json::json;

use tokio_stream::StreamExt;

#[tokio::test]
async fn load_chat_model_test() {
  let test = LocalAITest::new().unwrap();
  test.init_chat_plugin().await;

  let plugin_info = test.ollama_plugin.plugin_info().await.unwrap();
  println!("plugin info: {:?}", plugin_info);

  let chat_id = uuid::Uuid::new_v4().to_string();
  let resp = test
    .send_chat_message(&chat_id, "translate 你好 to english")
    .await;
  eprintln!("chat response: {:?}", resp);

  let score = test.calculate_similarity(&resp, "Hello").await;
  assert!(score > 0.8, "score: {}", score);
}

#[tokio::test]
async fn ci_chat_stream_test() {
  let test = LocalAITest::new().unwrap();
  test.init_chat_plugin().await;

  let chat_plugin = test
    .ollama_plugin
    .get_ai_plugin()
    .await
    .unwrap()
    .upgrade()
    .unwrap();
  let mut state_rx = chat_plugin.subscribe_running_state();
  tokio::spawn(async move {
    while let Some(state) = state_rx.next().await {
      eprintln!("chat state: {:?}", state);
    }
  });

  let chat_id = uuid::Uuid::new_v4().to_string();
  let resp = test
    .stream_chat_message(&chat_id, "what is banana?", None)
    .await;
  let answer = collect_json_stream(resp).await;
  println!("stream response: {:?}", answer);

  let expected = r#"banana is a fruit that belongs to the genus _______, which also includes other fruits such as apple and pear. It has several varieties with different shapes, colors, and flavors depending on where it grows. Bananas are typically green or yellow in color and have smooth skin that peels off easily when ripe. They are sweet and juicy, often eaten raw or roasted, and can also be used for cooking and baking. In some cultures, banana is considered a symbol of good luck, fertility, and prosperity. Bananas originated in Southeast Asia, where they were cultivated by early humans thousands of years ago. They are now grown around the world as a major crop, with significant production in many countries including the United States, Brazil, India, and China#"#;
  let score = test.calculate_similarity(&answer, expected).await;
  assert!(score > 0.7, "score: {}", score);

  let questions = test
    .ollama_plugin
    .get_related_question(&chat_id)
    .await
    .unwrap();
  assert_eq!(questions.len(), 3);
  println!("related questions: {:?}", questions)
}

#[tokio::test]
async fn ci_completion_text_v2_test() {
  let test = LocalAITest::new().unwrap();
  test.init_chat_plugin().await;

  let chat_plugin = test
    .ollama_plugin
    .get_ai_plugin()
    .await
    .unwrap()
    .upgrade()
    .unwrap();
  let mut state_rx = chat_plugin.subscribe_running_state();
  tokio::spawn(async move {
    while let Some(state) = state_rx.next().await {
      eprintln!("chat state: {:?}", state);
    }
  });

  let resp = test
    .ollama_plugin
    .complete_text_v2(
      "Me and him was going to the store, but we didn’t had enough money",
      CompleteTextType::SpellingAndGrammar as u8,
      None,
      Some(json!({
        "object_id": "123",
      })),
    )
    .await
    .unwrap();

  let (answer, comment) = collect_completion_stream(resp).await;
  eprintln!("answer: {:?}", answer);
  eprintln!("comment: {:?}", comment);

  let expected = r#"He and I were going to the store, but we didn’t have enough money"#;
  let score = test.calculate_similarity(&answer, expected).await;
  assert!(score > 0.7, "score: {}", score);

  let expected = r#"The subject "Me and him" was corrected to "He and I" because "I" is the correct subject pronoun when referring to oneself in the subject position. "Was" was changed to "were" to agree with the plural subject. "Didn’t had" was corrected to "didn’t have" as "didn't" requires the base form of"#;
  let score = test.calculate_similarity(&comment, expected).await;
  assert!(score > 0.7, "score: {}", score);
}

#[tokio::test]
async fn ci_completion_text_v2_unicode_test() {
  let test = LocalAITest::new().unwrap();
  test.init_chat_plugin().await;

  let chat_plugin = test
    .ollama_plugin
    .get_ai_plugin()
    .await
    .unwrap()
    .upgrade()
    .unwrap();
  let mut state_rx = chat_plugin.subscribe_running_state();
  tokio::spawn(async move {
    while let Some(state) = state_rx.next().await {
      eprintln!("chat state: {:?}", state);
    }
  });

  let resp = test
    .ollama_plugin
    .complete_text_v2(
      "He starts work everyday at 8 a.m. 然后他开始工作了一整天， 没有♨️",
      CompleteTextType::ImproveWriting as u8,
      None,
      Some(json!({
        "object_id": "123",
      })),
    )
    .await
    .unwrap();

  let (answer, comment) = collect_completion_stream(resp).await;
  eprintln!("answer: {:?}", answer);
  eprintln!("comment: {:?}", comment);
}

#[tokio::test]
async fn ci_chat_with_pdf() {
  let test = LocalAITest::new().unwrap();
  test.init_chat_plugin().await;
  let chat_id = uuid::Uuid::new_v4().to_string();
  let pdf = get_asset_path("AppFlowy_Values.pdf");
  test
    .ollama_plugin
    .embed_file(&chat_id, pdf, None)
    .await
    .unwrap();

  let resp = test
    .ollama_plugin
    .stream_question(&chat_id, "what is AppFlowy Values?", None, json!({}))
    .await
    .unwrap();
  let answer = collect_json_stream(resp).await;
  println!("chat with pdf response: {}", answer);

  let expected = r#"
  1. **Mission Driven**: Our mission is to enable everyone to unleash their potential and achieve more with secure workplace tools.
  2. **Collaboration**: We pride ourselves on being a great team. We foster collaboration, value diversity and inclusion, and encourage sharing.
  3. **Honesty**: We are honest with ourselves. We admit mistakes freely and openly. We provide candid, helpful, timely feedback to colleagues with respect, regardless of their status or whether they disagree with us.
  4. **Aim High and Iterate**: We strive for excellence with a growth mindset. We dream big, start small, and move fast. We take smaller steps and ship smaller, simpler features.
  5. **Transparency**: We make information about AppFlowy public by default unless there is a compelling reason not to. We are straightforward and kind with ourselves and each other.
  "#;
  let score = test.calculate_similarity(&answer, expected).await;
  assert!(score > 0.6, "score: {}", score);
}

#[tokio::test]
async fn ci_database_row_test() {
  let test = LocalAITest::new().unwrap();
  test.init_chat_plugin().await;

  // summary
  let mut params = HashMap::new();
  params.insert("book name".to_string(), "Atomic Habits".to_string());
  params.insert("finish reading at".to_string(), "2023-02-10".to_string());
  params.insert(
    "notes".to_string(),
    "An atomic habit is a regular practice or routine that is not
           only small and easy to do but is also the source of incredible power; a
           component of the system of compound growth. Bad habits repeat themselves
           again and again not because you don’t want to change, but because you
           have the wrong system for change. Changes that seem small and
           unimportant at first will compound into remarkable results if you’re
           willing to stick with them for years"
      .to_string(),
  );
  let resp = test
    .ollama_plugin
    .summary_database_row(params)
    .await
    .unwrap();
  let expected = r#"
  Finished reading "Atomic Habits" on 2023-02-10. The book emphasizes that
  small, regular practices can lead to significant growth over time. Bad
  habits persist due to flawed systems, and minor, consistent changes can
  yield impressive results when maintained over the long term.
  "#;
  let score = test.calculate_similarity(&resp, expected).await;
  assert!(score > 0.8, "score: {}", score);

  // translate
  let data = LocalAITranslateRowData {
    cells: vec![
      LocalAITranslateItem {
        title: "book name".to_string(),
        content: "Atomic Habits".to_string(),
      },
      LocalAITranslateItem {
        title: "score".to_string(),
        content: "8".to_string(),
      },
      LocalAITranslateItem {
        title: "finish reading at".to_string(),
        content: "2023-02-10".to_string(),
      },
    ],
    language: "chinese".to_string(),
    include_header: false,
  };
  let resp = test
    .ollama_plugin
    .translate_database_row(data)
    .await
    .unwrap();
  let resp_str: String = resp
    .items
    .into_iter()
    .flat_map(|map| map.into_iter().map(|(k, v)| format!("{}:{}", k, v)))
    .collect::<Vec<String>>()
    .join(",");

  let expected = r#"书名:原子习惯,评分:8,完成阅读日期:2023-02-10"#;
  let score = test.calculate_similarity(&resp_str, expected).await;
  assert!(score > 0.8, "score: {}, actural: {}", score, resp_str);
}

#[tokio::test]
async fn destroy_plugin_test() {
  let test = LocalAITest::new().unwrap();

  test.init_chat_plugin().await;
}
