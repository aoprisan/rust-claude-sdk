//! The client against a local stand-in for the API: request shape and headers, citations,
//! unknown block types, server tools, retries, deadlines, streaming, token counting,
//! batches, files and models.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::Router;
use rust_claude_sdk::{
    BatchOutcome, BatchRequest, CacheControl, CallOptions, Citation, Client, ClientConfig, CodeExecutionContent, CodeExecutionTool, ContentBlock,
    ContentBlockParam, Credential, Delta, Effort, Error, Fallbacks, ListParams, MessageParam, MessagesRequest, ProcessingStatus, StopReason, StreamEvent,
    ThinkingConfig, ThinkingDisplay, Tool, ToolLoopOptions, ToolOutput, ToolSearchTool, UserLocation, WebFetchContent, WebFetchTool, WebSearchContent,
    WebSearchTool,
};
use serde_json::{json, Value};

const KEY: &str = "sk-ant-test-SECRET";

#[derive(Clone)]
struct Reply {
    status: u16,
    headers: Vec<(&'static str, &'static str)>,
    body: String,
    delay: Duration,
}

fn reply(status: u16, body: impl Into<String>) -> Reply {
    Reply { status, headers: Vec::new(), body: body.into(), delay: Duration::ZERO }
}

/// Headers and JSON body of one request.
type Seen = (HeaderMap, Value);

#[derive(Clone)]
struct Server {
    replies: Arc<Vec<Reply>>,
    hits: Arc<AtomicUsize>,
    seen: Arc<Mutex<Vec<Seen>>>,
    /// Method, path with query, and raw body of each request.
    calls: Arc<Mutex<Vec<(String, String, String)>>>,
}

async fn handle(State(s): State<Server>, method: Method, uri: Uri, headers: HeaderMap, body: String) -> Response {
    let n = s.hits.fetch_add(1, Ordering::SeqCst);
    s.seen.lock().unwrap().push((headers, serde_json::from_str(&body).unwrap_or(Value::Null)));
    s.calls.lock().unwrap().push((method.to_string(), uri.to_string(), body));
    let r = s.replies.get(n).or(s.replies.last()).unwrap().clone();
    tokio::time::sleep(r.delay).await;
    let mut res = (StatusCode::from_u16(r.status).unwrap(), r.body).into_response();
    for (k, v) in r.headers {
        res.headers_mut().insert(k, v.parse().unwrap());
    }
    res
}

async fn serve(replies: Vec<Reply>) -> (String, Server) {
    let server = Server { replies: Arc::new(replies), hits: Arc::default(), seen: Arc::default(), calls: Arc::default() };
    let app = Router::new().fallback(handle).with_state(server.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, server)
}

fn client(url: &str, max_retries: u32) -> Client {
    Client::new(ClientConfig { base_url: format!("{url}/"), max_retries, ..ClientConfig::api_key(KEY) }).unwrap()
}

fn header<'a>(h: &'a HeaderMap, name: &str) -> Option<&'a str> {
    h.get(name).and_then(|v| v.to_str().ok())
}

const CITED: &str = r#"{
  "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
  "content": [
    {"type": "text", "text": "Cartea de identitate se eliberează în ", "citations": null},
    {"type": "text", "text": "30 de zile", "citations": [
      {"type": "char_location", "cited_text": "Termenul este de 30 de zile.", "document_index": 0,
       "document_title": "Ghid CI", "start_char_index": 0, "end_char_index": 28},
      {"type": "search_result_location", "cited_text": "30 de zile", "source": "https://hub.mai.gov.ro", "search_result_index": 0}
    ]},
    {"type": "brand_new_block", "id": "x_1", "payload": {"query": "ci"}},
    {"type": "text", "text": "."}
  ],
  "stop_reason": "end_turn", "stop_sequence": null,
  "usage": {"input_tokens": 1200, "output_tokens": 40, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 900, "service_tier": "standard"}
}"#;

fn grounded_request() -> MessagesRequest {
    MessagesRequest::new("claude-opus-5-5", 1500).system_cached("Answer only from the documents.").effort(Effort::Low).fallbacks(Fallbacks::Default).message(
        MessageParam::user(vec![
            ContentBlockParam::text_document("Termenul este de 30 de zile.", "Ghid CI").with_context("hub.mai.gov.ro").with_citations(),
            ContentBlockParam::text("În cât timp primesc buletinul?"),
        ]),
    )
}

#[tokio::test]
async fn sends_the_documented_request_shape_and_headers() {
    let (url, server) = serve(vec![reply(200, CITED)]).await;
    client(&url, 0).create(&grounded_request()).await.unwrap();

    let seen = server.seen.lock().unwrap();
    let (headers, body) = &seen[0];
    assert_eq!(header(headers, "x-api-key"), Some(KEY));
    assert_eq!(header(headers, "anthropic-version"), Some("2023-06-01"));
    assert_eq!(header(headers, "anthropic-beta"), Some("server-side-fallback-2026-07-01"));
    assert_eq!(header(headers, "content-type"), Some("application/json"));
    assert_eq!(
        body,
        &json!({
            "model": "claude-opus-5-5",
            "max_tokens": 1500,
            "messages": [{"role": "user", "content": [
                {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "Termenul este de 30 de zile."},
                 "title": "Ghid CI", "context": "hub.mai.gov.ro", "citations": {"enabled": true}},
                {"type": "text", "text": "În cât timp primesc buletinul?"}
            ]}],
            "system": [{"type": "text", "text": "Answer only from the documents.", "cache_control": {"type": "ephemeral"}}],
            "output_config": {"effort": "low"},
            "fallbacks": "default"
        })
    );
}

#[tokio::test]
async fn decodes_citations_and_keeps_unknown_blocks_for_echoing() {
    let (url, _) = serve(vec![reply(200, CITED)]).await;
    let m = client(&url, 0).create(&grounded_request()).await.unwrap();

    assert_eq!(m.text(), "Cartea de identitate se eliberează în 30 de zile.");
    assert_eq!(m.stop_reason, Some(StopReason::EndTurn));
    assert!(!m.is_refusal());
    assert_eq!(m.usage.cache_read_input_tokens, Some(900));
    assert_eq!(m.usage.extra.get("service_tier"), Some(&json!("standard")));

    let cites: Vec<_> = m.citations().collect();
    assert_eq!(cites.len(), 2);
    assert_eq!(cites[0].0, "30 de zile");
    assert!(matches!(cites[0].1, Citation::CharLocation { document_index: 0, end_char_index: 28, .. }));
    assert_eq!(cites[1].1.cited_text(), Some("30 de zile"));
    assert!(matches!(cites[1].1, Citation::Other(_)));

    // The unknown block survives a round trip unchanged, so an assistant turn echoes intact.
    let ContentBlock::Other(raw) = &m.content[2] else { panic!("{:?}", m.content[2]) };
    assert_eq!(raw["type"], "brand_new_block");
    let echoed: Vec<ContentBlockParam> = m.content.iter().cloned().map(ContentBlockParam::from).collect();
    let v = serde_json::to_value(MessageParam::assistant(echoed)).unwrap();
    assert_eq!(v["content"][2], serde_json::from_str::<Value>(CITED).unwrap()["content"][2]);
    assert_eq!(v["content"][1]["citations"][1]["type"], "search_result_location");
    assert!(v["content"][0].get("citations").is_none());
}

#[tokio::test]
async fn retries_overload_honouring_retry_after_then_succeeds() {
    let overloaded = r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
    let busy = Reply { headers: vec![("retry-after", "0")], ..reply(529, overloaded) };
    let (url, server) = serve(vec![busy, reply(200, CITED)]).await;
    let m = client(&url, 2).create(&grounded_request()).await.unwrap();
    assert_eq!(m.id, "msg_1");
    assert_eq!(server.hits.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn client_errors_are_not_retried_and_carry_the_request_id_but_not_the_key() {
    let bad = r#"{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: too large"},"request_id":"req_body"}"#;
    let r = Reply { headers: vec![("request-id", "req_header")], ..reply(400, bad) };
    let (url, server) = serve(vec![r]).await;
    let err = client(&url, 3).create(&grounded_request()).await.unwrap_err();
    assert_eq!(server.hits.load(Ordering::SeqCst), 1);
    match &err {
        Error::Api { status, kind, message, request_id } => {
            assert_eq!(*status, Some(400));
            assert_eq!(kind, "invalid_request_error");
            assert_eq!(message, "max_tokens: too large");
            assert_eq!(request_id.as_deref(), Some("req_header"));
        }
        e => panic!("{e:?}"),
    }
    assert!(!err.is_retryable());
    assert!(!format!("{err} {err:?}").contains(KEY));
    assert!(!format!("{:?}", client(&url, 0)).contains(KEY));
}

#[tokio::test]
async fn x_should_retry_overrides_the_status() {
    let r = Reply { headers: vec![("x-should-retry", "false")], ..reply(500, "oops") };
    let (url, server) = serve(vec![r]).await;
    let err = client(&url, 3).create(&grounded_request()).await.unwrap_err();
    assert_eq!(server.hits.load(Ordering::SeqCst), 1);
    assert_eq!(err.status(), Some(500));
}

#[tokio::test]
async fn the_deadline_cuts_slow_attempts_and_retries_short() {
    let slow = Reply { delay: Duration::from_secs(5), ..reply(200, CITED) };
    let (url, _) = serve(vec![slow]).await;
    let started = std::time::Instant::now();
    let err = client(&url, 5).create_with(&grounded_request(), CallOptions::within(Duration::from_millis(300))).await.unwrap_err();
    assert!(matches!(err, Error::Deadline), "{err:?}");
    assert!(started.elapsed() < Duration::from_secs(2));

    // A retry that would end past the deadline is not attempted: the real error comes back.
    let busy = Reply { headers: vec![("retry-after", "10")], ..reply(429, r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#) };
    let (url, server) = serve(vec![busy]).await;
    let err = client(&url, 5).create_with(&grounded_request(), CallOptions::within(Duration::from_secs(2))).await.unwrap_err();
    assert_eq!(err.status(), Some(429));
    assert_eq!(server.hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn fallback_list_tools_thinking_and_bearer_tokens() {
    let (url, server) = serve(vec![reply(200, CITED)]).await;
    let c = Client::new(ClientConfig { base_url: url, betas: vec!["oauth-2025-04-20".into()], ..ClientConfig::new(Credential::Bearer("tok".into())) }).unwrap();
    let req = MessagesRequest::new("claude-sonnet-5-5", 512)
        .user("salut")
        .thinking(ThinkingConfig::Adaptive { display: Some(ThinkingDisplay::Summarized) })
        .fallbacks(Fallbacks::Models(vec![rust_claude_sdk::FallbackModel { model: "claude-opus-4-8".into(), max_tokens: None }]))
        .tool(
            Tool::new(
                "cui",
                "Checks a CUI",
                json!({"type": "object", "properties": {"cui": {"type": "string"}}, "required": ["cui"], "additionalProperties": false}),
            )
            .strict()
            .with_cache_control(CacheControl::one_hour()),
        )
        .tool(json!({"type": "web_search_20260209", "name": "web_search"}))
        .beta("server-side-fallback-2026-06-01")
        .param("inference_geo", "eu");
    c.create(&req).await.unwrap();

    let seen = server.seen.lock().unwrap();
    let (h, body) = &seen[0];
    assert_eq!(header(h, "authorization"), Some("Bearer tok"));
    assert_eq!(header(h, "x-api-key"), None);
    assert_eq!(header(h, "anthropic-beta"), Some("oauth-2025-04-20,server-side-fallback-2026-06-01"));
    assert_eq!(body["messages"][0], json!({"role": "user", "content": "salut"}));
    assert_eq!(body["thinking"], json!({"type": "adaptive", "display": "summarized"}));
    assert_eq!(body["fallbacks"], json!([{"model": "claude-opus-4-8"}]));
    assert_eq!(body["tools"][0]["cache_control"], json!({"type": "ephemeral", "ttl": "1h"}));
    assert_eq!(body["tools"][0]["strict"], json!(true));
    assert_eq!(body["tools"][1], json!({"type": "web_search_20260209", "name": "web_search"}));
    assert_eq!(body["inference_geo"], "eu");
    assert!(body.get("stream").is_none());
}

#[tokio::test]
async fn refusals_are_visible_before_content() {
    let refused = r#"{"id":"msg_r","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],
        "stop_reason":"refusal","stop_sequence":null,"stop_details":{"type":"refusal","category":null,"explanation":"declined"},
        "usage":{"input_tokens":10,"output_tokens":0}}"#;
    let (url, _) = serve(vec![reply(200, refused)]).await;
    let m = client(&url, 0).create(&grounded_request()).await.unwrap();
    assert!(m.is_refusal());
    assert_eq!(m.stop_details.unwrap()["explanation"], "declined");
}

#[tokio::test]
async fn system_messages_task_budgets_structured_outputs_and_eager_tools() {
    let structured = r#"{"id":"msg_j","type":"message","role":"assistant","model":"claude-opus-5-5",
        "content":[{"type":"text","text":"{\"days\": 30}"}],"stop_reason":"end_turn","stop_sequence":null,
        "usage":{"input_tokens":10,"output_tokens":5}}"#;
    let refused = r#"{"id":"msg_r","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],
        "stop_reason":"refusal","stop_sequence":null,"stop_details":{"type":"refusal","category":"cyber","explanation":null},
        "usage":{"input_tokens":10,"output_tokens":0}}"#;
    let (url, server) = serve(vec![reply(200, structured), reply(200, refused)]).await;
    let schema = json!({"type": "object", "properties": {"days": {"type": "integer"}}, "required": ["days"], "additionalProperties": false});
    let request = MessagesRequest::new("claude-opus-5-5", 20000)
        .effort(Effort::High)
        .task_budget(64000)
        .json_schema(schema.clone())
        .beta("task-budgets-2026-03-13")
        .tool(Tool::new("lookup", "Look up a term", json!({"type": "object"})).eager_input_streaming())
        .user("În cât timp primesc buletinul?")
        .message(MessageParam::system("Answer in JSON."));
    let c = client(&url, 0);
    let m = c.create(&request).await.unwrap();

    #[derive(serde::Deserialize)]
    struct Answer {
        days: u32,
    }
    assert_eq!(m.json::<Answer>().unwrap().days, 30);
    assert!(m.json::<Vec<u32>>().is_err());
    {
        let seen = server.seen.lock().unwrap();
        let (headers, body) = &seen[0];
        assert_eq!(header(headers, "anthropic-beta"), Some("task-budgets-2026-03-13"));
        assert_eq!(
            body["output_config"],
            json!({"effort": "high", "format": {"type": "json_schema", "schema": schema}, "task_budget": {"type": "tokens", "total": 64000}})
        );
        assert_eq!(body["tools"][0]["eager_input_streaming"], true);
        assert_eq!(body["messages"][1], json!({"role": "system", "content": "Answer in JSON."}));
    }

    let m = c.create(&MessagesRequest::new("claude-opus-5-5", 100).user("x")).await.unwrap();
    assert_eq!(m.refusal_category(), Some("cyber"));
}

fn sse(events: &[Value]) -> String {
    events.iter().map(|e| format!("event: {}\ndata: {}\n\n", e["type"].as_str().unwrap(), e)).collect()
}

fn stream_body() -> String {
    sse(&[
        json!({"type": "message_start", "message": {"id": "msg_s", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
               "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 50, "output_tokens": 1, "cache_read_input_tokens": 40}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "ping"}),
        json!({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "În 30 "}}),
        json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "de zile"}}),
        json!({"type": "content_block_delta", "index": 1, "delta": {"type": "citations_delta", "citation": {"type": "char_location",
               "cited_text": "Termenul este de 30 de zile.", "document_index": 0, "document_title": "Ghid CI", "start_char_index": 0, "end_char_index": 28}}}),
        json!({"type": "content_block_stop", "index": 1}),
        json!({"type": "something_new", "detail": 1}),
        json!({"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "cui", "input": {}}}),
        json!({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"cui\": \"RO1"}}),
        json!({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "8547290\"}"}}),
        json!({"type": "content_block_stop", "index": 2}),
        json!({"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null}, "usage": {"output_tokens": 77}}),
        json!({"type": "message_stop"}),
    ])
}

#[tokio::test]
async fn streams_events_and_accumulates_the_final_message() {
    let (url, server) = serve(vec![Reply { headers: vec![("content-type", "text/event-stream"), ("request-id", "req_s")], ..reply(200, stream_body()) }]).await;
    let c = client(&url, 0);

    let mut s = c.stream(&grounded_request()).await.unwrap();
    assert_eq!(s.request_id(), Some("req_s"));
    let mut text = String::new();
    let mut kinds = Vec::new();
    while let Some(e) = s.next_event().await {
        let e = e.unwrap();
        if let StreamEvent::ContentBlockDelta { delta: Delta::TextDelta { text: t }, .. } = &e {
            text.push_str(t);
        }
        kinds.push(serde_json::to_value(&e).unwrap()["type"].as_str().unwrap().to_string());
    }
    assert_eq!(text, "În 30 de zile");
    assert!(kinds.contains(&"something_new".to_string()));
    assert_eq!(kinds.last().map(String::as_str), Some("message_stop"));
    assert_eq!(server.seen.lock().unwrap()[0].1["stream"], json!(true));

    let m = c.stream(&grounded_request()).await.unwrap().final_message().await.unwrap();
    assert_eq!(m.text(), "În 30 de zile");
    assert_eq!(m.stop_reason, Some(StopReason::ToolUse));
    assert_eq!(m.usage.input_tokens, 50);
    assert_eq!(m.usage.output_tokens, 77);
    assert_eq!(m.usage.cache_read_input_tokens, Some(40));
    assert_eq!(m.citations().count(), 1);
    assert!(matches!(&m.content[0], ContentBlock::Thinking { signature, .. } if signature == "sig"));
    assert_eq!(m.tool_uses().collect::<Vec<_>>(), vec![("toolu_1", "cui", &json!({"cui": "RO18547290"}))]);
}

#[tokio::test]
async fn an_error_event_or_a_cut_stream_fails_the_final_message() {
    let body = sse(&[
        json!({"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "model": "x", "content": [], "stop_reason": null, "usage": {}}}),
        json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
    ]);
    let (url, _) = serve(vec![reply(200, body)]).await;
    let err = client(&url, 0).stream(&grounded_request()).await.unwrap().final_message().await.unwrap_err();
    assert!(matches!(&err, Error::Api { status: None, kind, .. } if kind == "overloaded_error"));
    assert!(err.is_retryable());

    let cut = sse(&[
        json!({"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "model": "x", "content": [], "stop_reason": null, "usage": {}}}),
    ]);
    let (url, _) = serve(vec![reply(200, cut)]).await;
    let err = client(&url, 0).stream(&grounded_request()).await.unwrap().final_message().await.unwrap_err();
    assert!(matches!(err, Error::Decode(_)));
}

#[tokio::test]
async fn opening_a_stream_retries_but_errors_after_it_opens_do_not() {
    let (url, server) = serve(vec![Reply { headers: vec![("retry-after", "0")], ..reply(503, "") }, reply(200, stream_body())]).await;
    let m = client(&url, 1).stream(&grounded_request()).await.unwrap().final_message().await.unwrap();
    assert_eq!(m.id, "msg_s");
    assert_eq!(server.hits.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn counts_tokens_with_only_the_prompt_fields() {
    let (url, server) = serve(vec![reply(200, r#"{"input_tokens": 321}"#)]).await;
    let n = client(&url, 0).count_tokens(&grounded_request()).await.unwrap();
    assert_eq!(n, 321);
    let seen = server.seen.lock().unwrap();
    let body = &seen[0].1;
    assert_eq!(body["model"], "claude-opus-5-5");
    assert!(body.get("max_tokens").is_none());
    assert!(body.get("fallbacks").is_none());
    assert!(body.get("system").is_some());
}

#[tokio::test]
async fn server_tools_are_sent_as_documented() {
    let (url, server) = serve(vec![reply(200, CITED)]).await;
    let req = MessagesRequest::new("claude-opus-5-5", 4096)
        .user("Ce acte îmi trebuie pentru pașaport?")
        .tool(WebSearchTool::new().max_uses(3).allowed_domains(["mai.gov.ro"]).user_location(UserLocation {
            country: Some("RO".into()),
            timezone: Some("Europe/Bucharest".into()),
            ..UserLocation::default()
        }))
        .tool(WebFetchTool::new().with_citations().max_content_tokens(20_000))
        .tool(CodeExecutionTool::new())
        .tool(ToolSearchTool::bm25())
        .tool(Tool::new("cui", "Checks a CUI", json!({"type": "object"})).deferred());
    client(&url, 0).create(&req).await.unwrap();

    let seen = server.seen.lock().unwrap();
    assert_eq!(
        seen[0].1["tools"],
        json!([
            {"type": "web_search_20260209", "name": "web_search", "max_uses": 3, "allowed_domains": ["mai.gov.ro"],
             "user_location": {"type": "approximate", "country": "RO", "timezone": "Europe/Bucharest"}},
            {"type": "web_fetch_20260209", "name": "web_fetch", "citations": {"enabled": true}, "max_content_tokens": 20000},
            {"type": "code_execution_20260521", "name": "code_execution"},
            {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
            {"name": "cui", "description": "Checks a CUI", "input_schema": {"type": "object"}, "defer_loading": true}
        ])
    );
    assert!(header(&seen[0].0, "anthropic-beta").is_none());
}

fn searched() -> Value {
    json!({
      "id": "msg_w", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
      "content": [
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "acte pasaport"}},
        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
          {"type": "web_search_result", "url": "https://pasapoarte.mai.gov.ro/acte", "title": "Acte necesare",
           "encrypted_content": "EqQBCkY", "page_age": "3 days ago", "brand_new_field": 7}
        ]},
        {"type": "server_tool_use", "id": "srvtoolu_2", "name": "web_search", "input": {"query": "taxa"}},
        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_2",
         "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}},
        {"type": "text", "text": "Buletinul și taxa achitată.", "citations": [
          {"type": "web_search_result_location", "url": "https://pasapoarte.mai.gov.ro/acte", "title": "Acte necesare",
           "encrypted_index": "Eo8BCi", "cited_text": "Cartea de identitate și dovada plății taxei."}
        ]},
        {"type": "server_tool_use", "id": "srvtoolu_3", "name": "web_fetch", "input": {"url": "https://pasapoarte.mai.gov.ro/acte"}},
        {"type": "web_fetch_tool_result", "tool_use_id": "srvtoolu_3", "content": {
          "type": "web_fetch_result", "url": "https://pasapoarte.mai.gov.ro/acte", "retrieved_at": "2026-10-04T07:00:00Z",
          "content": {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "Acte necesare: ..."},
                      "title": "Acte necesare", "citations": {"enabled": true}}}},
        {"type": "server_tool_use", "id": "srvtoolu_4", "name": "bash_code_execution", "input": {"command": "python plot.py"}},
        {"type": "bash_code_execution_tool_result", "tool_use_id": "srvtoolu_4", "content": {
          "type": "bash_code_execution_result", "stdout": "ok\n", "stderr": "", "return_code": 0,
          "content": [{"type": "bash_code_execution_output", "file_id": "file_011"}]}},
        {"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_5", "content": {
          "type": "tool_search_tool_search_result", "tool_references": [{"type": "tool_reference", "tool_name": "cui"}]}}
      ],
      "stop_reason": "end_turn", "stop_sequence": null,
      "usage": {"input_tokens": 5000, "output_tokens": 120, "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 1}}
    })
}

#[tokio::test]
async fn server_tool_blocks_decode_typed_and_echo_back_unchanged() {
    let (url, _) = serve(vec![reply(200, searched().to_string())]).await;
    let m = client(&url, 0).create(&MessagesRequest::new("claude-opus-5-5", 4096).user("?")).await.unwrap();

    assert!(matches!(&m.content[0], ContentBlock::ServerToolUse { name, input, .. } if name == "web_search" && input["query"] == "acte pasaport"));
    let results: Vec<_> = m.web_search_results().collect();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].title, "Acte necesare");
    assert_eq!(results[0].page_age.as_deref(), Some("3 days ago"));
    let errors: Vec<_> = m.server_tool_errors().collect();
    assert_eq!(errors.len(), 1);
    assert_eq!((errors[0].0, errors[0].1.error_code.as_str()), ("srvtoolu_2", "max_uses_exceeded"));
    assert!(matches!(&m.content[3], ContentBlock::WebSearchToolResult { content: WebSearchContent::Error(_), .. }));

    let (span, cite) = m.citations().next().unwrap();
    assert_eq!(span, "Buletinul și taxa achitată.");
    assert!(matches!(cite, Citation::WebSearchResultLocation { .. }));
    assert_eq!(cite.url(), Some("https://pasapoarte.mai.gov.ro/acte"));
    assert_eq!(cite.cited_text(), Some("Cartea de identitate și dovada plății taxei."));
    assert_eq!(cite.document_index(), None);

    let ContentBlock::WebFetchToolResult { content: WebFetchContent::Page(page), .. } = &m.content[6] else { panic!("{:?}", m.content[6]) };
    assert_eq!(page.text(), Some("Acte necesare: ..."));
    assert_eq!(page.title(), Some("Acte necesare"));
    let ContentBlock::BashCodeExecutionToolResult { content: CodeExecutionContent::Output(out), .. } = &m.content[8] else { panic!() };
    assert_eq!((out.stdout.as_str(), out.return_code), ("ok\n", 0));
    assert_eq!(out.file_ids().collect::<Vec<_>>(), vec!["file_011"]);
    assert!(matches!(&m.content[9], ContentBlock::ToolSearchToolResult { content: rust_claude_sdk::ToolSearchContent::Found(f), .. }
        if f.tool_references[0].tool_name == "cui"));

    let usage = m.usage.server_tool_use.as_ref().unwrap();
    assert_eq!((usage.web_search_requests, usage.web_fetch_requests), (2, 1));

    // Every block, unknown fields included, goes back exactly as received.
    let echoed: Vec<ContentBlockParam> = m.content.iter().cloned().map(ContentBlockParam::from).collect();
    assert_eq!(serde_json::to_value(MessageParam::assistant(echoed)).unwrap()["content"], searched()["content"]);
    assert_eq!(serde_json::to_value(&m.usage).unwrap()["server_tool_use"], searched()["usage"]["server_tool_use"]);
}

#[tokio::test]
async fn streamed_server_tool_calls_accumulate_their_input() {
    let body = sse(&[
        json!({"type": "message_start", "message": {"id": "m", "type": "message", "role": "assistant", "model": "x", "content": [], "stop_reason": null, "usage": {}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"query\": "}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "\"taxa\"}"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "content_block_start", "index": 1, "content_block": searched()["content"][1]}),
        json!({"type": "content_block_stop", "index": 1}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 9, "server_tool_use": {"web_search_requests": 1}}}),
        json!({"type": "message_stop"}),
    ]);
    let (url, _) = serve(vec![reply(200, body)]).await;
    let m = client(&url, 0).stream(&grounded_request()).await.unwrap().final_message().await.unwrap();
    assert!(matches!(&m.content[0], ContentBlock::ServerToolUse { input, .. } if *input == json!({"query": "taxa"})));
    assert_eq!(m.web_search_results().count(), 1);
    assert_eq!(m.usage.server_tool_use.unwrap().web_search_requests, 1);
}

fn batch_json(status: &str) -> String {
    json!({"id": "msgbatch_1", "type": "message_batch", "processing_status": status,
           "request_counts": {"processing": 0, "succeeded": 1, "errored": 2, "canceled": 1, "expired": 0},
           "created_at": "2026-10-04T07:00:00Z", "expires_at": "2026-10-05T07:00:00Z", "ended_at": null,
           "cancel_initiated_at": null, "archived_at": null, "results_url": null})
    .to_string()
}

#[tokio::test]
async fn batches_create_poll_list_cancel_delete() {
    let page = format!(r#"{{"data": [{}], "has_more": true, "first_id": "msgbatch_1", "last_id": "msgbatch_1"}}"#, batch_json("ended"));
    let (url, server) = serve(vec![
        reply(200, batch_json("in_progress")),
        reply(200, batch_json("ended")),
        reply(200, page),
        reply(200, batch_json("canceling")),
        reply(200, r#"{"id": "msgbatch_1", "type": "message_batch_deleted"}"#),
    ])
    .await;
    let c = client(&url, 0);
    let req = MessagesRequest::new("claude-opus-5-5", 64).user("a").beta("context-management-2025-06-27");
    let created = c
        .create_batch(&[BatchRequest::new("r-1", req.clone()), BatchRequest::new("r-2", MessagesRequest::new("claude-haiku-4-5", 64).user("b"))])
        .await
        .unwrap();
    assert_eq!(created.processing_status, ProcessingStatus::InProgress);
    assert_eq!(created.request_counts.errored, 2);

    let polled = c.batch("msgbatch_1").await.unwrap();
    assert!(polled.is_ended());
    let params = ListParams::limit(1);
    let listed = c.list_batches(&params).await.unwrap();
    assert_eq!(listed.data[0].id, "msgbatch_1");
    assert_eq!(listed.next_params(&params).unwrap().after_id.as_deref(), Some("msgbatch_1"));
    assert_eq!(c.cancel_batch("msgbatch_1").await.unwrap().processing_status, ProcessingStatus::Canceling);
    assert_eq!(c.delete_batch("msgbatch_1").await.unwrap().extra["type"], "message_batch_deleted");

    let calls = server.calls.lock().unwrap();
    let routes: Vec<_> = calls.iter().map(|(m, u, _)| format!("{m} {u}")).collect();
    assert_eq!(
        routes,
        [
            "POST /v1/messages/batches",
            "GET /v1/messages/batches/msgbatch_1",
            "GET /v1/messages/batches?limit=1",
            "POST /v1/messages/batches/msgbatch_1/cancel",
            "DELETE /v1/messages/batches/msgbatch_1"
        ]
    );
    let seen = server.seen.lock().unwrap();
    assert_eq!(header(&seen[0].0, "anthropic-beta"), Some("context-management-2025-06-27"));
    assert_eq!(
        seen[0].1["requests"][0],
        json!({"custom_id": "r-1", "params": {"model": "claude-opus-5-5", "max_tokens": 64, "messages": [{"role": "user", "content": "a"}]}})
    );
    assert_eq!(seen[0].1["requests"][1]["custom_id"], "r-2");
    assert!(header(&seen[1].0, "content-type").is_none());
}

#[tokio::test]
async fn batch_results_stream_line_by_line() {
    let message = serde_json::from_str::<Value>(CITED).unwrap();
    let lines = [
        json!({"custom_id": "r-2", "result": {"type": "succeeded", "message": message}}),
        json!({"custom_id": "r-1", "result": {"type": "errored", "error": {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}}}),
        json!({"custom_id": "r-3", "result": {"type": "errored", "error": {"type": "overloaded_error", "message": "busy"}}}),
        json!({"custom_id": "r-4", "result": {"type": "canceled"}}),
        json!({"custom_id": "r-5", "result": {"type": "expired"}}),
        json!({"custom_id": "r-6", "result": {"type": "postponed", "until": "later"}}),
    ];
    // The last line has no trailing newline.
    let body = lines.iter().map(Value::to_string).collect::<Vec<_>>().join("\n");
    let (url, server) = serve(vec![reply(200, body)]).await;
    let results = client(&url, 0).batch_results("msgbatch_1").await.unwrap().collect().await.unwrap();

    assert_eq!(server.calls.lock().unwrap()[0].1, "/v1/messages/batches/msgbatch_1/results");
    assert_eq!(results.len(), 6);
    let BatchOutcome::Succeeded { message } = &results[0].result else { panic!() };
    assert_eq!(message.citations().count(), 2);
    assert_eq!(results[1].result.error().unwrap().kind, "invalid_request_error");
    assert_eq!(results[2].result.error().unwrap().kind, "overloaded_error");
    assert_eq!(results[3].result, BatchOutcome::Canceled);
    assert_eq!(results[4].result, BatchOutcome::Expired);
    assert_eq!(serde_json::to_value(&results[5].result).unwrap(), lines[5]["result"]);
}

const FILE: &str = r#"{"id": "file_011", "type": "file", "filename": "ghid.pdf", "mime_type": "application/pdf", "size_bytes": 4,
  "created_at": "2026-10-04T07:00:00Z", "downloadable": false}"#;

#[tokio::test]
async fn files_upload_list_get_download_delete() {
    let page = format!(r#"{{"data": [{FILE}], "has_more": false, "first_id": "file_011", "last_id": "file_011"}}"#);
    let (url, server) =
        serve(vec![reply(200, FILE), reply(200, page), reply(200, FILE), reply(200, "%PDF"), reply(200, r#"{"id": "file_011", "type": "file_deleted"}"#)])
            .await;
    let c = client(&url, 0);
    let uploaded = c.upload_file("ghid.pdf", "application/pdf", b"%PDF").await.unwrap();
    assert_eq!((uploaded.id.as_str(), uploaded.size_bytes, uploaded.downloadable), ("file_011", 4, false));
    let params = ListParams { limit: Some(5), after_id: Some("file_000".into()), ..ListParams::default() };
    let listed = c.list_files(&params).await.unwrap();
    assert_eq!(listed.data[0].filename, "ghid.pdf");
    assert!(listed.next_params(&params).is_none());
    assert_eq!(c.file("../v1/x?y").await.unwrap().mime_type, "application/pdf");
    assert_eq!(c.download_file("file_011").await.unwrap(), b"%PDF");
    assert_eq!(c.delete_file("file_011").await.unwrap().id, "file_011");

    let calls = server.calls.lock().unwrap();
    let routes: Vec<_> = calls.iter().map(|(m, u, _)| format!("{m} {u}")).collect();
    assert_eq!(
        routes,
        [
            "POST /v1/files",
            "GET /v1/files?limit=5&after_id=file_000",
            // An id cannot leave its path segment.
            "GET /v1/files/..%2Fv1%2Fx%3Fy",
            "GET /v1/files/file_011/content",
            "DELETE /v1/files/file_011"
        ]
    );
    let seen = server.seen.lock().unwrap();
    assert!(header(&seen[0].0, "content-type").unwrap().starts_with("multipart/form-data; boundary="));
    assert_eq!(header(&seen[0].0, "x-api-key"), Some(KEY));
    let upload = &calls[0].2;
    assert!(upload.contains(r#"Content-Disposition: form-data; name="file"; filename="ghid.pdf""#), "{upload}");
    assert!(upload.contains("Content-Type: application/pdf\r\n\r\n%PDF\r\n"), "{upload}");

    let block = serde_json::to_value(ContentBlockParam::file_document(&uploaded.id)).unwrap();
    assert_eq!(block, json!({"type": "document", "source": {"type": "file", "file_id": "file_011"}}));
}

#[tokio::test]
async fn models_list_and_retrieve_with_capabilities() {
    let model = json!({"type": "model", "id": "claude-opus-5-5", "display_name": "Claude Opus 5.5", "created_at": "2026-09-01T00:00:00Z",
        "max_input_tokens": 1000000, "max_tokens": 128000,
        "capabilities": {"image_input": {"supported": true}, "thinking": {"supported": true, "types": {"enabled": {"supported": false}, "adaptive": {"supported": true}}},
                         "effort": {"supported": true, "max": {"supported": true}}}});
    let page = json!({"data": [model], "has_more": false, "first_id": "claude-opus-5-5", "last_id": "claude-opus-5-5"});
    let (url, server) = serve(vec![reply(200, page.to_string()), reply(200, model.to_string())]).await;
    let c = client(&url, 0);
    let listed = c.list_models(&ListParams::default()).await.unwrap();
    assert_eq!(listed.data.len(), 1);
    let m = c.model("claude-opus-5-5").await.unwrap();
    assert_eq!((m.max_input_tokens, m.max_tokens), (Some(1_000_000), Some(128_000)));
    assert!(m.supports(&["image_input"]));
    assert!(m.supports(&["thinking", "types", "adaptive"]));
    assert!(!m.supports(&["thinking", "types", "enabled"]));
    assert!(m.supports(&["effort", "max"]));
    assert!(!m.supports(&["pdf_input"]));
    assert_eq!(serde_json::to_value(&m).unwrap(), model);

    let calls = server.calls.lock().unwrap();
    assert_eq!((calls[0].0.as_str(), calls[0].1.as_str()), ("GET", "/v1/models"));
    assert_eq!(calls[1].1, "/v1/models/claude-opus-5-5");
}

fn turn(stop_reason: &str, content: Value, input_tokens: u64) -> Reply {
    reply(
        200,
        json!({"id": "msg_t", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": content,
               "stop_reason": stop_reason, "stop_sequence": null, "usage": {"input_tokens": input_tokens, "output_tokens": 5}})
        .to_string(),
    )
}

fn cui_request() -> MessagesRequest {
    MessagesRequest::new("claude-opus-5-5", 1024)
        .tool(Tool::new("cui", "Checks a CUI", json!({"type": "object", "properties": {"cui": {"type": "string"}}, "required": ["cui"]})))
        .user("Check RO1 and RO2")
}

async fn check_cui(cui: String) -> Result<String, String> {
    tokio::time::sleep(Duration::from_millis(if cui == "RO1" { 50 } else { 0 })).await;
    if cui == "RO1" {
        Ok("active".into())
    } else {
        Err(format!("unknown CUI {cui}"))
    }
}

#[tokio::test]
async fn the_tool_loop_runs_parallel_calls_and_resumes_paused_turns() {
    let (url, server) = serve(vec![
        turn(
            "tool_use",
            json!([
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "tool_use", "id": "toolu_1", "name": "cui", "input": {"cui": "RO1"}},
                {"type": "tool_use", "id": "toolu_2", "name": "cui", "input": {"cui": "RO2"}}
            ]),
            10,
        ),
        turn("pause_turn", json!([{"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "RO2"}}]), 20),
        turn("end_turn", json!([{"type": "text", "text": "RO1 is active."}]), 30),
    ])
    .await;
    let mut req = cui_request();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let seen_calls = calls.clone();
    let run = client(&url, 0)
        .run_tools(&mut req, move |call| {
            seen_calls.lock().unwrap().push(call.name.clone());
            check_cui(call.input["cui"].as_str().unwrap_or_default().to_string())
        })
        .await
        .unwrap();

    assert!(run.is_finished());
    assert_eq!(run.requests, 3);
    assert_eq!(run.message.text(), "RO1 is active.");
    assert_eq!((run.usage.input_tokens, run.usage.output_tokens), (60, 15));
    assert_eq!(*calls.lock().unwrap(), vec!["cui", "cui"]);
    assert_eq!(req.messages.len(), 5);

    let seen = server.seen.lock().unwrap();
    let second = &seen[1].1["messages"];
    assert_eq!(second[1]["role"], "assistant");
    assert_eq!(second[1]["content"][0], json!({"type": "thinking", "thinking": "", "signature": "sig"}));
    // Both results in one user turn, in call order although RO1 finished last.
    assert_eq!(
        second[2],
        json!({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "active"},
            {"type": "tool_result", "tool_use_id": "toolu_2", "content": "unknown CUI RO2", "is_error": true}
        ]})
    );
    // A paused turn is sent back as the last message, with no user turn after it.
    let third = seen[2].1["messages"].as_array().unwrap();
    assert_eq!(third.len(), 4);
    assert_eq!(third[3]["role"], "assistant");
    assert_eq!(third[3]["content"][0]["type"], "server_tool_use");
}

#[tokio::test]
async fn a_tool_loop_cut_short_resumes_with_the_pending_calls() {
    let (url, server) = serve(vec![
        turn("tool_use", json!([{"type": "tool_use", "id": "toolu_1", "name": "cui", "input": {"cui": "RO1"}}]), 10),
        turn("end_turn", json!([{"type": "text", "text": "Done."}]), 10),
    ])
    .await;
    let c = client(&url, 0);
    let mut req = cui_request();
    let options = ToolLoopOptions { max_requests: 1, ..ToolLoopOptions::default() };
    let run = c.run_tools_with(&mut req, options, |_| async { ToolOutput::text("never") }).await.unwrap();
    assert!(!run.is_finished());
    assert_eq!(run.message.stop_reason, Some(StopReason::ToolUse));
    assert_eq!(req.messages.len(), 2);

    let run = c.run_tools(&mut req, |call| async move { format!("{} ok", call.input["cui"].as_str().unwrap()) }).await.unwrap();
    assert!(run.is_finished());
    assert_eq!(run.requests, 1);
    let sent = &server.seen.lock().unwrap()[1].1["messages"];
    assert_eq!(sent[2], json!({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "RO1 ok"}]}));
}
