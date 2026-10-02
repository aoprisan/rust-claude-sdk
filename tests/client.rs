//! The client against a local stand-in for the Messages API: request shape and headers,
//! citations, unknown block types, retries, deadlines, streaming and token counting.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use claude_sdk::{
    CacheControl, CallOptions, Citation, Client, ClientConfig, ContentBlock, ContentBlockParam, Credential, Delta, Effort, Error, Fallbacks, MessageParam,
    MessagesRequest, StopReason, StreamEvent, ThinkingConfig, ThinkingDisplay, Tool,
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
}

async fn handle(State(s): State<Server>, headers: HeaderMap, body: String) -> Response {
    let n = s.hits.fetch_add(1, Ordering::SeqCst);
    s.seen.lock().unwrap().push((headers, serde_json::from_str(&body).unwrap_or(Value::Null)));
    let r = s.replies.get(n).or(s.replies.last()).unwrap().clone();
    tokio::time::sleep(r.delay).await;
    let mut res = (StatusCode::from_u16(r.status).unwrap(), r.body).into_response();
    for (k, v) in r.headers {
        res.headers_mut().insert(k, v.parse().unwrap());
    }
    res
}

async fn serve(replies: Vec<Reply>) -> (String, Server) {
    let server = Server { replies: Arc::new(replies), hits: Arc::default(), seen: Arc::default() };
    let app = Router::new().route("/v1/messages", post(handle)).route("/v1/messages/count_tokens", post(handle)).with_state(server.clone());
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
    {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "ci"}},
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
    assert_eq!(raw["type"], "server_tool_use");
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
        .fallbacks(Fallbacks::Models(vec![claude_sdk::FallbackModel { model: "claude-opus-4-8".into(), max_tokens: None }]))
        .tool(Tool {
            name: "cui".into(),
            description: "Checks a CUI".into(),
            input_schema: json!({"type": "object", "properties": {"cui": {"type": "string"}}, "required": ["cui"], "additionalProperties": false}),
            strict: Some(true),
            cache_control: Some(CacheControl::one_hour()),
        })
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
