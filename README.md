# rust-claude-sdk

A small typed Rust client for Anthropic's Messages API (`POST /v1/messages`) and the endpoints around it, over raw HTTPS, following the documented wire format.

- Requests with system prompts, `document` blocks with citations, images, tools, thinking and effort settings, `cache_control` breakpoints and server-side refusal `fallbacks` (the matching `anthropic-beta` header is added automatically).
- Responses with text, citations, thinking, tool calls and fallback markers. Unknown block, citation, event and delta types are kept as raw JSON (`Other`) instead of failing, so a new API feature never breaks decoding and assistant turns echo back intact.
- Retries on connection errors, 408, 409, 429 and 5xx (honouring `retry-after` and `x-should-retry`), bounded by an optional deadline that also cuts retries short.
- Streaming over server-sent events, event by event or accumulated into the final message.
- Server tools: web search, web fetch, code execution and tool search, with typed definitions, typed result blocks (errors included), web search citations and `usage.server_tool_use`.
- A tool-use loop, `Client::run_tools`: runs the tools the model calls (a turn's calls concurrently), sends all results back in one turn and resumes `pause_turn`, bounded by `max_requests`.
- `POST /v1/messages/count_tokens`, Message Batches (results streamed line by line), the Files API and the Models API.

## Usage

```toml
[dependencies]
rust-claude-sdk = "0.2"
```

```rust
use rust_claude_sdk::{Client, ClientConfig, ContentBlockParam, MessageParam, MessagesRequest};

let client = Client::new(ClientConfig::from_env().expect("ANTHROPIC_API_KEY"))?;
let request = MessagesRequest::new("claude-opus-5-5", 2048)
    .system_cached("Answer only from the documents and cite them.")
    .message(MessageParam::user(vec![
        ContentBlockParam::text_document("Cartea de identitate se eliberează în 30 de zile.", "Ghid CI").with_citations(),
        ContentBlockParam::text("În cât timp primesc buletinul?"),
    ]));
let message = client.create(&request).await?;
println!("{}", message.text());
```

### Tools

Tools: the handler gets each call and returns a string, a `ToolOutput`, or a `Result` (an `Err` goes back to the model as an error result). `request.messages` keeps the whole conversation, so a run cut short or failed resumes by calling again.

```rust
use rust_claude_sdk::{Tool, ToolOutput};
use serde_json::json;

let mut request = MessagesRequest::new("claude-opus-5-5", 4096)
    .tool(Tool::new(
        "get_weather",
        "Current weather for a city",
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    ))
    .user("What's the weather in Cluj?");
let run = client
    .run_tools(&mut request, |call| async move {
        match call.name.as_str() {
            "get_weather" => ToolOutput::text(format!("Sunny in {}", call.input["city"])),
            other => ToolOutput::error(format!("unknown tool {other}")),
        }
    })
    .await?;
if run.is_finished() && !run.message.is_refusal() {
    println!("{}", run.message.text());
}
```

### Web search

```rust
use rust_claude_sdk::{MessagesRequest, WebSearchTool};

let request = MessagesRequest::new("claude-opus-5-5", 4096)
    .tool(WebSearchTool::new().max_uses(3).allowed_domains(["mai.gov.ro"]))
    .user("Ce acte îmi trebuie pentru pașaport?");
let message = client.create(&request).await?;
for (span, citation) in message.citations() {
    println!("{span} ← {:?}", citation.url());
}
for (id, error) in message.server_tool_errors() {
    eprintln!("{id}: {}", error.error_code);
}
```

### Batches, files and models

```rust
use rust_claude_sdk::{BatchOutcome, BatchRequest, ContentBlockParam, ListParams, MessageParam, MessagesRequest};

let batch = client.create_batch(&[BatchRequest::new("q-1", MessagesRequest::new("claude-opus-5-5", 1024).user("Salut"))]).await?;
// ...poll `client.batch(&batch.id)` until `is_ended()`, then:
let mut results = client.batch_results(&batch.id).await?;
while let Some(r) = results.next().await {
    let r = r?;
    if let BatchOutcome::Succeeded { message } = &r.result {
        println!("{}: {}", r.custom_id, message.text());
    }
}

let file = client.upload_file("ghid.pdf", "application/pdf", &std::fs::read("ghid.pdf")?).await?;
let request = MessagesRequest::new("claude-opus-5-5", 2048)
    .message(MessageParam::user(vec![ContentBlockParam::file_document(&file.id), ContentBlockParam::text("Rezumă.")]));

let opus = client.model("claude-opus-5-5").await?;
println!("{:?} tokens of context, adaptive thinking: {}", opus.max_input_tokens, opus.supports(&["thinking", "types", "adaptive"]));
let page = client.list_models(&ListParams::default()).await?;
```

A streaming, grounded example: `ANTHROPIC_API_KEY=... cargo run --example grounded -- "În cât timp primesc buletinul?"`

## Development

Rust ≥ 1.85.

```sh
cargo fmt --all --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
```

Tests run against a local mock server and need no API key.

## Releasing

Bump `version` in `Cargo.toml`, merge to `main`, then push a matching tag:

```sh
git tag v0.1.0
git push origin v0.1.0
```

The `Release` workflow checks that the tag matches the crate version, runs the checks above and publishes to crates.io with the `CARGO_REGISTRY_TOKEN` repository secret.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

Originally written as the `rogov-anthropic` crate in [aoprisan/govromania](https://github.com/aoprisan/govromania).
