# claude-sdk

A small typed Rust client for Anthropic's Messages API (`POST /v1/messages`) over raw HTTPS, following the documented wire format. There is no official Rust SDK.

- Requests with system prompts, `document` blocks with citations, images, tools, thinking and effort settings, `cache_control` breakpoints and server-side refusal `fallbacks` (the matching `anthropic-beta` header is added automatically).
- Responses with text, citations, thinking, tool calls and fallback markers. Unknown block, citation, event and delta types are kept as raw JSON (`Other`) instead of failing, so a new API feature never breaks decoding and assistant turns echo back intact.
- Retries on connection errors, 408, 409, 429 and 5xx (honouring `retry-after` and `x-should-retry`), bounded by an optional deadline that also cuts retries short.
- Streaming over server-sent events, event by event or accumulated into the final message.
- `POST /v1/messages/count_tokens`.

## Usage

```toml
[dependencies]
claude-sdk = { git = "https://github.com/aoprisan/rust-claude-sdk" }
```

```rust
use claude_sdk::{Client, ClientConfig, ContentBlockParam, MessageParam, MessagesRequest};

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

A streaming, grounded example: `ANTHROPIC_API_KEY=... cargo run --example grounded -- "În cât timp primesc buletinul?"`

## Development

Rust ≥ 1.85.

```sh
cargo fmt --all --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
```

Tests run against a local mock server and need no API key.

Originally written as the `rogov-anthropic` crate in [aoprisan/govromania](https://github.com/aoprisan/govromania).
