//! A small typed client for Anthropic's Messages API (`POST /v1/messages`) and the endpoints
//! around it, over raw HTTPS following the documented wire format:
//!
//! - requests with system prompts, `document` blocks with citations, images, tools, thinking
//!   and effort settings, `cache_control` breakpoints and server-side refusal `fallbacks`
//!   (the matching `anthropic-beta` header is added automatically);
//! - responses with text, citations, thinking, tool calls and fallback markers, keeping
//!   any block, citation, event or delta type it does not know as raw JSON instead of
//!   failing, so a new API feature never breaks decoding and assistant turns echo back intact;
//! - retries on connection errors, 408, 409, 429 and 5xx (honouring `retry-after` and
//!   `x-should-retry`), bounded by an optional deadline that also cuts retries short;
//! - streaming over server-sent events, event by event or accumulated into the final message;
//! - server tools (web search, web fetch, code execution, tool search) with typed
//!   definitions, result blocks, web search citations and `usage.server_tool_use`;
//! - `POST /v1/messages/count_tokens`, Message Batches, the Files API and the Models API.
//!
//! ```no_run
//! use rust_claude_sdk::{Client, ClientConfig, ContentBlockParam, MessageParam, MessagesRequest};
//!
//! # async fn run() -> Result<(), rust_claude_sdk::Error> {
//! let client = Client::new(ClientConfig::from_env().expect("ANTHROPIC_API_KEY"))?;
//! let request = MessagesRequest::new("claude-opus-5-5", 2048)
//!     .system_cached("Answer only from the documents and cite them.")
//!     .message(MessageParam::user(vec![
//!         ContentBlockParam::text_document("Cartea de identitate se eliberează în 30 de zile.", "Ghid CI").with_citations(),
//!         ContentBlockParam::text("În cât timp primesc buletinul?"),
//!     ]));
//! let message = client.create(&request).await?;
//! if message.is_refusal() {
//!     return Ok(());
//! }
//! println!("{}", message.text());
//! for (span, citation) in message.citations() {
//!     println!("{span:?} ← {:?}", citation.cited_text());
//! }
//! # Ok(())
//! # }
//! ```
#[macro_use]
mod open;

mod batches;
mod client;
mod error;
mod files;
mod models;
mod page;
mod request;
mod response;
mod stream;

pub use batches::{BatchOutcome, BatchRequest, BatchResult, BatchResults, MessageBatch, ProcessingStatus, RequestCounts};
pub use client::{CallOptions, Client, ClientConfig, Credential, API_VERSION, DEFAULT_BASE_URL};
pub use error::{ApiErrorBody, Error};
pub use files::FileMetadata;
pub use models::Model;
pub use page::{Deleted, ListParams, Page};
pub use request::{
    CacheControl, CitationsConfig, CodeExecutionTool, Content, ContentBlockParam, DocumentSource, Effort, FallbackModel, Fallbacks, ImageSource, MessageParam,
    MessagesRequest, OutputConfig, Role, System, SystemBlock, ThinkingConfig, ThinkingDisplay, Tool, ToolChoice, ToolDefinition, ToolResultContent,
    ToolSearchTool, UserLocation, WebFetchTool, WebSearchTool,
};
pub use response::{
    Citation, CodeExecutionContent, CodeExecutionOutput, ContentBlock, Message, ModelRef, ServerToolError, ServerToolUsage, StopReason, ToolReference,
    ToolSearchContent, ToolSearchFound, Usage, WebFetchContent, WebFetchResult, WebSearchContent, WebSearchResult,
};
pub use stream::{Delta, MessageStream, StreamEvent};
