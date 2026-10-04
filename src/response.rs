//! The message returned by `POST /v1/messages` (or accumulated from a stream).
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    /// The model that produced the message (the fallback model after a refusal fallback).
    pub model: String,
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<StopReason>,
    #[serde(default)]
    pub stop_sequence: Option<String>,
    /// Set when `stop_reason` is `refusal`: category, explanation, fallback credit...
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_details: Option<Value>,
    #[serde(default)]
    pub usage: Usage,
}

impl Message {
    /// All text blocks, concatenated.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// A safety classifier declined the request (and any fallback declined too). Check this
    /// before using `content`.
    pub fn is_refusal(&self) -> bool {
        self.stop_reason == Some(StopReason::Refusal)
    }

    /// Each cited text span with one of its citations, in order.
    pub fn citations(&self) -> impl Iterator<Item = (&str, &Citation)> {
        self.content.iter().flat_map(|b| match b {
            ContentBlock::Text { text, citations } => citations.iter().map(move |c| (text.as_str(), c)).collect::<Vec<_>>(),
            _ => Vec::new(),
        })
    }

    /// The web search results in this message, across all searches.
    pub fn web_search_results(&self) -> impl Iterator<Item = &WebSearchResult> {
        self.content.iter().flat_map(|b| match b {
            ContentBlock::WebSearchToolResult { content: WebSearchContent::Results(r), .. } => r.iter().collect::<Vec<_>>(),
            _ => Vec::new(),
        })
    }

    /// Server tool calls that failed, with the id of the call: `(tool_use_id, error)`.
    pub fn server_tool_errors(&self) -> impl Iterator<Item = (&str, &ServerToolError)> {
        self.content.iter().filter_map(|b| {
            let err = match b {
                ContentBlock::WebSearchToolResult { content: WebSearchContent::Error(e), .. }
                | ContentBlock::WebFetchToolResult { content: WebFetchContent::Error(e), .. }
                | ContentBlock::BashCodeExecutionToolResult { content: CodeExecutionContent::Error(e), .. }
                | ContentBlock::CodeExecutionToolResult { content: CodeExecutionContent::Error(e), .. }
                | ContentBlock::ToolSearchToolResult { content: ToolSearchContent::Error(e), .. } => e,
                _ => return None,
            };
            Some((b.server_tool_result_id()?, err))
        })
    }

    /// The tool calls the model asked for: `(id, name, input)`.
    pub fn tool_uses(&self) -> impl Iterator<Item = (&str, &str, &Value)> {
        self.content.iter().filter_map(|b| match b {
            ContentBlock::ToolUse { id, name, input } => Some((id.as_str(), name.as_str(), input)),
            _ => None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty", deserialize_with = "null_as_empty")]
        citations: Vec<Citation>,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default)]
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    /// A call of a server tool (web search, web fetch, code execution...), run by the API.
    ServerToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    WebSearchToolResult {
        tool_use_id: String,
        content: WebSearchContent,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    WebFetchToolResult {
        tool_use_id: String,
        content: WebFetchContent,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    BashCodeExecutionToolResult {
        tool_use_id: String,
        content: CodeExecutionContent,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    /// The result of an older code execution version.
    CodeExecutionToolResult {
        tool_use_id: String,
        content: CodeExecutionContent,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    /// A file the sandbox viewed, created or edited; `content` kept as sent.
    TextEditorCodeExecutionToolResult {
        tool_use_id: String,
        content: Value,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    ToolSearchToolResult {
        tool_use_id: String,
        content: ToolSearchContent,
        #[serde(flatten)]
        extra: Map<String, Value>,
    },
    /// Marks where a refused attempt handed over to a fallback model.
    Fallback {
        from: ModelRef,
        to: ModelRef,
    },
    /// Any other block type, kept verbatim.
    #[serde(skip)]
    Other(Value),
}
open_enum!(
    ContentBlock,
    [
        "text",
        "thinking",
        "redacted_thinking",
        "tool_use",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
        "bash_code_execution_tool_result",
        "code_execution_tool_result",
        "text_editor_code_execution_tool_result",
        "tool_search_tool_result",
        "fallback",
    ]
);

impl ContentBlock {
    /// The `tool_use_id` of a server tool's result block.
    pub fn server_tool_result_id(&self) -> Option<&str> {
        match self {
            ContentBlock::WebSearchToolResult { tool_use_id, .. }
            | ContentBlock::WebFetchToolResult { tool_use_id, .. }
            | ContentBlock::BashCodeExecutionToolResult { tool_use_id, .. }
            | ContentBlock::CodeExecutionToolResult { tool_use_id, .. }
            | ContentBlock::TextEditorCodeExecutionToolResult { tool_use_id, .. }
            | ContentBlock::ToolSearchToolResult { tool_use_id, .. } => Some(tool_use_id),
            _ => None,
        }
    }
}

/// A server tool that ran but failed (`max_uses_exceeded`, `too_many_requests`,
/// `unavailable`, `url_not_accessible`...). The response itself still succeeds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServerToolError {
    pub error_code: String,
    /// `type` and anything else.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// What a web search returned: a list of results, or an error object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WebSearchContent {
    Results(Vec<WebSearchResult>),
    Error(ServerToolError),
    Other(Value),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebSearchResult {
    pub url: String,
    #[serde(default)]
    pub title: String,
    /// How old the page is, when known (e.g. `"2 days ago"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_age: Option<String>,
    /// Opaque page content; must be echoed back for later turns to cite it.
    #[serde(default)]
    pub encrypted_content: String,
    /// `type` and anything else.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// What a web fetch returned: the page, or an error object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WebFetchContent {
    Page(WebFetchResult),
    Error(ServerToolError),
    Other(Value),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebFetchResult {
    pub url: String,
    /// A `document` block holding the page (`source.data` is its text for text pages).
    pub content: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieved_at: Option<String>,
    /// `type` and anything else.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl WebFetchResult {
    /// The fetched text, when the page came back as a text document.
    pub fn text(&self) -> Option<&str> {
        self.content.get("source").and_then(|s| s.get("data")).and_then(Value::as_str)
    }

    pub fn title(&self) -> Option<&str> {
        self.content.get("title").and_then(Value::as_str)
    }
}

/// What a code execution returned: its output, or an error object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CodeExecutionContent {
    Output(CodeExecutionOutput),
    Error(ServerToolError),
    Other(Value),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodeExecutionOutput {
    #[serde(default)]
    pub stdout: String,
    #[serde(default)]
    pub stderr: String,
    /// `0` on success.
    pub return_code: i64,
    /// Files the code wrote: `{"type": ..., "file_id": ...}`, downloadable with
    /// `Client::download_file`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<Value>,
    /// `type` and anything else.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl CodeExecutionOutput {
    /// Ids of the files the code wrote.
    pub fn file_ids(&self) -> impl Iterator<Item = &str> {
        self.content.iter().filter_map(|f| f.get("file_id").and_then(Value::as_str))
    }
}

/// What a tool search found: references to the tools now loaded, or an error object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolSearchContent {
    Found(ToolSearchFound),
    Error(ServerToolError),
    Other(Value),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSearchFound {
    pub tool_references: Vec<ToolReference>,
    /// `type` and anything else.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolReference {
    pub tool_name: String,
    /// `type` and anything else.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef {
    pub model: String,
}

/// Where a cited span comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", tag = "type", rename_all = "snake_case")]
pub enum Citation {
    /// Plain-text document: character range, end exclusive.
    CharLocation {
        cited_text: String,
        document_index: usize,
        #[serde(default)]
        document_title: Option<String>,
        start_char_index: usize,
        end_char_index: usize,
    },
    /// PDF: page range, 1-indexed, end exclusive.
    PageLocation {
        cited_text: String,
        document_index: usize,
        #[serde(default)]
        document_title: Option<String>,
        start_page_number: usize,
        end_page_number: usize,
    },
    /// Custom-content document: chunk range, end exclusive.
    ContentBlockLocation {
        cited_text: String,
        document_index: usize,
        #[serde(default)]
        document_title: Option<String>,
        start_block_index: usize,
        end_block_index: usize,
    },
    /// A web search result.
    WebSearchResultLocation {
        url: String,
        #[serde(default)]
        title: Option<String>,
        #[serde(default)]
        cited_text: String,
        /// Opaque; must be echoed back unchanged.
        #[serde(default)]
        encrypted_index: String,
    },
    /// Any other citation type (search results...), kept verbatim.
    #[serde(skip)]
    Other(Value),
}
open_enum!(Citation, ["char_location", "page_location", "content_block_location", "web_search_result_location"]);

impl Citation {
    /// The quoted source text, when the citation type has one.
    pub fn cited_text(&self) -> Option<&str> {
        match self {
            Citation::CharLocation { cited_text, .. }
            | Citation::PageLocation { cited_text, .. }
            | Citation::ContentBlockLocation { cited_text, .. }
            | Citation::WebSearchResultLocation { cited_text, .. } => Some(cited_text),
            Citation::Other(v) => v.get("cited_text").and_then(Value::as_str),
        }
    }

    /// Index of the cited document among the request's documents.
    pub fn document_index(&self) -> Option<usize> {
        match self {
            Citation::CharLocation { document_index, .. }
            | Citation::PageLocation { document_index, .. }
            | Citation::ContentBlockLocation { document_index, .. } => Some(*document_index),
            Citation::WebSearchResultLocation { .. } => None,
            Citation::Other(v) => v.get("document_index").and_then(Value::as_u64).map(|i| i as usize),
        }
    }

    /// The cited page of a web search result.
    pub fn url(&self) -> Option<&str> {
        match self {
            Citation::WebSearchResultLocation { url, .. } => Some(url),
            Citation::Other(v) => v.get("url").and_then(Value::as_str),
            _ => None,
        }
    }
}

fn null_as_empty<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<Citation>, D::Error> {
    Ok(Option::<Vec<Citation>>::deserialize(d)?.unwrap_or_default())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    PauseTurn,
    Refusal,
    Other(String),
}

impl StopReason {
    pub fn as_str(&self) -> &str {
        match self {
            StopReason::EndTurn => "end_turn",
            StopReason::MaxTokens => "max_tokens",
            StopReason::StopSequence => "stop_sequence",
            StopReason::ToolUse => "tool_use",
            StopReason::PauseTurn => "pause_turn",
            StopReason::Refusal => "refusal",
            StopReason::Other(s) => s,
        }
    }
}

impl From<String> for StopReason {
    fn from(s: String) -> StopReason {
        match s.as_str() {
            "end_turn" => StopReason::EndTurn,
            "max_tokens" => StopReason::MaxTokens,
            "stop_sequence" => StopReason::StopSequence,
            "tool_use" => StopReason::ToolUse,
            "pause_turn" => StopReason::PauseTurn,
            "refusal" => StopReason::Refusal,
            _ => StopReason::Other(s),
        }
    }
}

impl Serialize for StopReason {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for StopReason {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d).map(StopReason::from)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    /// Tokens written to the prompt cache by this request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    /// Tokens read from the prompt cache; zero on every request means the prefix changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
    /// How many times each server tool ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<ServerToolUsage>,
    /// Anything else (`iterations`, `service_tier`, `inference_geo`...).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Usage {
    /// Adds another request's token and server tool counts; `extra` is left as is.
    pub(crate) fn add(&mut self, other: &Usage) {
        fn sum(a: Option<u64>, b: Option<u64>) -> Option<u64> {
            if a.is_none() && b.is_none() {
                None
            } else {
                Some(a.unwrap_or(0) + b.unwrap_or(0))
            }
        }
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_creation_input_tokens = sum(self.cache_creation_input_tokens, other.cache_creation_input_tokens);
        self.cache_read_input_tokens = sum(self.cache_read_input_tokens, other.cache_read_input_tokens);
        if let Some(o) = &other.server_tool_use {
            let s = self.server_tool_use.get_or_insert_with(ServerToolUsage::default);
            s.web_search_requests += o.web_search_requests;
            s.web_fetch_requests += o.web_fetch_requests;
        }
    }

    /// Applies the cumulative counts of a `message_delta` event.
    pub(crate) fn merge(&mut self, delta: &Map<String, Value>) {
        for (k, v) in delta {
            if v.is_null() {
                continue;
            }
            match k.as_str() {
                "input_tokens" => self.input_tokens = v.as_u64().unwrap_or(self.input_tokens),
                "output_tokens" => self.output_tokens = v.as_u64().unwrap_or(self.output_tokens),
                "cache_creation_input_tokens" => self.cache_creation_input_tokens = v.as_u64(),
                "cache_read_input_tokens" => self.cache_read_input_tokens = v.as_u64(),
                "server_tool_use" => self.server_tool_use = serde_json::from_value(v.clone()).ok(),
                _ => {
                    self.extra.insert(k.clone(), v.clone());
                }
            }
        }
    }
}

/// `usage.server_tool_use`: billable server tool calls.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ServerToolUsage {
    #[serde(default)]
    pub web_search_requests: u64,
    #[serde(default)]
    pub web_fetch_requests: u64,
    /// Anything else.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
