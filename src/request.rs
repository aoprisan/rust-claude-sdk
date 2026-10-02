//! The body of `POST /v1/messages`.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::response::{Citation, ContentBlock};

/// Beta header for `fallbacks: "default"`.
pub(crate) const BETA_FALLBACK_DEFAULT: &str = "server-side-fallback-2026-07-01";
/// Beta header for the array form `fallbacks: [{"model": ...}]`.
pub(crate) const BETA_FALLBACK_LIST: &str = "server-side-fallback-2026-06-01";

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<MessageParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<System>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    /// Top-level automatic caching: the API places the breakpoint on the last cacheable block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    /// Server-side retry of a refused request on another model. The client adds the beta
    /// header this form needs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallbacks: Option<Fallbacks>,
    /// Set by `Client::stream`; leave it `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Further `anthropic-beta` values for this request.
    #[serde(skip)]
    pub betas: Vec<String>,
    /// Any other top-level parameter, sent as is (e.g. `inference_geo`, `metadata`).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl MessagesRequest {
    pub fn new(model: impl Into<String>, max_tokens: u32) -> MessagesRequest {
        MessagesRequest {
            model: model.into(),
            max_tokens,
            messages: Vec::new(),
            system: None,
            thinking: None,
            output_config: None,
            tools: Vec::new(),
            tool_choice: None,
            stop_sequences: Vec::new(),
            cache_control: None,
            fallbacks: None,
            stream: None,
            betas: Vec::new(),
            extra: Map::new(),
        }
    }

    pub fn system(mut self, text: impl Into<String>) -> Self {
        self.system = Some(System::Text(text.into()));
        self
    }

    /// A system prompt marked as a cache breakpoint: keep it byte-identical across requests
    /// (no dates or ids in it) so later requests read it from the cache.
    pub fn system_cached(mut self, text: impl Into<String>) -> Self {
        self.system = Some(System::Blocks(vec![SystemBlock { text: text.into(), cache_control: Some(CacheControl::ephemeral()) }]));
        self
    }

    pub fn message(mut self, message: MessageParam) -> Self {
        self.messages.push(message);
        self
    }

    pub fn user(self, text: impl Into<String>) -> Self {
        self.message(MessageParam::user_text(text))
    }

    pub fn thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Sets `output_config.effort`. Not every model accepts it: Claude Haiku 4.5 and
    /// Claude Sonnet 4.5 reject the parameter (use `ThinkingConfig::Enabled` there),
    /// Claude Opus 4.5 takes only `Low`/`Medium`/`High`, and Claude Opus 4.6 / Sonnet 4.6
    /// do not take `Xhigh`. The SDK does not check this; the API returns an error.
    pub fn effort(mut self, effort: Effort) -> Self {
        self.output_config.get_or_insert_with(OutputConfig::default).effort = Some(effort);
        self
    }

    pub fn tool(mut self, tool: impl Into<ToolDefinition>) -> Self {
        self.tools.push(tool.into());
        self
    }

    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    pub fn fallbacks(mut self, fallbacks: Fallbacks) -> Self {
        self.fallbacks = Some(fallbacks);
        self
    }

    pub fn beta(mut self, beta: impl Into<String>) -> Self {
        self.betas.push(beta.into());
        self
    }

    /// Any other top-level parameter.
    pub fn param(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// The `anthropic-beta` values this request needs: its own, then the one its
    /// `fallbacks` form requires (unless already present), without duplicates.
    pub fn beta_header_values(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let needed = match &self.fallbacks {
            Some(Fallbacks::Default) => Some(BETA_FALLBACK_DEFAULT),
            Some(Fallbacks::Models(_)) => Some(BETA_FALLBACK_LIST),
            None => None,
        };
        for b in self.betas.iter().map(String::as_str).chain(needed) {
            if !out.iter().any(|o| o == b) {
                out.push(b.to_string());
            }
        }
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MessageParam {
    pub role: Role,
    pub content: Content,
}

impl MessageParam {
    pub fn user_text(text: impl Into<String>) -> MessageParam {
        MessageParam { role: Role::User, content: Content::Text(text.into()) }
    }

    pub fn user(blocks: Vec<ContentBlockParam>) -> MessageParam {
        MessageParam { role: Role::User, content: Content::Blocks(blocks) }
    }

    pub fn assistant(blocks: Vec<ContentBlockParam>) -> MessageParam {
        MessageParam { role: Role::Assistant, content: Content::Blocks(blocks) }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Blocks(Vec<ContentBlockParam>),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockParam {
    Text {
        text: String,
        /// Only on echoed assistant text that carried citations.
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<Citation>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    Document {
        source: DocumentSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        /// Shown to the model but never cited from.
        #[serde(skip_serializing_if = "Option::is_none")]
        context: Option<String>,
        /// Citations must be enabled on all documents of a request or on none.
        #[serde(skip_serializing_if = "Option::is_none")]
        citations: Option<CitationsConfig>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: ToolResultContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Echoed back unchanged on the same model.
    Thinking {
        thinking: String,
        signature: String,
    },
    RedactedThinking {
        data: String,
    },
    /// Any other block, sent verbatim (e.g. an echoed block type this crate does not model).
    #[serde(untagged)]
    Raw(Value),
}

impl ContentBlockParam {
    pub fn text(text: impl Into<String>) -> ContentBlockParam {
        ContentBlockParam::Text { text: text.into(), citations: None, cache_control: None }
    }

    /// A plain-text document; the API splits it into sentences for citations.
    pub fn text_document(data: impl Into<String>, title: impl Into<String>) -> ContentBlockParam {
        ContentBlockParam::Document {
            source: DocumentSource::Text { media_type: "text/plain".into(), data: data.into() },
            title: Some(title.into()),
            context: None,
            citations: None,
            cache_control: None,
        }
    }

    /// A document made of chunks you choose; citations point at chunk indices.
    pub fn chunked_document(chunks: impl IntoIterator<Item = impl Into<String>>, title: impl Into<String>) -> ContentBlockParam {
        let content = chunks.into_iter().map(|c| ContentBlockParam::text(c)).collect();
        ContentBlockParam::Document {
            source: DocumentSource::Content { content },
            title: Some(title.into()),
            context: None,
            citations: None,
            cache_control: None,
        }
    }

    pub fn base64_pdf(data: impl Into<String>) -> ContentBlockParam {
        ContentBlockParam::Document {
            source: DocumentSource::Base64 { media_type: "application/pdf".into(), data: data.into() },
            title: None,
            context: None,
            citations: None,
            cache_control: None,
        }
    }

    pub fn tool_result(tool_use_id: impl Into<String>, content: impl Into<String>, is_error: bool) -> ContentBlockParam {
        ContentBlockParam::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: ToolResultContent::Text(content.into()),
            is_error: is_error.then_some(true),
            cache_control: None,
        }
    }

    /// Enables citations on a document; no effect on other blocks.
    pub fn with_citations(mut self) -> Self {
        if let ContentBlockParam::Document { citations, .. } = &mut self {
            *citations = Some(CitationsConfig { enabled: true });
        }
        self
    }

    /// Sets the document's non-citable context; no effect on other blocks.
    pub fn with_context(mut self, text: impl Into<String>) -> Self {
        if let ContentBlockParam::Document { context, .. } = &mut self {
            *context = Some(text.into());
        }
        self
    }

    /// Marks this block as a cache breakpoint; no effect on blocks that take none.
    pub fn with_cache_control(mut self, cc: CacheControl) -> Self {
        match &mut self {
            ContentBlockParam::Text { cache_control, .. }
            | ContentBlockParam::Image { cache_control, .. }
            | ContentBlockParam::Document { cache_control, .. }
            | ContentBlockParam::ToolResult { cache_control, .. } => *cache_control = Some(cc),
            _ => {}
        }
        self
    }
}

/// Echoing an assistant turn: every block goes back as received.
impl From<ContentBlock> for ContentBlockParam {
    fn from(block: ContentBlock) -> ContentBlockParam {
        match block {
            ContentBlock::Text { text, citations } => {
                ContentBlockParam::Text { text, citations: (!citations.is_empty()).then_some(citations), cache_control: None }
            }
            ContentBlock::Thinking { thinking, signature } => ContentBlockParam::Thinking { thinking, signature },
            ContentBlock::RedactedThinking { data } => ContentBlockParam::RedactedThinking { data },
            ContentBlock::ToolUse { id, name, input } => ContentBlockParam::ToolUse { id, name, input },
            other => ContentBlockParam::Raw(serde_json::to_value(&other).unwrap_or(Value::Null)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    Text { media_type: String, data: String },
    Base64 { media_type: String, data: String },
    Content { content: Vec<ContentBlockParam> },
    Url { url: String },
    File { file_id: String },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
    File { file_id: String },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ContentBlockParam>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CitationsConfig {
    pub enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub kind: String,
    /// `"5m"` (default) or `"1h"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

impl CacheControl {
    pub fn ephemeral() -> CacheControl {
        CacheControl { kind: "ephemeral".into(), ttl: None }
    }

    pub fn one_hour() -> CacheControl {
        CacheControl { kind: "ephemeral".into(), ttl: Some("1h".into()) }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum System {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

/// A system prompt block; serialised with `"type": "text"`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename = "text")]
pub struct SystemBlock {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ThinkingConfig {
    /// The model decides when and how much to think; `effort` sets the depth.
    Adaptive {
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<ThinkingDisplay>,
    },
    /// Rejected by several current models (see the model's documentation).
    Disabled,
    /// Older models only.
    Enabled { budget_tokens: u32 },
    /// Claude Sonnet 5.5's way to turn thinking off.
    BetweenTools,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingDisplay {
    Summarized,
    Omitted,
    Updates,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct OutputConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,
    /// Structured output format (cannot be combined with citations).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// A custom tool, or any other tool definition (server tools such as web search) as raw JSON.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ToolDefinition {
    Custom(Tool),
    Raw(Value),
}

impl From<Tool> for ToolDefinition {
    fn from(t: Tool) -> ToolDefinition {
        ToolDefinition::Custom(t)
    }
}

impl From<Value> for ToolDefinition {
    fn from(v: Value) -> ToolDefinition {
        ToolDefinition::Raw(v)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    None,
    /// Rejected by several current models; prefer `Auto` and say which tool to use.
    Any,
    /// Rejected by several current models; prefer `Auto` and say which tool to use.
    Tool {
        name: String,
    },
}

/// Which model the API retries a refused request on.
#[derive(Debug, Clone, PartialEq)]
pub enum Fallbacks {
    /// `"default"`: the API picks the fallback by refusal category.
    Default,
    /// Named models, tried in order.
    Models(Vec<FallbackModel>),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FallbackModel {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

impl Serialize for Fallbacks {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Fallbacks::Default => s.serialize_str("default"),
            Fallbacks::Models(models) => models.serialize(s),
        }
    }
}
