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

    /// A document uploaded with the Files API (PDF or plain text).
    pub fn file_document(file_id: impl Into<String>) -> ContentBlockParam {
        ContentBlockParam::Document {
            source: DocumentSource::File { file_id: file_id.into() },
            title: None,
            context: None,
            citations: None,
            cache_control: None,
        }
    }

    /// An image uploaded with the Files API.
    pub fn file_image(file_id: impl Into<String>) -> ContentBlockParam {
        ContentBlockParam::Image { source: ImageSource::File { file_id: file_id.into() }, cache_control: None }
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
    /// Loaded only when a tool search finds it (see `ToolSearchTool`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl Tool {
    pub fn new(name: impl Into<String>, description: impl Into<String>, input_schema: Value) -> Tool {
        Tool { name: name.into(), description: description.into(), input_schema, strict: None, defer_loading: None, cache_control: None }
    }

    /// Inputs always validate against the schema (which needs `additionalProperties: false`).
    pub fn strict(mut self) -> Self {
        self.strict = Some(true);
        self
    }

    pub fn deferred(mut self) -> Self {
        self.defer_loading = Some(true);
        self
    }

    pub fn with_cache_control(mut self, cc: CacheControl) -> Self {
        self.cache_control = Some(cc);
        self
    }
}

/// A tool definition: a custom tool, a server tool Anthropic runs, or anything else as raw JSON.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ToolDefinition {
    Custom(Tool),
    WebSearch(WebSearchTool),
    WebFetch(WebFetchTool),
    CodeExecution(CodeExecutionTool),
    ToolSearch(ToolSearchTool),
    Raw(Value),
}

impl From<Tool> for ToolDefinition {
    fn from(t: Tool) -> ToolDefinition {
        ToolDefinition::Custom(t)
    }
}

impl From<WebSearchTool> for ToolDefinition {
    fn from(t: WebSearchTool) -> ToolDefinition {
        ToolDefinition::WebSearch(t)
    }
}

impl From<WebFetchTool> for ToolDefinition {
    fn from(t: WebFetchTool) -> ToolDefinition {
        ToolDefinition::WebFetch(t)
    }
}

impl From<CodeExecutionTool> for ToolDefinition {
    fn from(t: CodeExecutionTool) -> ToolDefinition {
        ToolDefinition::CodeExecution(t)
    }
}

impl From<ToolSearchTool> for ToolDefinition {
    fn from(t: ToolSearchTool) -> ToolDefinition {
        ToolDefinition::ToolSearch(t)
    }
}

impl From<Value> for ToolDefinition {
    fn from(v: Value) -> ToolDefinition {
        ToolDefinition::Raw(v)
    }
}

/// Server-side web search. Results come back as `ContentBlock::WebSearchToolResult` blocks
/// and cited text carries `Citation::WebSearchResultLocation`. A failed search is an error
/// object inside the result block, not an API error.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WebSearchTool {
    /// The tool version, e.g. `web_search_20260209`.
    #[serde(rename = "type")]
    pub kind: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,
    /// Use `allowed_domains` or `blocked_domains`, not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<UserLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl WebSearchTool {
    /// The version with dynamic filtering (Claude Opus 4.6, Claude Sonnet 4.6 and later).
    /// Do not also declare `CodeExecutionTool`: it runs code on its own.
    pub fn new() -> WebSearchTool {
        WebSearchTool::version("web_search_20260209")
    }

    /// The basic version, for older models and Vertex AI.
    pub fn basic() -> WebSearchTool {
        WebSearchTool::version("web_search_20250305")
    }

    pub fn version(kind: impl Into<String>) -> WebSearchTool {
        WebSearchTool {
            kind: kind.into(),
            name: "web_search".into(),
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
            user_location: None,
            cache_control: None,
        }
    }

    pub fn max_uses(mut self, n: u32) -> Self {
        self.max_uses = Some(n);
        self
    }

    pub fn allowed_domains(mut self, domains: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.allowed_domains = Some(domains.into_iter().map(Into::into).collect());
        self
    }

    pub fn blocked_domains(mut self, domains: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.blocked_domains = Some(domains.into_iter().map(Into::into).collect());
        self
    }

    pub fn user_location(mut self, location: UserLocation) -> Self {
        self.user_location = Some(location);
        self
    }
}

impl Default for WebSearchTool {
    fn default() -> Self {
        WebSearchTool::new()
    }
}

/// Where the user roughly is, to localise search results; serialised with `"type": "approximate"`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(tag = "type", rename = "approximate")]
pub struct UserLocation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// ISO 3166-1 alpha-2, e.g. `RO`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    /// IANA time zone, e.g. `Europe/Bucharest`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

/// Server-side fetch of a URL already present in the conversation. The page comes back as
/// a `ContentBlock::WebFetchToolResult` block.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WebFetchTool {
    /// The tool version, e.g. `web_fetch_20260209`.
    #[serde(rename = "type")]
    pub kind: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,
    /// Lets the answer cite the fetched pages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<CitationsConfig>,
    /// Truncates long pages to about this many tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_content_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl WebFetchTool {
    /// The version with dynamic filtering (Claude Opus 4.6, Claude Sonnet 4.6 and later).
    pub fn new() -> WebFetchTool {
        WebFetchTool::version("web_fetch_20260209")
    }

    /// The basic version, for older models.
    pub fn basic() -> WebFetchTool {
        WebFetchTool::version("web_fetch_20250910")
    }

    pub fn version(kind: impl Into<String>) -> WebFetchTool {
        WebFetchTool {
            kind: kind.into(),
            name: "web_fetch".into(),
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None,
            citations: None,
            max_content_tokens: None,
            cache_control: None,
        }
    }

    pub fn max_uses(mut self, n: u32) -> Self {
        self.max_uses = Some(n);
        self
    }

    pub fn allowed_domains(mut self, domains: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.allowed_domains = Some(domains.into_iter().map(Into::into).collect());
        self
    }

    pub fn blocked_domains(mut self, domains: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.blocked_domains = Some(domains.into_iter().map(Into::into).collect());
        self
    }

    pub fn with_citations(mut self) -> Self {
        self.citations = Some(CitationsConfig { enabled: true });
        self
    }

    pub fn max_content_tokens(mut self, n: u32) -> Self {
        self.max_content_tokens = Some(n);
        self
    }
}

impl Default for WebFetchTool {
    fn default() -> Self {
        WebFetchTool::new()
    }
}

/// Server-side code execution in a sandbox. Output comes back as
/// `ContentBlock::BashCodeExecutionToolResult` and `TextEditorCodeExecutionToolResult` blocks;
/// files it writes can be downloaded with `Client::download_file`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CodeExecutionTool {
    /// The tool version, e.g. `code_execution_20260521`.
    #[serde(rename = "type")]
    pub kind: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl CodeExecutionTool {
    pub fn new() -> CodeExecutionTool {
        CodeExecutionTool::version("code_execution_20260521")
    }

    /// Another version, e.g. `code_execution_20260120` for programmatic tool calling.
    pub fn version(kind: impl Into<String>) -> CodeExecutionTool {
        CodeExecutionTool { kind: kind.into(), name: "code_execution".into(), cache_control: None }
    }
}

impl Default for CodeExecutionTool {
    fn default() -> Self {
        CodeExecutionTool::new()
    }
}

/// Server-side search over the request's deferred tools (`Tool::deferred`): only the tools
/// it finds are loaded. Keep at least one tool, this one included, not deferred.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolSearchTool {
    /// `tool_search_tool_regex_20251119` or `tool_search_tool_bm25_20251119`.
    #[serde(rename = "type")]
    pub kind: String,
    pub name: String,
}

impl ToolSearchTool {
    /// The model searches with regular expressions.
    pub fn regex() -> ToolSearchTool {
        ToolSearchTool { kind: "tool_search_tool_regex_20251119".into(), name: "tool_search_tool_regex".into() }
    }

    /// The model searches with natural-language queries.
    pub fn bm25() -> ToolSearchTool {
        ToolSearchTool { kind: "tool_search_tool_bm25_20251119".into(), name: "tool_search_tool_bm25".into() }
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
