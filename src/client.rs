//! HTTPS transport: headers, retries, deadlines.
use std::fmt;
use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use tokio::time::Instant;

use crate::error::{retryable_status, ApiErrorBody, Error};
use crate::request::{MessageParam, MessagesRequest, System, ThinkingConfig, ToolChoice, ToolDefinition};
use crate::response::Message;
use crate::stream::MessageStream;

pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// Value of the `anthropic-version` header.
pub const API_VERSION: &str = "2023-06-01";
/// Beta value an OAuth access token needs.
const BETA_OAUTH: &str = "oauth-2025-04-20";

#[derive(Clone, PartialEq, Eq)]
pub enum Credential {
    /// Sent as `x-api-key`.
    ApiKey(String),
    /// An OAuth access token, sent as `Authorization: Bearer`.
    Bearer(String),
}

impl fmt::Debug for Credential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Credential::ApiKey(_) => f.write_str("ApiKey(..)"),
            Credential::Bearer(_) => f.write_str("Bearer(..)"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub credential: Credential,
    /// API root, `https://api.anthropic.com` by default.
    pub base_url: String,
    /// Timeout of one non-streaming attempt, and of waiting for a stream's headers.
    pub timeout: Duration,
    pub connect_timeout: Duration,
    /// Extra attempts after a retryable failure.
    pub max_retries: u32,
    /// `anthropic-beta` values sent with every request.
    pub betas: Vec<String>,
}

impl ClientConfig {
    pub fn new(credential: Credential) -> ClientConfig {
        ClientConfig {
            credential,
            base_url: DEFAULT_BASE_URL.into(),
            timeout: Duration::from_secs(600),
            connect_timeout: Duration::from_secs(10),
            max_retries: 2,
            betas: Vec::new(),
        }
    }

    pub fn api_key(key: impl Into<String>) -> ClientConfig {
        ClientConfig::new(Credential::ApiKey(key.into()))
    }

    /// `ANTHROPIC_API_KEY`, else `ANTHROPIC_AUTH_TOKEN` (a bearer token), plus
    /// `ANTHROPIC_BASE_URL` when set, as the official SDKs read them.
    pub fn from_env() -> Option<ClientConfig> {
        let var = |k: &str| std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let credential = var("ANTHROPIC_API_KEY").map(Credential::ApiKey).or_else(|| var("ANTHROPIC_AUTH_TOKEN").map(Credential::Bearer))?;
        let mut c = ClientConfig::new(credential);
        if let Some(u) = var("ANTHROPIC_BASE_URL") {
            c.base_url = u;
        }
        Some(c)
    }
}

/// Per-call settings.
#[derive(Debug, Clone, Copy, Default)]
pub struct CallOptions {
    /// When the whole call (every attempt, every wait, and for a stream every read) must end.
    pub deadline: Option<Instant>,
    /// Overrides `ClientConfig::max_retries`.
    pub max_retries: Option<u32>,
}

impl CallOptions {
    pub fn within(budget: Duration) -> CallOptions {
        CallOptions { deadline: Some(Instant::now() + budget), max_retries: None }
    }
}

#[derive(Clone)]
pub struct Client {
    config: ClientConfig,
    http: reqwest::Client,
}

impl fmt::Debug for Client {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Client").field("config", &self.config).finish_non_exhaustive()
    }
}

/// The body of `POST /v1/messages/count_tokens`: the prompt-shaping fields only.
#[derive(Serialize)]
struct CountTokensBody<'a> {
    model: &'a str,
    messages: &'a [MessageParam],
    #[serde(skip_serializing_if = "Option::is_none")]
    system: &'a Option<System>,
    #[serde(skip_serializing_if = "<[ToolDefinition]>::is_empty")]
    tools: &'a [ToolDefinition],
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: &'a Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: &'a Option<ThinkingConfig>,
}

#[derive(Deserialize)]
struct CountTokensResponse {
    input_tokens: u64,
}

#[derive(Deserialize)]
struct ErrorEnvelope {
    error: ApiErrorBody,
    #[serde(default)]
    request_id: Option<String>,
}

/// A failed attempt, and whether it may be retried after `wait`.
struct Failure {
    error: Error,
    retry: bool,
    wait: Option<Duration>,
}

impl Client {
    pub fn new(config: ClientConfig) -> Result<Client, Error> {
        let http = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout)
            .user_agent(concat!("claude-sdk/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| Error::Config(e.to_string()))?;
        Ok(Client { config, http })
    }

    pub fn config(&self) -> &ClientConfig {
        &self.config
    }

    /// `POST /v1/messages`.
    pub async fn create(&self, request: &MessagesRequest) -> Result<Message, Error> {
        self.create_with(request, CallOptions::default()).await
    }

    pub async fn create_with(&self, request: &MessagesRequest, options: CallOptions) -> Result<Message, Error> {
        let mut body = request.clone();
        body.stream = None;
        let betas = request.beta_header_values();
        self.retrying(options, || async {
            let res = self.send("/v1/messages", &body, &betas, options).await?;
            let read = res.bytes();
            let bytes = within(self.attempt_end(options), read).await?.map_err(|e| transport(&e))?;
            serde_json::from_slice::<Message>(&bytes).map_err(|e| Failure { error: Error::Decode(e.to_string()), retry: false, wait: None })
        })
        .await
    }

    /// `POST /v1/messages` with `"stream": true`. Retries cover opening the stream only;
    /// an error after the first event is returned by the stream.
    pub async fn stream(&self, request: &MessagesRequest) -> Result<MessageStream, Error> {
        self.stream_with(request, CallOptions::default()).await
    }

    pub async fn stream_with(&self, request: &MessagesRequest, options: CallOptions) -> Result<MessageStream, Error> {
        let mut body = request.clone();
        body.stream = Some(true);
        let betas = request.beta_header_values();
        self.retrying(options, || async {
            let res = self.send("/v1/messages", &body, &betas, options).await?;
            let request_id = header(res.headers(), "request-id");
            Ok(MessageStream::new(res, options.deadline, request_id))
        })
        .await
    }

    /// `POST /v1/messages/count_tokens`: input tokens of the request's prompt on its model.
    pub async fn count_tokens(&self, request: &MessagesRequest) -> Result<u64, Error> {
        let body = CountTokensBody {
            model: &request.model,
            messages: &request.messages,
            system: &request.system,
            tools: &request.tools,
            tool_choice: &request.tool_choice,
            thinking: &request.thinking,
        };
        let options = CallOptions::default();
        let betas = request.betas.clone();
        self.retrying(options, || async {
            let res = self.send("/v1/messages/count_tokens", &body, &betas, options).await?;
            let bytes = within(self.attempt_end(options), res.bytes()).await?.map_err(|e| transport(&e))?;
            serde_json::from_slice::<CountTokensResponse>(&bytes).map(|r| r.input_tokens).map_err(|e| Failure {
                error: Error::Decode(e.to_string()),
                retry: false,
                wait: None,
            })
        })
        .await
    }

    async fn retrying<T, F, Fut>(&self, options: CallOptions, attempt: F) -> Result<T, Error>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, Failure>>,
    {
        let max = options.max_retries.unwrap_or(self.config.max_retries);
        let mut tries = 0;
        loop {
            if options.deadline.is_some_and(|d| Instant::now() >= d) {
                return Err(Error::Deadline);
            }
            let failure = match attempt().await {
                Ok(v) => return Ok(v),
                Err(f) => f,
            };
            if !failure.retry || tries >= max {
                return Err(failure.error);
            }
            let wait = failure.wait.unwrap_or_else(|| backoff(tries));
            // No time left for the wait and another attempt: report what went wrong.
            if options.deadline.is_some_and(|d| Instant::now() + wait >= d) {
                return Err(failure.error);
            }
            tries += 1;
            tokio::time::sleep(wait).await;
        }
    }

    fn attempt_end(&self, options: CallOptions) -> Instant {
        let end = Instant::now() + self.config.timeout;
        options.deadline.map_or(end, |d| d.min(end))
    }

    async fn send(&self, path: &str, body: &impl Serialize, betas: &[String], options: CallOptions) -> Result<reqwest::Response, Failure> {
        let url = format!("{}{path}", self.config.base_url.trim_end_matches('/'));
        let mut req = self.http.post(url).header("anthropic-version", API_VERSION).header(CONTENT_TYPE, "application/json");
        req = match &self.config.credential {
            Credential::ApiKey(k) => req.header("x-api-key", k),
            Credential::Bearer(t) => req.bearer_auth(t),
        };
        let oauth = matches!(self.config.credential, Credential::Bearer(_)).then_some(BETA_OAUTH);
        let mut all: Vec<&str> = Vec::new();
        for b in self.config.betas.iter().map(String::as_str).chain(betas.iter().map(String::as_str)).chain(oauth) {
            if !all.contains(&b) {
                all.push(b);
            }
        }
        if !all.is_empty() {
            req = req.header("anthropic-beta", all.join(","));
        }
        let body = serde_json::to_vec(body).map_err(|e| Failure { error: Error::Config(e.to_string()), retry: false, wait: None })?;
        let res = within(self.attempt_end(options), req.body(body).send()).await?.map_err(|e| transport(&e))?;
        let status = res.status().as_u16();
        if res.status().is_success() {
            return Ok(res);
        }
        let headers = res.headers().clone();
        let request_id = header(&headers, "request-id");
        let should_retry = header(&headers, "x-should-retry");
        let wait = header(&headers, "retry-after").and_then(|v| v.parse::<f64>().ok()).filter(|s| s.is_finite() && *s >= 0.0).map(Duration::from_secs_f64);
        let text = within(self.attempt_end(options), res.text()).await.ok().and_then(Result::ok).unwrap_or_default();
        let (kind, message, body_id) = match serde_json::from_str::<ErrorEnvelope>(&text) {
            Ok(e) => (e.error.kind, e.error.message, e.request_id),
            Err(_) => ("http_error".to_string(), format!("HTTP {status}"), None),
        };
        let retry = match should_retry.as_deref() {
            Some("true") => true,
            Some("false") => false,
            _ => retryable_status(status),
        };
        Err(Failure { error: Error::Api { status: Some(status), kind, message, request_id: request_id.or(body_id) }, retry, wait })
    }
}

fn header(headers: &HeaderMap, name: &str) -> Option<String> {
    headers.get(name).and_then(|v: &HeaderValue| v.to_str().ok()).map(str::to_string)
}

fn transport(e: &reqwest::Error) -> Failure {
    let error = if e.is_timeout() { Error::Deadline } else { Error::Transport(e.to_string()) };
    Failure { error, retry: true, wait: None }
}

/// Runs `fut` until `end`; past it the attempt failed with a retryable timeout.
async fn within<T>(end: Instant, fut: impl std::future::Future<Output = T>) -> Result<T, Failure> {
    tokio::time::timeout_at(end, fut).await.map_err(|_| Failure { error: Error::Deadline, retry: true, wait: None })
}

/// 0.5 s, 1 s, 2 s... capped at 8 s.
fn backoff(tries: u32) -> Duration {
    Duration::from_millis(500u64.saturating_mul(1 << tries.min(4)))
}
