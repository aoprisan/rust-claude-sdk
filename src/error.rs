use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The `error` object of an API error response or of an `error` stream event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiErrorBody {
    /// `invalid_request_error`, `authentication_error`, `rate_limit_error`, `overloaded_error`...
    #[serde(rename = "type")]
    pub kind: String,
    pub message: String,
}

/// Errors never carry the request body or the credential.
#[derive(Debug, Error)]
pub enum Error {
    /// The API answered with an error. `status` is `None` for an error sent inside a stream.
    #[error("Anthropic API error {kind} (status {status:?}): {message}")]
    Api { status: Option<u16>, kind: String, message: String, request_id: Option<String> },
    #[error("transport error: {0}")]
    Transport(String),
    #[error("could not decode the response: {0}")]
    Decode(String),
    /// The call's deadline passed (before an attempt, during one, or between retries).
    #[error("deadline exceeded")]
    Deadline,
    #[error("invalid configuration: {0}")]
    Config(String),
}

impl Error {
    /// HTTP status of an API error.
    pub fn status(&self) -> Option<u16> {
        match self {
            Error::Api { status, .. } => *status,
            _ => None,
        }
    }

    /// Whether the same request may succeed if sent again: connection errors, 408, 409, 429,
    /// 5xx (529 is "overloaded") and overload/server errors reported inside a stream.
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::Transport(_) => true,
            Error::Api { status: Some(s), .. } => retryable_status(*s),
            Error::Api { status: None, kind, .. } => matches!(kind.as_str(), "overloaded_error" | "api_error" | "rate_limit_error"),
            _ => false,
        }
    }
}

pub(crate) fn retryable_status(status: u16) -> bool {
    matches!(status, 408 | 409 | 429) || status >= 500
}
