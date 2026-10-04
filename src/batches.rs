//! Message Batches: many Messages requests processed asynchronously at half price.
//! Create, poll `processing_status` until `Ended`, then read the results (in any order;
//! match them by `custom_id`).
use std::collections::VecDeque;

use reqwest::Method;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::client::{to_json, CallOptions, Client, Payload};
use crate::error::{ApiErrorBody, Error};
use crate::page::{Deleted, ListParams, Page};
use crate::request::MessagesRequest;
use crate::response::Message;

/// One request of a batch. `params` is sent without `stream`; `fallbacks` is rejected here.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BatchRequest {
    /// Unique within the batch: 1-64 characters of `[a-zA-Z0-9_-]`.
    pub custom_id: String,
    pub params: MessagesRequest,
}

impl BatchRequest {
    pub fn new(custom_id: impl Into<String>, params: MessagesRequest) -> BatchRequest {
        BatchRequest { custom_id: custom_id.into(), params }
    }
}

#[derive(Serialize)]
struct CreateBody<'a> {
    requests: &'a [BatchRequest],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageBatch {
    pub id: String,
    pub processing_status: ProcessingStatus,
    #[serde(default)]
    pub request_counts: RequestCounts,
    /// RFC 3339 timestamps.
    #[serde(default)]
    pub created_at: String,
    #[serde(default)]
    pub expires_at: Option<String>,
    #[serde(default)]
    pub ended_at: Option<String>,
    #[serde(default)]
    pub cancel_initiated_at: Option<String>,
    #[serde(default)]
    pub archived_at: Option<String>,
    /// Set once the batch has ended.
    #[serde(default)]
    pub results_url: Option<String>,
    /// `type` and anything else.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl MessageBatch {
    pub fn is_ended(&self) -> bool {
        self.processing_status == ProcessingStatus::Ended
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessingStatus {
    InProgress,
    Canceling,
    Ended,
    Other(String),
}

impl ProcessingStatus {
    pub fn as_str(&self) -> &str {
        match self {
            ProcessingStatus::InProgress => "in_progress",
            ProcessingStatus::Canceling => "canceling",
            ProcessingStatus::Ended => "ended",
            ProcessingStatus::Other(s) => s,
        }
    }
}

impl Serialize for ProcessingStatus {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ProcessingStatus {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(match String::deserialize(d)?.as_str() {
            "in_progress" => ProcessingStatus::InProgress,
            "canceling" => ProcessingStatus::Canceling,
            "ended" => ProcessingStatus::Ended,
            other => ProcessingStatus::Other(other.to_string()),
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestCounts {
    #[serde(default)]
    pub processing: u64,
    #[serde(default)]
    pub succeeded: u64,
    #[serde(default)]
    pub errored: u64,
    #[serde(default)]
    pub canceled: u64,
    #[serde(default)]
    pub expired: u64,
}

/// One line of a batch's results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchResult {
    pub custom_id: String,
    pub result: BatchOutcome,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(remote = "Self", tag = "type", rename_all = "snake_case")]
pub enum BatchOutcome {
    Succeeded {
        message: Box<Message>,
    },
    /// `error` is the API's error response; see `BatchOutcome::error`.
    Errored {
        error: Value,
    },
    Canceled,
    /// The batch reached its 24-hour limit before this request ran.
    Expired,
    #[serde(skip)]
    Other(Value),
}
open_enum!(BatchOutcome, ["succeeded", "errored", "canceled", "expired"]);

impl BatchOutcome {
    /// The error of an `Errored` result. An `invalid_request_error` needs a fixed request;
    /// other kinds may succeed if resubmitted.
    pub fn error(&self) -> Option<ApiErrorBody> {
        let BatchOutcome::Errored { error } = self else { return None };
        // `{"type": "error", "error": {...}}`, or the inner object alone.
        let inner = error.get("error").filter(|e| e.is_object()).unwrap_or(error);
        serde_json::from_value(inner.clone()).ok()
    }
}

/// A batch's results, read line by line as they download.
pub struct BatchResults {
    response: reqwest::Response,
    buf: Vec<u8>,
    pending: VecDeque<Vec<u8>>,
    ended: bool,
}

impl BatchResults {
    /// The next result, `None` after the last.
    pub async fn next(&mut self) -> Option<Result<BatchResult, Error>> {
        loop {
            if let Some(line) = self.pending.pop_front() {
                return Some(serde_json::from_slice(&line).map_err(|e| Error::Decode(e.to_string())));
            }
            if self.ended {
                return None;
            }
            match self.response.chunk().await {
                Ok(Some(bytes)) => {
                    self.buf.extend_from_slice(&bytes);
                    while let Some(i) = self.buf.iter().position(|b| *b == b'\n') {
                        let line: Vec<u8> = self.buf.drain(..=i).collect();
                        self.push(line);
                    }
                }
                Ok(None) => {
                    self.ended = true;
                    let rest = std::mem::take(&mut self.buf);
                    self.push(rest);
                }
                Err(e) => {
                    self.ended = true;
                    return Some(Err(Error::Transport(e.without_url().to_string())));
                }
            }
        }
    }

    /// Every remaining result.
    pub async fn collect(mut self) -> Result<Vec<BatchResult>, Error> {
        let mut out = Vec::new();
        while let Some(r) = self.next().await {
            out.push(r?);
        }
        Ok(out)
    }

    fn push(&mut self, line: Vec<u8>) {
        if !line.iter().all(u8::is_ascii_whitespace) {
            self.pending.push_back(line);
        }
    }
}

impl Client {
    /// `POST /v1/messages/batches`. The beta headers the requests need are sent with it.
    pub async fn create_batch(&self, requests: &[BatchRequest]) -> Result<MessageBatch, Error> {
        let mut betas: Vec<String> = Vec::new();
        let mut requests = requests.to_vec();
        for r in &mut requests {
            r.params.stream = None;
            for b in r.params.beta_header_values() {
                if !betas.contains(&b) {
                    betas.push(b);
                }
            }
        }
        let json = to_json(&CreateBody { requests: &requests })?;
        self.call(Method::POST, &["v1", "messages", "batches"], &[], Payload::Json(&json), &betas, CallOptions::default()).await
    }

    /// `GET /v1/messages/batches/{id}`.
    pub async fn batch(&self, id: &str) -> Result<MessageBatch, Error> {
        self.call(Method::GET, &["v1", "messages", "batches", id], &[], Payload::Empty, &[], CallOptions::default()).await
    }

    /// `GET /v1/messages/batches`: one page, newest first.
    pub async fn list_batches(&self, params: &ListParams) -> Result<Page<MessageBatch>, Error> {
        self.call(Method::GET, &["v1", "messages", "batches"], &params.query(), Payload::Empty, &[], CallOptions::default()).await
    }

    /// `POST /v1/messages/batches/{id}/cancel`: the batch moves to `Canceling`, then `Ended`;
    /// requests already processed keep their results.
    pub async fn cancel_batch(&self, id: &str) -> Result<MessageBatch, Error> {
        self.call(Method::POST, &["v1", "messages", "batches", id, "cancel"], &[], Payload::Empty, &[], CallOptions::default()).await
    }

    /// `DELETE /v1/messages/batches/{id}`: only an ended batch can be deleted.
    pub async fn delete_batch(&self, id: &str) -> Result<Deleted, Error> {
        self.call(Method::DELETE, &["v1", "messages", "batches", id], &[], Payload::Empty, &[], CallOptions::default()).await
    }

    /// `GET /v1/messages/batches/{id}/results`: the results of an ended batch, as JSON lines.
    pub async fn batch_results(&self, id: &str) -> Result<BatchResults, Error> {
        let response = self.call_response(Method::GET, &["v1", "messages", "batches", id, "results"], &[], CallOptions::default()).await?;
        Ok(BatchResults { response, buf: Vec::new(), pending: VecDeque::new(), ended: false })
    }
}
