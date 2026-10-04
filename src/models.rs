//! `GET /v1/models`: the models available to the credential, with their limits and capabilities.
use reqwest::Method;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::client::{CallOptions, Client, Payload};
use crate::error::Error;
use crate::page::{ListParams, Page};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    #[serde(default)]
    pub display_name: String,
    /// RFC 3339.
    #[serde(default)]
    pub created_at: String,
    /// The context window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<u64>,
    /// The output cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// The capability tree, `supported: true/false` at each leaf; read it with `supports`.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub capabilities: Value,
    /// `type` and anything else.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Model {
    /// Whether the capability at `path` is supported, e.g. `["image_input"]`,
    /// `["thinking", "types", "adaptive"]` or `["effort", "max"]`. `false` when absent.
    pub fn supports(&self, path: &[&str]) -> bool {
        let node = path.iter().try_fold(&self.capabilities, |v, k| v.get(k));
        node.and_then(|n| n.get("supported")).and_then(Value::as_bool).unwrap_or(false)
    }
}

impl Client {
    /// `GET /v1/models`: one page, newest first.
    pub async fn list_models(&self, params: &ListParams) -> Result<Page<Model>, Error> {
        self.call(Method::GET, &["v1", "models"], &params.query(), Payload::Empty, &[], CallOptions::default()).await
    }

    /// `GET /v1/models/{id}`; `id` may be an alias.
    pub async fn model(&self, id: &str) -> Result<Model, Error> {
        self.call(Method::GET, &["v1", "models", id], &[], Payload::Empty, &[], CallOptions::default()).await
    }
}
