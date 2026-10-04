//! Cursor pagination shared by the list endpoints.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Which page of a list to fetch. Pages run newest first; `after_id` moves to older items,
/// `before_id` to newer ones.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ListParams {
    /// Items per page (the API's default is 20, its maximum 1000).
    pub limit: Option<u32>,
    pub before_id: Option<String>,
    pub after_id: Option<String>,
}

impl ListParams {
    pub fn limit(limit: u32) -> ListParams {
        ListParams { limit: Some(limit), ..ListParams::default() }
    }

    pub(crate) fn query(&self) -> Vec<(&'static str, String)> {
        let mut q = Vec::new();
        if let Some(l) = self.limit {
            q.push(("limit", l.to_string()));
        }
        if let Some(b) = &self.before_id {
            q.push(("before_id", b.clone()));
        }
        if let Some(a) = &self.after_id {
            q.push(("after_id", a.clone()));
        }
        q
    }
}

/// One page of a list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Page<T> {
    pub data: Vec<T>,
    #[serde(default)]
    pub has_more: bool,
    #[serde(default)]
    pub first_id: Option<String>,
    #[serde(default)]
    pub last_id: Option<String>,
}

impl<T> Page<T> {
    /// The parameters of the following page (same `limit`), or `None` on the last one.
    pub fn next_params(&self, current: &ListParams) -> Option<ListParams> {
        let last = self.last_id.clone().filter(|_| self.has_more)?;
        Some(ListParams { limit: current.limit, before_id: None, after_id: Some(last) })
    }
}

/// The answer to a `DELETE`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Deleted {
    pub id: String,
    /// `type` (`message_batch_deleted`, `file_deleted`...) and anything else.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}
