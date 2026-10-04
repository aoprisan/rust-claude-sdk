//! Files API: upload once, then refer to the file by id from `document` and `image` blocks
//! (`DocumentSource::File`, `ImageSource::File`), and download what code execution wrote.
use reqwest::Method;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::client::{CallOptions, Client, Payload};
use crate::error::Error;
use crate::page::{Deleted, ListParams, Page};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileMetadata {
    pub id: String,
    #[serde(default)]
    pub filename: String,
    #[serde(default)]
    pub mime_type: String,
    #[serde(default)]
    pub size_bytes: u64,
    /// RFC 3339.
    #[serde(default)]
    pub created_at: String,
    /// Only files created by tools (code execution, skills) can be downloaded.
    #[serde(default)]
    pub downloadable: bool,
    /// `type` and anything else.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Client {
    /// `POST /v1/files` (multipart). The block that uses the file must match its type:
    /// `document` for PDF and plain text, `image` for images.
    pub async fn upload_file(&self, filename: &str, mime_type: &str, data: &[u8]) -> Result<FileMetadata, Error> {
        self.call(Method::POST, &["v1", "files"], &[], Payload::File { filename, mime_type, data }, &[], CallOptions::default()).await
    }

    /// `GET /v1/files`: one page, newest first.
    pub async fn list_files(&self, params: &ListParams) -> Result<Page<FileMetadata>, Error> {
        self.call(Method::GET, &["v1", "files"], &params.query(), Payload::Empty, &[], CallOptions::default()).await
    }

    /// `GET /v1/files/{id}`.
    pub async fn file(&self, id: &str) -> Result<FileMetadata, Error> {
        self.call(Method::GET, &["v1", "files", id], &[], Payload::Empty, &[], CallOptions::default()).await
    }

    /// `GET /v1/files/{id}/content`: the file's bytes (downloadable files only).
    pub async fn download_file(&self, id: &str) -> Result<Vec<u8>, Error> {
        self.call_bytes(Method::GET, &["v1", "files", id, "content"], &[], Payload::Empty, &[], CallOptions::default()).await
    }

    /// `DELETE /v1/files/{id}`.
    pub async fn delete_file(&self, id: &str) -> Result<Deleted, Error> {
        self.call(Method::DELETE, &["v1", "files", id], &[], Payload::Empty, &[], CallOptions::default()).await
    }
}
