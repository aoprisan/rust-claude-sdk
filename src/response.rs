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

    /// The tool calls the model asked for: `(id, name, input)`.
    pub fn tool_uses(&self) -> impl Iterator<Item = (&str, &str, &Value)> {
        self.content.iter().filter_map(|b| match b {
            ContentBlock::ToolUse { id, name, input } => Some((id.as_str(), name.as_str(), input)),
            _ => None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
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
    /// Marks where a refused attempt handed over to a fallback model.
    Fallback {
        from: ModelRef,
        to: ModelRef,
    },
    /// Any other block type, kept verbatim.
    #[serde(skip)]
    Other(Value),
}
open_enum!(ContentBlock, ["text", "thinking", "redacted_thinking", "tool_use", "fallback"]);

// `remote = "Self"` derives `serialize` as an inherent fn; the derive must sit on the same
// item, so it is spelled out here.
impl ContentBlock {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum Out<'a> {
            Text {
                text: &'a str,
                #[serde(skip_serializing_if = "<[Citation]>::is_empty")]
                citations: &'a [Citation],
            },
            Thinking {
                thinking: &'a str,
                signature: &'a str,
            },
            RedactedThinking {
                data: &'a str,
            },
            ToolUse {
                id: &'a str,
                name: &'a str,
                input: &'a Value,
            },
            Fallback {
                from: &'a ModelRef,
                to: &'a ModelRef,
            },
        }
        let out = match self {
            ContentBlock::Text { text, citations } => Out::Text { text, citations },
            ContentBlock::Thinking { thinking, signature } => Out::Thinking { thinking, signature },
            ContentBlock::RedactedThinking { data } => Out::RedactedThinking { data },
            ContentBlock::ToolUse { id, name, input } => Out::ToolUse { id, name, input },
            ContentBlock::Fallback { from, to } => Out::Fallback { from, to },
            ContentBlock::Other(v) => return v.serialize(s),
        };
        out.serialize(s)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef {
    pub model: String,
}

/// Where a cited span comes from.
#[derive(Debug, Clone, PartialEq, Deserialize)]
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
    /// Any other citation type (search results, web pages...), kept verbatim.
    #[serde(skip)]
    Other(Value),
}
open_enum!(Citation, ["char_location", "page_location", "content_block_location"]);

impl Citation {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // Variant names are the wire tags.
        #[allow(clippy::enum_variant_names)]
        #[derive(Serialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum Out<'a> {
            CharLocation {
                cited_text: &'a str,
                document_index: usize,
                document_title: &'a Option<String>,
                start_char_index: usize,
                end_char_index: usize,
            },
            PageLocation {
                cited_text: &'a str,
                document_index: usize,
                document_title: &'a Option<String>,
                start_page_number: usize,
                end_page_number: usize,
            },
            ContentBlockLocation {
                cited_text: &'a str,
                document_index: usize,
                document_title: &'a Option<String>,
                start_block_index: usize,
                end_block_index: usize,
            },
        }
        let out = match self {
            Citation::CharLocation { cited_text, document_index, document_title, start_char_index, end_char_index } => Out::CharLocation {
                cited_text,
                document_index: *document_index,
                document_title,
                start_char_index: *start_char_index,
                end_char_index: *end_char_index,
            },
            Citation::PageLocation { cited_text, document_index, document_title, start_page_number, end_page_number } => Out::PageLocation {
                cited_text,
                document_index: *document_index,
                document_title,
                start_page_number: *start_page_number,
                end_page_number: *end_page_number,
            },
            Citation::ContentBlockLocation { cited_text, document_index, document_title, start_block_index, end_block_index } => Out::ContentBlockLocation {
                cited_text,
                document_index: *document_index,
                document_title,
                start_block_index: *start_block_index,
                end_block_index: *end_block_index,
            },
            Citation::Other(v) => return v.serialize(s),
        };
        out.serialize(s)
    }

    /// The quoted source text, when the citation type has one.
    pub fn cited_text(&self) -> Option<&str> {
        match self {
            Citation::CharLocation { cited_text, .. } | Citation::PageLocation { cited_text, .. } | Citation::ContentBlockLocation { cited_text, .. } => {
                Some(cited_text)
            }
            Citation::Other(v) => v.get("cited_text").and_then(Value::as_str),
        }
    }

    /// Index of the cited document among the request's documents.
    pub fn document_index(&self) -> Option<usize> {
        match self {
            Citation::CharLocation { document_index, .. }
            | Citation::PageLocation { document_index, .. }
            | Citation::ContentBlockLocation { document_index, .. } => Some(*document_index),
            Citation::Other(v) => v.get("document_index").and_then(Value::as_u64).map(|i| i as usize),
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
    /// Anything else (`iterations`, `service_tier`, `inference_geo`...).
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Usage {
    /// Adds another request's token counts; `extra` is left as is.
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
                _ => {
                    self.extra.insert(k.clone(), v.clone());
                }
            }
        }
    }
}
