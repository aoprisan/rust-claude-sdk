//! Streaming: server-sent events from `POST /v1/messages` with `"stream": true`.
use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::time::Instant;

use crate::error::{ApiErrorBody, Error};
use crate::response::{Citation, ContentBlock, Message, StopReason};

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(remote = "Self", tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart {
        message: Message,
    },
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: Delta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        #[serde(default)]
        delta: Map<String, Value>,
        #[serde(default)]
        usage: Map<String, Value>,
    },
    MessageStop,
    Ping,
    Error {
        error: ApiErrorBody,
    },
    /// Any other event type, kept verbatim (new types may be added at any time).
    #[serde(skip)]
    Other(Value),
}
open_enum!(
    StreamEvent,
    ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop", "ping", "error",]
);

impl StreamEvent {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum Out<'a> {
            MessageStart { message: &'a Message },
            ContentBlockStart { index: usize, content_block: &'a ContentBlock },
            ContentBlockDelta { index: usize, delta: &'a Delta },
            ContentBlockStop { index: usize },
            MessageDelta { delta: &'a Map<String, Value>, usage: &'a Map<String, Value> },
            MessageStop,
            Ping,
            Error { error: &'a ApiErrorBody },
        }
        let out = match self {
            StreamEvent::MessageStart { message } => Out::MessageStart { message },
            StreamEvent::ContentBlockStart { index, content_block } => Out::ContentBlockStart { index: *index, content_block },
            StreamEvent::ContentBlockDelta { index, delta } => Out::ContentBlockDelta { index: *index, delta },
            StreamEvent::ContentBlockStop { index } => Out::ContentBlockStop { index: *index },
            StreamEvent::MessageDelta { delta, usage } => Out::MessageDelta { delta, usage },
            StreamEvent::MessageStop => Out::MessageStop,
            StreamEvent::Ping => Out::Ping,
            StreamEvent::Error { error } => Out::Error { error },
            StreamEvent::Other(v) => return v.serialize(s),
        };
        out.serialize(s)
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(remote = "Self", tag = "type", rename_all = "snake_case")]
pub enum Delta {
    TextDelta {
        text: String,
    },
    /// A piece of a tool call's JSON input; only the concatenation parses.
    InputJsonDelta {
        partial_json: String,
    },
    ThinkingDelta {
        thinking: String,
    },
    SignatureDelta {
        signature: String,
    },
    /// One citation to add to the current text block.
    CitationsDelta {
        citation: Citation,
    },
    #[serde(skip)]
    Other(Value),
}
open_enum!(Delta, ["text_delta", "input_json_delta", "thinking_delta", "signature_delta", "citations_delta"]);

impl Delta {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // Variant names are the wire tags.
        #[allow(clippy::enum_variant_names)]
        #[derive(Serialize)]
        #[serde(tag = "type", rename_all = "snake_case")]
        enum Out<'a> {
            TextDelta { text: &'a str },
            InputJsonDelta { partial_json: &'a str },
            ThinkingDelta { thinking: &'a str },
            SignatureDelta { signature: &'a str },
            CitationsDelta { citation: &'a Citation },
        }
        let out = match self {
            Delta::TextDelta { text } => Out::TextDelta { text },
            Delta::InputJsonDelta { partial_json } => Out::InputJsonDelta { partial_json },
            Delta::ThinkingDelta { thinking } => Out::ThinkingDelta { thinking },
            Delta::SignatureDelta { signature } => Out::SignatureDelta { signature },
            Delta::CitationsDelta { citation } => Out::CitationsDelta { citation },
            Delta::Other(v) => return v.serialize(s),
        };
        out.serialize(s)
    }
}

/// Splits a byte stream into the `data` of each server-sent event.
#[derive(Debug, Default)]
pub(crate) struct SseParser {
    buf: Vec<u8>,
}

impl SseParser {
    /// Feeds bytes and returns the data of every event completed by them.
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some((end, sep)) = find_event_end(&self.buf) {
            let raw: Vec<u8> = self.buf.drain(..end + sep).collect();
            if let Some(data) = event_data(&String::from_utf8_lossy(&raw[..end])) {
                out.push(data);
            }
        }
        out
    }

    /// Whatever is left when the body ends (an event without its blank line).
    pub(crate) fn finish(&mut self) -> Option<String> {
        let rest = String::from_utf8_lossy(&std::mem::take(&mut self.buf)).into_owned();
        event_data(&rest)
    }
}

/// Position and length of the first blank-line separator (`\n\n`, `\r\n\r\n` or `\r\r`).
fn find_event_end(buf: &[u8]) -> Option<(usize, usize)> {
    (0..buf.len()).find_map(|i| {
        if buf[i..].starts_with(b"\r\n\r\n") {
            Some((i, 4))
        } else if buf[i..].starts_with(b"\n\n") || buf[i..].starts_with(b"\r\r") {
            Some((i, 2))
        } else {
            None
        }
    })
}

/// The `data` field of one event: its `data:` lines joined with newlines; `None` for an
/// event without data (comments, a bare `event:` line).
fn event_data(event: &str) -> Option<String> {
    let mut lines = Vec::new();
    for line in event.split(['\n', '\r']) {
        if let Some(rest) = line.strip_prefix("data") {
            if rest.is_empty() {
                lines.push("");
            } else if let Some(v) = rest.strip_prefix(':') {
                lines.push(v.strip_prefix(' ').unwrap_or(v));
            }
        }
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

/// Builds the final message from the events.
#[derive(Debug, Default)]
pub(crate) struct Accumulator {
    message: Option<Message>,
    partial_json: HashMap<usize, String>,
}

impl Accumulator {
    pub(crate) fn apply(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::MessageStart { message } => self.message = Some(message.clone()),
            StreamEvent::ContentBlockStart { index, content_block } => {
                if let Some(m) = &mut self.message {
                    while m.content.len() < *index {
                        m.content.push(ContentBlock::Other(Value::Null));
                    }
                    if *index < m.content.len() {
                        m.content[*index] = content_block.clone();
                    } else {
                        m.content.push(content_block.clone());
                    }
                }
            }
            StreamEvent::ContentBlockDelta { index, delta } => {
                if let Delta::InputJsonDelta { partial_json } = delta {
                    self.partial_json.entry(*index).or_default().push_str(partial_json);
                    return;
                }
                let Some(block) = self.message.as_mut().and_then(|m| m.content.get_mut(*index)) else { return };
                match (block, delta) {
                    (ContentBlock::Text { text, .. }, Delta::TextDelta { text: t }) => text.push_str(t),
                    (ContentBlock::Text { citations, .. }, Delta::CitationsDelta { citation }) => citations.push(citation.clone()),
                    (ContentBlock::Thinking { thinking, .. }, Delta::ThinkingDelta { thinking: t }) => thinking.push_str(t),
                    (ContentBlock::Thinking { signature, .. }, Delta::SignatureDelta { signature: s }) => signature.push_str(s),
                    _ => {}
                }
            }
            StreamEvent::ContentBlockStop { index } => {
                let Some(json) = self.partial_json.remove(index) else { return };
                if let Some(ContentBlock::ToolUse { input, .. }) = self.message.as_mut().and_then(|m| m.content.get_mut(*index)) {
                    // An empty input streams as no delta, or as an empty string.
                    *input = if json.trim().is_empty() { Value::Object(Map::new()) } else { serde_json::from_str(&json).unwrap_or(Value::String(json)) };
                }
            }
            StreamEvent::MessageDelta { delta, usage } => {
                if let Some(m) = &mut self.message {
                    if let Some(r) = delta.get("stop_reason").and_then(Value::as_str) {
                        m.stop_reason = Some(StopReason::from(r.to_string()));
                    }
                    if let Some(s) = delta.get("stop_sequence") {
                        m.stop_sequence = s.as_str().map(str::to_string);
                    }
                    if let Some(d) = delta.get("stop_details").filter(|d| !d.is_null()) {
                        m.stop_details = Some(d.clone());
                    }
                    m.usage.merge(usage);
                }
            }
            _ => {}
        }
    }

    pub(crate) fn message(&self) -> Option<&Message> {
        self.message.as_ref()
    }
}

/// An open stream. Read it event by event with `next_event`, or all at once with
/// `final_message`; either way the message is accumulated as events arrive.
pub struct MessageStream {
    response: reqwest::Response,
    parser: SseParser,
    pending: VecDeque<String>,
    acc: Accumulator,
    deadline: Option<Instant>,
    ended: bool,
    request_id: Option<String>,
}

impl MessageStream {
    pub(crate) fn new(response: reqwest::Response, deadline: Option<Instant>, request_id: Option<String>) -> MessageStream {
        MessageStream { response, parser: SseParser::default(), pending: VecDeque::new(), acc: Accumulator::default(), deadline, ended: false, request_id }
    }

    /// The `request-id` of the response, for support requests.
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    /// The message as accumulated so far (`None` before `message_start`).
    pub fn snapshot(&self) -> Option<&Message> {
        self.acc.message()
    }

    /// The next event, `None` once the stream has ended. An `error` event becomes
    /// `Err(Error::Api { status: None, .. })`.
    pub async fn next_event(&mut self) -> Option<Result<StreamEvent, Error>> {
        loop {
            if let Some(data) = self.pending.pop_front() {
                let event = match serde_json::from_str::<StreamEvent>(&data) {
                    Ok(e) => e,
                    Err(e) => return Some(Err(Error::Decode(e.to_string()))),
                };
                self.acc.apply(&event);
                if let StreamEvent::Error { error } = &event {
                    self.ended = true;
                    return Some(Err(Error::Api {
                        status: None,
                        kind: error.kind.clone(),
                        message: error.message.clone(),
                        request_id: self.request_id.clone(),
                    }));
                }
                return Some(Ok(event));
            }
            if self.ended {
                return None;
            }
            let chunk = match self.deadline {
                Some(d) => match tokio::time::timeout_at(d, self.response.chunk()).await {
                    Ok(c) => c,
                    Err(_) => {
                        self.ended = true;
                        return Some(Err(Error::Deadline));
                    }
                },
                None => self.response.chunk().await,
            };
            match chunk {
                Ok(Some(bytes)) => self.pending.extend(self.parser.push(&bytes)),
                Ok(None) => {
                    self.ended = true;
                    self.pending.extend(self.parser.finish());
                }
                Err(e) => {
                    self.ended = true;
                    return Some(Err(Error::Transport(e.without_url().to_string())));
                }
            }
        }
    }

    /// Reads the rest of the stream and returns the complete message. Fails if the stream
    /// reported an error or ended before `message_stop`.
    pub async fn final_message(mut self) -> Result<Message, Error> {
        let mut stopped = false;
        while let Some(event) = self.next_event().await {
            if matches!(event?, StreamEvent::MessageStop) {
                stopped = true;
            }
        }
        match self.acc.message {
            Some(m) if stopped => Ok(m),
            _ => Err(Error::Decode("the stream ended before message_stop".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sse_events_split_across_chunks_and_line_endings() {
        let mut p = SseParser::default();
        assert!(p.push(b"event: ping\ndata: {\"type\":").is_empty());
        assert_eq!(p.push(b"\"ping\"}\n\n: comment\n\nevent: x\r\ndata: a\r\ndata: b\r\n\r\ndata:c"), vec![r#"{"type":"ping"}"#, "a\nb"]);
        assert_eq!(p.finish().as_deref(), Some("c"));
    }

    #[test]
    fn unknown_events_and_deltas_are_kept() {
        let e: StreamEvent = serde_json::from_value(json!({"type": "brand_new", "x": 1})).unwrap();
        assert_eq!(e, StreamEvent::Other(json!({"type": "brand_new", "x": 1})));
        let d: Delta = serde_json::from_value(json!({"type": "future_delta", "y": 2})).unwrap();
        assert_eq!(serde_json::to_value(&d).unwrap(), json!({"type": "future_delta", "y": 2}));
    }

    #[test]
    fn a_known_event_with_a_bad_shape_is_an_error() {
        assert!(serde_json::from_value::<StreamEvent>(json!({"type": "content_block_stop"})).is_err());
    }
}
