//! The tool-use loop: send the request, run the tools the model asks for, send their
//! results back, and repeat until the model ends its turn.
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::task::Poll;

use serde_json::Value;

use crate::client::{CallOptions, Client};
use crate::error::Error;
use crate::request::{Content, ContentBlockParam, MessageParam, MessagesRequest, Role, ToolResultContent};
use crate::response::{Message, StopReason, Usage};

/// One tool call the model asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
}

/// What a tool returns to the model. A failure goes back with `is_error` set, so the model
/// can try another way; it does not end the loop.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutput {
    pub content: ToolResultContent,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn text(text: impl Into<String>) -> ToolOutput {
        ToolOutput { content: ToolResultContent::Text(text.into()), is_error: false }
    }

    pub fn error(message: impl Into<String>) -> ToolOutput {
        ToolOutput { content: ToolResultContent::Text(message.into()), is_error: true }
    }

    /// Text, images or documents.
    pub fn blocks(blocks: Vec<ContentBlockParam>) -> ToolOutput {
        ToolOutput { content: ToolResultContent::Blocks(blocks), is_error: false }
    }

    fn into_block(self, tool_use_id: String) -> ContentBlockParam {
        ContentBlockParam::ToolResult { tool_use_id, content: self.content, is_error: self.is_error.then_some(true), cache_control: None }
    }
}

impl From<String> for ToolOutput {
    fn from(text: String) -> ToolOutput {
        ToolOutput::text(text)
    }
}

impl From<&str> for ToolOutput {
    fn from(text: &str) -> ToolOutput {
        ToolOutput::text(text)
    }
}

/// `Err` goes back as an error result carrying the error's message.
impl<T: Into<ToolOutput>, E: fmt::Display> From<Result<T, E>> for ToolOutput {
    fn from(result: Result<T, E>) -> ToolOutput {
        result.map_or_else(|e| ToolOutput::error(e.to_string()), Into::into)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ToolLoopOptions {
    /// Most requests one call of the loop sends, `pause_turn` continuations included.
    pub max_requests: u32,
    /// Applied to every request; a deadline bounds the whole loop.
    pub call: CallOptions,
    /// Send each request streamed and accumulate it, for long outputs or large `max_tokens`.
    pub stream: bool,
}

impl Default for ToolLoopOptions {
    fn default() -> ToolLoopOptions {
        ToolLoopOptions { max_requests: 20, call: CallOptions::default(), stream: false }
    }
}

/// How a tool loop ended.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolRun {
    /// The last response. Its turn is also the last entry of the request's `messages`.
    pub message: Message,
    pub requests: u32,
    /// Token counts summed over the requests; `extra` is left empty.
    pub usage: Usage,
}

impl ToolRun {
    /// False when the loop stopped at `max_requests` with tool calls still to run or a
    /// paused turn to resume; calling the loop again with the same request carries on.
    pub fn is_finished(&self) -> bool {
        !matches!(self.message.stop_reason, Some(StopReason::ToolUse | StopReason::PauseTurn))
    }
}

impl Client {
    /// Runs the tool-use loop on `request`, whose `messages` grow with every turn.
    ///
    /// Each response is appended as an assistant turn. On `tool_use`, `handler` runs every
    /// call of the turn (concurrently, on this task) and the results go back in one user
    /// turn, in call order. On `pause_turn` the request is sent again as is, so the server
    /// resumes its own tool loop. Any other stop reason (`end_turn`, `max_tokens`,
    /// `refusal`...) ends the loop: check it on the returned message.
    ///
    /// `request.messages` stays valid to send whatever happens: after an error, or when
    /// `max_requests` cut the loop short, calling this again resumes, first running the
    /// calls of a trailing assistant turn that has no results yet. To ask something new
    /// after a finished run, push a user turn.
    pub async fn run_tools<F, Fut, O>(&self, request: &mut MessagesRequest, handler: F) -> Result<ToolRun, Error>
    where
        F: FnMut(ToolCall) -> Fut,
        Fut: Future<Output = O>,
        O: Into<ToolOutput>,
    {
        self.run_tools_with(request, ToolLoopOptions::default(), handler).await
    }

    pub async fn run_tools_with<F, Fut, O>(&self, request: &mut MessagesRequest, options: ToolLoopOptions, mut handler: F) -> Result<ToolRun, Error>
    where
        F: FnMut(ToolCall) -> Fut,
        Fut: Future<Output = O>,
        O: Into<ToolOutput>,
    {
        let unanswered = request.messages.last().map(unanswered_calls).unwrap_or_default();
        if !unanswered.is_empty() {
            let results = run_calls(&mut handler, unanswered).await;
            request.messages.push(MessageParam::user(results));
        }
        let mut requests = 0;
        let mut usage = Usage::default();
        loop {
            let message = if options.stream {
                self.stream_with(request, options.call).await?.final_message().await?
            } else {
                self.create_with(request, options.call).await?
            };
            requests += 1;
            usage.add(&message.usage);
            request.messages.push(MessageParam::assistant(message.content.iter().cloned().map(ContentBlockParam::from).collect()));
            let calls: Vec<ToolCall> =
                message.tool_uses().map(|(id, name, input)| ToolCall { id: id.into(), name: name.into(), input: input.clone() }).collect();
            let more = match message.stop_reason {
                Some(StopReason::ToolUse) => !calls.is_empty(),
                Some(StopReason::PauseTurn) => true,
                _ => false,
            };
            if !more || requests >= options.max_requests.max(1) {
                return Ok(ToolRun { message, requests, usage });
            }
            if message.stop_reason == Some(StopReason::ToolUse) {
                let results = run_calls(&mut handler, calls).await;
                request.messages.push(MessageParam::user(results));
            }
        }
    }
}

/// The tool calls of an assistant turn, which then still need their results.
fn unanswered_calls(turn: &MessageParam) -> Vec<ToolCall> {
    match (&turn.role, &turn.content) {
        (Role::Assistant, Content::Blocks(blocks)) => blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlockParam::ToolUse { id, name, input } => Some(ToolCall { id: id.clone(), name: name.clone(), input: input.clone() }),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

async fn run_calls<F, Fut, O>(handler: &mut F, calls: Vec<ToolCall>) -> Vec<ContentBlockParam>
where
    F: FnMut(ToolCall) -> Fut,
    Fut: Future<Output = O>,
    O: Into<ToolOutput>,
{
    let ids: Vec<String> = calls.iter().map(|c| c.id.clone()).collect();
    let outputs = join_all(calls.into_iter().map(handler).collect()).await;
    ids.into_iter().zip(outputs).map(|(id, out)| out.into().into_block(id)).collect()
}

/// Polls every future until all are done; outputs in input order.
async fn join_all<F: Future>(futures: Vec<F>) -> Vec<F::Output> {
    let mut futures: Vec<Pin<Box<F>>> = futures.into_iter().map(Box::pin).collect();
    let mut outputs: Vec<Option<F::Output>> = futures.iter().map(|_| None).collect();
    std::future::poll_fn(|cx| {
        let mut pending = false;
        for (future, output) in futures.iter_mut().zip(outputs.iter_mut()).filter(|(_, o)| o.is_none()) {
            match future.as_mut().poll(cx) {
                Poll::Ready(v) => *output = Some(v),
                Poll::Pending => pending = true,
            }
        }
        if pending {
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    })
    .await;
    outputs.into_iter().map(Option::unwrap).collect()
}
