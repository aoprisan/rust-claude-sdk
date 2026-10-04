//! A multi-turn chat in the terminal: streamed answers, adaptive thinking shown as it
//! happens, and the conversation kept in the request so each turn sees the earlier ones.
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run --example chat
//! ```
use std::io::{BufRead, Write};

use rust_claude_sdk::{Client, ClientConfig, ContentBlockParam, Delta, Effort, MessageParam, MessagesRequest, StreamEvent, ThinkingConfig, ThinkingDisplay};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(ClientConfig::from_env().ok_or("set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN)")?)?;
    let mut request = MessagesRequest::new("claude-opus-5-5", 8000)
        .system_cached("You are a concise, friendly assistant.")
        .thinking(ThinkingConfig::Adaptive { display: Some(ThinkingDisplay::Summarized) })
        .effort(Effort::Medium);

    println!("Type a message, an empty line to quit.");
    let stdin = std::io::stdin();
    loop {
        print!("\n> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 || line.trim().is_empty() {
            break;
        }
        request.messages.push(MessageParam::user_text(line.trim()));

        let mut stream = client.stream(&request).await?;
        let mut thinking = false;
        while let Some(event) = stream.next_event().await {
            match event? {
                StreamEvent::ContentBlockDelta { delta: Delta::ThinkingDelta { thinking: t }, .. } => {
                    if !thinking {
                        print!("\x1b[2m(thinking) ");
                        thinking = true;
                    }
                    print!("{t}");
                }
                StreamEvent::ContentBlockDelta { delta: Delta::TextDelta { text }, .. } => {
                    if thinking {
                        print!("\x1b[0m\n\n");
                        thinking = false;
                    }
                    print!("{text}");
                }
                _ => {}
            }
            std::io::stdout().flush()?;
        }
        if thinking {
            print!("\x1b[0m");
        }
        println!();

        let message = stream.final_message().await?;
        if message.is_refusal() {
            // A refused turn is not echoed back; drop the question so the chat can go on.
            request.messages.pop();
            println!("(declined)");
            continue;
        }
        // Echo the whole turn, thinking blocks included, as the next request's context.
        let blocks: Vec<ContentBlockParam> = message.content.into_iter().map(Into::into).collect();
        request.messages.push(MessageParam::assistant(blocks));
    }
    Ok(())
}
