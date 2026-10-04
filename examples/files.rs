//! The Files API: upload a PDF once, ask about it by id, then delete it.
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run --example files -- path/to/document.pdf "Summarise it in five bullet points."
//! ```
use rust_claude_sdk::{Client, ClientConfig, ContentBlockParam, ListParams, MessageParam, MessagesRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: files <document.pdf> [question]")?;
    let question = args.next().unwrap_or_else(|| "Summarise it in five bullet points.".into());

    let mut config = ClientConfig::from_env().ok_or("set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN)")?;
    config.betas.push("files-api-2025-04-14".into());
    let client = Client::new(config)?;

    let filename = std::path::Path::new(&path).file_name().map_or("document.pdf".into(), |n| n.to_string_lossy().into_owned());
    let file = client.upload_file(&filename, "application/pdf", &std::fs::read(&path)?).await?;
    println!("uploaded {} as {} ({} bytes)", file.filename, file.id, file.size_bytes);

    let request = MessagesRequest::new("claude-opus-5-5", 4096)
        .message(MessageParam::user(vec![ContentBlockParam::file_document(&file.id).with_citations(), ContentBlockParam::text(question)]));
    let answer = client.create(&request).await;

    // Clean up whether or not the question succeeded.
    client.delete_file(&file.id).await?;
    println!("deleted {}", file.id);
    let page = client.list_files(&ListParams::limit(5)).await?;
    println!("{} recent files left in the workspace\n", page.data.len());

    let message = answer?;
    if message.is_refusal() {
        println!("(declined)");
    } else {
        println!("{}", message.text());
        for (span, citation) in message.citations() {
            println!("  «{span}» ← {}", citation.cited_text().unwrap_or(""));
        }
    }
    Ok(())
}
