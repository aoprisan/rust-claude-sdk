//! Server-side code execution: the model writes and runs code in a sandbox, and any files
//! it writes are downloaded with the Files API.
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run --example code_execution
//! ```
use rust_claude_sdk::{Client, ClientConfig, CodeExecutionContent, CodeExecutionTool, ContentBlock, MessagesRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = ClientConfig::from_env().ok_or("set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN)")?;
    // Downloading the files the sandbox writes uses the Files API.
    config.betas.push("files-api-2025-04-14".into());
    let client = Client::new(config)?;

    let request = MessagesRequest::new("claude-opus-5-5", 8000).tool(CodeExecutionTool::new()).user(
        "Compute the first 30 Fibonacci numbers, report how many are prime, \
         and save them with a primality column to fibonacci.csv.",
    );
    let message = client.create(&request).await?;
    if message.is_refusal() {
        println!("(declined)");
        return Ok(());
    }

    let mut files = Vec::new();
    for block in &message.content {
        match block {
            ContentBlock::ServerToolUse { name, input, .. } => println!("$ {name}: {}", input.get("command").unwrap_or(input)),
            ContentBlock::BashCodeExecutionToolResult { content: CodeExecutionContent::Output(out), .. } => {
                print!("{}", out.stdout);
                eprint!("{}", out.stderr);
                if out.return_code != 0 {
                    println!("(exit {})", out.return_code);
                }
                files.extend(out.file_ids().map(String::from));
            }
            ContentBlock::Text { text, .. } => println!("{text}"),
            _ => {}
        }
    }
    for (id, error) in message.server_tool_errors() {
        eprintln!("code execution {id} failed: {}", error.error_code);
    }

    for id in files {
        let meta = client.file(&id).await?;
        let bytes = client.download_file(&id).await?;
        let name = std::path::Path::new(&meta.filename).file_name().ok_or("file without a name")?;
        std::fs::write(name, &bytes)?;
        println!("saved {} ({} bytes)", name.to_string_lossy(), bytes.len());
    }
    Ok(())
}
