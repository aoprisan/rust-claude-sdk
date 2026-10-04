//! The Models API and token counting: list every model available to the key with its limits
//! and capabilities, then count a prompt's tokens before sending it.
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run --example models
//! ```
use rust_claude_sdk::{Client, ClientConfig, ListParams, MessagesRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(ClientConfig::from_env().ok_or("set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN)")?)?;

    // Walk every page.
    let mut params = ListParams::limit(20);
    loop {
        let page = client.list_models(&params).await?;
        for m in &page.data {
            println!(
                "{:<32} {:<24} context {:>9}  output {:>7}  images {:<5} adaptive thinking {}",
                m.id,
                m.display_name,
                m.max_input_tokens.map_or("?".into(), |n| n.to_string()),
                m.max_tokens.map_or("?".into(), |n| n.to_string()),
                m.supports(&["image_input"]),
                m.supports(&["thinking", "types", "adaptive"]),
            );
        }
        match page.next_params(&params) {
            Some(next) => params = next,
            None => break,
        }
    }

    let model = client.model("claude-opus-5-5").await?;
    let request = MessagesRequest::new(&model.id, 1024)
        .system("You are a helpful assistant.")
        .user("Explain the difference between a buletin and a carte electronică de identitate.");
    let tokens = client.count_tokens(&request).await?;
    println!("\nthat prompt is {tokens} input tokens on {}", model.display_name);
    Ok(())
}
