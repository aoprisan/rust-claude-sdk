//! Server tools: the API searches the web and fetches pages itself, and the answer cites
//! the pages it used.
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run --example web_search -- "What documents do I need for a Romanian passport?"
//! ```
use rust_claude_sdk::{Client, ClientConfig, MessagesRequest, ToolLoopOptions, ToolOutput, WebFetchTool, WebSearchTool};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let question = std::env::args().nth(1).unwrap_or_else(|| "What documents do I need for a Romanian passport?".into());
    let client = Client::new(ClientConfig::from_env().ok_or("set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN)")?)?;

    let mut request = MessagesRequest::new("claude-opus-5-5", 8000)
        .system("Answer from official sources only and cite them.")
        .tool(WebSearchTool::new().max_uses(5).allowed_domains(["gov.ro", "mai.gov.ro"]))
        .tool(WebFetchTool::new().max_uses(3).allowed_domains(["gov.ro", "mai.gov.ro"]).with_citations().max_content_tokens(20_000))
        .user(question);

    // Server tools run inside the API, so there are no local calls to handle; the loop is
    // still useful because it resumes `pause_turn` when the server's own loop runs long.
    let options = ToolLoopOptions { stream: true, ..ToolLoopOptions::default() };
    let run = client.run_tools_with(&mut request, options, |call| async move { ToolOutput::error(format!("no local tool {}", call.name)) }).await?;
    let message = &run.message;
    if message.is_refusal() {
        println!("(declined)");
        return Ok(());
    }

    println!("{}\n", message.text());
    println!("Sources:");
    let mut seen = Vec::new();
    for (_, citation) in message.citations() {
        if let Some(url) = citation.url().filter(|u| !seen.contains(u)) {
            seen.push(url);
            println!("  {url}");
        }
    }
    for (id, error) in message.server_tool_errors() {
        eprintln!("server tool call {id} failed: {}", error.error_code);
    }
    if let Some(used) = &run.usage.server_tool_use {
        println!("\n{} searches, {} fetches", used.web_search_requests, used.web_fetch_requests);
    }
    Ok(())
}
