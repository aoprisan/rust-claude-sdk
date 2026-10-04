//! The tool-use loop: two local tools the model calls as it needs them, run concurrently
//! when it asks for several at once, with their results sent back until it answers.
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run --example tools -- "What's the weather in Cluj and Iași, and how much warmer is the warmer one?"
//! ```
use std::time::Duration;

use rust_claude_sdk::{CallOptions, Client, ClientConfig, MessagesRequest, Tool, ToolCall, ToolLoopOptions, ToolOutput};
use serde_json::json;

/// A stand-in for a real weather service.
async fn weather(city: &str) -> Result<String, String> {
    tokio::time::sleep(Duration::from_millis(200)).await;
    match city.to_lowercase().as_str() {
        "cluj" | "cluj-napoca" => Ok("14°C, cloudy".into()),
        "iași" | "iasi" => Ok("17°C, sunny".into()),
        "bucurești" | "bucuresti" | "bucharest" => Ok("19°C, sunny".into()),
        _ => Err(format!("no weather station in {city}")),
    }
}

fn calculate(op: &str, a: f64, b: f64) -> Result<String, String> {
    let r = match op {
        "add" => a + b,
        "subtract" => a - b,
        "multiply" => a * b,
        "divide" if b != 0.0 => a / b,
        "divide" => return Err("division by zero".into()),
        other => return Err(format!("unknown operation {other}")),
    };
    Ok(r.to_string())
}

async fn handle(call: ToolCall) -> ToolOutput {
    println!("  → {}({})", call.name, call.input);
    match call.name.as_str() {
        "get_weather" => weather(call.input["city"].as_str().unwrap_or_default()).await.into(),
        "calculate" => {
            let num = |k: &str| call.input[k].as_f64().unwrap_or_default();
            calculate(call.input["operation"].as_str().unwrap_or_default(), num("a"), num("b")).into()
        }
        other => ToolOutput::error(format!("unknown tool {other}")),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let question = std::env::args().nth(1).unwrap_or_else(|| "What's the weather in Cluj and Iași, and how much warmer is the warmer one?".into());
    let client = Client::new(ClientConfig::from_env().ok_or("set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN)")?)?;

    let mut request = MessagesRequest::new("claude-opus-5-5", 4096)
        .system("Use the tools for weather and arithmetic; do not guess.")
        .tool(Tool::new(
            "get_weather",
            "Current weather for a Romanian city.",
            json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        ))
        .tool(
            Tool::new(
                "calculate",
                "Exact arithmetic on two numbers.",
                json!({
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["operation", "a", "b"],
                    "additionalProperties": false
                }),
            )
            .strict(),
        )
        .user(question);

    let options = ToolLoopOptions { max_requests: 10, call: CallOptions::within(Duration::from_secs(120)), ..ToolLoopOptions::default() };
    let run = client.run_tools_with(&mut request, options, handle).await?;

    if !run.is_finished() {
        println!("(stopped after {} requests; calling run_tools again would carry on)", run.requests);
    } else if run.message.is_refusal() {
        println!("(declined)");
    } else {
        println!("\n{}", run.message.text());
    }
    println!("\n{} requests, {} tokens in, {} out", run.requests, run.usage.input_tokens, run.usage.output_tokens);
    Ok(())
}
