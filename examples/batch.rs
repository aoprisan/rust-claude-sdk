//! Message Batches: many requests at half price, processed asynchronously. Creates a
//! batch, polls until it ends, then streams the results.
//!
//! ```sh
//! ANTHROPIC_API_KEY=... cargo run --example batch
//! ```
use std::time::Duration;

use rust_claude_sdk::{BatchOutcome, BatchRequest, Client, ClientConfig, MessagesRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(ClientConfig::from_env().ok_or("set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN)")?)?;

    let words = ["primărie", "buletin", "pașaport", "permis de conducere"];
    let requests: Vec<BatchRequest> = words
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let params =
                MessagesRequest::new("claude-haiku-4-5", 200).system("Translate the Romanian term to English and explain it in one sentence.").user(*w);
            BatchRequest::new(format!("term-{i}"), params)
        })
        .collect();

    let mut batch = client.create_batch(&requests).await?;
    println!("created {}", batch.id);
    while !batch.is_ended() {
        tokio::time::sleep(Duration::from_secs(15)).await;
        batch = client.batch(&batch.id).await?;
        let c = &batch.request_counts;
        println!("{}: {} processing, {} succeeded, {} errored", batch.processing_status.as_str(), c.processing, c.succeeded, c.errored);
    }

    // Results come in any order; match them back by `custom_id`.
    let mut results = client.batch_results(&batch.id).await?;
    while let Some(result) = results.next().await {
        let result = result?;
        let i: usize = result.custom_id.trim_start_matches("term-").parse()?;
        match &result.result {
            BatchOutcome::Succeeded { message } => println!("\n{}: {}", words[i], message.text()),
            BatchOutcome::Errored { .. } => {
                let error = result.result.error();
                println!("\n{}: error {:?}", words[i], error.map(|e| e.message));
            }
            other => println!("\n{}: {other:?}", words[i]),
        }
    }
    Ok(())
}
