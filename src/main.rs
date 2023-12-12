use clap::Parser;
use orca::{
    llm::{bert::Bert, quantized::Quantized, Embedding},
    pipeline::simple::LLMPipeline,
    pipeline::Pipeline,
    prompt,
    prompt::context::Context,
    prompts,
    qdrant::Qdrant,
    record::{pdf::Pdf, Spin},
};
use serde_json::json;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(long)]
    /// The prompt to use to query the index
    prompt: String,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    println!("ARGS");
    let bert = Bert::new().build_model_and_tokenizer().await.unwrap();
    println!("BERT");

    let qdrant = Qdrant::new("http://localhost:6334");
    println!("QDRAVT");

    let query_embedding = bert.generate_embedding(prompt!(args.prompt)).await.unwrap();
    println!("query embedding");
    let search_results = qdrant.search("PRG", query_embedding.to_vec().unwrap().clone(), 5, None).await.unwrap();
    println!("RESULT");
    for result in search_results {
        println!("Id:{:}, Score:{:}", result.id, result.score);  // Modify based on how you want to display the results
    }
}

