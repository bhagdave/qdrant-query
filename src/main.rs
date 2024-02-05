use clap::Parser;
use orca::{
    llm::{bert::Bert, quantized::Quantized, Embedding},
    pipeline::simple::LLMPipeline,
    pipeline::Pipeline,
    prompt,
    prompt::context::Context,
    qdrant::Qdrant,
};
use serde_json::{Value, json};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[clap(long)]
    /// The prompt to use to query the index
    prompt: String,
    
    #[clap(long)]
    /// The qdrant collection to query
    collection: String,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let bert = Bert::new().build_model_and_tokenizer().await.unwrap();

    let qdrant = Qdrant::new("http://localhost:6334");

    let query_embedding = bert.generate_embedding(prompt!(args.prompt)).await.unwrap();
    let search_results = qdrant.search(&args.collection, query_embedding.to_vec().unwrap().clone(), 20, None).await.unwrap();
    for result in &search_results {
        if let Some(payload) = &result.payload {
            for (_key, value) in payload {
                let json_string = value.clone().into_json().to_string();
                match serde_json::from_str::<Value>(&json_string) {
                    Ok(outer_json) => {
                        if let Some(inner_json_string) = outer_json.as_str() {
                            match serde_json::from_str::<Value>(inner_json_string) {
                                Ok(inner_json) => {
                                    // Access and print the "content" field
                                    if let Some(content) = inner_json["content"].as_str() {
                                        println!("{}", content);
                                    } else {
                                        println!("Content field is missing or not a string");
                                    }
                                },
                                Err(e) => {
                                    println!("Failed to parse inner JSON: {}", e);
                                }
                            }
                        } else {
                            println!("Outer JSON is not a string");
                        }
                    },
                    Err(e) => {
                        println!("Failed to parse JSON: {}", e);
                    }
                }
            }
        } else {
            println!("No payload or payload is not in the expected format");
        }
    }


    let prompt_for_model = r#"
    {{#chat}}
        {{#system}}
        You are a highly advanced assistant for STEM learning. You receive a prompt from a user and relevant excerpts extracted from slack messages. You then answer truthfully to the best of your ability. If you do not know the answer, your response is I don't know.
        {{/system}}
        {{#user}}
        {{user_prompt}}
        {{/user}}
        {{#system}}
        Based on the retrieved information from the slack messages, here are the relevant excerpts:
        {{#each payloads}}
        {{this}}
        {{/each}}
        Please provide a comprehensive answer to the user's question, integrating insights from these excerpts and your general knowledge.
        {{/system}}
    {{/chat}}
    "#;

    let context = json!({
        "user_prompt": args.prompt,
        "payloads": search_results
            .iter()
            .filter_map(|found_point| {
                found_point.payload.as_ref().map(|payload| {
                    // Assuming you want to convert the whole payload to a JSON string
                    serde_json::to_string(payload).unwrap_or_else(|_| "{}".to_string())
                })
            })
            .collect::<Vec<String>>()
    });

    let mistral = Quantized::new()
        .with_model(orca::llm::quantized::Model::Mistral7bInstruct)
        .with_sample_len(7500)
        .load_model_from_path("/home/dave/Projects/candle-app/models/mistral-7b-instruct-v0.1.Q4_K_S.gguf")
        .unwrap()
        .build_model()
        .unwrap();


    let mut pipe = LLMPipeline::new(&mistral).with_template("query", prompt_for_model);
    pipe.load_context(&Context::new(context).unwrap()).await;

    let response = pipe.execute("query").await.unwrap();

    println!("Response: {}", response.content());

}

