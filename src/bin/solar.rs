use anyhow::{bail, Error, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models;
use candle_transformers::models::llama::{Llama, LlamaConfig};
use clap::Parser;
use log::info;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, io::Write};
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// the prompt
    #[arg(long, short = 'p', default_value = "Hello, how are you?")]
    prompt: String,

    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 10)]
    sample_len: usize,

    /// Use different dtype than f16
    #[arg(long)]
    dtype: Option<String>,

    /// Enable the key-value cache.
    #[arg(long)]
    use_kv_cache: bool,

    #[arg(long)]
    use_flash_attn: bool,

    /// the path of config file(in json format)
    #[arg(
        long,
        short = 'c',
        default_value = "model/SOLAR-10.7B-Instruct-v1.0/config.json"
    )]
    config: String,

    /// the path of safetensors index file(in json format)
    #[arg(
        long,
        short = 's',
        default_value = "model/SOLAR-10.7B-Instruct-v1.0/model.safetensors.index.json"
    )]
    safetensors_index: String,

    /// the path of model file(in gguf format)
    #[arg(
        long,
        short = 'm',
        default_value = "model/SOLAR-10.7B-Instruct-v1.0/solar-10.7b-instruct-v1.0.Q4_K_M.gguf"
    )]
    model: String,

    /// the path of tokenizer file(in json format)
    #[arg(
        long,
        short = 't',
        default_value = "model/SOLAR-10.7B-Instruct-v1.0/tokenizer.json"
    )]
    tokenizer: String,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

const EOS_TOKEN: &str = "</s>";

fn main() -> Result<()> {
    log4rs::init_file("log4rs.yaml", Default::default()).unwrap();
    info!("Initialized the logger");

    let args = Args::parse();

    let dtype = DType::F16;
    let (llama, tokenizer, cache) = {
        let tokenizer = std::path::PathBuf::from(args.tokenizer);
        let config = std::path::PathBuf::from(args.config);
        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config)?)?;
        let config = config.into_config(args.use_flash_attn);

        let safetensors = vec![
            std::path::PathBuf::from(
                "model/SOLAR-10.7B-Instruct-v1.0/model-00001-of-00005.safetensors",
            ),
            std::path::PathBuf::from(
                "model/SOLAR-10.7B-Instruct-v1.0/model-00002-of-00005.safetensors",
            ),
            std::path::PathBuf::from(
                "model/SOLAR-10.7B-Instruct-v1.0/model-00003-of-00005.safetensors",
            ),
            std::path::PathBuf::from(
                "model/SOLAR-10.7B-Instruct-v1.0/model-00004-of-00005.safetensors",
            ),
            std::path::PathBuf::from(
                "model/SOLAR-10.7B-Instruct-v1.0/model-00005-of-00005.safetensors",
            ),
        ];
        let cache = models::llama::Cache::new(args.use_kv_cache, dtype, &config, &Device::Cpu)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensors, dtype, &Device::Cpu)? };
        (Llama::load(vb, &cache, &config)?, tokenizer, cache) // died here!!!
    };
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(Error::msg)?;
    let eos_token_id = tokenizer.token_to_id(EOS_TOKEN);

    let prompt = args.prompt;
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(Error::msg)?
        .get_ids()
        .to_vec();

    info!("starting the inference loop");
    let mut logits_processor = LogitsProcessor::new(args.seed, args.temperature, args.top_p);
    let start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;
    for index in 0..args.sample_len {
        let context_size = if cache.use_kv_cache && index > 0 {
            1
        } else {
            tokens.len()
        };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &Device::Cpu)?.unsqueeze(0)?;
        let logits = llama.forward(&input, index_pos)?;
        let logits = logits.squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &tokens[start_at..],
            )?
        };
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        if let Some(text) = tokenizer.id_to_token(next_token) {
            let text = text.replace('▁', " ").replace("<0x0A>", "\n");
            print!("{text}");
            std::io::stdout().flush()?;
        }
        if Some(next_token) == eos_token_id {
            break;
        }
    }
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        token_generated as f64 / dt.as_secs_f64(),
    );
    Ok(())
}

#[derive(Deserialize, Serialize)]
struct SafetensorsIndex {
    metadata: HashMap<String, usize>,
    weight_map: HashMap<String, String>,
}

/// Loads the safetensors files for a model from the hub based on a json index file.
pub fn load_safetensors(json_file: &str) -> Result<Vec<std::path::PathBuf>> {
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| std::path::PathBuf::from(format!("model/SOLAR-10.7B-Instruct-v1.0/{}", v)))
        .collect::<Vec<_>>();
    Ok(safetensors_files)
}
