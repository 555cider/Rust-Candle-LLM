use anyhow::{bail, Error, Result};
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::{generation::LogitsProcessor, models::quantized_llama, utils};
use clap::Parser;
use log::{debug, info};
use std::{
    fs::File,
    io::{stdin, stdout, Write},
    path::PathBuf,
    time::Instant,
};
use tokenizers::Tokenizer;
use yobara::token_output_stream::TokenOutputStream;

fn main() -> Result<()> {
    log4rs::init_file("log4rs.yaml", Default::default()).unwrap();
    info!("Initialized the logger");

    let args = Args::parse();
    info!(
        "temperature: {:.2?}, repeat-penalty: {:.2}, repeat-last-n: {}",
        args.temperature, args.repeat_penalty, args.repeat_last_n
    );

    let stop_watch = Instant::now();
    let model_path = PathBuf::from(args.model);
    let mut model_reader = File::open(&model_path)?;
    let model_content =
        gguf_file::Content::read(&mut model_reader).map_err(|e| e.with_path(model_path))?;
    let model_weights = quantized_llama::ModelWeights::from_gguf(model_content, &mut model_reader)?;
    info!("Loaded the model weights in {:?}", stop_watch.elapsed());

    let stop_watch = Instant::now();
    let tokenizer_path = PathBuf::from(args.tokenizer);
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(Error::msg)?;
    info!("Loaded the tokenizer in {:?}", stop_watch.elapsed());

    let logits_processor = LogitsProcessor::new(args.seed, args.temperature, args.top_p);
    let mut text_generation = TextGeneration::new(
        model_weights,
        tokenizer,
        logits_processor,
        args.repeat_penalty,
        args.repeat_last_n,
    );
    text_generation.run(args.prompt, args.sample_len.saturating_sub(1))?;

    Ok(())
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// 1. 'interactive': multiple prompts in an interactive way
    /// 2. 'chat': multiple prompts in an interactive way where history of previous prompts and generated tokens is preserved
    /// 3. else: a single prompt
    #[arg(long, short = 'p', default_value = "interactive")]
    prompt: String,

    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 100)]
    sample_len: usize,

    /// the path of model file(in gguf format)
    #[arg(
        long,
        short = 'm',
        default_value = "model/Mistral-7B-Instruct-v0.2/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    )]
    model: String,

    /// the path of tokenizer file(in json format)
    #[arg(
        long,
        short = 't',
        default_value = "model/Mistral-7B-Instruct-v0.2/tokenizer.json"
    )]
    tokenizer: String,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The temperature used to generate samples, use 0 for greedy sampling. (default was 0.8)
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

#[derive(Debug)]
enum Prompt {
    Interactive,
    Chat,
    One(String),
}

pub struct TextGeneration {
    model_weights: quantized_llama::ModelWeights,
    token_output_stream: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_weights: quantized_llama::ModelWeights,
        tokenizer: Tokenizer,
        logits_processor: LogitsProcessor,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Self {
        Self {
            token_output_stream: TokenOutputStream::new(tokenizer),
            model_weights,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
        }
    }

    pub fn run(&mut self, prompt: String, sample_len: usize) -> Result<()> {
        let prompt = match prompt.as_str() {
            "interactive" => Prompt::Interactive,
            "chat" => Prompt::Chat,
            s => Prompt::One(s.to_string()),
        };

        let mut pre_tokens = vec![];
        for _input_index in 0.. {
            // 1. Input and tokenization section
            let stop_watch = Instant::now();
            let prompt_formatted = match &prompt {
                Prompt::One(prompt) => prompt.clone(),
                Prompt::Interactive | Prompt::Chat => {
                    print!("> ");
                    stdout().flush()?;
                    let mut prompt = String::new();
                    stdin().read_line(&mut prompt)?;
                    if prompt.ends_with('\n') {
                        prompt.pop();
                        if prompt.ends_with('\r') {
                            prompt.pop();
                        }
                    }
                    format!("[INST] {prompt} [/INST]")
                }
            };
            let cur_input_tokens = self
                .token_output_stream
                .tokenizer()
                .encode(prompt_formatted, true)
                .map_err(Error::msg)?;

            // If the number of tokens exceeds the maximum, the oldest tokens are removed.
            let all_tokens = [&pre_tokens, cur_input_tokens.get_ids()].concat();
            let all_tokens = if all_tokens.len() + sample_len > quantized_llama::MAX_SEQ_LEN - 10 {
                let old_cnt = (all_tokens.len() + sample_len) - (quantized_llama::MAX_SEQ_LEN - 10);
                all_tokens[all_tokens.len().saturating_sub(old_cnt)..].to_vec()
            } else {
                all_tokens
            };

            debug!(
                "{:4} prompt tokens processed: {:.2} token/s",
                all_tokens.len(),
                all_tokens.len() as f64 / stop_watch.elapsed().as_secs_f64(),
            );

            // 2. Processing and output section
            let stop_watch = Instant::now();
            let eos_token = match self
                .token_output_stream
                .tokenizer()
                .get_vocab(true)
                .get("</s>")
            {
                Some(token) => token.to_owned(),
                None => bail!("cannot find the </s> token"),
            };

            let tensor = Tensor::new(all_tokens.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
            let logits = self.model_weights.forward(&tensor, 0)?.squeeze(0)?;
            let mut next_token = self.logits_processor.sample(&logits)?;
            let mut output_tokens = vec![];
            output_tokens.push(next_token);
            if let Some(t) = self.token_output_stream.next_token(next_token)? {
                print!("{t}");
                stdout().flush()?;
            }

            let mut sampled_cnt = 0;
            for index in 0..sample_len {
                let tensor = Tensor::new(&[next_token], &Device::Cpu)?.unsqueeze(0)?;
                let logits = self
                    .model_weights
                    .forward(&tensor, all_tokens.len() + index)?
                    .squeeze(0)?;
                let logits = if self.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = output_tokens.len().saturating_sub(self.repeat_last_n);
                    utils::apply_repeat_penalty(
                        &logits,
                        self.repeat_penalty,
                        &output_tokens[start_at..],
                    )?
                };
                next_token = self.logits_processor.sample(&logits)?;
                output_tokens.push(next_token);
                if let Some(t) = self.token_output_stream.next_token(next_token)? {
                    print!("{t}");
                    stdout().flush()?;
                }
                sampled_cnt += 1;
                if next_token == eos_token {
                    break;
                };
            }
            if let Some(rest) = self.token_output_stream.decode_rest().map_err(Error::msg)? {
                print!("{rest}");
            }
            stdout().flush()?;

            debug!(
                "{:4} tokens generated ({:.2} token/s)",
                sampled_cnt,
                sampled_cnt as f64 / stop_watch.elapsed().as_secs_f64(),
            );

            match prompt {
                Prompt::One(_) => break,
                Prompt::Interactive => {}
                Prompt::Chat => {
                    pre_tokens = [all_tokens.as_slice(), output_tokens.as_slice()].concat()
                }
            }
            self.token_output_stream.clear();
            println!();
        }
        Ok(())
    }
}
