### What this is

- These are my records of testing LLMs locally Using Rust with [Candle](https://github.com/huggingface/candle).\
  I just modified [the sample code](https://github.com/huggingface/candle/tree/main/candle-examples/examples/quantized) to fit my needs, and currently [Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) was tested.

<br>

### How to run

1. Download the model file.\
  : mistral-7b-instruct-v0.2.Q4_K_M.gguf ([TheBloke/Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main))

2. Download the tokenizer file.\
  : tokenizer.json ([mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/main))

3. Place them somewhere in root project.\
  : /model/Mistral-7B-Instruct-v0.2/

4. Run.
```bash
cargo run --bin mistral --release -- -p "chat"
```

<br>

### About the argument of "-p"

1. interactive: multiple prompts in an interactive way
2. chat: multiple prompts in an interactive way where history is preserved
3. else: a single prompt
