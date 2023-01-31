# Model Safety with Retrieval-Augmented Language Models


## Setup

Run the following to create the environment. Packages will be installed as well.
```bash
conda env create -f environment.yml
conda activate model_safety
```

## Usage

### Text Generation

#### Prompted
Prompted Generation uses RealToxicityPrompts to generate continuations to each prompt. Settings are the ones from the RTP paper: 25 generations to each prompt with nucleus sampling, up to 20 tokens per continuation. Files are saved as `prompted_{model_name}_generations.jsonl`.

```bash
python -m scripts.generate \
    --filename gs://cohere-dev/data/realtoxicityprompts/prompts.jsonl \
    --model_name gpt2 \
    --num_return_sequences 25 \
    --max_new_tokens 20
```

#### Unprompted

Generates a sequence with up to 20 tokens from an EOS token. As in the paper, 10k samples are generated. Files are saved as `eos_{model_name}_generations.jsonl`

```bash
python -m scripts.generate \
    --model_name gpt2 \
    --use_eos True
```

### PerspectiveAPI Scoring

Files are saved as `{eos/prompted}_{model_name}_perspective.jsonl`.

```bash
python -m scripts.score \
    --filename outputs/prompted_gpt2_generations.jsonl \
    --perspective_rate_limit 50
```

#### API Key
For running the previous command, you need to export the `PERSPECTIVE_API_KEY` as an environment variable. They key is found in 1Password's "general" vault.

```bash
export PERSPECTIVE_API_KEY=$API_KEY
```

### Data Collation

In this step the prompts, generations and scores are collated. Files are saved as `{eos/prompted}_{model_name}_collated.jsonl`.

```bash
python -m scripts.score \
    outputs/prompted_gpt2_generations.jsonl \
    outputs/prompted_gpt2_perspective.jsonl
```

### Results

Expected Maximum Toxicity and Toxicity Probability are computed in `notebooks/02_realtoxicityprompts_results.ipynb`. That notebook was copied integrally from RTP's repo with minor modifications.
