from transformers import GPT2Tokenizer, GPT2LMHeadModel

import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Generator
import json
from pathlib import Path


def load_cache(file: Path):
    if file.exists():
        with file.open() as f:
            for line in tqdm(f, desc=f'Loading cache from {file}'):
                yield json.loads(line)

def generate(text: List[str], model, tokenizer, max_new_tokens: int, num_return_sequences: int):
    # Batched tokenization and generation
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=False
    )
    ## Nucleous sampling
    outputs = model.generate(
        **inputs,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        top_p=0.95,
        top_k=0,

    )
    continuations = tokenizer.batch_decode(
        outputs[:, inputs['input_ids'].shape[-1]:],
        clean_up_tokenization_spaces=True,
        skip_special_tokens=True,
    )
    # Group generations from same prompt
    continuations = np.array(continuations).reshape(
        (-1, num_return_sequences)).tolist()

    return continuations


def batched_generation(
    df: pd.DataFrame,
    model_name: str,
    batch_size: int,
    num_return_sequences: int,
    max_new_tokens: int,
    out_folder: str
) -> Generator:
    """https://github.com/allenai/real-toxicity-prompts/blob/master/generation/generation.py#L61"""

    out_file = Path(out_folder) / f"{model_name}_continuations.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load cached generations
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1

    # Remove prompts that have already been generated with
    df = df.iloc[num_cached_generations:]
    if df.empty:
        return

    # Setup model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    chunks = df.shape[0] // batch_size
    print(f"Iterating on {chunks} chunks...")
    for chunk in tqdm(np.array_split(df, chunks), total=chunks):
        prompts = pd.json_normalize(chunk['prompt'])['text'].values.tolist()
        continuations = generate(
            prompts,
            model,
            tokenizer,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
        )
        data = [
            {"prompt": p, "continuations": c}
            for p, c in zip(prompts, continuations)
        ]
        for d in data:
            with out_file.open('a') as f:
                print(json.dumps(d), file=f)
            yield d
