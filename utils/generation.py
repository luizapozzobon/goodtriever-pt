import json
from pathlib import Path
from typing import Callable, Generator, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_cache(file: Path):
    if file.exists():
        with file.open() as f:
            for line in tqdm(f, desc=f'Loading cache from {file}'):
                yield json.loads(line)

def generate(
    text: List[str],
    model: Callable,
    tokenizer: Callable,
    max_new_tokens: int,
    num_return_sequences: int
) -> np.array:
    """Generate sequences given a prompt.

    Args:
        text (List[str]): Batch of prompts.
        model (Callable): HuggingFace model instance.
        tokenizer (Callable): HuggingFace tokenizer instance.
        max_new_tokens (int): Number of tokens to generate.
        num_return_sequences (int): Number of sequences to generate for each prompt.

    Returns:
        np.array: Prompt continuations
    """
    # Batched tokenization and generation
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=False
    ).to(model.device)

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
    prompts: pd.DataFrame,
    model_name: str,
    batch_size: int,
    num_return_sequences: int,
    max_new_tokens: int,
    out_folder: str,
    use_eos: bool = False
) -> Generator:
    """https://github.com/allenai/real-toxicity-prompts/blob/master/generation/generation.py#L61"""

    out_file = Path(out_folder) / f'{"eos" if use_eos else "prompted"}_{model_name}_generations.jsonl'
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load cached generations
    num_cached_generations = 0
    for generation in load_cache(out_file):
        yield generation
        num_cached_generations += 1

    # Remove prompts that have already been generated with
    prompts = prompts.iloc[num_cached_generations:]
    if prompts.empty:
        return

    # Setup model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    chunks = len(prompts) // batch_size
    print(f"Iterating on {chunks} chunks...")
    for chunk in tqdm(np.array_split(prompts, chunks), total=chunks):
        if not use_eos:
            chunk = pd.json_normalize(chunk['prompt'])['text']

        chunk = chunk.values.tolist()

        continuations = generate(
            chunk,
            model,
            tokenizer,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
        )
        data = [
            {"prompt": p, "generations": c}
            for p, c in zip(prompts, continuations)
        ]
        for d in data:
            with out_file.open('a') as f:
                print(json.dumps(d), file=f)
            yield d
