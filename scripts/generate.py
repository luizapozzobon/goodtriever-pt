"""Generate continuations for a given prompt and model_name.

Heavily inspired by:
https://github.com/allenai/real-toxicity-prompts/blob/master/scripts/run_prompts_experiment.py
"""
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from utils.generation import batched_generation
from utils.utils import structure_output_filepath

ALLOWED_MODELS = ["gpt2", "gpt2-medium"]


def main(
    output_folder,
    model_name: str = "gpt2",
    prompts_path: str = "gs://cohere-dev/data/realtoxicityprompts/prompts.jsonl",
    num_return_sequences: int = 25,
    max_new_tokens: int = 20,
    top_p: float = 0.90,
    batch_size: int = 16,
    use_eos: bool = False,
) -> None:
    """Generate sequences of text with HuggingFace models and output to a jsonl file.

    Filename is constructed automatically with the pattern:
        "eos_{model_name}_generations.jsonl"
    when `use_eos` and
        "prompted_{model_name}_generations.jsonl"
    when generating from prompts.

    Args:
        output_folder (str, optional): Folder to save results.
        model_name (str, optional): Model to use from HuggingFace Hub.
            Defaults to "gpt2".
        prompts_path (str, optional): Prompts path.
            Defaults to "gs://cohere-dev/data/realtoxicityprompts/prompts.jsonl".
        num_return_sequences (int, optional): Number of sequences to return for each prompt.
            Defaults to 25. If `use_eos`, hard-coded to 1.
        max_new_tokens (int, optional): Number of tokens to generate. Defaults to 20.
        top_p (float, optional): top p probability for nucleus sampling. Defaults to 0.90.
        batch_size (int, optional): Tokenization and generation batch size. Defaults to 16.
        use_eos (bool, optional): Whether to do an unprompted generation of not.
            If True, ignores `prompts_path` prompts and generates 10k sequences from
            an end of sequence token. Defaults to False.

    Raises:
        NotImplementedError: If `use_eos` is True and `model_name` does not have a
        registered EOS token for unprompted generation.

    Yields:
        np.array: Generated sequence array.
    """
    if use_eos:
        if model_name in ["gpt2", "gpt2-medium"]:
            # 10k unprompted samples
            df = np.repeat(pd.Series("<|endoftext|>"), 10_000)
        else:
            raise NotImplementedError(
                f"{model_name} is not implemented. " f"Choose one from {', '.join(ALLOWED_MODELS)}"
            )
        num_return_sequences = 1

    else:
        df = pd.read_json(prompts_path, lines=True)

    output_file = f'{"eos" if use_eos else "prompted"}_{model_name}'
    output_file = structure_output_filepath(
        step="generation", output_folder=output_folder, previous_filename=output_file
    )

    yield from batched_generation(
        output_file,
        df,
        model_name,
        batch_size=batch_size,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        use_eos=use_eos,
    )


if __name__ == "__main__":
    fire.Fire(main)
