"""Generate continuations for a given prompt and model_name.

Heavily inspired by:
https://github.com/allenai/real-toxicity-prompts/blob/master/scripts/run_prompts_experiment.py
"""
import fire
import numpy as np
import pandas as pd

from utils.generation import batched_generation

ALLOWED_MODELS = ["gpt2", "gpt2-medium"]

def main(
    filename: str = "gs://cohere-dev/data/realtoxicityprompts/prompts.jsonl",
    model_name: str = "gpt2",
    num_return_sequences: int = 25,
    max_new_tokens: int = 20,
    batch_size: int = 16,
    out_folder: str = "./outputs/",
    use_eos: bool = False,
) -> None:
    """Generate sequences of text with HuggingFace models.

    Args:
        filename (str, optional): Prompts filename.
            Defaults to "gs://cohere-dev/data/realtoxicityprompts/prompts.jsonl".
        model_name (str, optional): Model to use from HuggingFace Hub.
            Defaults to "gpt2".
        num_return_sequences (int, optional): Number of sequences to return for each prompt.
            Defaults to 25. If `use_eos`, hard-coded to 1.
        max_new_tokens (int, optional): Number of tokens to generate. Defaults to 20.
        batch_size (int, optional): Tokenization and generation batch size. Defaults to 16.
        out_folder (str, optional): Folder to save results. Defaults to "./outputs/".
        use_eos (bool, optional): Whether to do an unprompted generation of not.
            If True, ignores `filename` prompts and generates 10k sequences from
            an end of sequence token. Defaults to False.

    Raises:
        NotImplementedError: If `use_eos` is True and `model_name` does not have a
        registered EOS token for unprompted generation.

    Yields:
        np.array: Generated sequence array.
    """
    if use_eos:
        if model_name in ['gpt2', 'gpt2-medium']:
            # 10k unprompted samples
            df = np.repeat(pd.Series('<|endoftext|>'), 10_000)
        else:
            raise NotImplementedError(
                f"{model_name} is not implemented. " f"Choose one from {', '.join(ALLOWED_MODELS)}"
            )
        num_return_sequences = 1

    else:
        df = pd.read_json(filename, lines=True)

    yield from batched_generation(
        df,
        model_name,
        batch_size=batch_size,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        out_folder=out_folder,
        use_eos=use_eos,
    )


if __name__ == "__main__":
    fire.Fire(main)
