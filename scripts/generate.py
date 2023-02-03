"""Generate continuations for a given prompt and model_name.

Heavily inspired by:
https://github.com/allenai/real-toxicity-prompts/blob/master/scripts/run_prompts_experiment.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import HfArgumentParser

from generation.args import GenerationArguments, KNNArguments
from generation.base import batched_generation
from generation.models import setup_model, setup_tokenizer
from utils.utils import load_cache, structure_output_filepath

ALLOWED_MODELS = ["gpt2", "gpt2-medium"]



def main() -> None:
    """Generate sequences of text with HuggingFace models.

    Raises:
        NotImplementedError: If `use_eos` is True and `model_name` does not have a
        registered EOS token for unprompted generation.

    Yields:
        np.array: Generated sequence array.
    """
    parser = HfArgumentParser((GenerationArguments, KNNArguments))
    gen_args, knn_args = parser.parse_args_into_dataclasses()

    if gen_args.use_eos:
        if gen_args.model_name in ["gpt2", "gpt2-medium"]:
            df = np.repeat(pd.Series("<|endoftext|>", name="text"), gen_args.eos_samples)
            df = df.to_frame().reset_index(drop=True)
        else:
            raise NotImplementedError(
                f"{gen_args.model_name} is not implemented. "
                f"Choose one from {', '.join(ALLOWED_MODELS)}"
            )
        gen_args.num_return_sequences = 1

    else:
        df = pd.read_json(gen_args.prompts_path, lines=True)
        df = pd.json_normalize(df["prompt"])

    if gen_args.out_filename is None:
        name = f'{"eos" if gen_args.use_eos else "prompted"}_{gen_args.model_name}'
        name += f'{"_knn" if knn_args.knn else ""}'
        name += f'{"_non-toxic" if knn_args.discourage_retrieved_nn else "toxic"}'
        name += '_generations.jsonl'

    output_file = structure_output_filepath(
        step="generation", output_folder=gen_args.output_folder, previous_filename=name
    )

    # Remove prompts that have already been generated
    lines = load_cache(output_file)
    df = df.iloc[lines:]
    if df.empty:
        return

    tokenizer = setup_tokenizer(gen_args.model_name)
    model = setup_model(gen_args.model_name, knn_args)


    yield from batched_generation(
        output_file,
        df,
        model,
        tokenizer,
        batch_size=gen_args.batch_size,
        num_return_sequences=gen_args.num_return_sequences,
        max_new_tokens=gen_args.max_new_tokens,
        top_p=gen_args.top_p,
        out_filename=output_file,
    )


if __name__ == "__main__":
    for _ in main():
        pass
