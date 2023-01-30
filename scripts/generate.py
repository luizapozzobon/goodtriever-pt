"""Generate continuations for a given prompt and model_name.

Heavily inspired by:
https://github.com/allenai/real-toxicity-prompts/blob/master/scripts/run_prompts_experiment.py
"""
import fire
import pandas as pd

from utils.generation import batched_generation

ALLOWED_MODELS = ["gpt2"]


def main(
    filename: str = "gs://cohere-dev/data/realtoxicityprompts/prompts.jsonl",
    model_name: str = "gpt2",
    num_return_sequences: int = 25,
    max_new_tokens: int = 20,
    batch_size: int = 16,
    out_folder: str = "./outputs/",
    use_eos: bool = False,
) -> None:
    if model_name not in ALLOWED_MODELS:
        raise NotImplementedError(
            f"{model_name} is not implemented. " f"Choose one from {', '.join(ALLOWED_MODELS)}"
        )

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
