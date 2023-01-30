"""Score sequence of texts with PerspectiveAPI for toxicity.

Heavily inspired by:
https://github.com/allenai/real-toxicity-prompts/blob/master/scripts/run_prompts_experiment.py
"""
import sys
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from utils.perspective_api import PerspectiveWorker
from utils.utils import load_cache


def main(
    filename: str = "outputs/prompted_gpt2_generations.jsonl",
    column_name: str = "generations",
    out_folder: str = "./outputs/",
    perspective_rate_limit: str = 50
) -> None:
    """Score sequences of text with PerspectiveAPI.

    Args:
        filename (str, optional): jsonl file with generated text to be scored.
            Defaults to "outputs/prompted_gpt2_generations.jsonl".
        column_name (str, optional): Name of the field where the text sequences are.
            Defaults to "generations".
        out_folder (str, optional): Output folder. Defaults to "./outputs/".
        perspective_rate_limit (str, optional): Maximum number of API calls per minute.
            Defaults to 50.

    Raises:
        ValueError: If `filename` does not exist.
        ValueError: If `column_name` values are not lists.
    """
    filename = Path(filename)
    if not filename.exists():
        raise ValueError(f"{filename} not found.")

    df = pd.read_json(filename, lines=True)
    out_file = Path(out_folder) / f'{filename.stem.replace("generations", "perspective")}.jsonl'

    if not isinstance(df.iloc[0][column_name], list):
        raise ValueError(f"`{column_name}` should have lists as value")

    out_file.parent.mkdir(exist_ok=True, parents=True)

    perspective = PerspectiveWorker(
        out_file=out_file,
        total=df.shape[0] * len(df.iloc[0][column_name]),
        rate_limit=perspective_rate_limit
    )

    # Load cached generations
    num_cached_scores = 0
    for scores in load_cache(out_file):
        # yield scores
        num_cached_scores += 1

    # Remove prompts that have already been generated with
    df = df.iloc[num_cached_scores:]
    if df.empty:
        perspective.stop()
        print("No more samples to score.")
        sys.exit()

    # Flatten and make list
    values = np.stack(df[column_name].values).reshape(-1).tolist()

    for i, text in enumerate(values):
        perspective(f'generation-{num_cached_scores + i}', text)

    perspective.stop()


if __name__ == "__main__":
    fire.Fire(main)
