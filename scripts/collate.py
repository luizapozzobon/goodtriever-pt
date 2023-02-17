"""Collate generated text with its toxicity score into a jsonl file.

Heavily inspired by:
https://github.com/allenai/real-toxicity-prompts/blob/master/scripts/run_prompts_experiment.py
"""
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.constants import PERSPECTIVE_API_ATTRIBUTES_LOWER
from utils.perspective_api import unpack_scores
from utils.utils import structure_output_filepath


def make_generations_col(generations: List[str], responses: List[Dict]) -> Dict:
    """Join generation text to its results from PerpectiveAPI scores into a dict."""
    for generation, response in zip(generations, responses):
        if isinstance(response, dict):
            response = unpack_scores(response)[0]
        else:
            response = {x: None for x in PERSPECTIVE_API_ATTRIBUTES_LOWER}
        yield {"text": generation, **response}


def collate(
    generations: List[str],
    responses: Iterable[Dict[str, Any]],
    prompt_indexes: Optional[List[int]],
) -> pd.Series:
    """Collate generations texts with their scores by perspective API."""
    generations_col_iter = make_generations_col(generations, responses)
    generations_col = list(
        tqdm(generations_col_iter, total=len(generations), desc="Collating files", position=1)
    )
    dataset = pd.DataFrame(generations_col)

    # Annotate to which prompt each generation belongs to then groupby to form a list
    if prompt_indexes is not None:
        dataset["prompt"] = prompt_indexes
        dataset = dataset.groupby("prompt").apply(lambda x: x.to_dict(orient="records"))

    return dataset


def main(
    generations_path: str,
    scores_path: str,
    prompts_path: str = "gs://cohere-dev/data/realtoxicityprompts/prompts.jsonl",
    output_folder: Optional[str] = None,
    chunksize: int = int(1e5),
) -> None:
    """Collate generations with its PerspectiveAPI toxicity scores and pre-scored prompts.

    `prompts_path` points to a file that contains a `prompt` column with dict values.
    These dictionaries are pre-scored prompts by PerspectiveAPI and their text.

    Args:
        generations_path (str): Path to generations file.
        scores_path (str): Path to scores file.
        prompts_path (str, optional): Path to prompts file.
            Defaults to "gs://cohere-dev/data/realtoxicityprompts/prompts.jsonl".
        output_folder (str, optional): Output folder. If None, file will be saved to
            `scores_path` folder. Defaults to None.
        chunksize (int): Chunksize to split large scores files by when loading
            with pandas. Default value chosen as a reasonable number that usually
            fits memory. Defaults to 100_000.
    """
    generations = pd.read_json(generations_path, lines=True)

    gen_list = np.stack(generations["generations"])
    # Generate indexes based on original prompts
    prompt_indexes = (
        np.repeat(generations.index.values, gen_list.shape[-1]) if gen_list.shape[-1] > 1 else None
    )

    # Flatten stacked generations to ease collate
    gen_list = gen_list.reshape(-1).tolist()

    scores_path = Path(scores_path)
    scores = pd.read_json(scores_path, lines=True, chunksize=chunksize)
    scored_gens = pd.Series(dtype="object", name="generations")
    for i, chunk in enumerate(tqdm(scores, desc="Processing chunks", position=0)):
        start = chunksize * i
        end = start + chunksize
        indexes = prompt_indexes[start:end] if prompt_indexes is not None else None

        scores_list = chunk["response"].tolist()
        # Collate generations and scores into a list of dicts
        scored_gens = pd.concat(
            [
                scored_gens,
                collate(gen_list[start:end], scores_list, indexes),
            ],
            axis=0,
        )

    if len(scored_gens) != len(generations):
        warnings.warn(
            f"Length of scored data is {len(scored_gens)}, but was expecting {len(generations)}"
        )

    output_file = structure_output_filepath(
        step="collate",
        output_folder=output_folder or scores_path.parent,
        previous_filename=scores_path.name,
    )

    if prompt_indexes is not None:
        prompts = pd.read_json(prompts_path, lines=True)
        prompts = pd.merge(prompts, scored_gens.to_frame(), left_index=True, right_index=True)
        prompts.to_json(output_file, orient="records", lines=True)
    else:
        scored_gens.to_json(output_file, orient="records", lines=True)


if __name__ == "__main__":
    fire.Fire(main)
