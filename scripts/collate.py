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


def make_generations_col(generations: List[str], responses: List[Dict]):
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
    output_folder: str = "./outputs/",
    chunksize: int = int(5e5),
) -> None:
    """Collate sequences with its PerspectiveAPI toxicity scores.

    Args:
        generations_path (str): Path to generations file.
        scores_path (str): Path to scores file.
        prompts_path (str, optional): Path to prompts file.
            Defaults to "gs://cohere-dev/data/realtoxicityprompts/prompts.jsonl".
        output_folder (str, optional): Output folder. Defaults to "./outputs/".
    """
    generations = pd.read_json(generations_path, lines=True)

    gen_list = np.stack(generations["generations"])
    # Generate indexes based on original prompts
    prompt_indexes = (
        np.repeat(generations.index.values, gen_list.shape[-1]) if gen_list.shape[-1] > 1 else None
    )

    # Flatten stacked generations to ease collate
    gen_list = gen_list.reshape(-1).tolist()

    scores = pd.read_json(scores_path, lines=True, chunksize=chunksize)
    scored_gens = pd.Series(dtype="object", name="generations")
    for i, chunk in enumerate(tqdm(scores, desc="Processing chunks", position=0)):
        start = chunksize * i
        end = start + chunksize

        scores_list = chunk["response"].tolist()
        # Collate generations and scores into a list of dicts
        scored_gens = pd.concat(
            [
                scored_gens,
                collate(gen_list[start:end], scores_list, prompt_indexes[start:end]),
            ],
            axis=0,
        )

    if len(scored_gens) != len(generations):
        warnings.warn(
            f"Length of scored data is {len(scored_gens)}, but was expecting {len(generations)}"
        )

    output_file = (
        Path(output_folder)
        / f'{Path(generations_path).stem.replace("generations", "collated")}.jsonl'
    )
    output_file.parent.mkdir(exist_ok=True, parents=True)

    if prompt_indexes is not None:
        prompts = pd.read_json(prompts_path, lines=True)
        prompts = pd.merge(prompts, scored_gens.to_frame(), left_index=True, right_index=True)
        prompts.to_json(output_file, orient="records", lines=True)
    else:
        scored_gens.to_json(output_file, orient="records", lines=True)


if __name__ == "__main__":
    fire.Fire(main)
