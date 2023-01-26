"""Collate generated text with its toxicity score into a jsonl file.

Heavily inspired by:
https://github.com/allenai/real-toxicity-prompts/blob/master/scripts/run_prompts_experiment.py
"""
from pathlib import Path
from typing import Any, Dict, Iterable, List

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


def collate(generations: List[str], responses: Iterable[Dict[str, Any]]):
    generations_col_iter = make_generations_col(generations, responses)
    generations_col = list(
        tqdm(generations_col_iter, total=len(generations), desc="Collating files")
    )
    dataset = pd.DataFrame(generations_col)
    return dataset


def main(generations_path: str, scores_path: str, output_folder: str = "./outputs/") -> None:
    generations = pd.read_json(generations_path, lines=True)
    scores = pd.read_json(scores_path, lines=True)

    output_file = (
        Path(output_folder) / f"{Path(generations_path).stem.split('_')[0]}_collated.jsonl"
    )
    output_file.parent.mkdir(exist_ok=True, parents=True)

    # Generate indexes based on original prompts
    gen_list = np.stack(generations["continuations"])
    prompt_indexes = np.repeat(generations.index.values, gen_list.shape[-1])

    # Flatten stacked generations to ease collate
    gen_list = gen_list.reshape(-1).tolist()
    scores_list = scores["response"].tolist()
    assert len(gen_list) == len(scores_list)

    dataset = collate(gen_list, scores_list, output_file)
    dataset["prompt"] = prompt_indexes
    dataset.to_json(output_file, orient="records", lines=True)


if __name__ == "__main__":
    fire.Fire(main)
