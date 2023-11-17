"""Score sequence of texts with PerspectiveAPI for toxicity.

Heavily inspired by:
https://github.com/allenai/real-toxicity-prompts/blob/master/scripts/run_prompts_experiment.py
"""
import logging
from pathlib import Path
from typing import List, Optional

import fire
import numpy as np
import pandas as pd

from utils.chatgpt import ChatGPTToxicityScorer
from utils.constants import PERSPECTIVE_API_KEY
from utils.perspective_api import PerspectiveWorker
from utils.utils import load_cache, structure_output_filepath


def main(
    input_filename: str,
    column_name: str = "generations",
    output_folder: Optional[str] = None,
    rate_limit: int = 1,
    custom_attrs: Optional[List[str]] = None,
    api: str = "perspective",
    language: str = "pt",
) -> None:
    """Score sequences of text with PerspectiveAPI.

    Args:
        input_filename (str, optional): jsonl file with generated text to be scored.
            Should be stored locally.
        column_name (str, optional): Name of the field where the text sequences are.
            Supports any dict or list of dict column as long as the dict contains a
            `text` keyword.
            Defaults to "generations".
        output_folder (str, optional): Output folder. If None, results will be saved
            to the same folder as `input_filename`. Defaults to None.
        rate_limit (int, optional): Maximum number of API calls per second.
            Defaults to 1.
        custom_attrs (list, optional): Custom attributes to request PAPI.
            If None, all will be requested. Defaults to None.
        api (str, optional): Which API to use between 'perspective' and 'chatgpt'.
            Defaults to "perspective".

    Raises:
        NotImplementedError: If `column_name` values are not lists or dicts or don't
            have a 'text' key.
    """

    if api not in ["perspective", "chatgpt"]:
        raise ValueError(f"API {api} not supported. Options: 'perspective', 'chatgpt'.")

    if api == "perspective":
        if PERSPECTIVE_API_KEY is None:
            raise ValueError(
                "Please run `export PERSPECTIVE_API_KEY='key'` if you wish to use PerspectiveAPI."
            )

    input_filename = Path(input_filename)
    if not input_filename.exists():
        raise ValueError(f"{input_filename} not found.")

    output_file = structure_output_filepath(
        step=api,
        output_folder=output_folder or input_filename.parent,
        previous_filename=input_filename,
    )

    df = pd.read_json(input_filename, lines=True)

    if isinstance(df.iloc[0][column_name], dict):
        df[column_name] = df[column_name].apply(lambda x: [x.get("text")])
    elif isinstance(df.iloc[0][column_name], list):
        df[column_name] = df[column_name].apply(
            lambda y: [x.get("text") if isinstance(x, dict) else x for x in y]
        )
    elif isinstance(df.iloc[0][column_name], str):
        df[column_name] = df[column_name].apply(lambda x: [x])
    else:
        raise NotImplementedError(
            "If dict or list of dicts, make sure there's a `text` key."
        )

    num_samples = len(df.iloc[0][column_name])

    if api == "perspective":
        worker = PerspectiveWorker(
            out_file=output_file,
            total=df.shape[0] * num_samples,
            rate_limit=rate_limit,
            custom_attrs=custom_attrs,
        )
    else:
        logging.info(f"Starting ChatGPT toxicity scorer for language {language}.")
        worker = ChatGPTToxicityScorer(output_file=output_file, language=language)

    # Flatten and make list
    values = np.stack(df[column_name].values).reshape(-1).tolist()
    del df

    num_cached_scores = load_cache(output_file)
    values = values[num_cached_scores:]

    if len(values) == 0:
        print("No more samples to score.")
        if api == "perspective":
            worker.stop()
        return output_file

    if api == "perspective":
        for i, text in enumerate(values):
            worker(f"generation-{num_cached_scores + i}", text)
        worker.stop()
    else:
        worker.current_request_id = num_cached_scores
        for response_dict in worker.run_parallel_requests(
            texts=values, num_workers=rate_limit
        ):
            pass

    return output_file


if __name__ == "__main__":
    fire.Fire(main)
