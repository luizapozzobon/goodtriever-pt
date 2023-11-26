import warnings
from pathlib import Path
from typing import Optional

import fire
import pandas as pd

from scripts.utils.evaluation_metrics import (
    compute_distinctness,
    compute_ppl,
    compute_toxicity_prompted,
    compute_toxicity_unprompted,
)
from utils.utils import structure_output_filepath


def main(
    unprompted_json: Optional[str] = None,
    prompted_json: Optional[str] = None,
    compute_perplexity: bool = True,
    compute_toxicity: bool = True,
    compute_diversity: bool = True,
    model_name: str = "gpt2-xl",
    sample_perplexity: Optional[int] = 1000,
    stride: int = 512,
    threshold: float = 0.5,
    group_toxicity_by: Optional[str] = None,
):
    """Compute toxicity and perplexity metrics for prompted or unprompted generations.

    The inputs files for this script are the ones from the "collate.py" script.

    Args:
        unprompted_json (Optional[str], optional): Path to unprompted generations jsonl file.
            If None, this will be skipped. Defaults to None.
        prompted_json (Optional[str], optional): Path to prompted generations jsonl file.
            If None, this will be skipped. Defaults to None.
        compute_perplexity (bool, optional): Whether to compute perplexity or not.
            Defaults to True.
        compute_toxicity (bool, optional): Whether to compute toxicity metrics or not.
            Defaults to True.
        compute_toxicity (bool, optional): Whether to compute diversity metrics or not.
            Defaults to True.
        model_name (str, optional): Which model to compute perplexity with.
            Defaults to "gpt2-xl".
        sample_perplexity (int, optional): The amount of prompt samples to
            from each toxicity condition to compute perplexity.
            If None, computes for all samples.
            Defaults to None.
        stride (int, optional): Stride to compute perplexity. It usually is the model's
            maximum sequence length of a model.
            Defaults to 512. More details on:
            https://www.reddit.com/r/MachineLearning/comments/oye64h/r_struggling_to_reproduce_perplexity_benchmarks/
        threshold (float, optional): Toxicity threshold.
            If higher than `threshold`, the text is toxic. Defaults to 0.5.
        group_toxicity_by (str, optional): Column to group toxicity results by
            (i.e. a column containing different classes of interest). Only
            possible for prompted generation. Classes should be present in the
            `prompted_json` file. Defaults to None.

    """
    for path, prompted in zip([unprompted_json, prompted_json], [False, True]):
        if path is not None:
            path = Path(path)
            df = pd.read_json(path, lines=True)

            if compute_toxicity:
                output_file = structure_output_filepath(
                    step="toxicity",
                    previous_filename=path,
                )
                if not output_file.exists():
                    if prompted:
                        compute_toxicity_prompted(
                            df,
                            output_file,
                            threshold=threshold,
                            group_toxicity_by=group_toxicity_by,
                        )
                    else:
                        compute_toxicity_unprompted(
                            df, output_file, threshold=threshold
                        )
                else:
                    warnings.warn(f"{output_file} already exists. Skipping.")

            if compute_perplexity:
                output_file = structure_output_filepath(
                    step="perplexity",
                    previous_filename=path,
                )
                compute_ppl(
                    df,
                    model_name,
                    output_file,
                    prompted=prompted,
                    sample_perplexity=sample_perplexity,
                    stride=stride,
                    threshold=threshold,
                )

            if compute_diversity:
                output_file = structure_output_filepath(
                    step="diversity",
                    previous_filename=path,
                )
                compute_distinctness(df, output_file)


if __name__ == "__main__":
    fire.Fire(main)
