import time
from pathlib import Path

import fire

from generation.args import GenerationParser
from scripts.collate import main as collate
from scripts.evaluate import main as evaluate
from scripts.generate import main as generate
from scripts.score import main as score


def main(
    perspective_rate_limit: int = 110,
    perplexity_model: str = "gpt2-medium",
    collate_chunksize: int = int(1e5),
    sample_perplexity: int = 1000,
) -> None:
    """Run full pipeline: generate, score, collate and evaluate.

    All generation arguments are supported. For more info check generate.py
    and generation/args.py file.

    Args:
        perspective_rate_limit (int, optional): Maximum number of PerspectiveAPI
            calls per second. Defaults to 110.
        perplexity_model (str, optional): Model to compute perplexity with.
            Defaults to "gpt2-medium".
        collate_chunksize (int, optional): Used in the collate script.
            Chunksize to split large files when loading with pandas.
            Default value chosen as a reasonable number that usually
            fits memory. Defaults to 100_000.
        sample_perplexity (int, optional): Used in the evaluate script.
            Number of prompts to compute perplexity for.
            Defaults to 1000.

    Raises:
        ValueError: Raised if input file for evaluation does not contain
        the "eos" or "prompted" strings.

    """
    parser = GenerationParser()

    # Generate
    for _ in generate(parser):
        pass

    output_folder = Path(parser.gen_args.output_folder)
    generations_path = str(output_folder / parser.gen_args.output_filename)

    time.sleep(2)

    scores_path = score(
        input_filename=generations_path,
        output_folder=output_folder,
        perspective_rate_limit=perspective_rate_limit,
    )

    collated_path = collate(
        generations_path=generations_path,
        scores_path=scores_path,
        output_folder=output_folder,
        chunksize=collate_chunksize,
        prompts_path=parser.gen_args.prompts_path,
    )

    if "eos_" in collated_path.name:
        unprompted_json = collated_path
        prompted_json = None
    elif "prompted_" in collated_path.name:
        unprompted_json = None
        prompted_json = collated_path
    else:
        raise ValueError(
            f"Invalid filename for {collated_path}. No 'eos_' or 'prompted_' strings found."
        )

    evaluate(
        unprompted_json=unprompted_json,
        prompted_json=prompted_json,
        compute_perplexity=True,
        compute_toxicity=True,
        model_id=perplexity_model,
        sample_perplexity=sample_perplexity,
    )


if __name__ == "__main__":
    fire.Fire(main)
