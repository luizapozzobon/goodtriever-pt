from pathlib import Path

import fire

from generation.args import GenerationParser
from scripts.collate import main as collate
from scripts.evaluate import main as evaluate
from scripts.generate import main as generate
from scripts.score import main as score


def main(perspective_rate_limit=110, perplexity_model="gpt2-medium", collate_chunksize: int = 1e5):

    parser = GenerationParser()

    # Generate
    for _ in generate(parser):
        pass

    out_folder = parser.gen_args.out_folder
    generations_path = str(Path(out_folder) / parser.gen_args.out_filename)

    scores_path = score(
        filename=generations_path,
        out_folder=out_folder,
        perspective_rate_limit=perspective_rate_limit,
    )

    collated_path = collate(
        generations_path=generations_path,
        scores_path=scores_path,
        out_folder=out_folder,
        chunksize=collate_chunksize,
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
    )


if __name__ == "__main__":
    fire.Fire(main)
