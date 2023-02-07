from pathlib import Path

import fire
import pandas as pd
from datasets import load_dataset


def main(output_folder: str):
    dataset = load_dataset("s-nlp/paradetox")

    toxic_col = "en_toxic_comment"
    neutral_col = "en_neutral_comment"

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    for col in [toxic_col, neutral_col]:
        ds = dataset["train"][col]

        output_file = output_folder / f"paradetox_{col}.json"
        pd.DataFrame(ds, columns=["text"]).to_json(output_file, orient="records")

    print(f"json files saved at {output_folder}")


if __name__ == "__main__":
    fire.Fire(main)
