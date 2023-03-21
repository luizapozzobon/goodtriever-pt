from pathlib import Path

import fire
import pandas as pd
from datasets import load_dataset


def main(
    output_folder: str,
    use_paradetox: bool = True,
    use_rtp: bool = True,
    rtp_path: str = "data/rescored/realtoxicityprompts-data/rtp_joint_sequences_rescored.jsonl",
    threshold: float = 0.5,
):
    toxic_ds = pd.DataFrame()
    nontoxic_ds = pd.DataFrame()

    if use_paradetox:
        dataset = load_dataset("s-nlp/paradetox")
        toxic = dataset["train"]["en_toxic_comment"]
        nontoxic = dataset["train"]["en_neutral_comment"]

        toxic_ds = pd.concat([toxic_ds, pd.DataFrame(toxic, columns=["text"])])
        nontoxic_ds = pd.concat([nontoxic_ds, pd.DataFrame(nontoxic, columns=["text"])])

    if use_rtp:
        rtp = pd.read_json(rtp_path, lines=True)
        toxic = rtp[rtp["toxicity"] > threshold]
        nontoxic = rtp[rtp["toxicity"] <= threshold]

        toxic_ds = pd.concat([toxic_ds, pd.DataFrame(toxic["text"], columns=["text"])])
        nontoxic_ds = pd.concat([nontoxic_ds, pd.DataFrame(nontoxic["text"], columns=["text"])])

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    for mode, dataset in zip(["toxic", "nontoxic"], [toxic_ds, nontoxic_ds]):
        name = mode
        if use_paradetox:
            name += "_paradetox"
        if use_rtp:
            name += "_rtp"
        output_file = output_folder / f"{name}.json"

        dataset.to_json(output_file, orient="records")

    print(f"json files saved at {output_folder}")


if __name__ == "__main__":
    fire.Fire(main)
