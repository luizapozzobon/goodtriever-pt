import logging
import tempfile
from collections import defaultdict
from typing import Iterable, Optional, Tuple

import fire
import pandas as pd

from experiments.continual_learning.cl_experiment import (
    build_dstore,
    setup_output_folder,
    train_expert,
)
from experiments.continual_learning.utils import evaluate
from experiments.logger import configure_logger

logger = logging.getLogger(__name__)


def main(
    toxic_tokens: Optional[Tuple] = (10000, 50000, 75000, 100000, 150000, None),
    nontoxic_tokens: Optional[Tuple] = None,
    model_name: str = "gpt2-large",
    toxicity_choices: Tuple = ("toxic", "nontoxic"),
    prompts_path: str = "data/continual_mitigation/prompts/wilds_5_clusters_200_samples_toxic.jsonl",
    rtp_prompts_path: Optional[str] = None,
    toxic_train_file: Optional[str] = None,
    nontoxic_train_file: Optional[str] = None,
    output_folder: str = "outputs/experiments/continual_learning",
    experiment_name: str = "continual_mitigation",
    rate_limit: str = 30,
    batch_size: int = 4,
    group_toxicity_by: str = "cluster",
    filter_prompts_by_group_val: Optional[str] = None,
    kind: str = "knn",
    pretrained_toxic: Optional[str] = None,
    pretrained_nontoxic: Optional[str] = None,
    num_prompts: Optional[int] = None,
    flat_index: Optional[bool] = False,
):
    # Folder setup
    output_folder = setup_output_folder(
        output_folder,
        toxicity_choices,
        experiment_name,
        model_name,
        kind,
    )

    # Logger setup
    (output_folder / "logs").mkdir(parents=True, exist_ok=True)
    done_dstore_sizes = (
        open(output_folder / "logs/done_dstore_sizes.txt").readlines()
        if (output_folder / "logs/done_dstore_sizes.txt").exists()
        else []
    )
    configure_logger(output_folder / "logs/experiment.log")

    train_files = {"toxic": toxic_train_file, "nontoxic": nontoxic_train_file}
    pretrained = {"toxic": pretrained_toxic, "nontoxic": pretrained_nontoxic}

    if filter_prompts_by_group_val is not None:
        prompts = pd.read_json(prompts_path, lines=True)
        prompts = prompts[prompts[group_toxicity_by] == filter_prompts_by_group_val]
        temp_prompts = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=True)
        prompts.to_json(temp_prompts.name, orient="records", lines=True)
        prompts_path = temp_prompts.name

    logger.info(f"{'====' * 5}")
    logger.info(f"Starting '{output_folder}' experiment.")
    logger.info(f"{'====' * 5}")

    if not isinstance(toxic_tokens, Iterable) and isinstance(nontoxic_tokens, Iterable):
        toxic_tokens = (toxic_tokens,) * len(nontoxic_tokens)
    if not isinstance(nontoxic_tokens, Iterable) and isinstance(toxic_tokens, Iterable):
        nontoxic_tokens = (nontoxic_tokens,) * len(toxic_tokens)

    paths = defaultdict(lambda: None)
    try:
        for d, (toxic_size, nontoxic_size) in enumerate(zip(toxic_tokens, nontoxic_tokens)):
            for toxicity in toxicity_choices:
                if toxicity == "toxic":
                    dstore_size = toxic_size
                else:
                    dstore_size = nontoxic_size

                # Check which dstore_size were already added
                done = False
                if any(f"{dstore_size}, {toxicity}" in done for done in done_dstore_sizes):
                    logger.info(f"Skipped {dstore_size} , {toxicity} build or train")
                    done = True

                logger.info(
                    f"dstore_size ({d}/{len(toxic_tokens)}): {dstore_size} // Toxicity: {toxicity}"
                )

                file = train_files[toxicity]
                if file is None:
                    if pretrained[toxicity] is None:
                        raise NotImplementedError(
                            "You have to provide either a pretrained dstore/model or the train file."
                        )
                    # This will skip dstore building, but will return the pretrained path if any
                    logger.info(
                        f"No train files found for {dstore_size}-{toxicity}. Using {pretrained[toxicity]}"
                    )
                    paths[toxicity] = pretrained[toxicity]
                    continue

                # We have to run even if done=True to get the dstore/model path
                model_path = (
                    output_folder
                    / f"toxic={toxic_size}_nontoxic={nontoxic_size}/checkpoints/{toxicity}"
                )
                if kind == "knn":
                    path = build_dstore(
                        output_folder=model_path,
                        train_file=file,
                        model_name=model_name,
                        toxicity=toxicity,
                        dstore_size=dstore_size,
                        pretrained_path=pretrained[toxicity],
                        log_folder=output_folder / f"logs/build_dstore_{toxicity}.log",
                        flat_index=flat_index,
                        done=done,
                    )

                elif kind == "dexperts":
                    path = train_expert(
                        output_folder=model_path,
                        expert_name=f"finetune_{model_name}_{d}_{dstore_size}_{toxicity}",
                        train_file=file,
                        model_name=model_name,
                        epochs=1,
                        pretrained_path=pretrained[toxicity],
                        log_folder=output_folder / f"logs/train_{toxicity}.log",
                        done=done,
                    )
                else:
                    raise RuntimeError("")

                paths[toxicity] = path

                if not done:
                    with open(output_folder / "logs/done_dstore_sizes.txt", "a") as f:
                        f.write(f"{dstore_size}, {toxicity}\n")

            evaluate(
                domain=filter_prompts_by_group_val,
                output_folder=output_folder / f"toxic={toxic_size}_nontoxic={nontoxic_size}",
                model_name=model_name,
                prompts_path=prompts_path,
                rtp_prompts_path=rtp_prompts_path,
                toxic_model=paths["toxic"],
                nontoxic_model=paths["nontoxic"],
                rate_limit=rate_limit,
                group_toxicity_by=group_toxicity_by,
                num_prompts=num_prompts,
                kind=kind,
                batch_size=batch_size,
                flat_index=flat_index,
            )

    except KeyboardInterrupt:
        logger.info("Interrupted")


if __name__ == "__main__":
    fire.Fire(main)
