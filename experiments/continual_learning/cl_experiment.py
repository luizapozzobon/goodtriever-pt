import logging
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import fire
import pandas as pd

from experiments.continual_learning.utils import (
    evaluate,
    extract_domains_from_file_list,
    run,
)
from experiments.logger import configure_logger

logger = logging.getLogger(__name__)


def build_dstore(
    output_folder,
    train_file,
    model_name,
    toxicity,
    dstore_size=None,
    pretrained_path=None,
    done=False,
    log_folder="logs",
):
    output_folder.mkdir(exist_ok=True, parents=True)

    if done:
        return output_folder

    if pretrained_path is not None:
        logger.info(f"Copying base {toxicity} dstore from {pretrained_path} to {output_folder}.")
        subprocess.run(f"cp -r {pretrained_path}/* {output_folder}/", shell=True)
        logger.info(f"Base {toxicity} dstore copied.")

    ds_cmd = f"""
        python -u -m knn_transformers.run_clm \
            --model_name_or_path {model_name} \
            --train_file {train_file} \
            --eval_subset train \
            --output_dir {output_folder} \
            --dstore_dir {output_folder} \
            --save_knnlm_dstore \
            --continue_writing \
            {f'--dstore_size {dstore_size} --limit_eval_to_dstore' if dstore_size else ''} \
            --do_eval | tee -a {log_folder}
    """

    train_cmd = f"""
        python -u -m knn_transformers.run_clm \
            --model_name_or_path {model_name} \
            --train_file {train_file} \
            --eval_subset train \
            --output_dir {output_folder} \
            --dstore_dir {output_folder} \
            {f'--dstore_size {dstore_size}' if dstore_size else ''} \
            --build_index | tee -a {log_folder}
    """

    logger.info(f"Running `datastore build` command: {ds_cmd}")
    run(ds_cmd)

    logger.info(f"Running `index train` command: {train_cmd}")
    run(train_cmd)

    return output_folder


def train_expert(
    output_folder,
    expert_name,
    train_file,
    model_name,
    pretrained_path=None,
    epochs=1,
    block_size=128,
    batch_size=4,
    grad_accum=16,
    log_folder="logs/",
    done=False,
):
    expert_path = output_folder / model_name / expert_name

    if done:
        return expert_path

    if pretrained_path is not None:
        model_name = pretrained_path

    train_cmd = f"""
        python -m scripts.finetuning.finetune_gpt2 \
            --output_dir {expert_path} \
            --model_type gpt2 \
            --model_name_or_path {model_name} \
            --do_train \
            --num_train_epochs {epochs} \
            --block_size {block_size} \
            --save_total_limit 1 \
            --dataloader_drop_last \
            --per_device_train_batch_size {batch_size} \
            --gradient_accumulation_steps {grad_accum} \
            --train_data_file {train_file} \
            --overwrite_cache | tee -a {log_folder}
    """
    # TODO check log with tee

    run(train_cmd)

    return expert_path


def setup_output_folder(output_folder, toxicity_choices, experiment_name, model_name, kind):
    if not all(choice in ["toxic", "nontoxic"] for choice in toxicity_choices):
        raise ValueError("`toxicity_choices` should contain only 'toxic' or 'nontoxic'.")

    # Toxicity and dstores setup
    toxic_added = True if "toxic" in toxicity_choices else False
    nontoxic_added = True if "nontoxic" in toxicity_choices else False

    # Folder setup
    output_folder = (
        Path(output_folder)
        / experiment_name
        / model_name
        / kind
        / f"toxic={toxic_added}_nontoxic={nontoxic_added}"
    )

    return output_folder


def main(
    domains: Optional[Tuple] = None,
    model_name: str = "gpt2-large",
    toxicity_choices: Tuple = ("toxic", "nontoxic"),
    prompts_path: str = "data/continual_mitigation/prompts/wilds_single_identity_4k_nontoxic_prompts.jsonl",
    rtp_prompts_path: str = "data/dexperts/prompts/nontoxic_prompts-10k.jsonl",
    train_folder: str = "data/continual_mitigation/domains/train",
    output_folder: str = "outputs/experiments/continual_learning",
    experiment_name: str = "continual_mitigation",
    rate_limit: str = 90,
    group_toxicity_by: str = "domain",
    kind: str = "knn",
    pretrained_toxic: Optional[str] = None,
    pretrained_nontoxic: Optional[str] = None,
    dstore_size: Optional[int] = None,
    num_prompts: Optional[int] = None,
    multitask: Optional[bool] = False,
):
    # Folder setup
    output_folder = setup_output_folder(
        output_folder,
        toxicity_choices,
        experiment_name,
        model_name,
        kind,
    )
    if multitask:
        multitask_files = {
            "toxic": tempfile.NamedTemporaryFile(suffix=".json", mode="w+"),
            "nontoxic": tempfile.NamedTemporaryFile(suffix=".json", mode="w+"),
        }

    # Logger setup
    (output_folder / "logs").mkdir(parents=True, exist_ok=True)
    done_domains = (
        open(output_folder / "logs/done_domains.txt").readlines()
        if (output_folder / "logs/done_domains.txt").exists()
        else []
    )
    configure_logger(output_folder / "logs/experiment.log")

    # Domains setup
    train_folder = Path(train_folder)
    files = {
        "toxic": sorted(list(train_folder.glob("wilds_*_toxic.json"))),
        "nontoxic": sorted(list(train_folder.glob("wilds_*_nontoxic.json"))),
    }
    pretrained = {"toxic": pretrained_toxic, "nontoxic": pretrained_nontoxic}
    domains = domains or extract_domains_from_file_list(files["toxic"])
    domains = [str(d) for d in domains]
    logger.info(f"Domains: {', '.join(domains)}")

    logger.info(f"{'====' * 5}")
    logger.info(f"Starting '{output_folder}' experiment.")
    logger.info(f"{'====' * 5}")

    paths = defaultdict(lambda: None)
    try:
        for d, domain in enumerate(domains):
            for toxicity in toxicity_choices:
                # Check which domains were already added
                done = False
                if any(f"{domain}, {toxicity}" in done for done in done_domains):
                    logger.info(f"Skipped {domain}, {toxicity} build or train")
                    done = True

                logger.info(f"Domain ({d}/{len(domains)}): {domain} // Toxicity: {toxicity}")

                file = [f for f in files[toxicity] if domain in str(f)][0]

                if multitask:
                    curr_df = pd.read_json(file)
                    logger.info(f"Current number of samples: {curr_df.shape[0]}")

                    if d > 0:
                        previous_df = pd.read_json(multitask_files[toxicity].name)
                        logger.info(f"Previously saved number of samples: {previous_df.shape[0]}")
                        curr_df = pd.concat([previous_df, curr_df], axis=0)
                        curr_df = curr_df.reset_index(drop=True)
                        logger.info(f"New number of samples: {curr_df.shape[0]}")

                    curr_df.to_json(multitask_files[toxicity].name, orient="records")
                    # Read file with appended data instead of unique domain ones
                    file = multitask_files[toxicity].name

                # We have to run even if done=True to get the dstore/model path
                model_path = output_folder / f"domain={d}-{domain}/checkpoints/{toxicity}"
                if kind == "knn":
                    path = build_dstore(
                        output_folder=model_path,
                        train_file=file,
                        model_name=model_name,
                        toxicity=toxicity,
                        dstore_size=dstore_size,
                        pretrained_path=pretrained[toxicity],
                        log_folder=output_folder / f"logs/build_dstore_{toxicity}.log",
                        done=done,
                    )

                elif kind == "dexperts":
                    path = train_expert(
                        output_folder=model_path,
                        expert_name=f"finetune_{model_name}_{d}_{domain}_{toxicity}",
                        train_file=file,
                        model_name=model_name,
                        epochs=1,
                        pretrained_path=pretrained[toxicity],
                        log_folder=output_folder / f"logs/train_{toxicity}.log",
                        done=done,
                    )
                else:
                    raise RuntimeError("")

                # We'll have every intermediate model saved
                pretrained[toxicity] = path
                paths[toxicity] = path

                if not done:
                    with open(output_folder / "logs/done_domains.txt", "a") as f:
                        f.write(f"{domain}, {toxicity}\n")

            evaluate(
                domain=domain,
                output_folder=output_folder / f"domain={d}-{domain}",
                model_name=model_name,
                prompts_path=prompts_path,
                rtp_prompts_path=rtp_prompts_path,
                toxic_model=paths["toxic"],
                nontoxic_model=paths["nontoxic"],
                rate_limit=rate_limit,
                group_toxicity_by=group_toxicity_by,
                num_prompts=num_prompts,
                kind=kind,
            )

    except KeyboardInterrupt:
        logger.info("Interrupted")


if __name__ == "__main__":
    fire.Fire(main)
