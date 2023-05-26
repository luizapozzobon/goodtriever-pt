import logging
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import fire

from experiments.continual_learning.utils import (
    evaluate,
    extract_domains_from_file_list,
    run,
)
from experiments.logger import configure_logger

logger = logging.getLogger(__name__)


def build_dstore(
    output_folder, train_file, model_name, toxicity, pretrained_path, dstore_size=None, done=False
):
    dstore_path = output_folder / f"checkpoints/{model_name}_cl/{toxicity}"
    dstore_path.mkdir(exist_ok=True, parents=True)

    if done:
        return dstore_path

    if pretrained_path is not None:
        logger.info(f"Copying base {toxicity} dstore from {pretrained_path} to {dstore_path}.")
        subprocess.run(f"cp -r {pretrained_path}/* {dstore_path}/", shell=True)
        logger.info(f"Base {toxicity} dstore copied.")

    ds_cmd = f"""
        python -u -m knn_transformers.run_clm \
            --model_name_or_path {model_name} \
            --train_file {train_file} \
            --eval_subset train \
            --output_dir {dstore_path} \
            --dstore_dir {dstore_path} \
            --save_knnlm_dstore \
            --continue_writing \
            {f'--dstore_size {dstore_size} --limit_eval_to_dstore' if dstore_size else ''} \
            --do_eval | tee -a {output_folder / f"logs/build_dstore_{toxicity}.log"}
    """

    train_cmd = f"""
        python -u -m knn_transformers.run_clm \
            --model_name_or_path {model_name} \
            --train_file {train_file} \
            --eval_subset train \
            --output_dir {dstore_path} \
            --dstore_dir {dstore_path} \
            {f'--dstore_size {dstore_size}' if dstore_size else ''} \
            --build_index | tee -a {output_folder / f"logs/build_index_{toxicity}.log"}
    """

    logger.info(f"Running `datastore build` command: {ds_cmd}")
    run(ds_cmd)

    logger.info(f"Running `index train` command: {train_cmd}")
    run(train_cmd)

    return dstore_path


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
            --overwrite_cache
    """

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
):
    # Folder setup
    output_folder = setup_output_folder(
        output_folder, toxicity_choices, experiment_name, model_name, kind
    )

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
        "toxic": sorted(list(train_folder.glob("*_toxic.json"))),
        "nontoxic": sorted(list(train_folder.glob("*_nontoxic.json"))),
    }
    pretrained = {"toxic": pretrained_toxic, "nontoxic": pretrained_nontoxic}
    domains = domains or extract_domains_from_file_list(files["toxic"])
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

                path = None
                if kind == "knn":
                    path = build_dstore(
                        output_folder=output_folder,
                        train_file=file,
                        model_name=model_name,
                        toxicity=toxicity,
                        pretrained_path=pretrained[toxicity],
                        dstore_size=dstore_size,
                        done=done,
                    )

                elif kind == "dexperts":
                    path = train_expert(
                        expert_name=f"finetune_{model_name}_{d}_{domain}_{toxicity}",
                        train_file=file,
                        model_name=model_name,
                        epochs=1,
                        output_folder=output_folder / f"models/experts/{domain}",
                        pretrained_path=pretrained[toxicity],
                        done=done,
                    )
                else:
                    raise RuntimeError("")

                if not done:
                    with open(output_folder / "logs/done_domains.txt", "a") as f:
                        f.write(f"{domain}, {toxicity}\n")

                paths[toxicity] = path

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
