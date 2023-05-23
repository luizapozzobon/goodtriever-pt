import gc
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

import fire
import torch

from experiments.logger import configure_logger


def extract_domains_from_file_list(file_list):
    def _extract_domain(text):
        pattern = r"identity_(.*?)_(toxic|nontoxic)\.json"
        matches = re.findall(pattern, text)

        if matches:
            for match in matches:
                return match[0]

    return sorted(set([_extract_domain(str(filename)) for filename in file_list]))


def main(
    domains: Optional[Tuple] = None,
    toxicity_choices: Tuple = ("toxic", "nontoxic"),
    model_name: str = "gpt2-large",
    prompts_path: str = "data/continual_mitigation/prompts/wilds_single_identity_4k_nontoxic_prompts.jsonl",
    train_folder: str = "data/continual_mitigation/domains/train",
    output_folder: str = "outputs/experiments/continual_learning",
    experiment_name: str = "continual_mitigation",
    filename_format: str = "wilds_single_identity_{domain}_{toxicity}.json",
    rate_limit: str = 90,
    group_toxicity_by: str = "domain",
):
    train_folder = Path(train_folder)

    toxic_added = True if "toxic" in toxicity_choices else False
    nontoxic_added = True if "nontoxic" in toxicity_choices else False
    output_folder = (
        Path(output_folder)
        / experiment_name
        / model_name
        / f"toxic={toxic_added}_nontoxic={nontoxic_added}"
    )

    (output_folder / "logs").mkdir(parents=True, exist_ok=True)
    configure_logger(output_folder / "logs/experiment.log")
    logger = logging.getLogger(__name__)

    files = list(train_folder.glob("*.json"))
    domains = domains or extract_domains_from_file_list(files)

    logger.info(f"{'====' * 5}")
    logger.info(
        f"Starting '{experiment_name}/{model_name}/toxic={toxic_added}_nontoxic={nontoxic_added}' experiment."
    )
    logger.info(f"{'====' * 5}")

    dstore_dirs = []
    for d, domain in enumerate(domains):
        logger.info(f"Domain ({d}/{len(domains)}): {domain}")
        for toxicity in toxicity_choices:
            dstore = output_folder / f"checkpoints/{model_name}_cl/{toxicity}"
            if toxicity == "toxic":
                dstore_dirs.append(f"--dstore_dir {dstore}")
            elif toxicity == "nontoxic":
                dstore_dirs.append(f"--other_dstore_dir {dstore}")
            train_file = train_folder / filename_format.format(domain=domain, toxicity=toxicity)

            ds_cmd = f"""
                python -u -m knn_transformers.run_clm \
                    --model_name_or_path {model_name} \
                    --train_file {train_file} \
                    --eval_subset train \
                    --output_dir {dstore} \
                    --dstore_dir {dstore} \
                    --save_knnlm_dstore \
                    --continue_writing \
                    --do_eval | tee -a {output_folder / f"logs/build_dstore_{toxicity}.log"}
            """

            train_cmd = f"""
                python -u -m knn_transformers.run_clm \
                    --model_name_or_path {model_name} \
                    --train_file {train_file} \
                    --eval_subset train \
                    --output_dir {dstore} \
                    --dstore_dir {dstore} \
                    --build_index | tee -a {output_folder / f"logs/build_index_{toxicity}.log"}
            """

            logger.info(f"Running `datastore build` command {domain}/{toxicity}: {ds_cmd}")
            start_time = time.time()
            os.system(ds_cmd)
            logger.info(f"Execution time: {time.time() - start_time}")
            torch.cuda.empty_cache()
            gc.collect()

            logger.info(f"Running `index train` command {domain}/{toxicity}: {train_cmd}")
            start_time = time.time()
            os.system(train_cmd)
            logger.info(f"Execution time: {time.time() - start_time}")
            torch.cuda.empty_cache()
            gc.collect()

        gens_output = output_folder / f"domain={d}-{domain}"
        generate_cmd = f"""
            python -m scripts.run_all \
                --output_folder {gens_output} \
                --model_name {model_name} \
                --prompts_path {prompts_path} \
                {" ".join(dstore_dirs)} \
                --perspective_rate_limit {rate_limit} \
                --knn True \
                --method ensemble \
                --lmbda 2.0 \
                --knn_temp 100 \
                --method ensemble \
                --num_prompts 100 \
                --group_toxicity_by {group_toxicity_by} \
                --batch_size 4
        """

        logger.info(f"Datastores paths: {' '.join(dstore_dirs)}")
        logger.info(f"Running `run_all` command: {generate_cmd}")
        start_time = time.time()
        os.system(generate_cmd)
        logger.info(f"Execution time: {time.time() - start_time}")
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    fire.Fire(main)
