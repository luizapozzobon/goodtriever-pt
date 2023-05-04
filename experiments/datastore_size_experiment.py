import gc
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Union

import fire
import torch
from logger import configure_logger


def main(
    model_name: str,
    experiment_name: str,
    toxic_tokens: Optional[Union[int, List]] = None,
    nontoxic_tokens: Optional[Union[int, List]] = None,
    num_prompts: Optional[int] = None,
    output_folder: str = "outputs/experiments/",
    dstores: str = "both",
    toxic_train_file: str = "data/jigsaw/toxicity_gte0.5_clean.json",
    nontoxic_train_file: str = "data/jigsaw/toxicity_eq0_half_clean.json",
    method: str = "ensemble",
    lmbda: float = 2.0,
    temperature: int = 100,
    prompts_path: str = "data/dexperts/prompts/nontoxic_prompts-10k.jsonl",
    only_generate: bool = False,
):
    base_folder = Path(output_folder) / experiment_name / model_name

    if not isinstance(toxic_tokens, tuple):
        toxic_tokens = (toxic_tokens,)
    if not isinstance(nontoxic_tokens, tuple):
        nontoxic_tokens = (nontoxic_tokens,)

    assert len(toxic_tokens) == len(nontoxic_tokens), "Must have same number of dstore sizes."

    for dstore_tokens, other_dstore_tokens in zip(toxic_tokens, nontoxic_tokens):
        dstore_dirs = ["--dstore_dir", "--other_dstore_dir"]
        output_folder = base_folder / f"toxic={dstore_tokens}_nontoxic={other_dstore_tokens}"
        (output_folder / "logs").mkdir(parents=True, exist_ok=True)

        configure_logger(output_folder / "logs/experiment.log")
        logger = logging.getLogger(__name__)
        logger.info(f"Starting '{experiment_name}/{model_name}' experiment.")

        for i, train_file in enumerate([toxic_train_file, nontoxic_train_file]):
            if i == 0:
                tokens = dstore_tokens
            elif i == 1:
                tokens = other_dstore_tokens
                if dstores == "toxic":
                    continue
                elif dstores == "nontoxic":
                    raise NotImplementedError(
                        "Currently, just using both datastores or just the toxic one is supported."
                    )

            dstore = base_folder / "checkpoints" / f"gpt2_{Path(train_file).stem}_{tokens}"
            dstore_dirs[i] = f"{dstore_dirs[i]} {dstore}"

            if only_generate or dstore.exists():
                logger.info(f"Skipping datastore build and index train for datastore {i}.")
                continue

            ds_cmd = f"""
                python -u -m knn_transformers.run_clm \
                    --model_name_or_path {model_name} \
                    --train_file {train_file} \
                    --eval_subset train \
                    --output_dir {dstore} \
                    --dstore_dir {dstore} \
                    {f'--dstore_size {tokens} --limit_eval_to_dstore' if tokens else ''} \
                    --save_knnlm_dstore \
                    --do_eval | tee {output_folder / f"logs/build_dstore_{i}.log"}
            """

            train_cmd = f"""
                python -u -m knn_transformers.run_clm \
                    --model_name_or_path {model_name} \
                    --train_file {train_file} \
                    --eval_subset train \
                    --output_dir {dstore} \
                    --dstore_dir {dstore} \
                    {f'--dstore_size {tokens}' if tokens else ''} \
                    --build_index | tee {output_folder / f"logs/build_index_{i}.log"}
            """
            logger.info(f"Running `datastore build` command {i}: {ds_cmd}")
            start_time = time.time()
            os.system(ds_cmd)
            logger.info(f"Execution time: {time.time() - start_time}")
            torch.cuda.empty_cache()
            gc.collect()

            logger.info(f"Running `index train` command {i}: {train_cmd}")
            start_time = time.time()
            os.system(train_cmd)
            logger.info(f"Execution time: {time.time() - start_time}")
            torch.cuda.empty_cache()
            gc.collect()

        generate_cmd = f"""
            python -m scripts.run_all \
                --output_folder {output_folder} \
                --model_name {model_name} \
                --prompts_path {prompts_path} \
                {" ".join(dstore_dirs)} \
                --knn True \
                --method ensemble \
                --lmbda {lmbda} \
                --knn_temp {temperature} \
                --method {method} \
                {f'--num_prompts {num_prompts}' if num_prompts else ''} \
                --batch_size 4
        """

        logger.info(f"Running `run_all` command: {generate_cmd}")
        start_time = time.time()
        os.system(generate_cmd)
        logger.info(f"Execution time: {time.time() - start_time}")
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    fire.Fire(main)
