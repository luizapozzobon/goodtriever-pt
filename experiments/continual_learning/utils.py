import gc
import logging
import os
import re
import subprocess
import time

import torch

logger = logging.getLogger(__name__)


def cleanup(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        logger.info(f"Execution time: {time.time() - start_time}")
        torch.cuda.empty_cache()
        gc.collect()

    return wrapper


def run(cmd):
    return cleanup(subprocess.run(cmd, shell=True))


def evaluate_on_prompts(
    domain,
    output_folder,
    model_name,
    prompts_path,
    toxic_model,
    nontoxic_model,
    rate_limit,
    group_toxicity_by=None,
    num_prompts=None,
    kind="dexperts",
):
    generate_cmd = f"""
        python -m scripts.run_all \
            --output_folder {output_folder} \
            --model_name {model_name} \
            --prompts_path {prompts_path} \
            --dstore_dir {toxic_model} \
            {f'--other_dstore_dir {nontoxic_model}' if nontoxic_model else ''} \
            {'--dexperts True' if kind == 'dexperts' else '--knn True'} \
            --perspective_rate_limit {rate_limit} \
            --lmbda 2.0 \
            --method ensemble \
            --filter_p 0.9 \
            --group_toxicity_by {group_toxicity_by} \
            {f'--num_prompts {num_prompts}' if num_prompts is not None else ''} \
            --batch_size 4
    """
    logger.info(f"Running domain {domain} `run_all` command: {generate_cmd}")
    return run(generate_cmd)


def evaluate(
    domain,
    output_folder,
    model_name,
    prompts_path,
    rtp_prompts_path,
    toxic_model,
    nontoxic_model,
    rate_limit,
    group_toxicity_by=None,
    num_prompts=None,
    kind="knn",
):
    evaluate_on_prompts(
        domain=domain,
        output_folder=output_folder,
        model_name=model_name,
        prompts_path=prompts_path,
        toxic_model=toxic_model,
        nontoxic_model=nontoxic_model,
        rate_limit=rate_limit,
        group_toxicity_by=group_toxicity_by,
        num_prompts=num_prompts,
        kind=kind,
    )

    # RTP 1k evaluation
    if rtp_prompts_path is not None:
        evaluate_on_prompts(
            domain="rtp",
            output_folder=output_folder / "RTP",
            model_name=model_name,
            prompts_path=rtp_prompts_path,
            toxic_model=toxic_model,
            nontoxic_model=nontoxic_model,
            rate_limit=rate_limit,
            group_toxicity_by=None,
            num_prompts=num_prompts or 1000,
            kind=kind,
        )


def extract_domains_from_file_list(file_list):
    def _extract_domain(text):
        pattern = r"identity_(.*?)_(toxic|nontoxic)\.json"
        matches = re.findall(pattern, text)

        if matches:
            for match in matches:
                return match[0]

    return sorted(set([_extract_domain(str(filename)) for filename in file_list]))
