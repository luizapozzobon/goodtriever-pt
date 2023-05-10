import gc
import os

import fire
import torch
from typing import Tuple, Iterable


def main(
    toxic_ks: Tuple = (1, 2, 8, 64, 256, 1024),
    nontoxic_ks: Tuple = (1, 2, 8, 64, 256, 1024),
    lmbda: float = 2.0,
    temperature: int = 100,
    dstores: str = "both",
    model_name: str = "gpt2-large",
    rate_limit: int = 20
) -> None:

    if isinstance(toxic_ks, Iterable) and nontoxic_ks is None or isinstance(nontoxic_ks, int):
        nontoxic_ks = [nontoxic_ks] * len(toxic_ks)

    if isinstance(nontoxic_ks, Iterable) and toxic_ks is None or isinstance(toxic_ks, int):
        toxic_ks = [toxic_ks] * len(nontoxic_ks)

    assert len(toxic_ks) == len(nontoxic_ks), (
        "Define the same amount of Ks for both toxic and non-toxic datastores."
    )

    toxic_dstore = "checkpoints/gpt2-large/gpt2_toxicity_gte0.5_clean"
    nontoxic_dstore = "checkpoints/gpt2-large/gpt2_toxicity_eq0_half_clean"

    dstore_dirs = ""
    if dstores in ["toxic", "both"]:
        dstore_dirs += f"--dstore_dir {toxic_dstore}"
    if dstores == "both":
        dstore_dirs += f" --other_dstore_dir {nontoxic_dstore}"

    cmd = """
        python -m scripts.run_all \
            --model_name {model_name} \
            --prompts_path data/dexperts/prompts/nontoxic_prompts-10k.jsonl \
            --output_folder outputs/experiments/k/{dstores}/ktoxic={toxic_k}_knontoxic={nontoxic_k} \
            {dstore_dirs} \
            --k {toxic_k} \
            --other_k {nontoxic_k} \
            --knn True \
            --method ensemble \
            --lmbda {lmbda} \
            --knn_temp {temperature} \
            --num_prompts 100 \
            --perspective_rate_limit {rate_limit} \
            --batch_size 4
    """

    for toxic_k, nontoxic_k in zip(toxic_ks, nontoxic_ks):
        print(f"\nRunning with k_toxic = {toxic_k} and k_nontoxic = {nontoxic_k}")
        filled_cmd = cmd.format(
            model_name=model_name,
            toxic_k=toxic_k,
            nontoxic_k=nontoxic_k,
            lmbda=lmbda,
            temperature=temperature,
            dstore_dirs=dstore_dirs,
            dstores=dstores,
            rate_limit=rate_limit,
        )
        os.system(filled_cmd)
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    fire.Fire(main)
