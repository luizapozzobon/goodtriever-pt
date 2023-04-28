import gc
import os

import fire
import numpy as np
import torch


def main(param="lambda", temperature=1, dstores="both", min_alpha=0.5, max_alpha=2.0, step=0.5):

    toxic_dstore = "checkpoints/gpt2_jigsaw_toxic"
    nontoxic_dstore = "checkpoints/gpt2_jigsaw_nontoxic"

    dstore_dirs = ""
    if dstores in ["toxic", "both"]:
        dstore_dirs += f"--dstore_dir {toxic_dstore}"
    if dstores == "both":
        dstore_dirs += f" --other_dstore_dir {nontoxic_dstore}"

    cmd = """
        python -m scripts.run_all \
            --prompts_path data/dexperts/prompts/nontoxic_prompts-10k.jsonl \
            --output_folder outputs/gpt2_jigsaw/ensemble/small_prompted_exp_patch_min/temperature_{temperature}/{param}/{dstores}/{param}_{value} \
            {dstore_dirs} \
            --knn True \
            --method ensemble \
            --lmbda {value} \
            --knn_temp {temperature} \
            --num_prompts 100 \
            --batch_size 4
    """

    for value in np.arange(min_alpha, max_alpha + step, step):
        print("\nRunning with {} = {}".format(param, value))
        filled_cmd = cmd.format(
            param=param,
            value=value,
            temperature=temperature,
            dstore_dirs=dstore_dirs,
            dstores=dstores,
        )
        os.system(filled_cmd)
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    fire.Fire(main)
