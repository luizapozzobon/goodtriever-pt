
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from knn_transformers.knnlm import KNNWrapper


def setup_tokenizer(model_name: str):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def setup_model(model_name: str, knn_args):
    model = GPT2LMHeadModel.from_pretrained(model_name)

    dimension = model.config.hidden_size
    knn_wrapper = None
    # knn_args.seed = training_args.seed

    if knn_args.knn:
        knn_wrapper = KNNWrapper(
            dstore_dir=knn_args.dstore_dir,
            other_dstore_dir=knn_args.other_dstore_dir,
            dimension=dimension,
            flat_index=knn_args.flat_index,
            knn_sim_func=knn_args.knn_sim_func,
            knn_keytype=knn_args.knn_keytype,
            no_load_keys=knn_args.no_load_keys,
            move_dstore_to_mem=knn_args.move_dstore_to_mem,
            knn_gpu=knn_args.knn_gpu,
            recompute_dists=knn_args.recompute_dists,
            k=knn_args.k,
            other_k=knn_args.other_k,
            lmbda=knn_args.lmbda,
            knn_temp=knn_args.knn_temp,
            probe=knn_args.probe,
            method=knn_args.method
        )

    if knn_wrapper is not None:
        knn_wrapper.break_into(model)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model
