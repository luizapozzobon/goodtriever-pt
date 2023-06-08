from dataclasses import dataclass, field

from transformers import HfArgumentParser

from knn_transformers.knnlm import DIST, KEY_TYPE


class GenerationParser:
    """Handle arguments involved in the generation process (kNN-LM + generate.py script)."""

    def __init__(self):
        parser = HfArgumentParser((GenerationArguments, KNNArguments))
        self.gen_args, self.knn_args, self.other_strings = parser.parse_args_into_dataclasses(
            return_remaining_strings=True
        )

        self.all_args = {
            "GenerationArguments": vars(self.gen_args),
            "KNNArguments": vars(self.knn_args),
            "OtherStrings": self.other_strings,
        }


@dataclass
class GenerationArguments:
    """
    Args:
        output_folder (str, optional): Folder to save results.
        prompts_path (str, optional): Prompts filename.
            Defaults to "gs://cohere-dev/data/realtoxicityprompts/prompts.jsonl".
        model_name (str, optional): Model to use from HuggingFace Hub.
            Defaults to "gpt2".
        num_return_sequences (int, optional): Number of sequences to return for each prompt.
            Defaults to 25. If `use_eos`, hard-coded to 1.
        max_new_tokens (int, optional): Number of tokens to generate. Defaults to 20.
        top_p (float, optional): Top p for nucleus sampling. Defaults to 0.90.
        batch_size (int, optional): Tokenization and generation batch size. Defaults to 16.
        use_eos (bool, optional): Whether to do an unprompted generation of not.
            If True, ignores `filename` prompts and generates `eos_samples` sequences from
            an end of sequence token. Defaults to False.
        eos_samples (int, optional): Number of eos generations. Defaults to 10k
        output_filename (str, optional): Output filename. If None, will be built
            automatically from user parameters. Defaults to None.
        num_prompts (int, optional): Number of prompts to use. If None, will use all.
    """

    output_folder: str
    prompts_path: str = field(default="gs://cohere-dev/data/realtoxicityprompts/prompts.jsonl")
    model_name: str = field(default="gpt2")
    num_return_sequences: int = field(default=25)
    max_new_tokens: int = field(default=20)
    top_p: float = field(default=0.90)
    batch_size: int = field(default=16)
    use_eos: bool = field(default=False)
    eos_samples: int = field(default=10_000)
    output_filename: str = field(default=None)
    num_prompts: int = field(default=None)


@dataclass
class KNNArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    knn: bool = field(default=False)
    dexperts: bool = field(default=False)
    knn_gpu: bool = field(default=True)
    dstore_size: int = field(default=None, metadata={"help": "The size of the dstore."})
    knn_keytype: KEY_TYPE.from_string = field(default=KEY_TYPE.last_ffn_input)
    save_knnlm_dstore: bool = field(default=False)
    dstore_dir: str = field(default="knn_transformers/checkpoints")
    other_dstore_dir: str = field(default=None)
    knn_sim_func: DIST.from_string = field(default=DIST.l2)
    lmbda: float = field(default=0.5)
    k: int = field(default=1024)
    other_k: int = field(default=None)
    knn_temp: float = field(default=100)
    # Args for building the faiss index:
    build_index: bool = field(default=False)
    # To use flat index instead of quantized index in Faiss, set to True
    # Flat index is slower but returns 100% accurate nearest neighbor results
    flat_index: bool = field(default=False)
    # faiss_index: str = field(default="checkpoints/index")
    ncentroids: int = field(default=4096)
    code_size: int = field(default=64)
    probe: int = field(default=32)
    num_keys_to_add_at_a_time: int = field(default=1000000)
    move_dstore_to_mem: bool = field(default=True)
    no_load_keys: bool = field(default=True)
    recompute_dists: bool = field(default=False)
    method: str = field(default="interpolation")
    filter_p: float = field(default=0.9)
    ensemble_order: tuple = field(default=("subtract", "add"))

    ## RetoMaton args:
    retomaton: bool = field(default=False)
    cluster_dstore: bool = field(default=False)
    no_pointer: bool = field(default=False)
    min_knns: int = field(default=1)
    max_knns: int = field(default=1024)
    num_clusters: int = field(default=500000)
    sample_size: int = field(default=20000000)
    members: str = field(default=None)
