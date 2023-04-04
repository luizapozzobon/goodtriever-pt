import logging
from enum import Enum, auto

import numpy as np
import torch
from torch import nn

from knn_transformers.datastore import DIST, Datastore

logger = logging.getLogger(__name__)
logger.setLevel(20)


class KEY_TYPE(Enum):
    last_ffn_input = auto()
    last_ffn_output = auto()

    @staticmethod
    def from_string(s):
        try:
            return KEY_TYPE[s.lower()]
        except KeyError:
            raise ValueError()


class METHODS(Enum):
    interpolate = auto()
    interpolate_discourage = auto()

    @staticmethod
    def from_string(s):
        try:
            return METHODS[s.lower()]
        except KeyError:
            raise ValueError()


class KNNWrapper(object):
    def __init__(
        self,
        dstore_dir,
        dimension,
        flat_index=False,
        knn_sim_func=None,
        knn_keytype=None,
        no_load_keys=False,
        move_dstore_to_mem=False,
        knn_gpu=True,
        recompute_dists=False,
        k=1024,
        lmbda=0.25,
        knn_temp=1.0,
        probe=32,
        method="interpolate",
    ):
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.flat_index = flat_index
        self.lmbda = lmbda
        self.k = k
        self.knn_temperature = knn_temp
        self.probe = probe

        self.knn_sim_func = DIST.l2 if knn_sim_func is None else knn_sim_func
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype
        self.no_load_keys = no_load_keys
        self.recompute_dists = recompute_dists
        self.move_dstore_to_mem = move_dstore_to_mem
        self.knn_gpu = knn_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_input_ids = None
        self.prompt_attention_mask = None
        self.model = None
        self.vocab_size = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.hook_handles = []

        method_to_method_func = {
            METHODS.interpolate: KNNWrapper.interpolate,
            METHODS.interpolate_discourage: KNNWrapper.interpolate_discourage,
        }
        self.method = method
        self.method_func = method_to_method_func[self.method]

    def setup_datastore(self):
        self.datastore = Datastore(
            dstore_dir=self.dstore_dir,
            dimension=self.dimension,
            model_type=self.model.config.model_type,
            device=self.device,
            knn_gpu=self.knn_gpu,
            knn_sim_func=self.knn_sim_func,
            knn_temperature=self.knn_temperature,
            move_dstore_to_mem=self.move_dstore_to_mem,
            flat_index=self.flat_index,
        ).setup_faiss()

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.setup_datastore()

        self.is_encoder_decoder = model.config.is_encoder_decoder

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[
            model.config.model_type
        ][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(
            layer_to_capture, capture_input=capture_input
        )
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)
        self.vocab_size = final_layer.out_features

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self.labels = labels
        return self.original_forward_func(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs
        )

    def post_forward_hook(self, module, input, output):
        batch, time_dim, vocab_size = output.shape
        shift = 0 if self.is_encoder_decoder else 1
        lm_logits = output
        lm_logits = torch.nn.functional.log_softmax(lm_logits, dim=-1)  # (batch, time, vocab)
        queries = self.activation_capturer.captured  # (batch, time, dim)

        if self.labels is None:
            nonpad_mask = torch.cat(
                [
                    torch.zeros([batch, time_dim - 1], dtype=torch.bool),
                    torch.ones([batch, 1], dtype=torch.bool),
                ],
                axis=-1,
            ).to(self.device)
        else:
            nonpad_mask = torch.cat(
                [
                    self.labels[:, shift:] != -100,
                    torch.zeros([self.labels.shape[0], shift], dtype=torch.bool).to(self.device),
                ],
                axis=-1,
            )

        lm_logits = lm_logits[nonpad_mask]
        queries = queries[nonpad_mask]  # (nonpad, dim)

        new_scores = self.modify_probabilities()
        output[nonpad_mask] = new_scores

        return output

    def modify_probabilities(self, lm_logits, queries):
        dists, knns = self.datastore.get_knns(queries)  # (nonpad batch * time, k)
        neg_dists = -dists
        knn_log_probs, _ = self.datastore.knns_to_log_prob(knns, neg_dists)

        interpolated_scores = self.method_func(
            knn_log_probs, lm_logits, self.lmbda
        )  # (nonpad, vocab)

        return interpolated_scores

    def register_hook(self, layer, func, pre=False):
        handle = (
            layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        )
        self.hook_handles.append(handle)

    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None

    def get_metrics(self):
        return {}

    @staticmethod
    def interpolate(knn_log_probs, lm_log_probs, lmbda):
        return torch.logaddexp(lm_log_probs + np.log(1 - lmbda), knn_log_probs + np.log(lmbda))

    @staticmethod
    def interpolate_discourage(knn_log_probs, lm_log_probs, lmbda):
        return torch.log(
            torch.nn.functional.relu(
                torch.exp(np.log(1 + lmbda) + lm_log_probs)
                - torch.exp(np.log(lmbda) + knn_log_probs)
            )
        )

    @staticmethod
    def get_model_last_layer(model_type):
        # works for gpt2, marian, t5. If a model does not have an ".lm_head" layer,
        # add an "if model_type is ..." statement here, and return the output embedding layer
        return lambda model: model.lm_head

    @staticmethod
    def get_model_embedding_layer(model_type):
        if model_type.startswith("gpt2"):
            return lambda model: model.transformer.wte

    # For every model name and key type, returns a lambda that returns the relevant layer in the model,
    # and whether the input of that layer should be captured (True) or the output (False)
    model_layer_to_capture = {
        "bart": {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        "gpt2": {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.h[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.h[-1], False),
        },
        "marian": {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        "t5": {
            KEY_TYPE.last_ffn_input: (
                lambda model: model.base_model.decoder.block[-1].layer[2].DenseReluDense,
                True,
            ),
            KEY_TYPE.last_ffn_output: (
                lambda model: model.base_model.decoder.block[-1].layer[2],
                False,
            ),
        },
    }


class KNNSaver(object):
    def __init__(self, dstore_size, dstore_dir, dimension, knn_keytype=None, flat_index=False):
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.flat_index = flat_index
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.dstore_idx = 0
        self.labels = None
        self.hook_handles = []

        logger.info(f"keytype being saved: {self.knn_keytype}")
        logger.info("Saving fp16")

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[
            model.config.model_type
        ][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(
            layer_to_capture, capture_input=capture_input
        )
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook

        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)

        self.datastore = Datastore(
            dstore_dir=self.dstore_dir,
            dimension=self.dimension,
            model_type=self.model.config.model_type,
            device=self.device,
            flat_index=self.flat_index,
            dstore_size=self.dstore_size,
        ).load_keys_and_vals()

    def build_index(self):
        self.datastore.build_index()

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            raise ValueError(
                "labels must be provided when saving a datastore. Are you using --predict_with_generate by mistake? If so, disable it"
            )
        self.labels = labels
        return self.original_forward_func(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs
        )

    def post_forward_hook(self, module, input, output):
        shift = 0 if self.is_encoder_decoder else 1
        captured_keys = self.activation_capturer.captured
        if shift == 1:
            captured_keys = captured_keys[:, :-shift]
        captured_keys = captured_keys.flatten(0, 1)  # (batch * time, dim)
        captured_values = self.labels[:, shift:].flatten(0, 1)  # (batch * time)

        nonpad_mask = captured_values != -100
        keys = captured_keys[nonpad_mask]
        values = captured_values[nonpad_mask]

        batch_time_size = keys.shape[0]
        # if shape[0] == args.tokens_per_sample:
        if self.dstore_idx + batch_time_size > self.dstore_size:
            batch_time_size = max(self.dstore_size - self.dstore_idx, 0)
            keys = keys[:batch_time_size]
            values = values[:batch_time_size]
        try:
            self.datastore.dstore_keys[self.dstore_idx : (batch_time_size + self.dstore_idx)] = (
                keys.cpu().numpy().astype(np.float16)
            )
            self.datastore.dstore_vals[self.dstore_idx : (batch_time_size + self.dstore_idx)] = (
                values.unsqueeze(-1).cpu().numpy().astype(np.int32)
            )
        except ValueError as ex:
            logger.error(
                f"Error saving datastore with mode {self.datastore.dstore_keys.mode}, did you try to save an already existing datastore?"
            )
            logger.error(
                f"Delete the files {self.datastore.dstore_keys.filename} and {self.datastore.dstore_vals.filename} and try again"
            )
            raise ex

        self.dstore_idx += batch_time_size

        return output

    def register_hook(self, layer, func, pre=False):
        handle = (
            layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        )
        self.hook_handles.append(handle)

    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None

    def get_metrics(self):
        return {}


class ActivationCapturer(nn.Module):
    def __init__(self, layer, capture_input=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input

        self.captured = None

    def forward(self, module, input, output):
        if self.capture_input:
            self.captured = input[0].detach()
        else:
            self.captured = output.detach()


# Keeping here for compatibility with Retomaton file
def get_dstore_path(dstore_dir, model_type, dstore_size, dimension):
    return f"{dstore_dir}/dstore_{model_type}_{dstore_size}_{dimension}"
