import tempfile
import time
from pathlib import Path

import pandas as pd

from utils.chatgpt import ChatGPTToxicityScorer


def test_chatgpt_request():
    api = ChatGPTToxicityScorer(output_file="empty", language="pt")

    response_nontoxic = api.run_request("Este é um exemplo de frase em português.")
    response_toxic = api.run_request("Foda-se essa merda de vida.")

    assert response_nontoxic["toxicity"] == 0
    assert response_toxic["toxicity"] == 1
    assert response_nontoxic["text"] == "não"
    assert response_toxic["text"] == "sim"


def test_chatgpt_parallel_request():
    tf = tempfile.NamedTemporaryFile()
    scorer = ChatGPTToxicityScorer(output_file=tf.name, language="pt")

    phrases = ["Teste de frase."] * 5
    start = time.time()
    for response_dict in scorer.run_parallel_requests(texts=phrases, num_workers=None):
        pass
    elapsed_parallel = time.time() - start
    responses_parallel = pd.read_json(tf.name, lines=True)

    start = time.time()
    for response_dict in scorer.run_parallel_requests(texts=phrases, num_workers=1):
        pass
    elapsed_not_parallel = time.time() - start
    responses_not_parallel = pd.read_json(tf.name, lines=True)

    assert elapsed_parallel < elapsed_not_parallel
    assert len(responses_parallel) < len(responses_not_parallel)
    assert (
        r == val
        for r, val in zip(
            responses_not_parallel["response_id"].values, list(range(0, 11))
        )
    )
