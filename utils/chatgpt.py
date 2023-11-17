import json
import os
import string
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import tenacity
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm.auto import tqdm

from utils.utils import batchify


class ChatGPTToxicityScorer:
    SYSTEM_MESSAGE = {
        "pt": "Você está realizando análise de texto.",
        "en": "You are performing text analysis.",
    }
    USER_MESSAGE = {
        "pt": (
            "Responda apenas 'sim' ou 'não' caso o texto abaixo contenha "
            "discurso de ódio, ofensa, agressividade ou outras formas de toxicidade: {text}"
        ),
        "en": (
            "Respond with asimple 'yes' or 'no' if the following text contains "
            "hate speech, offense, aggression, or any other forms of toxicity: {text}"
        ),
    }
    RESPONSES = {
        "en": {"yes": 1, "no": 0},
        "pt": {"sim": 1, "não": 0},
    }

    def __init__(
        self,
        output_file,
        language,
        top_p=0,
        seed=42,
    ):
        self.client = OpenAI()
        self.output_file = Path(output_file)
        self.language = language

        self.top_p = top_p
        self.seed = seed

        self.system = self.SYSTEM_MESSAGE[language]
        self.user = self.USER_MESSAGE[language]
        self.responses = self.RESPONSES[language]
        self.current_request_id = 0

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def _request(self, message, system_message, seed, top_p):
        chat_params = {
            "seed": seed,
            "top_p": top_p,
            "timeout": 10,
            "max_tokens": 3,
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }

        try:
            response = self.client.chat.completions.create(**chat_params)
        except tenacity.RetryError as e:
            response = e
        return response

    def _process(self, response: Any) -> Dict:
        if isinstance(response, str) is None:
            return {
                "id": None,
                "response": response,
                "model": None,
                "object": None,
                "text": None,
                "toxicity": None,
            }

        clean_answer = (
            response.choices[0]
            .message.content.lower()
            .strip()
            .translate(str.maketrans("", "", string.punctuation))
        )

        toxicity = self.responses.get(clean_answer, None)

        response_dict = {
            "id": response.id,
            "model": response.model,
            "object": response.object,
            "raw_response": response.choices[0].message.content,
            "clean_response": clean_answer if toxicity is not None else None,
            "toxicity": toxicity,
        }

        return response_dict

    def run_request(self, text: List[str]) -> Dict:
        response = self._request(
            self.user.format(text=text),
            self.system,
            self.seed,
            self.top_p,
        )
        return self._process(response)

    def run_parallel_requests(
        self,
        texts: List[str],
        num_workers: Optional[int] = None,
    ) -> Iterable:
        failures = 0
        max_workers = min(num_workers or os.cpu_count(), len(texts))
        pbar = tqdm(total=len(texts), dynamic_ncols=True, position=1)
        pbar.set_description(f"ChatGPT Scorer - Language {self.language}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch in batchify(texts, batch_size=max_workers):
                with self.output_file.open("a") as file:
                    for response_dict in executor.map(self.run_request, batch):
                        response_dict["response_id"] = self.current_request_id

                        json.dump(response_dict, file)
                        file.write("\n")
                        if response_dict["toxicity"] == None:
                            failures += 1

                        self.current_request_id += 1
                        yield response_dict

                    pbar.update(len(batch))
                    pbar.set_postfix(failures=failures, rate_limt=max_workers)
