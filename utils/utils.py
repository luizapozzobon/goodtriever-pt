"""Utility functions.

Some of the functions are from
https://github.com/allenai/real-toxicity-prompts/blob/master/utils/utils.py
"""
import json
from pathlib import Path
from typing import Iterable, List, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    """Create batches of `batch_size` from an iterable."""
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def load_cache(file: Path):
    """Load json file cache."""
    if file.exists():
        with file.open() as f:
            for line in tqdm(f, desc=f"Loading cache from {file}"):
                yield json.loads(line)
