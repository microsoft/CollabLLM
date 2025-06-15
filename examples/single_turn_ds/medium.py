# medium_dataset.py
from __future__ import annotations

import os
import json
import random
from typing import List, Dict, Any
import tiktoken

from datasets import load_dataset
from tqdm import tqdm

from collabllm.datasets.single_turn import SingleTurnDataset


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of (approximated) tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens
    

class Medium(SingleTurnDataset):
    """
    Medium-Articles → SingleTurnDataset adaptor.

    • Keeps only articles ≤ `max_tokens` tokens.  
    • Creates fresh train / test splits from the most recent articles
      (based on timestamp) according to `train_ratio` & `test_ratio`.  
    • Each example exposes:
        prompt      – “Write an article … about <title>”
        completion  – the full article text
        split       – train / test
      plus metadata columns (url, authors, timestamp, tags, num_tokens).
    """

    def __init__(
        self,
        repo_id: str = "Kamaljp/medium_articles",
        *,
        train_ratio: float = 0.020,
        test_ratio: float = 0.005,
        max_tokens: int = 512,
        seed: int = 42,
    ):
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.max_tokens = max_tokens
        self.seed = seed

        raw = load_dataset(repo_id, trust_remote_code=True)
        processed = self._preprocess(raw)
        super().__init__(processed, eval_ratio=test_ratio, seed=seed)  # eval_ratio unused because we set `split`

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    def _preprocess(self, raw_ds) -> List[Dict[str, Any]]:
        # The HF repo has a single 'train' split containing all articles.
        full = [row for row in raw_ds["train"] if row.get("timestamp") is not None]

        # Sort by timestamp so “most recent” = tail of the list
        full.sort(key=lambda x: x["timestamp"])

        n_total = len(full)
        n_test = int(n_total * self.test_ratio)
        n_train = int(n_total * self.train_ratio)

        # Take the most recent `n_train + n_test` articles,
        # then split them deterministically.
        recent = full[-(n_train + n_test) :]
        random.Random(self.seed).shuffle(recent)

        test_set = recent[: n_test]
        train_set = recent[n_test:]

        split_map = {id(row): "test" for row in test_set}
        split_map.update({id(row): "train" for row in train_set})

        examples: List[Dict[str, Any]] = []
        for row in tqdm(recent, desc="Processing Medium articles"):
            tokens = num_tokens_from_string(row["text"])
            if tokens > self.max_tokens:
                continue

            title = row["title"]
            prompt = f"Please write an article in less than 500 words about:\n\n{title}\n\n{row['text']}"

            examples.append(
                {
                    # Required columns
                    "prompt": prompt,
                    "completion": row["text"],
                    "split": split_map[id(row)],
                    # Metadata columns
                    "url": row.get("url"),
                    "authors": row.get("authors"),
                    "timestamp": row.get("timestamp"),
                    "tags": row.get("tags"),
                    "num_tokens": tokens,
                }
            )

        return examples
