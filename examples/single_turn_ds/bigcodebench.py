# bigcodebench_dataset.py
from __future__ import annotations

import json
import random
from typing import List, Dict, Any

from datasets import load_dataset
from collabllm.datasets.single_turn import SingleTurnDataset


class BigCodeBench(SingleTurnDataset):
    """
    BigCodeBench ➟ SingleTurnDataset adaptor.

    * Every row exposes the coding prompt as **`prompt`** and a JSON-string
      with ground-truth fields as **`completion`**.
    * A train/test split is created locally (default 80 / 20).
    * Extra columns (`split`, `task_id`, `entry_point`, `test`) are treated
      as metadata and end up in the `metadata` column when you call
      `to_hf_dataset()`.
    """

    def __init__(
        self,
        repo_id: str = "bigcode/bigcodebench",
        *,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.train_ratio = train_ratio
        self.seed = seed

        raw_ds = load_dataset(repo_id, split="v0.1.2", trust_remote_code=True)
        processed = self._preprocess(raw_ds)

        # Inherit eval_ratio = (1 − train_ratio)
        super().__init__(processed, eval_ratio=1.0 - train_ratio, seed=seed)

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    def _preprocess(self, raw_ds) -> List[Dict[str, Any]]:
        """
        Build the flat dict format required by `SingleTurnDataset`.

        Each output item has at least:
            • prompt
            • completion
            • split          (train / test)
        plus metadata fields used later by the PassRate metric.
        """
        n_total = len(raw_ds)
        n_train = int(n_total * self.train_ratio)

        # reproducible shuffle
        random.seed(self.seed)
        indices = list(range(n_total))
        random.shuffle(indices)

        # map index ➝ split
        split_map = {idx: ("train" if i < n_train else "test") for i, idx in enumerate(indices)}

        processed = []
        for idx, row in enumerate(raw_ds):
            split_tag = split_map[idx]

            # JSON package required by PassRateMetric
            ground_truth = {
                "dataset": "bigcodebench",
                "task_id": row["task_id"],
                "test": row["test"],
                "entry_point": row["entry_point"],
                "answer": row["code_prompt"] + row["canonical_solution"],
            }

            processed.append(
                {
                    # mandatory
                    "prompt": row["instruct_prompt"],
                    "completion": json.dumps(ground_truth),
                    # optional / metadata
                    "split": split_tag,
                    "task_id": row["task_id"],
                    "entry_point": row["entry_point"],
                    "test": row["test"],
                    "extraction_requirement": f"Your extraction should be executable code without any the need of processing. You should start with the following code:\n\n{row['code_prompt']}\n"
                }
            )

        return processed
