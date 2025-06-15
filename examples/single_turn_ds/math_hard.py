# math_dataset.py
from __future__ import annotations

from typing import Dict, List, Any

from datasets import load_dataset
from collabllm.datasets.single_turn import SingleTurnDataset


class MATH(SingleTurnDataset):
    """
    MATH-Hard ➟ SingleTurnDataset adaptor.

    Each row produced by `_preprocess` has at minimum:
        • prompt      - the math problem
        • completion  - the reference solution
        • split       - "train" / "test" / "dev" (propagated from HF dataset)
    Plus extra metadata fields (`level`, `type`, …) that SingleTurnDataset
    will automatically place inside the `metadata` column.
    """

    def __init__(
        self,
        repo_id: str = "lighteval/MATH-Hard",
        *,
        eval_ratio: float = 0.1,
        seed: int = 42,
    ):
        raw_ds = load_dataset(repo_id, trust_remote_code=True)
        processed = self._preprocess(raw_ds)
        super().__init__(processed, eval_ratio=eval_ratio, seed=seed)

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _preprocess(raw_ds) -> List[Dict[str, Any]]:
        """
        Convert the original HF splits into the flat dict format expected
        by `SingleTurnDataset`.

        Returns
        -------
        List[dict]
            Items with keys:
                prompt, completion, split, level, type
        """
        processed: List[Dict[str, Any]] = []

        for split_name, split in raw_ds.items():
            for row in split:
                processed.append(
                    {
                        # required columns
                        "prompt": row["problem"],
                        "completion": row["solution"],
                        # optional / metadata (SingleTurnDataset will group them)
                        "split": split_name,
                        "level": row.get("level"),
                        "type": row.get("type"),
                    }
                )

        return processed
