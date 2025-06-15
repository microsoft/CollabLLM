# abgcoqa_dataset.py
from __future__ import annotations

import json
import os.path as osp
from typing import List, Dict, Any

from collabllm.datasets.single_turn import SingleTurnDataset


class AbgCoQA(SingleTurnDataset):
    """
    Abg-CoQA → SingleTurnDataset adaptor.

    * Task: binary classification — **ambiguous** vs **non_ambiguous**.
    * Each example exposes
        • `prompt`      – story + (optional) history + target question  
        • `completion`  – the label string: "ambiguous" or "non_ambiguous"  
        • `split`       – train / val / test  
      plus metadata (`id`, `source`, `ambiguity` flag).  
    """

    def __init__(
        self,
        root: str = "data/",
        *,
        add_history: bool = False,
        eval_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.add_history = add_history
        raw_ds = self._load_json_splits(root)
        processed = self._preprocess(raw_ds)
        super().__init__(processed, eval_ratio=eval_ratio, seed=seed)

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_json_splits(root: str) -> Dict[str, Any]:
        splits = {}
        for name in ("train", "val", "test"):
            path = osp.join(root, f"abg-coqa/coqa_abg_{name}.json")
            with open(path, "r", encoding="utf-8") as f:
                splits[name] = json.load(f)
        return splits

    # ------------------------------------------------------------------ #
    def _preprocess(self, raw_ds) -> List[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []

        for split_name, split_blob in raw_ds.items():
            for story in split_blob["data"]:
                label: str = story["ambiguity"]  # 'ambiguous' | 'non_ambiguous'

                # --- build prompt --------------------------------------------------- #
                context_parts: List[str] = [
                    "Can you help me answer a question about the following story?",
                    "",
                    story["story"],
                    "",
                ]

                if self.add_history and story["history_turns"]:
                    context_parts.append("Below are some previous Q&A pairs:")
                    for t in sorted(story["history_turns"], key=lambda x: x["turn_id"]):
                        context_parts.append(f"Q: {t['question']}")
                        context_parts.append(f"A: {t['answer']}")
                    context_parts.append("")

                # target question
                target_q = story["target_turn"]["question"]
                context_parts.append(f"My question is: {target_q}")

                prompt_text = "\n".join(context_parts)

                # --- package example ------------------------------------------------ #
                examples.append(
                    {
                        # required keys for SingleTurnDataset
                        "prompt": prompt_text,
                        "completion": label,
                        "split": split_name,
                        # metadata
                        "id": story.get("id"),
                        "source": story.get("source"),
                        "ambiguity": label == "ambiguous",
                    }
                )

        return examples
