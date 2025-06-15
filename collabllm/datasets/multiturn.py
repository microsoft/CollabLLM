"""
collabllm.datasets.multiturn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unified loader + wrapper for multi-turn chat data.

Initialization supports three input styles:

1. **Flat list** (`List[dict]`) with required keys per row:
   {'prompt', 'completion', 'conv_id', 'score',
    'single_turn_prompt', 'single_turn_completion', 'single_turn_metadata'}

2. **Nested structure** (`List[dict]`) per conversation:
   [
     {
       "conv_id": ...,
       "single_turn_prompt": ...,
       "single_turn_completion": ...,
       "single_turn_metadata": ...,
       "turns": [
         {
           "prompt": [ {role,content}, ... ],
           "responses": [
             {"completion": ..., "score": ...},
             {"completion": ..., "score": ...},
             ...
           ]
         },
         ...
       ]
     },
     ...
   ]

3. **Local HF dataset directory** (path) or **HF Hub repo ID** (string).

In all cases, the internal `self.data` will be a flat `List[dict]` with keys:
{prompt, completion, conv_id, score, single_turn_prompt, single_turn_completion,
 single_turn_metadata, turn_id}

Derived field
-------------
• `turn_id` is set to `len(prompt)` if not provided explicitly.

Converters (all use a uniform random splitter)
-----------------------------------------------
• `to_sft_dataset()`   → DatasetDict {text}  
• `to_dpo_dataset()`   → DatasetDict {prompt, chosen, rejected, score_*}  
• `to_inputs_dataset()`→ DatasetDict {prompt, single_turn_*}
"""

from __future__ import annotations

import os
import random
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Union
from collabllm.prompts import SYSTEM_PROMPT
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

_REQUIRED: set[str] = {
    "prompt",
    "completion",
    "conv_id",
    "score",
    "single_turn_prompt",
    "single_turn_completion",
    "single_turn_metadata",
}

import logging
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# uniform splitter                                                            #
# --------------------------------------------------------------------------- #
def _uniform_split(
    full_ds: Dataset,
    *,
    eval_ratio: float,
    n_eval: Optional[int],
    seed: int,
) -> DatasetDict:
    k = n_eval if n_eval is not None else int(eval_ratio * len(full_ds))
    k = min(k, len(full_ds))

    random.seed(seed)
    eval_idx = set(random.sample(range(len(full_ds)), k=k))
    train_idx = [i for i in range(len(full_ds)) if i not in eval_idx]

    return DatasetDict(
        {
            "train": full_ds.select(train_idx),
            "eval": full_ds.select(sorted(eval_idx)),
        }
    )


# --------------------------------------------------------------------------- #
# main dataclass                                                              #
# --------------------------------------------------------------------------- #
class MultiturnDataset:
    def __init__(
        self,
        data_or_local_dir_or_hf_repo_or_nested: Union[List[Dict[str, Any]], str],
        *,
        seed: int = 42,
        add_system_prompt: bool = True,
    ):
        """
        Parameters
        ----------
        data_or_local_dir_or_hf_repo_or_nested :
            • Flat list of dicts with required keys (old style), OR
            • Nested list of conversations (new style), OR
            • Local path saved by `Dataset.save_to_disk`, OR
            • HF Hub repo ID (e.g. "org/dataset").
        seed : int
            RNG seed for uniform splitting.
        """
        self.seed = seed
        self.sys_msg = [{"role": "system", "content": SYSTEM_PROMPT}] if add_system_prompt else []

        # 1) Load raw data into `raw_list` of dicts
        if isinstance(data_or_local_dir_or_hf_repo_or_nested, list):
            raw_list = data_or_local_dir_or_hf_repo_or_nested
        elif os.path.exists(str(data_or_local_dir_or_hf_repo_or_nested)):
            ds_dict = load_from_disk(str(data_or_local_dir_or_hf_repo_or_nested))  # type: ignore
            raw_list = [dict(r) for r in ds_dict.flatten()]
        else:
            ds_dict = load_dataset(str(data_or_local_dir_or_hf_repo_or_nested), trust_remote_code=True)  # type: ignore
            raw_list = [dict(r) for _, split in ds_dict.items() for r in split]

        if not raw_list:
            raise ValueError("Loaded dataset is empty.")

        # 2) Detect nested structure: presence of "turns" key in first element
        if isinstance(raw_list[0], dict) and "turns" in raw_list[0]:
            self.data = self._flatten_nested(raw_list)
        else:
            # Assume flat structure; validate required keys
            if not _REQUIRED.issubset(raw_list[0]):
                missing = _REQUIRED - set(raw_list[0])
                raise ValueError(f"Missing required keys in flat data: {missing}")

            # Auto-fill turn_id if missing
            for row in raw_list:
                if not isinstance(row["prompt"], Sequence):
                    raise TypeError("`prompt` must be a list of messages.")
                row.setdefault("turn_id", len(row["prompt"]))

            self.data = raw_list  # type: ignore

        if not self.data:
            raise ValueError("No valid rows after processing input.")

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool = False,
        token: Optional[str] = None,
        split: Optional[str] = None,
    ) -> DatasetDict:

        """
        Push the dataset to the Hugging Face Hub.

        Parameters
        ----------
        repo_id : str
            The repository ID on the Hugging Face Hub.
        private : bool
            Whether to create a private repository.
        token : Optional[str]
            Optional authentication token for the Hub.
        split : Optional[str]
            If provided, will save only this split (e.g., "train", "eval").

        Returns
        -------
        DatasetDict
            The pushed dataset.
        """
        ds = Dataset.from_dict({k: [row[k] for row in self.data] for k in self.data[0]})
        return ds.push_to_hub(repo_id, private=private, token=token, split=split)


    def _flatten_nested(self, nested: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert nested conversation format to flat list of rows.

        Nested format per conversation:
        {
          "conv_id": ...,
          "single_turn_prompt": ...,
          "single_turn_completion": ...,
          "single_turn_metadata": ...,
          "turns": [
             {
               "prompt": [...],
               "responses": [
                  {"completion": ..., "score": ..., **kwargs}, ...
               ]
             },
             ...
          ]
        }

        Output per row:
        {
          "prompt": [...],
          "completion": ...,
          "conv_id": ...,
          "score": ...,
          "single_turn_prompt": ...,
          "single_turn_completion": ...,
          "single_turn_metadata": ...,
          "turn_id": len(prompt)
        }
        """
        flat = []
        for base_conv_id, convo in enumerate(nested):
            # Validate presence of required conversation-level keys
            for key in {"single_turn_prompt", "single_turn_completion", "single_turn_metadata", "turns"}:
                if key not in convo:
                    raise ValueError(f"Missing key '{key}' in nested conversation.")
            st_prompt = convo["single_turn_prompt"]
            st_completion = convo["single_turn_completion"]
            st_metadata = convo["single_turn_metadata"]

            for turn in convo["turns"]:
                if "prompt" not in turn or "responses" not in turn:
                    raise ValueError("Each turn must have 'prompt' and 'responses'.")
                prompt_msgs = turn["prompt"]
                if not isinstance(prompt_msgs, Sequence):
                    raise TypeError("`turn['prompt']` must be a list of messages.")
                turn_id = len(prompt_msgs)
                for resp in turn["responses"]:
                    if "completion" not in resp or "score" not in resp:
                        raise ValueError("Each response must have 'completion' and 'score'.")
                    flat.append(
                        {
                            "prompt": prompt_msgs,
                            "completion": resp["completion"],
                            "conv_id": base_conv_id,
                            "score": resp["score"],
                            "single_turn_prompt": st_prompt,
                            "single_turn_completion": st_completion,
                            "single_turn_metadata": st_metadata,
                            "turn_id": turn_id,
                            **{k: resp.get(k) for k in resp if k not in {"completion", "score"}},
                        }
                    )
        return flat

    # ------------------------------------------------------------------ #
    # SFT                                                                #
    # ------------------------------------------------------------------ #
    def to_sft_dataset(
        self,
        *,
        n_eval: Optional[int] = None,
        eval_ratio: Optional[float] = 0.0,
        lower_bound_metric: Optional[str] = None,
        lower_bound: Optional[float] = 0.0,
    ) -> DatasetDict:

        # Select best example per conversation ID: prefer latest turn, then highest score
        best_examples = {}
        for row in self.data:
            cid = row["conv_id"]
            prev = best_examples.get(cid)
            if prev is None or row["turn_id"] > prev["turn_id"] or (
                row["turn_id"] == prev["turn_id"] and row["score"] > prev["score"]
            ):
                best_examples[cid] = row

        # Build SFT dialogues, filtering by optional metric threshold
        serialized_dialogues = []
        for row in best_examples.values():
            if lower_bound_metric:
                try:
                    metric = row
                    for key in lower_bound_metric.split("."):
                        metric = metric.get(key, {})
                    value = np.asarray(metric).mean().item()
                except Exception as e:
                    logger.error(f"Failed to extract metric '{lower_bound_metric}' from row: {row} — {e}")
                    continue

                if value < lower_bound:
                    logger.warning(
                        f"Filtered out conv_id={row['conv_id']} (turn_id={row['turn_id']}) "
                        f"due to {lower_bound_metric}={value:.3f} < {lower_bound:.3f}"
                    )
                    continue

            if not isinstance(row["prompt"], list):
                raise TypeError("Expected `prompt` to be a list of messages.")

            messages = self.sys_msg + row["prompt"] + [{"role": "assistant", "content": row["completion"]}]
            serialized_dialogues.append(messages)

        logger.info(
            f"Converted {len(serialized_dialogues)} dialogues "
            f"(filter: {lower_bound_metric} ≥ {lower_bound}); "
            f"retention ratio: {len(serialized_dialogues)/len(best_examples):.2f}"
        )

        full_dataset = Dataset.from_dict({"messages": serialized_dialogues})
        return _uniform_split(full_dataset, eval_ratio=eval_ratio, n_eval=n_eval, seed=self.seed)

    # ------------------------------------------------------------------ #
    # DPO                                                                #
    # ------------------------------------------------------------------ #
    def to_dpo_dataset(
        self,
        *,
        minimum_gap: float = 0.0,
        n_eval: Optional[int] = None,
        eval_ratio: Optional[float] = 0.0,
    ) -> DatasetDict:

        # Group rows by (conv_id, turn_id)
        grouped: Dict[tuple, List[Dict[str, Any]]] = {}
        for r in self.data:
            grouped.setdefault((r["conv_id"], r["turn_id"]), []).append(r)

        pairs = []
        for items in grouped.values():
            if len(items) < 2:
                continue
            items = sorted(items, key=lambda r: r["score"], reverse=True)
            if items[0]["score"] - items[-1]["score"] < minimum_gap:
                continue
            pairs.append(
                {
                    "prompt": self.sys_msg + items[0]["prompt"],
                    "chosen": items[0]["completion"],
                    "rejected": items[-1]["completion"],
                    "score_chosen": items[0]["score"],
                    "score_rejected": items[-1]["score"],
                }
            )

        logger.info(f"Converted {len(pairs)} pairs (minimum_gap={minimum_gap}, ratio={len(pairs)/len(self.data):.2f})")

        if not pairs:
            return DatasetDict({"train": Dataset.from_dict({}), "eval": Dataset.from_dict({})})

        full_ds = Dataset.from_dict({k: [p[k] for p in pairs] for k in pairs[0]})
        return _uniform_split(full_ds, eval_ratio=eval_ratio, n_eval=n_eval, seed=self.seed)

    # ------------------------------------------------------------------ #
    # Inputs                                                             #
    # ------------------------------------------------------------------ #
    def to_inputs_dataset(
        self,
        *,
        n_eval: Optional[int] = None,
        eval_ratio: Optional[float] = 0.0,
    ) -> DatasetDict:

        # Keep exactly one row per (conv_id, turn_id)
        unique: Dict[tuple, Dict[str, Any]] = {}
        for r in self.data:
            key = (r["conv_id"], r["turn_id"])
            if key not in unique:
                unique[key] = r
            r['prompt'] = self.sys_msg + r["prompt"]

        keep_keys = [
            "prompt",
            "single_turn_prompt",
            "single_turn_completion",
            "single_turn_metadata",
        ]
        records = [{k: row[k] for k in keep_keys} for row in unique.values()]
        if not records:
            return DatasetDict({"train": Dataset.from_dict({}), "eval": Dataset.from_dict({})})

        full_ds = Dataset.from_dict({k: [rec[k] for rec in records] for k in keep_keys})
        return _uniform_split(full_ds, eval_ratio=eval_ratio, n_eval=n_eval, seed=self.seed)

    # ------------------------------------------------------------------ #
    # misc                                                               #
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
