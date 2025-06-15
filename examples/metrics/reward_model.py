import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from collabllm.metric import SingleTurnOrChatMetric, BaseMetric

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Utility: lazy-load + cache any reward model / tokenizer pair                #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _load_reward_model(model_name: str, device: str, dtype: str, **hf_kwargs):
    """
    Loads model & tokenizer exactly once per (model_name, device, dtype) combo.
    Returns (model, tokenizer) ready for inference.
    """
    logger.info("Loading reward model %s …", model_name)
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        **hf_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model.eval(), tokenizer


# --------------------------------------------------------------------------- #
# Metric implementation                                                       #
# --------------------------------------------------------------------------- #
@SingleTurnOrChatMetric.register_metric("rm_score")
class RewardModelMetric(BaseMetric):
    """
    Computes a scalar reward for an entire chat using a seq-classification model.

    Signature usage examples
    ------------------------
    • `SingleTurnOrChatMetric("rm_score")`
    • `SingleTurnOrChatMetric("document->rm_score", model_name="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")`
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        **hf_kwargs,
    ):
        self.model, self.tokenizer = _load_reward_model(model_name, device, dtype, **hf_kwargs)

    # --------------------------------------------------------------------- #
    def score(
        self,
        prompt: str,
        groundtruth: str,
        completion: Optional[str],
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        if messages is None:
            raise ValueError("`messages` must be provided for rm_score.")

        # Tokenise with chat template → tensor on same device as model.
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            score = self.model(input_ids).logits[0][0].item()

        return float(score)


if __name__ == "__main__":
    # Example usage
    metric = RewardModelMetric(
        model_name="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        attn_implementation="flash_attention_2",
        num_labels=1
    )
    result = metric.score(
        prompt="What is the capital of France?",
        groundtruth="Paris",
        completion=None,
        messages=[
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
    )
    print(result)
