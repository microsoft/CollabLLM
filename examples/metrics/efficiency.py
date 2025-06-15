# token_amount_metric.py
from typing import Any, Dict, List, Optional
import tiktoken

from collabllm.metric import SingleTurnOrChatMetric, BaseMetric


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of (approximated) tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


@SingleTurnOrChatMetric.register_metric("token_amount")
class TokenAmountMetric(BaseMetric):
    """
    Counts assistant / user tokens (in k-tokens) and dialogue turns.
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name

    def score(
        self,
        prompt: str,
        groundtruth: str,
        completion: Optional[str],
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        if messages is None:
            raise ValueError("`messages` must be provided for TokenAmountMetric.")

        assistant_text = " ".join(m["content"] for m in messages if m["role"] == "assistant")
        user_text = " ".join(m["content"] for m in messages if m["role"] == "user")

        full_results =  {
            "num_tokens_read(k)": num_tokens_from_string(assistant_text, self.encoding_name) / 1000.0,
            "num_tokens_typed(k)": num_tokens_from_string(user_text, self.encoding_name) / 1000.0,
            "num_turns": float(len(messages) // 2),
        }
        return full_results["num_tokens_read(k)"]

