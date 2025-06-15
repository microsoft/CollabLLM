# sentence_bleu_metric.py
from typing import Any, Dict, List, Optional
from nltk.translate.bleu_score import sentence_bleu

from collabllm.metric import SingleTurnOrChatMetric, BaseMetric


@SingleTurnOrChatMetric.register_metric("bleu")
class BLEUMetric(BaseMetric):
    """
    Corpus-free sentence-level BLEU:
        reference = `groundtruth`
        hypothesis = `completion`
    """

    def score(
        self,
        prompt: str,
        groundtruth: str,
        completion: Optional[str],
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        if completion is None:
            raise ValueError("`completion` must not be None for BLEU scoring.")

        bleu = sentence_bleu([groundtruth], completion)
        return float(bleu)


if __name__ == "__main__":
    # Example usage
    conv_metric = SingleTurnOrChatMetric("document->bleu", model="gpt-4o-mini")
    messages = [
        {"role": "user", "content": "Please write a document about optimism and its benefits."},
        {"role": "assistant", "content": "Optimism is a positive outlook on life..."},
        {"role": "user", "content": "What are the benefits of optimism?"},
        {"role": "assistant", "content": "Optimism can lead to better mental health, increased resilience, and improved physical health."}
    ]
    score = conv_metric(
        messages=messages,
        single_turn_prompt="Write a short essay on the benefits of optimism.",
        single_turn_completion="Embrace optimism for a brighter future. Optimism can lead to better mental health, increased resilience, and improved physical health.",
    )
    print(f"BLEU score: {score}")
