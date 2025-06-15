# accuracy_metric.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import json
import litellm
from collabllm.metric import SingleTurnOrChatMetric, BaseMetric
from collabllm.utils.extract_json_reliable import extract_json

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompt template                                                             #
# --------------------------------------------------------------------------- #
ACCURACY_PROMPT = '''You are a helpful and meticulous evaluator. Your task is to \
evaluate the *accuracy* of an AI model's answer to a target question. \
You will be given the target question, the ground truth answer, and the model's response.

Provided Information:

<|The Start of Target Question and Ground Truth Answer|>
Target Question: {single_turn_prompt}
Ground Truth Answer: {groundtruth}
<|The End of Target Question and Ground Truth Answer|>

<|The Start of The Model's Response|>
{completion}
<|The End of The Model's Response|>

You should determine whether the model's final response to the target question is \
factually correct and consistent with the provided ground truth.

Rating criteria (binary):
  • 1 = Correct   — the response matches the ground truth.
  • 0 = Incorrect — the response contradicts or misses the ground truth.

Output format (JSON):
{{
    "thought": "<your reasoning here>",
    "accuracy": <0 or 1>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. Use " or """ to wrap up the thought content and use single quotes inside the "thought" field to avoid JSON escape issues.

Your evaluation:
'''


# --------------------------------------------------------------------------- #
# Metric implementation                                                       #
# --------------------------------------------------------------------------- #
@SingleTurnOrChatMetric.register_metric("accuracy")
class AccuracyMetric(BaseMetric):
    """
    Calls an LLM to judge factual accuracy against a provided ground-truth answer.
    """

    def __init__(self, num_retries: int = 50, retry_after: int = 60, **llm_kwargs):
        self.num_retries = num_retries
        self.retry_after = retry_after
        self.llm_kwargs: Dict[str, Any] = {"temperature": 0.0, **llm_kwargs}

    # --------------------------------------------------------------------- #
    def score(
        self,
        prompt: str,
        groundtruth: str,
        completion: Optional[str],
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Parameters
        ----------
        prompt : str
            The target question.
        groundtruth : str
            The reference / correct answer.
        completion : Optional[str]
            The model's answer if already extracted; otherwise `None`.
        messages : Optional[List[dict]]
            Full chat messages. Used to locate the model answer if `completion` is None.
        metadata : dict
            Unused but kept for API symmetry.

        Returns
        -------
        {"accuracy": score}
            `score` is 1.0 for correct, 0.0 for incorrect, −1.0 if evaluation fails.
        """
        # ------------------------------------------------------------------ #
        # 1) Get the model response text                                     #
        # ------------------------------------------------------------------ #
        if completion is None:
            if not messages:
                raise ValueError(
                    "`completion` is None and could not infer an assistant answer "
                    "from `messages`."
                )
            completion_text = json.dumps(messages)
        else:
            completion_text = completion

        # ------------------------------------------------------------------ #
        # 2) Compose the evaluation prompt                                   #
        # ------------------------------------------------------------------ #
        eval_prompt = ACCURACY_PROMPT.format(
            single_turn_prompt=prompt.strip(),
            groundtruth=groundtruth.strip(),
            completion=completion_text.strip(),
        )

        # ------------------------------------------------------------------ #
        # 3) Call the judge LLM                                              #
        # ------------------------------------------------------------------ #

        for i in range(self.num_retries):
            try:
                full_response = litellm.completion(
                    **self.llm_kwargs, messages=[{"role": "user", "content": eval_prompt}], num_retries=self.num_retries
                ).choices[0].message.content
            except Exception as e:
                import time
                time.sleep(self.retry_after)
                logger.error(f"[retry={i + 1}] Error during LLM call: {e}")
                continue

            # ------------------------------------------------------------------ #
            # 4) Parse JSON                                                      #
            # ------------------------------------------------------------------ #
            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.error(f"Error extracting JSON: {e}")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {'thought', 'accuracy'}.issubset(keys):
                    accuracy = full_response.pop('accuracy')
                    break
                else:
                    logger.error(f"Keys {keys} do not match expected keys. Retrying...")
                    continue

        return accuracy
