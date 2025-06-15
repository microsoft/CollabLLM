import logging
from typing import Any, Dict, List, Optional

import litellm

from collabllm.metric import SingleTurnOrChatMetric, BaseMetric
from collabllm.utils.extract_json_reliable import extract_json

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompt template                                                             #
# --------------------------------------------------------------------------- #
INTERACTIVITY_PROMPT = '''You are a helpful and meticulous conversation evaluator. \
Your task is to evaluate the *interactivity* of the responses provided by an AI assistant \
to user questions in a given conversation:

<|The Start of the Conversation to be Evaluated|>
{chat_history}
<|The End of the Conversation to be Evaluated|>

You should assess the assistant's engagement, clarity, and ability to understand the user's needs. \
Give a float number between 0 and 1, where:
    1 = Highly interactive: The assistant is very engaging, asks all relevant questions, and significantly enhances understanding and problem-solving.
     - Example: The assistant thoroughly understands the user's question, asks for necessary clarifications, such as "It sounds like you're asking about the causes of climate change. Are you looking for specific examples or a general overview?"
    0.5 = Moderately interactive: The assistant is engaging, asks some relevant questions, but can be substantially improved.
     - Example: The assistant asks some relevant questions about the user's inquiry but misses key details, such as "Are you asking about the effects of climate change?" but does not probe further for clarification.
    0 = Low interactivity: The assistant shows low engagement, asks few relevant questions, and barely try to understand the user's needs.
     - Example: The assistant provides a vague or incomplete response without fully understanding the user's intent, such as "Climate change is bad," without asking any follow-up questions or providing detailed information.


Output format (JSON):
{{
    "thought": "<How interactive is the assistant?>",
    "interactivity": <score>
}}

Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured. Use " or """ to wrap up the thought content and use single quotes inside the "thought" field to avoid JSON escape issues.

Your evaluation:
'''


# --------------------------------------------------------------------------- #
# Metric implementation                                                       #
# --------------------------------------------------------------------------- #
@SingleTurnOrChatMetric.register_metric("interactivity")
class InteractivityMetric(BaseMetric):
    """
    Uses an LLM judge to produce an interactivity score in [-1, 1].
    """

    def __init__(self, num_retries: int = 50, retry_after: int = 60, **llm_kwargs):
        self.num_retries = num_retries
        self.retry_after = retry_after
        # Default to a deterministic model unless overridden.
        self.llm_kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "model": "claude-3-5-sonnet-latest",
            **llm_kwargs,
        }

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
        `prompt`, `groundtruth`, and `completion` are unused here;
        the full conversation in `messages` is what matters.
        """
        if not messages:
            raise ValueError("`messages` must be provided for InteractivityMetric.")

        # ------------------------------------------------------------------ #
        # 1) Build chat history string                                       #
        # ------------------------------------------------------------------ #
        chat_history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in messages
        )

        eval_prompt = INTERACTIVITY_PROMPT.format(chat_history=chat_history)

        logger.debug("Accuracy evaluator prompt:\n%s", eval_prompt)

        for i in range(self.num_retries):
            try:
                full_response = litellm.completion(
                    **self.llm_kwargs, messages=[{"role": "user", "content": eval_prompt}], num_retries=1
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
                if {'thought', 'interactivity'}.issubset(keys):
                    interactivity = full_response.pop('interactivity')
                    break
                else:
                    logger.error(f"Keys {keys} do not match expected keys. Retrying...")
                    continue
        return interactivity
