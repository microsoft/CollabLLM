import abc
import logging
from typing import Any, Dict, List, Optional, Tuple

import litellm
from collabllm.prompts import EXTRACT_MULTITURN_COMPLETION_PROMPT
from collabllm.utils.template import parse_messages, strip_system_prompt
from collabllm.utils.extract_json_reliable import extract_json

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. Abstract interface                                                       #
# --------------------------------------------------------------------------- #
class BaseMetric(abc.ABC):
    """Every metric must implement `score` and declare the keys it returns."""

    @abc.abstractmethod
    def score(  # noqa: D401  (imperative mood is OK here)
        self,
        prompt: str,
        groundtruth: str,
        completion: str,
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Compute the metric(s) for a prompt–completion pair."""


# --------------------------------------------------------------------------- #
# 2. Generic driver                                                           #
# --------------------------------------------------------------------------- #
class SingleTurnOrChatMetric:
    """
    A wrapper that (optionally) turns a multi-turn chat log into a *final
    completion*, then runs a concrete metric on the resulting text pair.

    The *signature* string is inspired by DSPy:

        "<extract_type>-><metric_name>"   e.g.  "document->bert_score"
        "<metric_name>"                   e.g.  "toxicity"

    • If `->` is present we first extract a `<extract_type>` artefact
      (document, answer, policy, …) from the full history via an LLM call.
    • In either case we then run `<metric_name>` to obtain numeric scores.
    """

    # Registry so users can add metrics with a one-liner
    _METRIC_REGISTRY: Dict[str, type[BaseMetric]] = {}

    def __init__(self, signature: str, **llm_kwargs: Any):
        self.extract_type, self.metric_name = self._parse_signature(signature)
        self.llm_kwargs = llm_kwargs

        try:
            metric_cls = self._METRIC_REGISTRY[self.metric_name]
        except KeyError as e:
            raise ValueError(
                f"Metric '{self.metric_name}' is not registered. "
                f"Available: {list(self._METRIC_REGISTRY)}"
            ) from e

        try:
            self.metric: BaseMetric = metric_cls(**self.llm_kwargs)
        except Exception as e:
            self.metric: BaseMetric = metric_cls()

    # -------------------------- public API --------------------------------- #
    def __call__(  # noqa: D401
        self,
        messages: List[Dict[str, str]],
        single_turn_prompt: str,
        single_turn_completion: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Main entry-point."""
        if self.extract_type:
            completion = self._extract_final_completion(messages, metadata)
        else:
            completion = None

        return self.metric.score(single_turn_prompt, single_turn_completion, completion, messages, metadata)

    # -------------------------- helpers ------------------------------------ #
    @staticmethod
    def _parse_signature(sig: str) -> Tuple[Optional[str], str]:
        return sig.split("->", 1) if "->" in sig else (None, sig)

    def _extract_final_completion(
        self,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Ask an LLM to distil the final artefact from `messages`."""
        prefix_msg = "Addtional requirement:\n" if metadata and "extraction_requirement" in metadata else ""
        prompt = EXTRACT_MULTITURN_COMPLETION_PROMPT.format(
            extract_type=self.extract_type,
            chat_history=parse_messages(messages, strip_sys_prompt=True),
            extraction_requirement=prefix_msg + metadata.get("extraction_requirement", ""),
        )

        response = litellm.completion(
            **self.llm_kwargs, messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

        try:
            payload = (
                extract_json(response)
                if isinstance(response, str)
                else response  # Already parsed
            )
            logger.info("Extractor output: %s", payload)
            if not (
                isinstance(payload, dict)
                and {"thought", "final_completion"} <= payload.keys()
            ):
                raise ValueError("Unexpected keys in extraction payload.")
            return payload["final_completion"]

        except Exception as e:
            logger.error("Failed to extract JSON: %s", e)
            raise RuntimeError(
                "Could not parse extractor output; see logs for details."
            ) from e

    # ---------------------- registration decorator ------------------------- #
    @classmethod
    def register_metric(cls, name: str):
        """Decorator to make `metric_cls` available in the registry."""

        def _decorator(metric_cls: type[BaseMetric]):
            if name in cls._METRIC_REGISTRY:
                logger.warning(
                    f"Overwriting existing metric '{name}' with {metric_cls.__name__}."
                )
            cls._METRIC_REGISTRY[name] = metric_cls
            return metric_cls

        return _decorator

