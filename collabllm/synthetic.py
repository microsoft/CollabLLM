import json
import numpy as np
from typing import Any, Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from collabllm.simulation import ChatSessionSimulator
from collabllm.reward import multiturn_aware_reward
from collabllm.utils.template import strip_system_prompt

import logging
logger = logging.getLogger(__name__)


def generate_multiturn_dataset(
    *,
    task_desc: str,
    single_turn_prompt: str,
    single_turn_completion: str,
    single_turn_metadata: Dict[str, Any],
    metric_names: List[str],
    user_generation_kwargs: Dict[str, Any],
    assistant_generation_kwargs: Dict[str, Any],
    reward_generation_kwargs: Optional[Dict[str, Any]] = None,
    metric_weights: Optional[List[float]] = None,
    proact_prompt_ratio: float = 0.5,
    add_system_prompt_ratio: float = 0.5,
    local_model: Optional[AutoModelForCausalLM] = None,
    local_tokenizer: Optional[AutoTokenizer] = None,
    vllm_base_model: Optional[Any] = None,
    num_candidate_responses: int = 5,
    max_total_turns: int = 10,
    max_new_turns: int = 1,
    num_samples: int = 3,
    max_workers: int = 8,
    max_metric_workers: int = 8,
) -> Dict[str, Any]:
    """
    Generate a synthetic conversation in nested format:
    {
      "single_turn_prompt": ...,
      "single_turn_completion": ...,
      "single_turn_metadata": ...,
      "turns": [
        {
          "prompt": [ ... messages up to user turn ... ],
          "responses": [
            {"completion": ..., "score": ...}, ...
          ]
        },
        ...
      ]
    }
    """
    reward_generation_kwargs = reward_generation_kwargs or {}
    metric_weights = metric_weights or [1.0] * len(metric_names)

    sim = ChatSessionSimulator()
    chat_history: List[Dict[str, str]] = []

    # Shared simulation args
    base_sim_args = {
        "task_desc": task_desc,
        "single_turn_prompt": single_turn_prompt,
        "local_model": local_model,
        "local_tokenizer": local_tokenizer,
        "vllm_base_model": vllm_base_model,
        "assistant_generation_kwargs": assistant_generation_kwargs,
        "user_generation_kwargs": user_generation_kwargs,
    }

    # Nested structure to return
    multiturn_data: Dict[str, Any] = {
        "single_turn_prompt": single_turn_prompt,
        "single_turn_completion": single_turn_completion,
        "single_turn_metadata": single_turn_metadata,
        "turns": [],
    }

    # 1) initial user turn
    first_user_msg = sim.run_chat_simulation(
        **base_sim_args, num_samples=1, chat_history=chat_history, max_new_turns=1, max_workers=1, verbose=False
    )[0][-1]
    chat_history.append(first_user_msg)

    # 2) loop to fill up to max_total_turns
    while len(chat_history) < max_total_turns:
        # a) sample assistant candidates
        candidate_hists = sim.run_chat_simulation(
            **base_sim_args,
            proact_prompt_ratio=proact_prompt_ratio,
            num_samples=num_candidate_responses,
            chat_history=chat_history,
            add_system_prompt_ratio=add_system_prompt_ratio,
            max_workers=max_workers,
            max_new_turns=1,
            verbose=False,
        )
        candidate_completions = [hist[-1]["content"] for hist in candidate_hists]

        # b) score each candidate and record
        turn_prompt = list(chat_history)  # copy up to user turn
        responses_with_scores: List[Dict[str, Any]] = []
        scores: List[float] = []
        for completion in candidate_completions:
            temp_history = chat_history + [{"role": "assistant", "content": completion}]
            rewards, sessions = multiturn_aware_reward(
                **base_sim_args,
                single_turn_completion=single_turn_completion,
                metric_names=metric_names,
                reward_generation_kwargs=reward_generation_kwargs,
                metadata=single_turn_metadata,
                metric_weights=metric_weights,
                chat_history=temp_history,
                max_new_turns=max_new_turns,
                num_samples=num_samples,
                max_workers=max_workers,
                max_metric_workers=max_metric_workers,
                return_details=True,
                verbose=False,
            )
            score = np.array(rewards["MR"]).mean()
            responses_with_scores.append({
                "completion": completion,
                "score": score,
                "sessions": sessions,
                "rewards": rewards,
            })
            scores.append(score)

        logger.info(f"\n\nResponses and scores (Turn {len(chat_history) // 2}):")
        logger.info(
            json.dumps(
                [{"completion": r["completion"], "rewards": {k: np.mean(r["rewards"][k]) for k in r["rewards"]}} for r in responses_with_scores], 
                indent=2)
        )

        multiturn_data["turns"].append(
            {"prompt": strip_system_prompt(turn_prompt.copy()), "responses": responses_with_scores}
        )

        # c) pick best assistant response
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        best_response = responses_with_scores[best_idx]["completion"]
        chat_history.append({"role": "assistant", "content": best_response})

        if len(chat_history) >= max_total_turns:
            break

        # d) select one user response
        # get session with the max number of length
        sessions = sorted(responses_with_scores[best_idx]['sessions'].copy(), key=lambda x: len(x))
        next_user_msg = sessions[-1][len(chat_history)]["content"]
        chat_history.append({"role": "user", "content": next_user_msg})

        if sim._should_terminate_conversation(next_user_msg):
            logger.info("Conversation terminated by user.")
            break
        
    return multiturn_data
