from typing import List, Dict, Optional
from vllm import LLM
import torch
import numpy as np

from collabllm.models.generation import run_one_chat_session
from collabllm.utils.token_count import num_tokens_from_string
from collabllm.metrics import registered_task_metrics, registered_general_metrics, metric_info, LLMJudge

def get_one_multiturn_reward(
    task_name: str,
    single_turn_data: List[Dict],
    chat_history: List[Dict],
    response: str,
    prompt_method: str = "none",
    window_size: int = 2,
    user_generation_kwargs: dict = {},
    assistant_generation_kwargs: dict = {},
    reward_generation_kwargs: dict = {},
    local_model: torch.nn.Module = None,
    local_tokenizer=None,
    vllm_base_model: Optional[LLM] = None,
    verbose=False,
    compute_llm_metrics=True,
):
    '''
    This function calculates the reward for a response given the chat history.
    Args:
        task_name: str, task name
        single_turn_data: List[Dict], single turn data
        chat_history: List[Dict], chat history
        response: str, response
        window_size: int, window size
        user_generation_kwargs: dict, user generation kwargs
        assistant_generation_kwargs: dict, assistant generation kwargs
        reward_generation_kwargs: dict, reward generation kwargs
        model: model, model
        tokenizer: tokenizer, tokenizer
        verbose: bool, verbose
        compute_llm_metrics: bool, whether to compute llm metrics
        rescale_func: function, rescale function for llm metrics
    Returns:
        llm_reward: float, llm reward
        task_metric: float, task metric value
        total_length: int, total length
        llm_reward: Dict, llm reward info
    '''
    all_metrics = registered_general_metrics + \
                  [registered_task_metrics[task_name]['task_specific']]
    if compute_llm_metrics:
        all_metrics += registered_task_metrics[task_name]['llm_metrics']

    metrics = set([metric.split('->')[0] for metric in all_metrics])
    task_specific_metric = registered_task_metrics[task_name]['task_specific']

    chat = run_one_chat_session(
        task_name=task_name,
        single_turn_data=single_turn_data,
        chat_history=chat_history + [{"role": "assistant", "content": response}],
        prompt_method=prompt_method,
        max_new_turns=window_size,
        is_api_model="auto",
        verbose=verbose,
        user_generation_kwargs=user_generation_kwargs,
        assistant_generation_kwargs=assistant_generation_kwargs,
        local_model=local_model,
        local_tokenizer=local_tokenizer,
        vllm_base_model=vllm_base_model,
    )

    forward_turns = chat[len(chat_history):]
    llm_judge = LLMJudge(task_name=task_name, **reward_generation_kwargs)

    if 'llm_judge' in metrics:
        llm_reward = llm_judge(single_turn_data=single_turn_data, chat_eval=forward_turns, chat_history=chat_history)
        if llm_reward is None:
            return None, None, None, None
        llm_reward['forward_chat'] = forward_turns[1:]
        llm_rewards = [
            llm_reward[key]['score']
            for key in llm_reward
            if 'score' in llm_reward[key] and isinstance(llm_reward[key]['score'], (float, int)) and key != task_specific_metric.split('->')[-1]
        ]
        llm_reward['average_score'] = np.mean(llm_rewards)
    else:
        llm_reward = {'average_score': 0}
    
    if 'llm_judge' in task_specific_metric:
        metric_name = task_specific_metric.split('->')[-1]
        task_metric = llm_reward[metric_name]['score']
    else:
        metric_name = registered_task_metrics[task_name]['task_specific']
        metric_func = metric_info[metric_name][0]
        task_metric = metric_func(**reward_generation_kwargs)(single_turn_data, chat_eval=chat, chat_history=None)[metric_name]

    total_length = num_tokens_from_string('\n'.join([t['content'] for t in forward_turns]))

    return llm_reward['average_score'], task_metric, total_length, llm_reward