from typing import List, Dict, Optional
from collabllm.core.multiturn_reward import get_one_multiturn_reward
import concurrent.futures
import numpy as np
import torch
from vllm import LLM


def get_multiturn_rewards(
    task_name: str,
    single_turn_ds: List[List[Dict]],
    chat_histories: List[List[Dict]],
    responses: List[str],
    prompt_method: str = "none",
    max_workers: int = 30,
    num_samples: int = 3,
    window_size: int = 2,
    llm_rw_weight: float = 1,
    task_weight: float = 1,
    cost_weight: float = 1e-3,
    user_generation_kwargs: dict = {},
    assistant_generation_kwargs: dict = {},
    reward_generation_kwargs: dict = {},
    local_model: torch.nn.Module = None,
    local_tokenizer=None,
    vllm_base_model: Optional[LLM] = None,
    verbose=False,
) -> List[torch.Tensor]:
    """
    This function samples multiple chat sessions to caculate the reward for each response.
    Args:
        task_name: str, task name
        single_turn_ds: List[List[Dict]], list of single turn data
        chat_histories: List[List[Dict]], list of chat histories
        responses: List[str], list of responses
        max_workers: int, number of workers
        num_samples: int, number of samples for each response
        window_size: int, window size
        llm_rw_weight: float, weight for llm reward
        task_weight: float, weight for task metric
        cost_weight: float, weight for token cost
        user_generation_kwargs: dict, user generation kwargs
        assistant_generation_kwargs: dict, assistant generation kwargs
        reward_generation_kwargs: dict, reward generation kwargs
        local_model: local_model, local_model
        local_tokenizer: local_tokenizer, local_tokenizer
    Returns:
        rewards: List[torch.Tensor], list of rewards
        reward_logs: List[Dict], list of reward logs
    """
    # assert all of the last chat from chat_history is from user
    assert all([chat[-1]['role'] == 'user' for chat in chat_histories])

    reward_lst, reward_logs = [], []
    
    if prompt_method == 'none':
        assistant_generation_kwargs['json_object'] = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of tasks to run in parallel
        tasks = [
            (
                task_name,
                single_turn_ds[i],
                chat_histories[i],
                responses[i],
                prompt_method,
                window_size,
                user_generation_kwargs,
                assistant_generation_kwargs,
                reward_generation_kwargs,
                local_model,
                local_tokenizer,
                vllm_base_model,
                verbose,
                llm_rw_weight > 0,
            )
            for i in range(len(single_turn_ds))
            for _ in range(num_samples)
        ]
        results = list(executor.map(
            lambda p: get_one_multiturn_reward(*p), tasks
        ))

    # Process results
    for i in range(len(single_turn_ds)):
        llm_rewards, task_metric_scores, total_lengths, llm_reward_samples = [], [], [], {}

        for j in range(num_samples):
            index = i * num_samples + j
            llm_reward, task_metric, total_length, llm_reward_detail = results[index]

            if llm_reward is None:
                continue

            task_metric_scores.append(task_metric)
            llm_rewards.append(llm_reward) # offset 2 to ensure range(0, 1)
            total_lengths.append(total_length)
            llm_reward_samples[str(j)] = llm_reward_detail

        llm_rewards = llm_rw_weight * np.array(llm_rewards)
        token_costs = cost_weight * np.array(total_lengths)
        token_costs[token_costs > 1] = 1 # gated the maximum cost 
        task_metric_scores = task_weight * np.array(task_metric_scores)

        llm_reward_avg = np.mean(llm_rewards).item()
        llm_reward_std = np.std(llm_rewards).item()
        token_cost_avg = np.mean(token_costs).item()
        token_cost_std = np.std(token_costs).item()
        task_metric_avg = np.mean(task_metric_scores).item()
        task_metric_std = np.std(task_metric_scores).item()

        print(llm_rw_weight, np.array(llm_rewards), 
              task_weight, np.array(task_metric_scores),  
              cost_weight, np.array(token_costs))
        rewards = np.array(llm_rewards) + np.array(task_metric_scores) - np.array(token_costs)
        reward_std = np.std(rewards).item()
        reward_avg = torch.Tensor([np.mean(rewards)])

        length_avg = np.mean(total_lengths)
        reward_logs.append({
            'reward': reward_avg.item(),
            'reward_std': reward_std,
            'llm_rw_avg': llm_reward_avg, 
            'llm_rw_std': llm_reward_std,
            'token_cost_avg': token_cost_avg,
            'token_cost_std': token_cost_std,
            'task_metric_avg': task_metric_avg,
            'task_metric_std': task_metric_std,
            'rs': llm_reward_samples,
            'length_avg': length_avg
        })
        reward_lst.append(reward_avg)

    return reward_lst, reward_logs