import torch
import random
from typing import List, Dict, Callable
from collabllm.metrics import metric_info, registered_general_metrics, registered_task_metrics, \
                           BertScore, SentenceBLEU, BLEURTScore, WMDDistance, LLMJudge, TokenAmount


class ChatEvaluator:
    def __init__(self, 
                 task_name, 
                 max_new_tokens=4096,
                 judge_model='gpt-4o', 
                 bert_score_model='microsoft/deberta-xlarge-mnli',
                 wmd_model="word2vec-google-news-300",
                 temperature=0,
                 seed=42,):
        
        self.seed = seed
        self.task_name = task_name

        rescale_func = registered_task_metrics[task_name]["llm_judge_rescale_func"]
        self.all_metrics = registered_general_metrics + \
                           registered_task_metrics[task_name]['llm_metrics'] + \
                           registered_task_metrics[task_name]['others'] + \
                           [registered_task_metrics[task_name]['task_specific']]
        self.metrics = set([metric.split('->')[0] for metric in self.all_metrics])

        # Mapping of metric names to their corresponding evaluator initialization functions
        model = judge_model
        local_variables = locals()
        self.evaluators = {
            metric: metric_info[metric][0](**{arg: local_variables[arg] for arg in metric_info[metric][1]}) for metric in self.metrics
        }

    def evaluate(self, single_turn_data, chat_eval, final_answer=None, task_specific_only=False):
        eval_result = {}

        if task_specific_only:
            metrics = registered_general_metrics + [registered_task_metrics[self.task_name]['task_specific']]
            metrics = set([m.split('->')[0] for m in metrics])
        else:
            metrics = self.metrics

        if 'llm_judge' in metrics:
            judge_eval_results = self.evaluators['llm_judge'](single_turn_data, chat_eval, chat_history=None)
            eval_result['llm_judge'] = judge_eval_results

        for metric, metric_func in self.evaluators.items():
            if metric == 'llm_judge':
                continue
                
            elif metric == 'token_amount':
                eval_result['token_amount'] = metric_func(chat_eval)

            else:
                if final_answer is None:
                    final_answer = metric_func.extract_final_answer(chat_eval)
                eval_result.update(metric_func(single_turn_data, chat_eval, final_answer=final_answer))
        
        return eval_result
