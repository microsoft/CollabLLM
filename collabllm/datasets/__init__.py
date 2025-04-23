import copy
import random
import re
import numpy as np
from typing import List

from datasets import load_dataset, Dataset, DatasetDict, interleave_datasets
from collabllm.prompts import SYSTEM_PROMPT

from .abg_coqa import AbgCoQA
from .asqa import ASQA
from .paqa import PAQA
from .ppc import PPC
from .math_hard import MATH
from .medium import Medium
from .mtbp import MTBP
from .humaneval import HumanEval
from .bigcodebench import BigCodeBench
# ADD NEW DATASET IMPORT ABOVE


# ADD NEW DATASET BELOW
datasets_info = {
    'math': {
        'task': 'question-answering',
        'class': MATH,
        'kwargs': {'repo_id': 'lighteval/MATH'}
        },
    'math-hard': {
        'task': 'question-answering',
        'class': MATH,
        'kwargs': {'repo_id': 'lighteval/MATH-Hard'}
    },
    'abg-coqa': {
        'task': 'maybe-ambiguous-qa',
        'class': AbgCoQA,
        'kwargs': {'add_history': False}
    },
    'medium': {
        'task': 'document-editing',
        'class': Medium,
        'kwargs': {},
    },
    'mtbp': {
        'task': 'code-generation',
        'class': MTBP,
        'kwargs': {}
        },
    'humaneval': {
        'task': 'code-generation',
        'class': HumanEval,
        'kwargs': {}
    },
    'bigcodebench': {
        'task': 'code-generation',
        'class': BigCodeBench,
        'kwargs': {}
    },
}

def add_sys_prompt(dataset: DatasetDict, 
                   key: str
                   ) -> DatasetDict:
    """Add a system prompt to each entry in the dataset."""
    def _add_for_example(example):
        if example[key] is None:
            print('\n', example.keys())
        example[key] = [{'role': 'system', 'content': SYSTEM_PROMPT}] + example[key]
        return example

    for split in dataset.keys():
        dataset[split] = dataset[split].map(_add_for_example)
    return dataset


# ADD NEW DATASET HERE
def load_single_turn_dataset(dataset_name: str, 
                             add_system_prompt: bool=True,
                             **kwargs
                             ) -> DatasetDict:
    """Load a single-turn dataset."""
    try:
        dataset = datasets_info[dataset_name]['class'](**datasets_info[dataset_name]['kwargs'], **kwargs)
    except KeyError:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    dataset = dataset.to_hf_dataset()

    if add_system_prompt:
        return add_sys_prompt(dataset, key='chat')
    
    return dataset


def load_dpo_dataset(dataset_name: str, 
                     minimum_gap: float=0.01,
                     add_system_prompt: bool=True,
                     to_sft_dataset: bool=False
                     ) -> DatasetDict:
    """
    Load a multi-turn DPO dataset.
    
    Key fields: idx, prompt, chosen, reject
    """
    dataset = load_dataset(dataset_name, trust_remote_code=True)
    
    if add_system_prompt:
        dataset = add_sys_prompt(dataset, key='prompt')

    single_turn_ds_name = re.search(r'collabllm-([a-zA-Z0-9_-]+)', dataset_name).group(1)
    # single_turn_ds_name = re.search(r'collabllm-(.*?)-dpo', dataset_name).group(1)
    task = datasets_info[single_turn_ds_name]['task']

    if to_sft_dataset:
        single_turn_ds = load_single_turn_dataset(single_turn_ds_name, add_system_prompt=False)

        dataset_dict = {}
        for split in dataset.keys():
            prompts = dataset[split]['prompt']
            last_responses = dataset[split]['chosen']
            indices = dataset[split]['idx']

            sft_lst = [
                prompts[max(subset, key=lambda j: len(prompts[j]))] + 
                [{'role': 'assistant', 'content': last_responses[max(subset, key=lambda j: len(prompts[j]))]}]
                for subset in (np.where(indices == i)[0] for i in np.unique(indices))
            ]
            
            idx_lst = list(np.unique(indices))
            
            qa_lst = [single_turn_ds[split]['chat'][i] for i in idx_lst]
            dataset_dict[split] = Dataset.from_dict({'chat': sft_lst, 'idx': idx_lst, 'qa': qa_lst})
        
        dataset = DatasetDict(dataset_dict)
    else:
        dataset = filter_dpo(dataset, minimum_gap)
    
    return dataset


def filter_dpo(dataset: DatasetDict,
               minimum_gap: float=0.01):
    """Filter DPO dataset based on gap and reward criteria."""
    dataset = copy.deepcopy(dataset)
    gaps = []
    
    for split in dataset:
        selected = []
        for i, example in enumerate(dataset[split]):
            gap = example['chosen_eval']['reward'] - example['rejected_eval']['reward']
            if gap >= minimum_gap:
                selected.append(i)
            gaps.append(gap)

        dataset[split] = dataset[split].select(selected)
    
    print(f'Ratio of gaps >= {minimum_gap}:', len(selected) / len(gaps))
    return dataset


def load_multiturn_dataset(dataset_name: str, 
                           root: str='data',
                           add_system_prompt: bool=True, 
                           is_dpo: bool=False, 
                           **dpo_kwargs):
    """
    Load a multi-turn dataset.
    
    Key fields: chat
    """
    if is_dpo:
        return load_dpo_dataset(dataset_name, 
                                add_system_prompt=add_system_prompt, 
                                **dpo_kwargs)
    
    if dataset_name == 'asqa':
        dataset = ASQA(root=root).to_hf_dataset()
    elif dataset_name == 'paqa':
        dataset = PAQA(root=root).to_hf_dataset()
    elif dataset_name == 'abg_coqa':
        dataset = AbgCoQA(root=root).to_hf_dataset()
    elif dataset_name == 'ppc':
        dataset = PPC().to_hf_dataset()
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    if add_system_prompt:
        return add_sys_prompt(dataset, key='chat')

    return dataset


def split_train_dev_datasets(dataset_names: list,
                             is_multiturn: bool=True,
                             is_dpo: str=False,
                             n_eval_per_dataset: int=None,
                             split_ratio: float=None,
                             probs: List[float]=None,
                             add_system_prompt: bool=False,
                             return_indices: bool=False,
                             return_eval_as_dict: bool=True,
                             seed: int=42,
                             **dpo_kwargs):
    """Split datasets into training and evaluation sets."""
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    
    probs = probs or [1. / len(dataset_names)] * len(dataset_names)

    assert n_eval_per_dataset is not None or split_ratio is not None
    assert is_multiturn if is_dpo else True

    train_datasets, dev_datasets = [], {}

    for dataset_name in dataset_names:

        if is_multiturn:
            full_dataset = load_multiturn_dataset(
                dataset_name, add_system_prompt=add_system_prompt, is_dpo=is_dpo, **dpo_kwargs
            )['train']
            
        else:
            full_dataset = load_single_turn_dataset(dataset_name, add_system_prompt=add_system_prompt)['train']

        n_eval = n_eval_per_dataset if n_eval_per_dataset is not None else int(split_ratio * len(full_dataset))
        
        random.seed(seed)
        eval_indices = random.sample(range(len(full_dataset)), k=min(n_eval, len(full_dataset)))
        train_indices = list(set(range(len(full_dataset))) - set(eval_indices))

        train_datasets.append(full_dataset.select(train_indices))
        dev_datasets[dataset_name] = full_dataset.select(eval_indices)

    train_dataset = interleave_datasets(train_datasets, probabilities=probs, seed=seed)
    
    if not return_eval_as_dict:
        dev_datasets = interleave_datasets(list(dev_datasets.values()), probabilities=probs, seed=seed)
    
    if return_indices:
        return full_dataset, train_indices, eval_indices
    else:
        return train_dataset, dev_datasets
