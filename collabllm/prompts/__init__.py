import os
import os.path as osp
from collabllm.prompts.prompt_handler import PromptHandler
from typing import Union


current_dir = osp.dirname(__file__)

###############################################################################
##        Load System Prompt & Assistant Prompt (for prompting methods)      ##
###############################################################################
with open(osp.join(current_dir, 'llm_assistant', 'system_prompt.txt'), 'r') as f:
    SYSTEM_PROMPT = f.read()

with open(osp.join(current_dir, 'funtional', 'extract_answer.txt'), 'r') as f:
    EXTRACT_ANSWER = f.read()

with open(osp.join(current_dir, 'llm_assistant', 'cot.txt'), 'r') as f:
    LLM_ASSISTANT_PROMPT_COT = PromptHandler(f.read(), input_keys=['chat_history'], output_format=str)

with open(osp.join(current_dir, 'llm_assistant', 'zero_shot.txt'), 'r') as f:
    LLM_ASSISTANT_PROMPT_ZS = PromptHandler(f.read(), input_keys=['chat_history'], output_format=str)
    
with open(osp.join(current_dir, 'llm_assistant', 'proact_with_gt_cot.txt'), 'r') as f:
    LLM_ASSISTANT_PROMPT_PROACT_COT_GT = PromptHandler(f.read(), input_keys=['chat_history', 'question', 'answer', 'max_new_tokens'], output_format=str)

with open(osp.join(current_dir, 'llm_assistant', 'proact_cot.txt'), 'r') as f:
    LLM_ASSISTANT_PROMPT_PROACT_COT = PromptHandler(f.read(), input_keys=['chat_history', 'max_new_tokens'], output_format=str)

with open(osp.join(current_dir, 'llm_assistant', 'proact.txt'), 'r') as f:
    LLM_ASSISTANT_PROMPT_PROACT = PromptHandler(f.read(), input_keys=['chat_history', 'max_new_tokens'], output_format=str)
###############################################################################
##        Load System Prompt & Assistant Prompt (for prompting methods)      ##
###############################################################################
USER_SIMULATOR_PRONPTS = {}
LLM_REWARD_PROMPTS = {}
LLM_JUDGE_PROMPTS = {}

for task in ['question-answering', 'document-editing', 'code-generation']:
    with open(osp.join(current_dir, 'user_simulator_cot', f'{task}.txt'), 'r') as f:
        if task == 'question-answering':
            USER_SIMULATOR_PRONPTS[task] = PromptHandler(f.read(), input_keys=['chat_history', 'question'], output_format=str)
        elif task == 'document-editing':
            USER_SIMULATOR_PRONPTS[task] = PromptHandler(f.read(), input_keys=['chat_history', 'question', 'answer'], output_format=str)
        elif task == 'code-generation':
            USER_SIMULATOR_PRONPTS[task] = PromptHandler(f.read(), input_keys=['chat_history', 'question'], output_format=str)
        else:
            raise NotImplementedError(f"Please declare the prompt for the task: {task}")


    with open(osp.join(current_dir, 'llm_judge', f'{task}.txt'), 'r') as f:
        if task == 'question-answering':
            LLM_JUDGE_PROMPTS[task] = PromptHandler(f.read(), 
                                                    input_keys=['chat_history', 'chat', 'question', 'answer'], 
                                                    output_format={
                                                            'interactivity': {'thought': str, 'score': Union[float, int]},
                                                            'accuracy': {'thought': str, 'score': Union[float, int]}
                                                    })
        elif task == 'document-editing':
            LLM_JUDGE_PROMPTS[task] = PromptHandler(f.read(), 
                                                    input_keys=['chat_history', 'chat'], 
                                                    output_format={
                                                        'interactivity': {'thought': str, 'score': Union[float, int]}
                                                    })
        elif task == 'code-generation':
            LLM_JUDGE_PROMPTS[task] = PromptHandler(f.read(), 
                                                    input_keys=['chat_history', 'chat'], 
                                                    output_format={
                                                        'interactivity': {'thought': str, 'score': Union[float, int]}
                                                    })
        else:
            raise NotImplementedError(f"Please declare the prompt for the task: {task}")


with open(osp.join(current_dir, 'llm_judge', 'maybe-ambiguous-qa.txt'), 'r') as f:
    LLM_JUDGE_PROMPTS['maybe-ambiguous-qa'] = PromptHandler(f.read(), 
                                            input_keys=['question', 'answer', 'chat'], 
                                            output_format={
                                                'clr_or_answer_acc': {'thought': str, 'score': Union[float, int]},
                                            })


with open(osp.join(current_dir, 'llm_judge', 'helpfulness.txt'), 'r') as f:
    LLM_JUDGE_PROMPTS["general"] = PromptHandler(f.read(), 
                                            input_keys=['chat_history', 'chat', 'question'], 
                                            output_format={
                                                'helpfulness': {'thought': str, 'score': Union[float, int]}
                                            })
