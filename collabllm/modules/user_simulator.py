from typing import List
from collabllm.utils.api import get_llm_output
from collabllm.prompts import USER_SIMULATOR_PRONPTS
from collabllm.utils.template import chat_template
from collabllm.models import get_meta_info_from_model_name, is_api_model_auto
from transformers import AutoTokenizer
from rich import print


class UserSimulator(object):
    def __init__(self, task_name, single_turn_data, **llm_kwargs):
        """
        Initialize the UserSimulator model.
        """
        super().__init__()
        self.task_name = task_name
        self.prompt_handler = USER_SIMULATOR_PRONPTS[task_name]
        self.question = [reply['content'] for reply in single_turn_data if reply['role'] == 'user'][0]
        self.answer = [reply['content'] for reply in single_turn_data if reply['role'] == 'assistant'][0]
        self.llm_kwargs = llm_kwargs
        if is_api_model_auto(llm_kwargs['model']):
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_kwargs['model'])

    def __call__(self, messages: List[dict]):
        if len(messages) and messages[0]['role'] == 'system':
            messages = messages[1:]
        
        prompt = self.prompt_handler(question=self.question,
                                     answer=self.answer,
                                     chat_history=chat_template(messages)
                                     )
        if self.tokenizer is not None:
            chat = [{'content': prompt, 'role': 'system'}]
            meta_info = get_meta_info_from_model_name(self.llm_kwargs['model'])
            prompt = self.tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
            prompt = prompt + meta_info['response_template'] + "\n\n**user**: "
        
        cnt = 0
        while True:
            cnt += 1
            response = get_llm_output(prompt, **self.llm_kwargs)
            if isinstance(response, dict):
                try:
                    keys = response.keys()
                    current_answer = response.pop('current_answer')
                    thought = response.pop('thought')
                    response = response['response']
                    with open('logs/user_simulator.txt', 'a+') as f:
                        f.write(f'\n\n[UserSimulator] `current_answer`={current_answer} | `thought`={thought}\n\n')
                    break
                except Exception as e:
                    print(f'[UserSimulator] {e}')
            else:
                break
            if cnt > 5:
                import pdb; pdb.set_trace()
        return response.strip()
