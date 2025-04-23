from typing import List
from collabllm.utils.api import get_llm_output
from collabllm.utils.template import chat_template
from collabllm.prompts import LLM_ASSISTANT_PROMPT_COT, LLM_ASSISTANT_PROMPT_ZS, \
    LLM_ASSISTANT_PROMPT_PROACT, LLM_ASSISTANT_PROMPT_PROACT_COT, LLM_ASSISTANT_PROMPT_PROACT_COT_GT


class LLMAssistant(object):
    registered_prompts = {
        'none': None,
        'zero-shot': LLM_ASSISTANT_PROMPT_ZS,
        'cot': LLM_ASSISTANT_PROMPT_COT,
        'proact': LLM_ASSISTANT_PROMPT_PROACT,
        'proact_cot': LLM_ASSISTANT_PROMPT_PROACT_COT,
        'proact_gt_cot': LLM_ASSISTANT_PROMPT_PROACT_COT_GT
    }
    def __init__(self, method='zero-shot', **llm_kwargs):
        """
        Initialize the LLMAssistant model.
        """
        super().__init__()
        self.method = method
        self.prompt_handler = self.registered_prompts[method]
        self.max_new_tokens = llm_kwargs.get('max_new_tokens', 512)
        self.llm_kwargs = llm_kwargs

    def __call__(self, messages: List[dict], **kwargs):
        """
        Forward pass of the LLMAssistant model.
        
        Args:
            messages (List[dict]): A list of message dictionaries with the last message being the user message.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        assert messages[-1]['role'] == 'user'

        if not self.method in ['proact_gt', 'proact_gt_cot']:
            kwargs = {}
            
        if self.method == 'none':
            prompt = messages
            if len(prompt) and prompt[0]['role'] == 'system':
                print('[LLMAssistant] System message detected.')
        else:
            prompt = self.prompt_handler(chat_history=chat_template(messages),
                                         max_new_tokens=self.max_new_tokens,
                                         **kwargs)
        cnt = 0
        while True:
            cnt += 1
            response = get_llm_output(prompt, **self.llm_kwargs)
            if isinstance(response, dict):
                try:
                    keys = response.keys()
                    current_problem = response.pop('current_problem')
                    thought = response.pop('thought')
                    response = response['response']
                    with open('logs/llm_assistant.txt', 'a+') as f:
                        f.write(f'\n\n[LLMAssistant] `current_problem`={current_problem} | `thought`={thought}\n\n')
                    break
                except Exception as e:
                    print(f'[LLMAssistant] {e}')
                    import pdb; pdb.set_trace()
            else:
                break
            if cnt > 5:
                import pdb; pdb.set_trace()
        return response.strip()