from typing import List
import logging
import litellm

from collabllm import ENABLE_COLLABLLM_LOGGING
from collabllm.utils.template import parse_messages
from collabllm.utils.extract_json_reliable import extract_json
from collabllm.prompts import SYSTEM_PROMPT, PROACT_MODEL_PROMPT

logger = logging.getLogger(__name__)


class LLMCollaborator(object):
    registered_prompts = {
        'none': None,
        'proact': PROACT_MODEL_PROMPT
    }
    def __init__(self, method='none', num_retries=10, **llm_kwargs):
        """
        Initialize the LLMAssistant model.
        """
        super().__init__()
        self.method = method
        assert method in self.registered_prompts, f"Prompting method {method} not registered. Available methods: {list(self.registered_prompts.keys())}"

        self.num_retries = num_retries
        self.llm_kwargs = {"temperature": 0.8, "max_tokens": 2048, **llm_kwargs}

    def __call__(self, messages: List[dict], **kwargs):
        """
        Forward pass of the LLMAssistant model.
        
        Args:
            messages (List[dict]): A list of message dictionaries with the last message being the user message.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        assert messages[-1]['role'] == 'user'

        if self.method == 'none':
            if len(messages) and messages[0]['role'] == 'system':
                logger.info('System message detected.')
        else:
            kwargs = {}
            prompt = PROACT_MODEL_PROMPT.format(
                chat_history=parse_messages(messages, strip_sys_prompt=True),
                max_new_tokens=self.llm_kwargs.get('max_new_tokens', 1024),
                additional_info=kwargs.get('additional_info', '')
            )
            messages = [{"role": "user", "content": prompt}]

        for _ in range(self.num_retries):
            full_response = litellm.completion(
                **self.llm_kwargs,
                messages=messages,
                num_retries=self.num_retries
            ).choices[0].message.content
            
            try:
                if isinstance(full_response, str) and not (self.method == 'none'):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.error(f"[LLMCollaborator] Error extracting JSON: {e}")
                continue
            
            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {'current_problem', 'thought', 'response'}.issubset(keys):
                    response = full_response.pop('response')
                    break
                else:
                    logger.error(f"[LLMCollaborator] Keys {keys} do not match expected keys. Retrying...")
                    continue
            else:
                response = full_response
                break
                
        return response.strip()

