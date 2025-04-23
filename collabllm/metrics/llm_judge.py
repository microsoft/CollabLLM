from collabllm.utils.api import get_llm_output
from collabllm.prompts import LLM_JUDGE_PROMPTS
from collabllm.utils.template import chat_template
from collabllm.metrics.multiturn_metric import MultiturnMetric


class LLMJudge(MultiturnMetric):
    def __init__(self, rescale_func=lambda x: x/3.0, **llm_kwargs):
        """
        Initialize the LLMJudge model.
        Args:
            task_name: str, the task name
            llm_kwargs: dict, the kwargs for the LLM model
            rescale_func: the rescale function to map the LLM output to a score between 0 and 1
        """
        super().__init__()
        self.rescale_func = rescale_func
        self.task_name = llm_kwargs.pop('task_name')
        self.llm_kwargs = llm_kwargs
        self.llm_kwargs['json_object'] = True
        self.prompt_handler = LLM_JUDGE_PROMPTS[self.task_name]

    def __call__(self, single_turn_data, chat_eval, chat_history, **kwargs):
        
        assert single_turn_data[-2]['role'] == 'user'
        assert single_turn_data[-1]['role'] == 'assistant'

        question = single_turn_data[-2]['content']
        answer = single_turn_data[-1]['content']
        
        if chat_history is None:
            chat_history = []
        
        prompt = self.prompt_handler(question=question, answer=answer,
                                     chat=chat_template(chat_eval),
                                     chat_history=chat_template(chat_history))
        print(prompt)
        while True:
            response = get_llm_output(prompt, **self.llm_kwargs)
            if response == ' ':
                return None
            if self.prompt_handler.check_format(response):
                break
            else:
                print('[LLMJudge] Format check failed.')
        
        # recale numerical values in response to a score between 0 and 1
        for key in response:
            if key != 'accuracy' and isinstance(response[key]['score'], (int, float)):
                response[key]['score'] = self.rescale_func(response[key]['score'])
        return response