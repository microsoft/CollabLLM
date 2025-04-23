from collabllm.prompts import EXTRACT_ANSWER
from collabllm.utils.api import get_llm_output
from collabllm.utils.template import chat_template


class MultiturnMetric(object):
    def __call__(single_turn_data, 
                 chat_eval, 
                 chat_history=None, **kwargs):
        self.llm_kwargs = {}
        '''
        This general class call takes in single turn data and messages and returns the multiturn metric.
        Args:
            single_turn_data: List[Dict[str, str]], following the format:
                [{'role': 'user', 'content': <a question or task description>}, 
                 {'role': 'assistant', 'content': <the task completion or unit tests for code generation>}]
            chat_eval: List[Dict[str, str]], conversation to be evaluated
            chat_history: List[Dict[str, str]], conversation history (to provide context for the evaluation)
            kwargs: dictionary of keyword arguments
        Returns:
            multiturn_metric: Dict[str, float]
        '''
        raise NotImplementedError
        
    def extract_final_answer(self, chat_eval):
        '''
        Extract the final answer from the chat_eval.
        Args:
            chat_eval: List[Dict[str, str]], conversation to be evaluated
        Returns:
            final_answer: str
        '''

        prompt = EXTRACT_ANSWER.format(chat=chat_template(chat_eval))
        output = get_llm_output(prompt, **self.llm_kwargs)
        return output["final_answer"]
