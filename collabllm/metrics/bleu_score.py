from nltk.translate.bleu_score import sentence_bleu
from collabllm.metrics.multiturn_metric import MultiturnMetric


class SentenceBLEU(MultiturnMetric):
    def __init__(self, **llm_kwargs):
        super().__init__()
        self.llm_kwargs = llm_kwargs
        self.llm_kwargs['json_object'] = True

    def __call__(self, single_turn_data, chat_eval, final_answer=None, **kwargs):
        '''
        Args:
            final_answer: str, the generated document
            reference_answer: str, the reference

        Returns:
            dict: a dictionary containing the bert score precision, recall, and f1
        '''
        if final_answer is None:
            final_answer = self.extract_final_answer(chat_eval)
        
        reference_answer = single_turn_data[-1]['content']
        bleu_score = sentence_bleu([reference_answer], final_answer)
        
        return {'bleu_score': bleu_score, 'final_answer': final_answer}   

