import bert_score
from collabllm.metrics.multiturn_metric import MultiturnMetric


class BertScore(MultiturnMetric):

    def __init__(self, bert_score_model='microsoft/deberta-xlarge-mnli', **llm_kwargs):
        super().__init__()
        self.bert_score_model = bert_score_model
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
        P, R, F1 = bert_score.score([final_answer], [reference_answer], lang='en', model_type=self.bert_score_model)
        P_rescale, R_rescale, F1_rescale = bert_score.score([final_answer], [reference_answer], lang='en', 
                                                            model_type=self.bert_score_model, 
                                                            rescale_with_baseline=True)

        eval_res = {
            'bert_score': F1.item(),

            'bert_score_f1': F1.item(),
            'bert_score_precision': P.item(), 
            'bert_score_recall': R.item(),

            'bert_score_f1_rescale': F1_rescale.item(),
            'bert_score_precision_rescale': P_rescale.item(), 
            'bert_score_recall_rescale': R_rescale.item(), 

            'final_answer': final_answer
            }
        print(eval_res)
        return eval_res
        
        