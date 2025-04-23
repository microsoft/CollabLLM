import gensim.downloader as api
from gensim.similarities import WmdSimilarity
from collabllm.metrics.multiturn_metric import MultiturnMetric


class WMDDistance(MultiturnMetric):
    def __init__(self, wmd_model="word2vec-google-news-300", **llm_kwargs):
        super().__init__()
        self.wmd_model = wmd_model
        self.llm_kwargs = llm_kwargs
        self.llm_kwargs['json_object'] = True
        self.model = None

    def _load_if_needed(self):
        if self.model is None:
            self.model = api.load(self.wmd_model)

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
        
        # Preprocess both articles
        self._load_if_needed()

        # Compute Word Mover's Distance
        wmdistance = self.model.wmdistance(final_answer.lower().split(), 
                                           reference_answer.lower().split())
        return {'wmd': wmdistance, 'final_answer': final_answer}   
