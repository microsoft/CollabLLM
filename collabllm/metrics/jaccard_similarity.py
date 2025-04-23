import tiktoken
from collabllm.metrics.multiturn_metric import MultiturnMetric


class JaccardSimilarity(MultiturnMetric):
    def __init__(self, encoding_name="cl100k_base", **llm_kwargs):
        super().__init__()
        """
        Args:
            encoding_name: str, name of the token encoding to use (default: "cl100k_base").
            llm_kwargs: additional arguments for the language model.
        """
        self.encoding_name = encoding_name
        self.llm_kwargs = llm_kwargs
        self.llm_kwargs['json_object'] = True
        self.encoding = None

    def _load_if_needed(self):
        """Load the encoding if it hasn't been loaded yet."""
        if self.encoding is None:
            self.encoding = tiktoken.get_encoding(self.encoding_name)

    def _tokenize(self, text):
        """
        Tokenize the input text using the tiktoken encoding.
        Args:
            text: str, the input text to tokenize.
        Returns:
            set: a set of tokens from the input text.
        """
        self._load_if_needed()
        return set(self.encoding.encode(text.lower(), disallowed_special=()))

    def __call__(self, single_turn_data, chat_eval, final_answer=None, **kwargs):
        '''
        Args:
            single_turn_data: List[dict], contains the reference text.
            chat_eval: List[dict], contains the generated document for evaluation.

        Returns:
            dict: a dictionary containing the Jaccard similarity and the token sets for both answers.
        '''
        # Generate final answer using the provided prompt
        if final_answer is None:
            final_answer = self.extract_final_answer(chat_eval)

        reference_answer = single_turn_data[-1]['content']
        
        # Tokenize both documents
        final_answer_tokens = self._tokenize(final_answer)
        reference_answer_tokens = self._tokenize(reference_answer)
        
        # Compute Jaccard Similarity
        intersection = final_answer_tokens.intersection(reference_answer_tokens)
        union = final_answer_tokens.union(reference_answer_tokens)
        jaccard_similarity = len(intersection) / len(union) if union else 0.0

        return {
            'jaccard_similarity': jaccard_similarity,
            'final_answer': final_answer 
        }
