import os
import zipfile
import urllib.request
from bleurt import score
from collabllm.metrics.multiturn_metric import MultiturnMetric


class BLEURTScore(MultiturnMetric):

    def __init__(self, data_dir="../data", **llm_kwargs):
        super().__init__()

        if not os.path.exists(os.path.join(data_dir, "BLEURT-20")):
            os.makedirs(data_dir, exist_ok=True)
            zip_file = os.path.join(data_dir, "BLEURT-20.zip")
            
            urllib.request.urlretrieve("https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip", zip_file)
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        
        self.llm_kwargs = llm_kwargs
        self.llm_kwargs['json_object'] = True
        self.data_dir = data_dir
        self.scorer = None
    
    def _init_when_needed(self):
        
        if self.scorer is None:
            checkpoint = os.path.join(self.data_dir, "BLEURT-20")
            self.scorer = score.BleurtScorer(checkpoint)

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
        self._init_when_needed()
        scores = self.scorer.score(references=[reference_answer], 
                                   candidates=[final_answer])
        assert isinstance(scores, list) and len(scores) == 1
        return {'bleurt_score': scores[0], 'final_answer': final_answer}   

    
if __name__ == "__main__":
    metric = BLEURTScore(model='gpt-4o')
    chat_eval = [{'role': 'user', 'content': "Please help me write the function from typing import List def has_close_elements(numbers: List[float], threshold: float) -> bool: \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True \"\"\"" },
    {'role': 'assistant', 'content': "I can help you with that. Here is the function you requested: def has_close_elements(numbers: List[float], threshold: float) -> bool: \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True \"\"\" return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i + 1:])"},
    {'role': 'user', 'content': "It doesn't seem right. Can you please check again? Also can you add some comments to the code?"},
    {'role': 'assistant', 'content': "Sure, here is the updated function with comments: def has_close_elements(numbers: List[float], threshold: float) -> bool: \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True \"\"\" return any(abs(a - b) < threshold for j, a in enumerate(numbers) for b in numbers[j + 1:]) # Check if any two numbers are closer than the threshold"}]

    single_turn_data = [{'role': 'user', 'content': "Please help me write the function from typing import List def has_close_elements(numbers: List[float], threshold: float) -> bool: \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True \"\"\"" },
    {'role': 'assistant', 'content': "from typing import List def has_close_elements(numbers: List[float], threshold: float) -> bool: \"\"\" Check if in given list of numbers, are any two numbers closer to each other than given threshold. >>> has_close_elements([1.0, 2.0, 3.0], 0.5) False >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) True \"\"\"\nfor idx, elem in enumerate(numbers): for idx2, elem2 in enumerate(numbers): if idx != idx2: distance = abs(elem - elem2) if distance < threshold: return True return False"}
    ]

    result = metric(single_turn_data, chat_eval)
    print(result)