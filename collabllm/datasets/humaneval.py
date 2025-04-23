import json
import random
from datasets import load_dataset
from collabllm.datasets.dataset import ChatDataset


class HumanEval(ChatDataset):

    def __init__(self, repo_id='openai/openai_humaneval',
                 train_ratio=0.8):
        """
        Initializes the HumanEval dataset with raw data.

        Parameters:
        raw_data (dict): The raw MATH data to be processed.
        """
        self.train_ratio = train_ratio
        raw_data = load_dataset(repo_id, trust_remote_code=True)['test']
        processed_data = self.preprocess(raw_data)
        super().__init__(processed_data)

    def preprocess(self, raw_data):
        """
        Processes the raw MATH data into the format expected by ChatDataset.

        Parameters:
        raw_data (dict): The raw MATH data to be processed.

        Returns:
        list: A list of processed chats with metadata.
        """
        processed_data = []
        
        splits = ['train'] * int(len(raw_data) * self.train_ratio) + \
                 ['test'] * (len(raw_data) - int(len(raw_data) * self.train_ratio))
        random.seed(42)
        random.shuffle(splits)
        for entry, split in zip(raw_data, splits):
            metadata = {
                'split': split,
            }
            gt = {'dataset': 'humaneval',
                  'task_id': entry.get('task_id'),
                  'test': entry.get('test'),
                  'entry_point': entry.get('entry_point'),
                  'answer': entry.get('canonical_solution'),
                  }
            turns = [
                {'role': 'user', 'content': entry.get('prompt')},
                {'role': 'assistant', 'content': json.dumps(gt)}
            ]
            processed_data.append({"metadata": metadata, "chat": turns})

        return processed_data
    

