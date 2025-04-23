import json
from collections import OrderedDict
from datasets import load_dataset
from collabllm.datasets.dataset import ChatDataset


class MATH(ChatDataset):

    def __init__(self, repo_id='lighteval/MATH-Hard'):
        """
        Initializes the MATH dataset with raw data.

        Parameters:
        raw_data (dict): The raw MATH data to be processed.
        """
        raw_data = load_dataset(repo_id, trust_remote_code=True)
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
        for set_type in raw_data.keys():
            for entry in raw_data[set_type]:
                metadata = {
                    'split': set_type,
                    'level': entry.get('level'),
                    'type': entry.get('type'),
                }
                turns = [
                    {'role': 'user', 'content':  entry.get('problem')},
                    {'role': 'assistant', 'content':  entry.get('solution')}
                ]
                processed_data.append({"metadata": metadata, "chat": turns})

        return processed_data