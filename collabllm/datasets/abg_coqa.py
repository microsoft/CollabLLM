import json
import copy
import os.path as osp
from collections import OrderedDict
from datasets import load_dataset
from collabllm.datasets.dataset import ChatDataset


class AbgCoQA(ChatDataset):

    def __init__(self, root='data/', add_history=False):
        """
        Initializes the AbgCoQA dataset with raw data.

        Parameters:
        raw_data (dict): The raw AbgCoQA data to be processed.
        """

        self.add_history = add_history
        raw_data = {}
        for split in ['train', 'val', 'test']:
            raw_data_path = osp.join(root, f'abg-coqa/coqa_abg_{split}.json')
            raw_data[split] = json.load(open(raw_data_path, 'r'))
            
        processed_data = self.preprocess(raw_data)
        super().__init__(processed_data)

    def preprocess(self, raw_data):
        """
        Processes the raw AbgCoQA data into the format expected by ChatDataset.

        Parameters:
        raw_data (dict): The raw AbgCoQA data to be processed.

        Returns:
        list: A list of processed chats with metadata.
        """
        processed_data = []
        for set_type, entries in raw_data.items():
            for entry in entries['data']:
                metadata = {
                    'split': set_type,
                    "id": entry.get('id'),
                    "source": entry.get('source'),
                    'ambiguity': entry.get('ambiguity') == 'ambiguous'
                }
                if entry.get('ambiguity') == 'non_ambiguous':
                    if self.add_history:
                        turns = self._process_historical_turns(entry)
                    else:
                        turns = []

                    target_turn = entry['target_turn']
                    turns.append({'role': 'user', 'content': target_turn['question']})
                    turns.append({'role': 'assistant', 'content': entry.get('ambiguity')}) # target_turn['answer']
                    turns[0]['content'] = "Can you help me answer a question about the following story?\n\n" + entry.get('story') + '\n\nMy question is: ' + turns[0]['content']
                    processed_data.append({"metadata": metadata, "chat": turns})

                elif entry.get('ambiguity') == 'ambiguous':
                    # for every possible chat combination, create a new chat
                    if self.add_history:
                        turns = self._process_historical_turns(entry)
                    else:
                        turns = []
                    clarification_turn, target_turn = entry['clarification_turn'], entry['target_turn']
                    turns.append({'role': 'user', 'content': target_turn['question']})
                    turns.append({'role': 'assistant', 'content': entry.get('ambiguity')}) # clarification_turn['question']
                    turns[0]['content'] = "Can you help me answer a question about the following story?\n\n" + entry.get('story') + '\n\nMy question is: ' + turns[0]['content']

                    # for clar_turn in clarification_turn['answers']:
                    #     new_turn = copy.deepcopy(turns)
                    #     new_turn.append({'role': 'user', 'content': clar_turn['clr_ans']})
                    #     new_turn.append({'role': 'assistant', 'content': clar_turn['org_ans']})
                    #     processed_data.append({"metadata": metadata, "chat": new_turn})
                    processed_data.append({"metadata": metadata, "chat": turns})
        return processed_data

    def _process_historical_turns(self, entry):
        """
        Processes history and target turns for non-ambiguous entries.

        Parameters:
        entry (dict): A single entry from the raw data.

        Returns:
        OrderedDict: Processed turns.
        """
        turns = []
        history_turns = entry['history_turns']
        history_turns = sorted(history_turns, key=lambda x: x['turn_id'])
        for i, turn in enumerate(history_turns):
            turns.append({'role': 'user', 'content': turn['question']})
            turns.append({'role': 'assistant', 'content': turn['answer']})
        return turns

