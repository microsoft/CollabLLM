import os
import os.path as osp

from datasets import load_dataset
from tqdm import tqdm

from collabllm.datasets.dataset import ChatDataset
from collabllm.utils.token_count import num_tokens_from_string
from collabllm.utils.api import get_llm_output
import json

class Medium(ChatDataset):

    def __init__(self, 
                 repo_id='Kamaljp/medium_articles', 
                 train_ratio=0.020, 
                 test_ratio=0.005, 
                 max_tokens=512,
                #  extract_test_summary=False
                 ):
        """
        Initializes the Medium dataset with raw data.

        Parameters:
        raw_data (dict): The raw Medium data to be processed.
        """
        self.cache_dir = 'outputs/cache'
        os.makedirs(self.cache_dir, exist_ok=True)

        self.test_ratio = test_ratio
        self.train_ratio = train_ratio
        self.max_tokens = max_tokens

        raw_data = load_dataset(repo_id, trust_remote_code=True)
        processed_data = self.preprocess(
            raw_data, 
            # extract_test_summary=extract_test_summary
        )
        super().__init__(processed_data)

    def preprocess(self, 
                   raw_data, 
                   # extract_test_summary=False
                   ):
        """
        Processes the raw Medium data into the format expected by ChatDataset.

        Parameters:
        raw_data (dict): The raw Medium data to be processed.

        Returns:
        list: A list of processed chats with metadata.
        """
        # drop the data with no timestamp
        raw_data['train'] = [entry for entry in raw_data['train'] if entry.get('timestamp') is not None]
        # order the data by timestamp
        raw_data['train'] = sorted(raw_data['train'], key=lambda x: x['timestamp']) 

        # set the train and test ratio in the most recent
        test_size = int(len(raw_data['train']) * self.test_ratio)
        train_size = int(len(raw_data['train']) * self.train_ratio)
        raw_data['test'] = raw_data['train'][-test_size:]
        raw_data['train'] = raw_data['train'][-test_size-train_size:-test_size]

        # need to generate document summary for testing under single-turn setting
        # test_summary_path = osp.join(self.cache_dir, 'medium_summaries.json')
        # exist_summary_cache = osp.exists(test_summary_path)

        summaries = {}
        processed_data = []
        for set_type in raw_data.keys():

            # extract_summary = set_type == 'test' and extract_test_summary
            # if extract_summary and exist_summary_cache:
            #     with open(test_summary_path, 'r') as f:
            #         summaries = json.load(f)

            for entry in tqdm(raw_data[set_type]):
                num_tokens = num_tokens_from_string(entry.get('text'))
                if num_tokens > self.max_tokens:
                    continue
                
                metadata = {
                    'split': set_type,
                    'url': entry.get('url'),
                    'authors': entry.get('authors'),
                    'timestamp': entry.get('timestamp'),
                    'tags': entry.get('tags'),
                    'num_tokens': num_tokens
                }
                # hash_key = str(hash(entry.get('text')))

                # generate summary 
                # if extract_summary and not hash_key in summaries:
                #     summaries[hash_key] = get_llm_output(
                #         f"""
                #         You should summarize the following text in less than 25 words:\n
                #         {entry.get('text')}\n
                #         Please directly output the summary: \n
                #         """, 
                #         model='gpt-4o')
                #     print(summaries[hash_key])

                # prompt = summaries.get(hash_key) if extract_summary else entry.get('title')
                prompt = entry.get('title')
                turns = [
                    {'role': 'user', 'content': 'Please write an article in less than 500 words about: ' + prompt},
                    {'role': 'assistant', 'content': entry.get('text')}
                ]
                processed_data.append({"metadata": metadata, "chat": turns})
        
        # if extract_test_summary:
        #     with open(test_summary_path, 'w') as f:
        #         json.dump(summaries, f, indent=4)

        return processed_data