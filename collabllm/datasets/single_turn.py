from datasets import Dataset, DatasetDict
from typing import List, Dict, Any
import random


class SingleTurnDataset:
    """A dataset wrapper for single-turn chat data with HuggingFace integration."""
    
    def __init__(self, data: List[Dict[str, Any]], eval_ratio: float = 0.1, seed: int = 42):
        """
        Initializes the SingleTurnDataset with chat data.

        Args:
            data: A list of dictionaries, each representing a single chat turn.
                  Each dictionary must contain 'prompt' and 'completion' keys,
                  and may contain additional metadata fields.
            eval_ratio: Proportion of data to use for evaluation split when 
                       'split' field is not present in data (default: 0.1).
            seed: Random seed for train/eval splitting (default: 42).
        
        Raises:
            ValueError: If data is empty or missing required fields.
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        # Validate required fields
        required_fields = {'prompt', 'completion'}
        self.fields = set(data[0].keys())
        
        if not required_fields.issubset(self.fields):
            missing = required_fields - self.fields
            raise ValueError(f"Missing required fields: {missing}")
        
        # Validate all entries have consistent keys
        for i, entry in enumerate(data):
            if set(entry.keys()) != self.fields:
                raise ValueError(f"Entry {i} has inconsistent keys. "
                               f"Expected: {self.fields}, Got: {set(entry.keys())}")
        
        self.data = data
        self.eval_ratio = eval_ratio
        self.seed = seed

    def to_hf_dataset(self) -> DatasetDict:
        """
        Converts the dataset to HuggingFace DatasetDict format.
        
        If 'split' field exists in data, uses those splits.
        Otherwise, randomly splits data into train/eval based on eval_ratio.
        
        Returns:
            DatasetDict with train and/or eval splits containing:
            - single_turn_prompt: The input prompts
            - single_turn_completion: The expected completions  
            - metadata: Dictionary of additional fields
        """
        # Check if split information exists
        if 'split' in self.fields:
            splits = [entry['split'] for entry in self.data]
            unique_splits = list(set(splits))
            split_indices = {
                split: [i for i, x in enumerate(splits) if x == split] 
                for split in unique_splits
            }
        else:
            # Create random train/eval split

            random.seed(self.seed)
            eval_size = int(len(self.data) * self.eval_ratio)
            eval_indices = random.sample(range(len(self.data)), k=min(eval_size, len(self.data)))
            train_indices = list(set(range(len(self.data))) - set(eval_indices))
            
            split_indices = {
                'train': train_indices,
                'eval': eval_indices
            }
        
        # Build metadata fields (exclude prompt, completion, and split)
        metadata_fields = self.fields - {'prompt', 'completion', 'split'}
        
        dataset_dict = {}
        for split, indices in split_indices.items():
            if not indices:  # Skip empty splits
                continue
                
            dataset_dict[split] = Dataset.from_dict({
                "single_turn_prompt": [self.data[i]['prompt'] for i in indices],
                "single_turn_completion": [self.data[i]['completion'] for i in indices],
                "single_turn_metadata": [
                    {field: self.data[i][field] for field in metadata_fields}
                    for i in indices
                ]
            })
        
        return DatasetDict(dataset_dict)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a specific chat entry by index.

        Args:
            idx: The index of the chat entry to retrieve.

        Returns:
            The chat entry at the specified index.
        
        Raises:
            IndexError: If index is out of range.
        """
        return self.data[idx]

    def __len__(self) -> int:
        """
        Returns the number of chat entries in the dataset.

        Returns:
            The number of chat entries in the dataset.
        """
        return len(self.data)
    
    def get_splits_info(self) -> Dict[str, int]:
        """
        Returns information about data splits.
        
        Returns:
            Dictionary mapping split names to their sizes.
        """
        if 'split' in self.fields:
            splits = [entry['split'] for entry in self.data]
            split_counts = {}
            for split in set(splits):
                split_counts[split] = splits.count(split)
            return split_counts
        else:
            eval_size = int(len(self.data) * self.eval_ratio)
            return {
                'train': len(self.data) - eval_size,
                'eval': eval_size
            }

