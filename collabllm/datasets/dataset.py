from datasets import Dataset, DatasetDict


class ChatDataset:
    def __init__(self, data):
        """
        Initializes the ChatDataset with chat data.

        Parameters:
        data (list): A list of dictionaries, each representing a chat.
                          Each dictionary contains:
                            - metadata: A dictionary containing metadata about the chat.
                            - chat: A list of dictionaries, each representing a reply in the chat.
        """
        self.data = data
        self.convs, self.metadata = self._parse_convs()

    def _parse_convs(self):
        """
        Parses the input chat data into ordered dictionaries and metadata.

        Returns:
        tuple: A tuple containing ordered dictionaries of chats and metadata.
        """
        parsed_convs = []
        metadata = []

        for conv in self.data:
            metadata.append(conv['metadata'])
            parsed_convs.append(conv["chat"])

        return parsed_convs, metadata

    def to_hf_dataset(self):
        # get unique splits
        splits = [data['split'] for data in self.metadata]
        unique_splits = list(set(splits))
        split_indices = [[i for i, x in enumerate(splits) if x == split] for split in unique_splits]
        dataset = DatasetDict({
                split: Dataset.from_dict({"chat": [self.convs[i] for i in indices], 
                                          "metadata": [self.metadata[i] for i in indices]}) 
                                          for split, indices in zip(unique_splits, split_indices)
            })
        return dataset

    def parse_to_paragraph(self, idx):
        """
        Parses a specific chat into a paragraph format.

        Parameters:
        idx (int): The index of the chat.

        Returns:
        str: The chat in a paragraph format.
        """
        conv = self.convs[idx]
        chat_paragraph = "\n".join([f"{reply['role']}: {reply['content']}" for reply in conv])
        return chat_paragraph

    def __getitem__(self, idx):
        """
        Retrieves a specific chat by index.

        Parameters:
        idx (int): The index of the chat to retrieve.

        Returns:
        list: The chat at the specified index.
        """
        return self.convs[idx]

    def __len__(self):
        """
        Returns the number of chats in the dataset.

        Returns:
        int: The number of chats in the dataset.
        """
        return len(self.convs)


if __name__ == "__main__":
    # Example usage
    data = [
        {
            "metadata": {"context": "User is asking for project assistance"},
            0: {'role': 'user', 'content': 'Hello!'},
            1: {'role': 'assistant', 'content': 'Hi there! How can I help you today?'},
            2: {'role': 'user', 'content': 'I need help with my project.'},
            3: {'role': 'assistant', 'content': 'Sure, what kind of help do you need?'}
        },
        {
            "metadata": {"context": "User has a question about their account"},
            0: {'role': 'user', 'content': 'Good morning!'},
            1: {'role': 'assistant', 'content': 'Good morning! How can I assist you?'},
            2: {'role': 'user', 'content': 'I have a question about my account.'}
        }
    ]

    dataset = ChatDataset(data)

    # Accessing chats by index
    print(dataset[0])  # First chat
    print(dataset[1])  # Second chat

    # Getting the context of the first chat
    print(dataset.get_context(0))  # "User is asking for project assistance"

    # Getting the number of chats
    print(len(dataset))  # 2

    # Retrieving the entire first chat
    for reply in dataset.get_conv(0):
        print(f"{reply['role']}: {reply['content']}")

    # Parsing a chat into a paragraph
    paragraph = dataset.parse_to_paragraph(0)
    print(paragraph)
