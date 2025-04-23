from collabllm.utils.token_count import num_tokens_from_string

class TokenAmount(object):

    def __init__(self, encoding_name="cl100k_base"):
        self.encoding_name = encoding_name

    def __call__(self, chat_eval, verbose=False, **kwargs): 
        '''
        Args:
            chat_eval: List[dict]
                List of dictionaries with keys 'role' and 'content'
                Example: chat_eval = [{'role': 'user', 'content': 'Hello!'}, 
                                     {'role': 'assistant', 'content': 'Hi!'}, ...]

        Returns:
            result: dict
                A dictionary with keys 'num_tokens_read' and 'num_tokens_typed' representing
                the number of tokens in the assistant's responses and the user's chat_eval respectively.
        '''
        assistant_content = ""
        user_content = ""

        for message in chat_eval:
            if message['role'] == 'assistant':
                assistant_content += message['content'] + " "
            elif message['role'] == 'user':
                user_content += message['content'] + " "

        assistant_tokens = num_tokens_from_string(assistant_content, self.encoding_name) / 1000.
        user_tokens = num_tokens_from_string(user_content, self.encoding_name) / 1000.

        if verbose:
            print(f"Total assistant tokens: {assistant_tokens}")
            print(f"Total user tokens: {user_tokens}")

        result = {
            "num_tokens_read(k)": assistant_tokens,
            "num_tokens_typed(k)": user_tokens,
            "num_turns": float(len(chat_eval) // 2)
        }

        if verbose:
            print(f"Result: {result}")

        return result
