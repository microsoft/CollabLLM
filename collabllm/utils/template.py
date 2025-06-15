def parse_messages(messages, strip_sys_prompt=True):
    '''
    Args:
        messages: List[dict]
            List of dictionaries with keys 'role' and 'content'
            Example: messages = [{'role': 'user', 'content': 'Hello!'}, 
                                 {'role': 'assistant', 'content': 'Hi!'}, ...]
    '''
    if messages is None: return ''

    if strip_sys_prompt:
        messages = strip_system_prompt(messages)
    
    chat = "\n".join(
        f"**{m['role'].capitalize()}**: {m['content']}" for m in messages
    )

    return chat

def strip_system_prompt(messages):
    '''
    Args:
        messages: List[dict]
            List of dictionaries with keys 'role' and 'content'
            Example: messages = [{'role': 'user', 'content': 'Hello!'}, 
                                 {'role': 'assistant', 'content': 'Hi!'}, ...]
    '''
    return [msg for msg in messages if msg['role'] != 'system']
