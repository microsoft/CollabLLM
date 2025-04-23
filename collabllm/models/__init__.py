import os.path as osp


def get_meta_info_from_model_name(model_name):
    meta_info  = {}
    if 'mistral' in model_name.lower():
        meta_info['response_template'] = "[/INST]"
        meta_info['instruction_template'] = "[INST]"
        meta_info['chat_template_name'] = 'mistral-instruct.jinja'
        meta_info['ignore_sequence'] = ['[INST]', '[/INST]', '<s>', '</s>']
        meta_info['stop_sequence'] = ['</s>']
        meta_info['eos_token'] = '</s>'
        
    elif 'llama-3' in model_name.lower():
        meta_info['response_template'] = '<|start_header_id|>assistant<|end_header_id|>'
        meta_info['instruction_template'] = '<|start_header_id|>user<|end_header_id|>'
        meta_info['chat_template_name'] = 'llama-3-instruct.jinja'
        meta_info['ignore_sequence'] = ['<|start_header_id|>', '<|end_header_id|>', '<|begin_of_text|>', '<|eot_id|>']
        meta_info['stop_sequence'] = ['<|eot_id|>']
        meta_info['eos_token'] = '<|eot_id|>'

    else:
        raise ValueError(f"Model name {model_name} not registered. Please add here.")
    
    # load chat template
    cur_dir = osp.dirname(osp.abspath(__file__))
    root = osp.join(cur_dir, '..')
    chat_template = open(osp.join(root, f'chat_templates/{meta_info["chat_template_name"]}')).read()
    meta_info['chat_template'] = chat_template.replace('    ', '').replace('\n', '')

    return meta_info

def is_api_model_auto(model_name):
    return 'gpt' in model_name or 'claude' in model_name

def is_unsloth_model_auto(model_name):
    return 'unsloth' in model_name

def is_base_model_auto(model_name):
    return 'meta-llama' in model_name or 'mistralai' in model_name or 'unsloth' in model_name

unsloth_fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",          
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             
]