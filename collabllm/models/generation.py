import os
import copy
from typing import Optional
import warnings
from tqdm import tqdm
from vllm import LLM
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams


from collabllm.modules import LLMAssistant, UserSimulator
from collabllm.models import get_meta_info_from_model_name, is_api_model_auto
from collabllm.utils.api_lib.huggingface import generation_pipeline_hf
from collabllm.utils.template import chat_template


def run_one_chat_session(
    task_name,
    single_turn_data,
    chat_history=None,
    prompt_method="proact",
    is_api_model="auto",
    local_model=None,
    local_tokenizer=None,
    vllm_base_model: Optional[LLM] = None,
    max_new_turns=None,
    verbose=False,
    assistant_generation_kwargs={},
    user_generation_kwargs={},
):
    """
    Run a single chat session with the given single_turn_data and chat_history.

    Args:
        single_turn_data (list): A list of two dictionaries, where the first dictionary
            represents the user's message and the second dictionary represents the assistant's response.
        chat_history (list, optional): A list of dictionaries representing the chat history. Defaults to None.
        prompt_method (str, optional): The prompting method to use. Defaults to 'proact'.
        is_api_model (str, optional): Flag to indicate if the assistant model is an API model. Defaults to 'auto'.
        local_model (str, optional): The model to use for local generation. Defaults to None.
        local_tokenizer (str, optional): The local_tokenizer to use for local generation. Defaults to None.
        max_new_turns (int, optional): The maximum number of new turns to generate. Defaults to None.
        verbose (bool, optional): Flag to indicate if the chat should be printed. Defaults to False.
        assistant_generation_kwargs (dict, optional): Additional keyword arguments for the assistant generation. Defaults to {}. Example: {'model_name': 'gpt-4o', 'temperature': 0.5, 'max_new_tokens': 1024}
        user_generation_kwargs (dict, optional): Additional keyword arguments for the user generation. Defaults to {}.

    Returns:
        list: A list of dictionaries representing the chat history.

    """
    if chat_history is None:
        chat_history = []
    assert single_turn_data[0]['role'] == 'user', "First role must be 'user'"
    assert single_turn_data[1]['role'] == 'assistant', "Second role must be 'assistant'"
    
    user = UserSimulator(task_name=task_name,
                         single_turn_data=single_turn_data, 
                         **user_generation_kwargs)

    chat = copy.copy(chat_history)
    cur_role = 'assistant' if chat_history and chat_history[-1]['role'] == 'user' else 'user'
    exit_flag = False

    for _ in tqdm(range(2 * (max_new_turns or 1)), desc="Generating chat", disable=not verbose):
        if cur_role == 'assistant':
            response = generate_assistant_response(
                is_api_model,
                prompt_method,
                chat,
                local_model,
                local_tokenizer,
                vllm_base_model,
                **assistant_generation_kwargs,
            )
        elif cur_role == 'user':
            response = user(chat)
        if os.environ.get('RANK') == '0' and verbose:
            print('*' * 75, f'\n**{cur_role}**: {response}\n', '*' * 75)
        
        if check_for_termination(response):
            exit_flag = True

        chat.append({'role': cur_role, 'content': response})
        cur_role = 'user' if cur_role == 'assistant' else 'assistant'

        if exit_flag:
            break

    return chat


def generate_assistant_response(
    is_api_model,
    prompt_method,
    chat,
    local_model=None,
    local_tokenizer=None,
    vllm_base_model: Optional[LLM] = None,
    **generation_kwargs,
):
    """Generate a response from the assistant model."""

    assistant_model_name = generation_kwargs['model']
    
    if is_api_model == 'auto':
        is_api_model = is_api_model_auto(assistant_model_name)
    
    if is_api_model:
        assistant = LLMAssistant(method=prompt_method, **generation_kwargs)
        return assistant(chat)
    else:
        if local_model is None or local_tokenizer is None:
            raise ValueError("Both model and local_tokenizer must be provided for local generation.")

        meta_info = get_meta_info_from_model_name(assistant_model_name)
        generation_kwargs.pop("model", "NA")

        if vllm_base_model is not None:
            RUN_USER_DIR = f"/run/user/{os.getuid()}"
            if not os.path.exists(RUN_USER_DIR):
                raise FileNotFoundError(
                    f"Unable to locate temporary directory {RUN_USER_DIR}"
                )

            # Create PEFT directory if it doesn't exist
            PEFT_DIR = os.path.join(
                RUN_USER_DIR, f"{assistant_model_name}-peft-checkpoint"
            )
            if not os.path.exists(PEFT_DIR):
                os.makedirs(PEFT_DIR)

            # Store the local_model as PEFT checkpoint
            local_model.save_pretrained(PEFT_DIR)

            return vllm_base_model.chat(
                messages=chat,
                sampling_params=convert_to_sampling_params(generation_kwargs),
                lora_request=LoRARequest("interactive_adapter", 1, PEFT_DIR),
            )

        else:
            local_tokenizer.padding_side = "left"
            local_tokenizer.pad_token = local_tokenizer.eos_token
            # local_tokenizer.chat_template = meta_info['chat_template'] # only for mistral

            return generation_pipeline_hf(
                chat,
                local_model,
                local_tokenizer,
                stop_sequence=meta_info["ignore_sequence"],
                **generation_kwargs,
            )


def check_for_termination(response):
    """Check if the response contains a termination signal."""
    try:
        return '[[TERMINATE CHAT]]' in response
    except Exception as e:
        warnings.warn(f"Error checking for chat termination: {e}")
        return False

def convert_to_sampling_params(generation_kwargs: dict) -> SamplingParams:
    """Convert generation kwargs to vllm SamplingParams."""

    # Valid sampling parameter keys from SamplingParams class
    valid_params = {
        "n",
        "best_of",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "stop",
        "stop_token_ids",
        "bad_words",
        "ignore_eos",
        "max_tokens",
        "min_tokens",
        "logprobs",
        "prompt_logprobs",
        "detokenize",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "truncate_prompt_tokens",
    }

    # Filter valid params and log unmapped ones
    sampling_kwargs = {}
    for key, value in generation_kwargs.items():
        if key in valid_params:
            sampling_kwargs[key] = value
        else:
            print(
                f"Warning: Parameter '{key}' not found in VLLM-supported sampling parameters"
            )

    # Create SamplingParams object
    return SamplingParams.from_optional(**sampling_kwargs)