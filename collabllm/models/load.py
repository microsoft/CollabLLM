import os
import torch
from peft import PeftModel, PeftConfig, get_peft_model
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def is_unsloth_model_auto(model_name):
    return 'unsloth' in model_name

def is_base_model_auto(model_name):
    return 'meta-llama' in model_name or 'mistralai' in model_name or 'unsloth' in model_name


def load_model_and_tokenizer(model_name, 
                             max_new_tokens, 
                             peft_config=None,
                             model_class=AutoModelForCausalLM,
                             is_eval=True,
                             device=None):
    if device is None:
        local_rank = os.getenv("LOCAL_RANK")
        device = "cuda:" + str(local_rank)
    
    if is_unsloth_model_auto(model_name):
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_new_tokens,
            dtype=None,
            use_cache=False,
            load_in_4bit=True
            )
        if is_eval:
            FastLanguageModel.for_inference(model)
        model = model.to(device)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.bfloat16
            )
        if is_base_model_auto(model_name):
            print(f'[load_model_and_tokenizer] Loading {model_class}: {model_name}')
            if model_class == AutoModelForCausalLM:
                model = model_class.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_cache=False,
                    device_map={'': device},
                    quantization_config=bnb_config,
                    # is_trainable=not eval
                )
                if peft_config is not None:
                    model = get_peft_model(model, peft_config)
            elif model_class == AutoModelForCausalLMWithValueHead:
                assert peft_config is not None
                model = model_class.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        device_map={'': device},
                        peft_config=peft_config,
                        quantization_config=bnb_config,
                        is_trainable=not eval
                    )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        else:
            print(f'[load_model_and_tokenizer] Load peft model from local {model_name}')
            config = PeftConfig.from_pretrained(model_name)
            if model_class == AutoModelForCausalLM:
                base_model = model_class.from_pretrained(config.base_model_name_or_path, 
                                                         device_map={'': device},
                                                         quantization_config=bnb_config)
                model = PeftModel.from_pretrained(base_model, model_name, is_trainable=not eval)

            if model_class == AutoModelForCausalLMWithValueHead:
                model = model_class.from_pretrained(model_name,
                                                    device_map={'': device},
                                                    # local_files_only=True, # need to disable for azure
                                                    quantization_config=bnb_config,
                                                    is_trainable=not eval
                                                    )

            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    if is_eval:
        tokenizer.padding_side = 'left'
    else:
        tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    print(f'[load_model_and_tokenizer] Set default padding side to {tokenizer.padding_side}')
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())
    n_trainable_params, n_params = count_parameters(model)

    print(f'[load_model_and_tokenizer] Number of trainable parameters: {n_trainable_params}'
            f' / {n_params} ({n_trainable_params / n_params * 100.:.2f}%)')
    return model.eval(), tokenizer
