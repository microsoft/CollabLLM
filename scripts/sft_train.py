import json
import argparse
import wandb
import os
import os.path as osp
from rich import print

from datasets import interleave_datasets
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import LoraConfig
from transformers import TrainingArguments

import random
import sys
sys.path.append('.')
from collabllm.models import get_meta_info_from_model_name, is_unsloth_model_auto
from collabllm.models.load import load_model_and_tokenizer
from collabllm.datasets import split_train_dev_datasets
from collabllm.utils.blob import upload_dir_to_blob
from collabllm.utils.dir import keep_levels


# args run_name name
def parse_args():
    parser = argparse.ArgumentParser()
    def list_of_strings(arg): return arg.split(',')
    def list_of_integers(arg): return [int(x) for x in arg.split(',')]

    parser.add_argument('--datasets', type=list_of_strings, default='ppc') # first will be used for model selection
    parser.add_argument('--probs', type=list_of_integers, default='1')
    parser.add_argument('--assistant_model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct") 
    
    parser.add_argument('--zero_stage', type=int, default=2)

    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--n_eval_per_dataset', type=int, default=100) 
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--save_total_limit', type=int, default=5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=6)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--eval_steps', type=int, default=50)
    
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--resume_ckpt_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="./outputs")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--push_to_hub', action='store_true', help='push to hub')
    parser.add_argument('--push_to_blob', action='store_true', help='push to blob storage')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--hf_org', type=str, default='org_name')
    return parser.parse_args()


args = parse_args()
######################## OUTPUT PATH ########################
dataset_str = '-'.join([d.split('/')[-1] for d in args.datasets])
run_name = args.run_name if args.run_name else \
    f"{args.assistant_model_name.split('/')[-1]}_epoch-{args.num_train_epochs}_lr-{args.learning_rate}"

output_dir = osp.join(args.output_dir, dataset_str, run_name)
os.makedirs(output_dir, exist_ok=True)

if os.environ.get('LOCAL_RANK', '0') == '0':
    with open(osp.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

######################## LOAD DATASETS ########################
train_dataset, eval_datasets = split_train_dev_datasets(args.datasets, 
                                                        is_dpo=True,
                                                        to_sft_dataset=True,
                                                        n_eval_per_dataset=args.n_eval_per_dataset, 
                                                        probs=args.probs, 
                                                        add_system_prompt=True,
                                                        seed=args.seed)

######################## MODEL ########################
model, tokenizer = load_model_and_tokenizer(args.assistant_model_name, 
                                            max_new_tokens=args.max_new_tokens, eval=False)
is_unsloth_model = is_unsloth_model_auto(args.assistant_model_name)
print('padding_side', tokenizer.padding_side)
print('len(tokenizer)', len(tokenizer))
print('pad_token', tokenizer.pad_token)
print('eos_token', tokenizer.eos_token)

# Load model and tokenizer
if is_unsloth_model:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=16,
        lora_dropout=0.1, 
        bias="none",    
        use_gradient_checkpointing="unsloth", # True or "unsloth" for very long context
        random_state=42,
        use_rslora=False,  
        loftq_config=None,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",]
    )
    ds_config = None
    peft_config = None
else:
    ds_config = {
        "zero_optimization": {
            "stage": args.zero_stage, 
            "overlap_comm": False,
            "reduce_bucket_size": "auto",
            "contiguous_gradients": True,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"}, 
        },
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_print": 200,
    }
    # PEFT config
    peft_config = LoraConfig(
        r=32, # 64
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",]
    )
# Args
train_args = SFTConfig(
    output_dir=output_dir,
    logging_steps=1,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    optim="adamw_torch",
    report_to="wandb",
    do_eval=True,
    eval_steps=args.eval_steps, # 50
    save_strategy='epoch',
    eval_strategy="steps",
    group_by_length=True,
    gradient_checkpointing=True,  
    lr_scheduler_type="cosine",
    metric_for_best_model=f"eval_{args.datasets[0]}_loss",
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_train_epochs,
    save_total_limit=args.save_total_limit,
    max_seq_length=args.max_new_tokens,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    run_name=keep_levels(output_dir, 3),
    deepspeed=ds_config, 
    fp16=not is_bfloat16_supported() if is_unsloth_model else False,
    bf16=is_bfloat16_supported() if is_unsloth_model else False
)

log_config = train_args.to_dict() 
if os.environ.get('LOCAL_RANK', '0') == '0':
    wandb.init(
	    project="interactivity", 
	    entity="dsp-team",
        name=keep_levels(output_dir, 3),
	    config=log_config,
        save_code=True,
        job_type='debug' if args.debug else 'train'
    )

######################## PROCESS DATASETS ########################
meta_info = get_meta_info_from_model_name(args.assistant_model_name)
# tokenizer.chat_template = meta_info['chat_template'] # only for mistral
collator = DataCollatorForCompletionOnlyLM(instruction_template=meta_info['instruction_template'],
                                           response_template=meta_info['response_template'],
                                           tokenizer=tokenizer)

def process_dataset(dataset):
    return dataset.map(
        lambda x: {"formatted_chat": tokenizer.apply_chat_template(
               x["chat"], tokenize=False, add_generation_prompt=False)})

train_dataset = process_dataset(train_dataset)
eval_datasets = {k: process_dataset(v) for k, v in eval_datasets.items()}
######################## TRAINING ########################
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_datasets,
    dataset_text_field="formatted_chat",
    data_collator=collator,
    tokenizer=tokenizer,
    peft_config=peft_config,
    args=train_args
)
trainer.model.print_trainable_parameters()
trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)

######################## SAVING ########################
trainer.save_model(output_dir) # save the LoRA adapters
trainer.model.save_pretrained(output_dir) #, save_embedding_layers=True) # save full model
tokenizer.save_pretrained(output_dir) # save tokenizer
if args.push_to_hub:
    surfix = output_dir.split("/")[-1].replace('_', '-')
    trainer.model.push_to_hub(f'{args.hf_org}/sft-model-{surfix}', private=True)
    tokenizer.push_to_hub(f'{args.hf_org}/sft-tokenizer-{surfix}', private=True)

if args.push_to_blob:
    upload_dir_to_blob(output_dir)
wandb.finish()