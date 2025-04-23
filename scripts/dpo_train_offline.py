import json
import argparse
import wandb
import datetime
import os
import os.path as osp
from rich import print

from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

import sys
sys.path.append('.')
from collabllm.datasets import split_train_dev_datasets
from collabllm.utils.distributed import init_distributed_mode
from collabllm.models import get_meta_info_from_model_name, is_unsloth_model_auto
from collabllm.models.load import load_model_and_tokenizer
from collabllm.utils.blob import upload_dir_to_blob
from collabllm.utils.dir import keep_levels


# args run_name name
def parse_args():
    parser = argparse.ArgumentParser()
    def list_of_strings(arg): return arg.split(',')
    def list_of_integers(arg): return [int(x) for x in arg.split(',')]

    parser.add_argument('--datasets', type=list_of_strings, default='ppc')
    parser.add_argument('--probs', type=list_of_integers, default='1')
    parser.add_argument('--assistant_model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct") 
    
    parser.add_argument('--n_eval_per_dataset', type=int, default=200) 
    parser.add_argument('--max_prompt_length', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--save_total_limit', type=int, default=5)
    parser.add_argument('--minimum_gap', type=float, default=0.1)
    
    parser.add_argument('--per_device_train_batch_size', type=int, default=6)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--eval_steps', type=int, default=50)
    
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--resume_ckpt_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="./outputs")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--push_to_hub', action='store_true', help='push to hub')
    parser.add_argument('--push_to_blob', action='store_true', help='push to blob storage')
    return parser.parse_args()

args = parse_args()
init_distributed_mode()

######################## OUTPUT PATH ########################
date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
dataset_str = '-'.join([d.split('/')[-1] for d in args.datasets])
model_name = args.assistant_model_name.split("/")[-1]
if model_name.startswith('checkpoint'):
   model_name =  args.assistant_model_name.split("/")[-2] + '_' + model_name
model_name = 'dpo_' + model_name + f'_epoch-{args.num_train_epochs}' + \
             f"_lr-{args.learning_rate}_gap-{args.minimum_gap}_{date_str}"

output_dir = osp.join(args.output_dir, dataset_str, model_name)
os.makedirs(output_dir, exist_ok=True)

if os.environ.get('LOCAL_RANK', '0') == '0':
    with open(osp.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

######################## LOAD DATASETS ########################
args.probs = [p / sum(args.probs) for p in args.probs]
train_dataset, eval_dataset = split_train_dev_datasets(args.datasets, 
                                                       is_dpo=True,
                                                       to_sft_dataset=False,
                                                       minimum_gap=args.minimum_gap,
                                                       n_eval_per_dataset=args.n_eval_per_dataset, 
                                                       probs=args.probs, 
                                                       add_system_prompt=True,
                                                       return_eval_as_dict=False,
                                                       seed=args.seed)

######################## MODEL ########################
# PEFT config
peft_config = LoraConfig(
    r=32, # 64
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    init_lora_weights="gaussian",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
)
is_unsloth_model = is_unsloth_model_auto(args.assistant_model_name)
model, tokenizer = load_model_and_tokenizer(args.assistant_model_name, 
                                            max_new_tokens=args.max_new_tokens, 
                                            peft_config=peft_config,
                                            eval=False)
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
            "stage": 2, 
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

if os.environ.get('LOCAL_RANK', '0') == '0':
    wandb.init(
	    project="interactivity", 
	    entity="dsp-team",
        name=keep_levels(output_dir, 3),
	    config=args,
        save_code=True,
        job_type='train'
    )

# Args
train_args = DPOConfig(
    beta=0.1, # The beta factor in DPO loss. Higher beta means less divergence
    loss_type="sigmoid", # The loss type for DPO.
    logging_steps=5,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    optim="adamw_torch",
    report_to="wandb",
    do_eval=True,
    eval_steps=args.eval_steps, 
    save_strategy='epoch',
    eval_strategy="steps",
    gradient_checkpointing=True,  
    lr_scheduler_type="cosine",
    metric_for_best_model="eval_loss",
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_train_epochs,
    save_total_limit=args.save_total_limit,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    max_length=args.max_new_tokens,
    max_prompt_length=args.max_prompt_length,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    run_name=keep_levels(output_dir, 3),
    output_dir=output_dir,
    deepspeed=ds_config, 
    fp16=not is_bfloat16_supported() if is_unsloth_model else False,
    bf16=is_bfloat16_supported() if is_unsloth_model else False,   
)

######################## PROCESS DATASETS ########################
meta_info = get_meta_info_from_model_name(args.assistant_model_name)
# tokenizer.chat_template = meta_info['chat_template'] # only for mistral

def process(row):
    prompt_chosen = tokenizer.apply_chat_template(row["prompt"] + [{'role': 'assistant', 'content': row["chosen"]}], tokenize=False)
    row["prompt"] = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
    row["chosen"] = row["chosen"].strip() + meta_info['eos_token']
    row["rejected"] = row["rejected"].strip() + meta_info['eos_token']
    assert prompt_chosen == row["prompt"] + row["chosen"]
    return row

train_dataset = train_dataset.map(process, load_from_cache_file=False)
eval_dataset = eval_dataset.map(process, load_from_cache_file=False)

######################## TRAINING ########################
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    # peft_config=peft_config,
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
    trainer.model.push_to_hub(f'{args.hf_org}/dpo-offline-model-{surfix}', private=True)
    tokenizer.push_to_hub(f'{args.hf_org}/dpo-offline-tokenizer-{surfix}', private=True)
    trainer.push_to_hub(f'{args.hf_org}/dpo-offline-trainer-{surfix}', private=True)

if args.push_to_blob:
    upload_dir_to_blob(output_dir)
wandb.finish()