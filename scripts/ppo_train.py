from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import os
import os.path as osp
import json
import argparse
import torch
import wandb
from rich import print
import sys
import random
import numpy as np
from tqdm import tqdm
import re
import copy
import datetime
from datasets import Dataset
from collections import defaultdict
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sys.path.append('.')
from collabllm.utils.distributed import init_distributed_mode
from collabllm.datasets import datasets_info, split_train_dev_datasets
from collabllm.utils.blob import upload_dir_to_blob
from collabllm.models import is_unsloth_model_auto, get_meta_info_from_model_name
from collabllm.models.load import load_model_and_tokenizer
from collabllm.core.multithread import get_multiturn_rewards
from collabllm.utils.dir import keep_levels



# args model_name name
def parse_args():
    parser = argparse.ArgumentParser()
    def list_of_strings(arg): return arg.split(',')
    def list_of_integers(arg): return [int(x) for x in arg.split(',')]
    parser.add_argument('--datasets', type=list_of_strings, default='ppc')
    parser.add_argument('--probs', type=list_of_integers, default='1')
    parser.add_argument('--n_eval_per_dataset', type=int, default=200) 

    parser.add_argument('--output_dir', type=str, default="./outputs")

    parser.add_argument('--reward_model', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--assistant_model_name', type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--summarization_model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--user_model_name', type=str, default='gpt-4o')
    parser.add_argument('--ref_model_name', type=str, default=None) 
    
    parser.add_argument('--reward', type=str, default="bert_score", 
                        choices=['bert_score', 'llm_reward'])
    parser.add_argument('--ppo_epochs', type=int, default=1)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=5e-4)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--save_total_limit', type=int, default=5)
    parser.add_argument('--window_size', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mini_batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--max_query_tokens', type=int, default=4096)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--save_step', type=int, default=50)

    # generation kwargs
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--max_new_tokens', type=int, default=768)
    
    parser.add_argument('--llm_rw_weight', type=float, default=1)
    parser.add_argument('--task_weight', type=float, default=10)
    parser.add_argument('--cost_weight', type=float, default=1e-3)
    parser.add_argument('--max_workers', type=int, default=30)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--push_to_hub', action='store_true', help='push to hub')
    parser.add_argument('--push_to_blob', action='store_true', help='push to blob storage')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--hf_org', type=str, default='org_name')
    parser.add_argument("--use_vllm", action="store_true", help="use vllm")
    return parser.parse_args()


args = parse_args()
init_distributed_mode()
######################## OUTPUT PATH ########################
date_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
dataset_str = '-'.join([d.split('/')[-1] for d in args.datasets])
model_name = args.assistant_model_name.split("/")[-1]
if model_name.startswith('checkpoint'):
   model_name = args.assistant_model_name.split("/")[-2] + '_' + model_name
model_name = 'ppo_' + model_name + f'_{date_str}'
output_dir = osp.join(args.output_dir, dataset_str, model_name)
os.makedirs(output_dir, exist_ok=True)

if os.environ.get('LOCAL_RANK', '0') == '0':
    print(args)
    with open(osp.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

assistant_generation_kwargs = {
   "model": args.assistant_model_name,
   "top_p": args.top_p,
   "temperature": args.temperature,
   "max_new_tokens": args.max_new_tokens
}

reward_generation_kwargs = {
   "model": args.reward_model,
   "top_p": 0.9,
   "temperature": 0.2,
   "max_new_tokens": 4096
}

user_generation_kwargs = {
    "model": args.user_model_name,
    "top_p": 1.0,
    "temperature": 1.0,
    "max_new_tokens": 4096,
    "json_object": True
}
######################## LOAD DATASETS ########################
args.probs = [p / sum(args.probs) for p in args.probs]

train_dataset, _ = split_train_dev_datasets(args.datasets, 
                                            is_multiturn=True,
                                            is_dpo=True,
                                            n_eval_per_dataset=args.n_eval_per_dataset, 
                                            probs=args.probs, 
                                            add_system_prompt=True,
                                            seed=args.seed,
                                            to_sft_dataset=True)

######################## LOAD MODEL AND TOKENIZER ########################
is_unsloth_model = is_unsloth_model_auto(args.assistant_model_name)
local_rank = os.getenv("LOCAL_RANK")
device_string = "cuda:" + str(local_rank)
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
model, tokenizer = load_model_and_tokenizer(
    model_name=args.assistant_model_name,
    max_new_tokens=args.max_new_tokens,
    peft_config=peft_config,
    model_class=AutoModelForCausalLMWithValueHead,
    eval=False
)
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
model = model.bfloat16().to(device_string)
if args.ref_model_name is not None:
    ref_model, _ = load_model_and_tokenizer(
        model_name=args.ref_model_name,
        max_new_tokens=args.max_new_tokens,
        peft_config=peft_config,
        model_class=AutoModelForCausalLMWithValueHead,
        eval=False
    )
    ref_model = ref_model.bfloat16().to(device_string)
else:
    ref_model = None

print('padding_side before', tokenizer.padding_side)
print('len(tokenizer) before', len(tokenizer))
print('pad_token before', tokenizer.pad_token)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token
print('padding_side after', tokenizer.padding_side)
print('len(tokenizer) after', len(tokenizer))
print('pad_token after', tokenizer.pad_token)
print('eos_token', tokenizer.eos_token)

generation_kwargs = {
    "min_length": -1,
    "do_sample": True,
    "top_p": args.top_p,
    "temperature": args.temperature,
    "max_new_tokens": args.max_new_tokens,
    "pad_token_id": tokenizer.eos_token_id,
    "no_repeat_ngram_size": 10
}
######################## PROCESS DATASETS ########################
meta_info = get_meta_info_from_model_name(args.assistant_model_name)
# ds_name = re.search(r'collabllm-(.*?)-dpo', args.datasets[0]).group(1)
ds_name = re.search(r'collabllm-([a-zA-Z0-9_-]+)', args.datasets[0]).group(1)
task_name = datasets_info[ds_name]['task']

def process_dataset(dataset):
    def tokenize(x):
        even_indices = np.array(list(range(2, len(x["chat"]), 2)))
        filtered_indices = even_indices[even_indices < len(even_indices) - args.window_size]
        
        if len(filtered_indices) == 0:
            filtered_indices = even_indices
        
        new_samples = []
        for input_size in filtered_indices:
            new_x = copy.deepcopy(x)
            new_x["chat_history"] = x["chat"][:input_size]
            new_x["query"] = tokenizer.apply_chat_template(new_x["chat_history"], 
                                                           tokenize=False, 
                                                           add_generation_prompt=True)
            new_x["input_ids"] = tokenizer.encode(new_x["query"],
                                                  max_length=args.max_query_tokens,
                                                  truncation=True)
            new_samples.append(new_x)
        
        return new_samples

    processed_slices = defaultdict(list)
    
    for item in dataset:
        tokenized_samples = tokenize(item)
        for sample in tokenized_samples:
            for key, value in sample.items():
                processed_slices[key].append(value)
    
    new_dataset = Dataset.from_dict(dict(processed_slices))
    new_dataset.set_format(type="torch")
    return new_dataset

train_dataset = process_dataset(train_dataset)
def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

######################## CONFIG TRAINER AND LOADER ########################
if os.environ.get('LOCAL_RANK', '0') == '0':
    wandb.init(
	    project="interactivity", 
	    entity="dsp-team",
        name=keep_levels(output_dir, 3),
	    config=args,
        save_code=True,
        job_type='debug' if args.debug else 'train'
    )

ppo_config = PPOConfig(
    batch_size=args.batch_size,
    mini_batch_size=args.mini_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    ppo_epochs=args.ppo_epochs,
    log_with='wandb',
    exp_name=model_name,
    model_name=args.assistant_model_name,
    remove_unused_columns=False,
    is_peft_model=True if not is_unsloth_model else False,
    use_score_scaling=True, 
    use_score_norm=False
)
trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    config=ppo_config,
    dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=collator
)

def dataloader():
    epoch = 0
    dataloader_iter = iter(trainer.dataloader)
    while True:
        try:
            yield next(dataloader_iter)
        except StopIteration:
            epoch += 1
            if epoch >= args.num_train_epochs:
                break
            dataloader_iter = iter(trainer.dataloader)
            yield next(dataloader_iter)

total_steps = sum(1 for _ in tqdm(dataloader()))
print(f'************** total steps = {total_steps} **************')

######################## TRAINING ########################
if args.use_vllm:
    # Load base model with vllm
    vllm_base_model = LLM(
        model=args.assistant_model_name,
        dtype="bfloat16",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_lora=True,
        max_lora_rank=32,
    )


step = 0
# for batch in tqdm(trainer.dataloader):
for batch in tqdm(dataloader(), total=total_steps):
    step += 1
    single_turn_ds = batch["qa"]
    chat_histories = batch["chat_history"]
    query_tensors = batch["input_ids"]
    print(f'size of query_tensors {[len(q) for q in query_tensors]}')
    # Get response
    model.train()
    generate_ref_response = False
    if generate_ref_response:
        response_tensors, ref_response_tensors = trainer.generate(query_tensors, 
                                                                  return_prompt=False, 
                                                                  generate_ref_response=True, 
                                                                  **generation_kwargs)
        batch["responses"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        batch["ref_responses"] = tokenizer.batch_decode(ref_response_tensors, skip_special_tokens=True)
        batch["ref_rewards"], _ = get_multiturn_rewards(
                task_name=task_name, # assume tasks are the same for now
                single_turn_ds=single_turn_ds,
                chat_histories=chat_histories,
                prompt_method='none',
                responses=batch["ref_response"],
                max_workers=args.max_workers,
                num_samples=args.num_samples,
                llm_rw_weight=args.llm_rw_weight,
                window_size=args.window_size,
                task_weight=args.task_weight,
                cost_weight=args.cost_weight,
                user_generation_kwargs=user_generation_kwargs,
                assistant_generation_kwargs=assistant_generation_kwargs,
                reward_generation_kwargs=reward_generation_kwargs,
                local_model=model.pretrained_model, local_tokenizer=tokenizer,
                verbose=True
            )
    else:
        response_tensors = trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        batch["responses"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    rewards, reward_logs = get_multiturn_rewards(
            task_name=task_name,
            single_turn_ds=single_turn_ds,
            chat_histories=chat_histories,
            responses=batch["responses"],
            prompt_method='none',
            max_workers=args.max_workers,
            num_samples=args.num_samples,
            llm_rw_weight=args.llm_rw_weight,
            window_size=args.window_size,
            task_weight=args.task_weight,
            cost_weight=args.cost_weight,
            user_generation_kwargs=user_generation_kwargs,
            assistant_generation_kwargs=assistant_generation_kwargs,
            reward_generation_kwargs=reward_generation_kwargs,
            local_model=model.pretrained_model, local_tokenizer=tokenizer,
            verbose=True
        )
    
    for chat_history, response, reward in zip(chat_histories, batch["responses"], rewards):
        print("\n\n", "*" * 100)
        print(f"Chat History: {chat_history[1:]}\n=> Response: {response}")
        print(f"=> Reward: {reward}\n", "*" * 100, '\n\n')

    keys = list(reward_logs[0].keys())
    for key in keys:
        batch[key] = [log[key] for log in reward_logs]

    batch["rewards"] = rewards
    
    # Run PPO step
    model.train()
    stats = trainer.step(query_tensors, response_tensors, rewards)
    trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "responses", "rewards"] + keys)
    torch.cuda.empty_cache()
    
    ######################## SAVING ########################
    if step % args.save_step == 0:
        trainer.save_pretrained(osp.join(output_dir, f"step_{step}")) # save the LoRA adapters
        tokenizer.save_pretrained(osp.join(output_dir, f"step_{step}")) # save tokenizer
    print(f"[Rank: {int(os.environ['RANK'])}] Step: {step}, Rewards: {rewards} | Done!")

trainer.save_pretrained(output_dir) # save the LoRA adapters
tokenizer.save_pretrained(output_dir) # save tokenizer

if args.push_to_hub:
    surfix = output_dir.split("/")[-1].replace('_', '-')
    trainer.push_to_hub(f'{args.hf_org}/ppo-model-{surfix}', private=True)
    tokenizer.push_to_hub(f'{args.hf_org}/ppo-tokenizer-{surfix}', private=True)

if args.push_to_blob:
    upload_dir_to_blob(output_dir)
wandb.finish()