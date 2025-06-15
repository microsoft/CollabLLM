# IN PROGRESS
    
#!/usr/bin/env python3
"""
PPO training script for multiturn conversation models.
Example usage:
ENABLE_COLLABLLM_LOGGING=0 LLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --master_port=56500 --nnodes=1 --nproc_per_node=8 -m scripts.train.ppo \
    --dataset_name math-hard \
    --metric_names "accuracy" "interactivity" "token_amount" \
    --metric_weights 1 1 -0.5 \
    --user_generation_kwargs '{"model": "gpt-4o-mini"}' \
    --assistant_generation_kwargs '{"model": "sft-math-hard-Llama-3.1-8B-Instruct", "temperature": 0.6}' \
    --reward_generation_kwargs '{"model": "claude-3-5-sonnet-latest"}' \
    --dataset_repo collabllm/collabllm-multiturn-math-hard \
    --model_name outputs/sft/collabllm-multiturn-math-hard \
    --output_dir outputs/ppo/collabllm-multiturn-math-hard \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 10 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --gpu_memory_utilization 0.6 \
    --logging_steps 1 \
    --wandb_entity dsp-team \
    --wandb_project collabllm \
    --num_samples 3 \
    --max_new_turns 4 \
    --max_metric_workers 2 \
    --use_4bit
"""
from __future__ import annotations

import argparse, os, json
import torch.distributed as dist
import wandb
import hashlib
from typing import Tuple, Optional
from dotenv import load_dotenv
from datetime import timedelta
import numpy as np
import copy
from tqdm import tqdm

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model
from collabllm.datasets.multiturn import MultiturnDataset
from collabllm.reward import multiturn_aware_reward
from examples.single_turn_ds import datasets_info
from collabllm.simulation import ChatSessionSimulator
from examples.metrics import *

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Parameter-free multiturn PPO trainer")

    # Data / paths
    p.add_argument("--dataset_repo", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--metric_names", nargs="+", required=True)
    p.add_argument("--user_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--assistant_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--reward_generation_kwargs", type=json.loads, default="{}")
    p.add_argument("--metric_weights", type=float, nargs="+", default=None)
    p.add_argument("--max_new_turns", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=3)

    p.add_argument("--output_dir",   type=str, required=True)
    p.add_argument("--resume_ckpt_dir", type=str, default=None)

    # Base / adapter models
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--ref_model_name", type=str, default=None)
    p.add_argument("--peft_r",     type=int,   default=32)
    p.add_argument("--peft_alpha", type=int,   default=16)
    p.add_argument("--peft_dropout", type=float, default=0.1)
    p.add_argument("--target_modules",
                   type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # PPO specific
    p.add_argument("--ppo_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--mini_batch_size", type=int, default=32)
    p.add_argument("--use_score_scaling", action="store_true", default=True)
    p.add_argument("--use_score_norm", action="store_true", default=False)

    # Optim & schedule
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=8196)
    p.add_argument("--max_new_tokens", type=int, default=2048) 
    p.add_argument("--max_metric_workers", type=int, default=4)
    p.add_argument("--window_size", type=int, default=2)
    
    # Generation parameters
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--no_repeat_ngram_size", type=int, default=10)
    
    # Precision / hardware
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--use_4bit", action="store_true", default=False)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.6)

    # Tracking
    p.add_argument("--wandb_project", type=str)
    p.add_argument("--wandb_entity",  type=str)
    p.add_argument("--push_to_hub",   action="store_true")
    p.add_argument("--hf_org",        type=str)
    p.add_argument("--debug", action="store_true")

    # Optional JSON/YAML override
    p.add_argument("--config_file", type=str)

    args = p.parse_args()
    if args.config_file:
        with open(args.config_file) as f:
            override = json.load(f) if args.config_file.endswith(".json") else \
                       __import__("yaml").safe_load(f)
        for k, v in override.items():
            setattr(args, k, v)
    return args

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(
    model_name: str,
    bnb_cfg: Optional[BitsAndBytesConfig],
    lora_cfg: Optional[LoraConfig],
    device: str = "cuda",
    is_eval: bool = False,
    gpu_memory_utilization: float = 0.6,
    max_model_len: int = 8196
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    try:
        pc = PeftConfig.from_pretrained(model_name)
        base = AutoModelForCausalLM.from_pretrained(
            pc.base_model_name_or_path,
            device_map={"": device},
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_name, is_trainable=not is_eval)
        tok = AutoTokenizer.from_pretrained(pc.base_model_name_or_path, trust_remote_code=True)
        base_model_name = pc.base_model_name_or_path
    except Exception:
        logger.error(f"Failed to load PeftConfig for {model_name}, loading as base model.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": device},
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if lora_cfg:
            model = get_peft_model(model, lora_cfg)
        base_model_name = model_name

    tok.padding_side, tok.pad_token = ("left" if is_eval else "right"), tok.eos_token
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,}/{total:,} ({trainable/total:.2%})")

    try:
        from vllm import LLM

        vllm_base_model = LLM(
            model=base_model_name,
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            quantization="bitsandbytes" if bnb_cfg else None,
            enable_lora=True if lora_cfg else False,
            max_lora_rank=lora_cfg.r if lora_cfg else None,
            # Use `distributed_executor_backend="external_launcher"` so that
            # this llm engine/instance only creates one worker.
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    except ImportError:
        vllm_base_model = None
    return model, tok, vllm_base_model


def process_dataset(dataset, tokenizer, max_query_tokens, window_size):
    """Process dataset to create query tensors for PPO training"""
    from collections import defaultdict
    from datasets import Dataset
    
    def tokenize(x):
        even_indices = np.array(list(range(2, len(x["prompt"]), 2)))
        filtered_indices = even_indices[even_indices < len(even_indices) - window_size]
        
        if len(filtered_indices) == 0:
            filtered_indices = even_indices
        
        new_samples = []
        for input_size in filtered_indices:
            new_x = copy.deepcopy(x)
            new_x["chat_history"] = x["prompt"][:input_size]
            new_x["query"] = tokenizer.apply_chat_template(new_x["chat_history"], 
                                                           tokenize=False, 
                                                           add_generation_prompt=True)
            new_x["input_ids"] = tokenizer.encode(new_x["query"],
                                                  max_length=max_query_tokens,
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

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Important for initializing vllm base model per GPU
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    dist.init_process_group(backend='nccl', init_method=None)
    torch.cuda.set_device(local_rank)
    dist.barrier()

    # Dataset
    ds = MultiturnDataset(args.dataset_repo).to_inputs_dataset(eval_ratio=0.)

    # Bits-and-bytes
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if args.use_4bit else None

    # LoRA
    lora_cfg = LoraConfig(
        r=args.peft_r,
        lora_alpha=args.peft_alpha,
        lora_dropout=args.peft_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
        target_modules=args.target_modules.split(","),
    )

    # Load model
    model, tok, vllm = load_model_and_tokenizer(
        args.model_name,
        bnb_cfg=bnb_cfg,
        lora_cfg=lora_cfg,
        device=args.device,
        is_eval=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        model_class=AutoModelForCausalLMWithValueHead
    )
    
    # Load reference model if specified
    ref_model = None
    if args.ref_model_name:
        ref_model, _, _ = load_model_and_tokenizer(
            args.ref_model_name,
            bnb_cfg=bnb_cfg,
            lora_cfg=lora_cfg,
            device=args.device,
            is_eval=False,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            model_class=AutoModelForCausalLMWithValueHead
        )

    # Process dataset
    train_dataset = process_dataset(
        ds["train"], 
        tok, 
        args.max_model_len - args.max_new_tokens,
        args.window_size
    )

    # Generation kwargs
    generation_kwargs = {
        "min_length": -1,
        "do_sample": True,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tok.eos_token_id,
        "no_repeat_ngram_size": args.no_repeat_ngram_size
    }

    # W&B
    if args.wandb_project and os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.output_dir.replace("/", "_"),
            config=vars(args),
            save_code=True,
            job_type="debug" if args.debug else "train",
        )

    # PPO Config
    ppo_config = PPOConfig(
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        ppo_epochs=args.ppo_epochs,
        log_with='wandb',
        exp_name=args.output_dir.replace("/", "_"),
        model_name=args.model_name,
        remove_unused_columns=False,
        is_peft_model=True,
        use_score_scaling=args.use_score_scaling, 
        use_score_norm=args.use_score_norm
    )

    # PPO Trainer
    trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        config=ppo_config,
        dataset=train_dataset,
        tokenizer=tok,
        data_collator=collator
    )

    ######################## REWARD FUNCTION ########################
    def compute_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()
    
    str_prompt_to_multiturn_data_map = {}
    def process_prompt_mapping(row):
        str_prompt = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        str_prompt_to_multiturn_data_map.setdefault(
            compute_hash(str_prompt), 
            {k: row[k] for k in ["single_turn_prompt", "single_turn_completion", "single_turn_metadata", "prompt"]}
        )
        return row

    # Create mapping for reward computation
    ds["train"] = ds["train"].map(process_prompt_mapping, load_from_cache_file=False)
    
    collabllm_model_kwargs = {
        "local_model": model.pretrained_model if hasattr(model, 'pretrained_model') else model,
        "local_tokenizer": tok,
        "vllm_base_model": vllm,
    }

    def compute_rewards(prompts, responses):
        """Compute rewards for PPO training"""
        rewards = []
        for prompt, response in zip(prompts, responses):
            multiturn_data = str_prompt_to_multiturn_data_map.get(compute_hash(prompt))
            if multiturn_data is None:
                rewards.append(0.0)
                continue
                
            chat_history = multiturn_data["prompt"] + [{"role": "assistant", "content": response}]
            
            reward_info = multiturn_aware_reward(
                chat_history=chat_history,
                task_desc=datasets_info[args.dataset_name]["task_desc"],
                single_turn_prompt=multiturn_data["single_turn_prompt"],
                single_turn_completion=multiturn_data["single_turn_completion"],
                metadata=multiturn_data["single_turn_metadata"],
                metric_names=args.metric_names,
                metric_weights=args.metric_weights,
                user_generation_kwargs=args.user_generation_kwargs,
                assistant_generation_kwargs=args.assistant_generation_kwargs,
                reward_generation_kwargs=args.reward_generation_kwargs,
                num_samples=args.num_samples,
                max_new_turns=args.max_new_turns,
                max_metric_workers=args.max_metric_workers,
                **collabllm_model_kwargs
            )
            rewards.append(np.mean(reward_info["MR"]))
        return rewards

    ######################## TRAINING LOOP ########################
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

    step = 0
    for batch in tqdm(dataloader(), total=total_steps):
        step += 1
        
        # Extract data from batch
        query_tensors = batch["input_ids"]
        prompts = [tok.decode(query, skip_special_tokens=True) for query in query_tensors]
        
        # Generate responses
        model.train()
        response_tensors = trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
        responses = tok.batch_decode(response_tensors, skip_special_tokens=True)
        
        # Compute rewards
        rewards = compute_rewards(prompts, responses)
        
        print(f"\n{'='*50} Step {step} {'='*50}")
        for i, (prompt, response, reward) in enumerate(zip(prompts, responses, rewards)):
            print(f"Sample {i+1}:")
            print(f"Prompt: {prompt[-200:]}")  # Show last 200 chars
            print(f"Response: {response}")
            print(f"Reward: {reward}")
            print("-" * 100)
        
        # Update batch with rewards and responses
        batch["responses"] = responses
        batch["rewards"] = rewards
        
        # Run PPO step
        stats = trainer.step(query_tensors, response_tensors, rewards)
        trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "responses", "rewards"])
        
        torch.cuda.empty_cache()
        
        # Save checkpoint
        if step % args.save_steps == 0:
            trainer.save_pretrained(os.path.join(args.output_dir, f"step_{step}"))
            tok.save_pretrained(os.path.join(args.output_dir, f"step_{step}"))
        
        print(f"[Rank: {local_rank}] Step: {step}, Mean Reward: {np.mean(rewards):.4f}")

    # Final save
    trainer.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hf_org:
        repo = f"ppo-{args.dataset_repo.replace('/', '_')}"
        trainer.push_to_hub(f"{args.hf_org}/{repo}", private=True)
        tok.push_to_hub(f"{args.hf_org}/{repo}", private=True)

    if args.wandb_project:
        wandb.finish()

if __name__ == "__main__":
    load_dotenv(".env")
    main()
