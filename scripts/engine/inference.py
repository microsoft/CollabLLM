#!/usr/bin/env python3
"""
Parallel multiturn inference with per-example metric collection.

Example run:
Inference for the base model:
On medium dataset:
-------
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=56900 -m scripts.engine.inference \
    --dataset_name medium \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir outputs/base/collabllm-multiturn-medium/inference-user4o \
    --eval_metric_names "document->bleu" "interactivity" "token_amount" \
    --user_generation_kwargs '{"model": "gpt-4o"}' \
    --assistant_generation_kwargs '{"model": "meta-llama/Llama-3.1-8B-Instruct", "temperature": 0.8}' \
    --eval_generation_kwargs '{"model": "claude-3-5-sonnet-latest"}' \
    --eval_size 20

On math-hard dataset:
-------
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master_port=56900 -m scripts.engine.inference \
    --dataset_name math-hard \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir outputs/base/collabllm-multiturn-math-hard/inference-4o \
    --eval_metric_names accuracy interactivity token_amount \
    --user_generation_kwargs '{"model": "gpt-4o"}' \
    --assistant_generation_kwargs '{"model": "meta-llama/Llama-3.1-8B-Instruct", "temperature": 0.6}' \
    --eval_generation_kwargs '{"model": "claude-3-5-sonnet-latest"}' \
    --eval_size 50

Inference for collallm models:
On medium dataset:
-------
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=56900 -m scripts.engine.inference \
    --dataset_name medium \
    --model_name outputs/offline_dpo_from_base/collabllm-multiturn-medium/ \
    --output_dir outputs/offline_dpo_from_base/collabllm-multiturn-medium/inference-user4o \
    --eval_metric_names "document->bleu" "interactivity" "token_amount" \
    --user_generation_kwargs '{"model": "gpt-4o"}' \
    --assistant_generation_kwargs '{"model": "offline_dpo_from_base-medium-Llama-3.1-8B-Instruct", "temperature": 0.8}' \
    --eval_generation_kwargs '{"model": "claude-3-5-sonnet-latest"}' \
    --eval_size 20 \
    --use_lora \
    --add_system_prompt

On math-hard dataset:
-------
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=56900 -m scripts.engine.inference \
    --dataset_name math-hard \
    --model_name outputs/offline_dpo_from_base/collabllm-multiturn-math-hard/ \
    --output_dir outputs/offline_dpo_from_base/collabllm-multiturn-math-hard/inference-4o \
    --eval_metric_names accuracy interactivity token_amount \
    --user_generation_kwargs '{"model": "gpt-4o"}' \
    --assistant_generation_kwargs '{"model": "offline_dpo_from_base-math-hard-Llama-3.1-8B-Instruct", "temperature": 0.6}' \
    --eval_generation_kwargs '{"model": "claude-3-5-sonnet-latest"}' \
    --eval_size 50 \
    --use_lora \
    --add_system_prompt
"""
import os
import json
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist

from dotenv import load_dotenv
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model

from collabllm.reward import multiturn_aware_reward
from collabllm.simulation import ChatSessionSimulator
from examples.single_turn_ds import datasets_info
from examples.metrics import *

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser("Distributed multiturn inference")

    # LoRA config
    p.add_argument("--peft_r", type=int, default=32)
    p.add_argument("--peft_alpha", type=int, default=16)
    p.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )

    # Evaluation + model
    p.add_argument("--eval_size", type=int, default=500)
    p.add_argument("--max_new_tokens", type=int, default=2048) 
    p.add_argument("--max_model_len", type=int, default=8196)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--add_system_prompt", action="store_true", default=False)
    p.add_argument("--max_metric_workers", type=int, default=4)
    p.add_argument("--max_turns", type=int, default=14)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--eval_metric_names", nargs="+", required=True)
    p.add_argument("--eval_generation_kwargs", type=json.loads, default={})
    p.add_argument("--user_generation_kwargs", type=json.loads, default={})
    p.add_argument("--assistant_generation_kwargs", type=json.loads, default={})
    p.add_argument("--gpu_memory_utilization", type=float, default=0.8)

    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--use_4bit", action="store_true", default=False)

    return p.parse_args()



def load_model_and_tokenizer(
    model_name: str,
    bnb_cfg: Optional[BitsAndBytesConfig],
    lora_cfg: Optional[LoraConfig],
    device: str = "cuda",
    is_eval: bool = False,
    gpu_memory_utilization: float = 0.8,
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
            trust_remote_code=True
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
            max_model_len=max_model_len
        )
    except ImportError:
        vllm_base_model = None
    
    return model.eval(), tok, vllm_base_model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Distributed init
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    # Load data and slice
    dataset_cls = datasets_info[args.dataset_name]["class"]
    task_desc = datasets_info[args.dataset_name]["task_desc"]
    dataset = dataset_cls().to_hf_dataset()
    full_testset = dataset["test"].select(range(args.eval_size)) if args.eval_size > 0 else dataset["test"]
    total_size = len(full_testset)
    shard_size = total_size // world_size
    start = rank * shard_size
    end = total_size if rank == world_size - 1 else (rank + 1) * shard_size
    testset = full_testset.select(range(start, end))

    lora_cfg = LoraConfig(
        r=args.peft_r,
        lora_alpha=args.peft_alpha,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
        target_modules=args.target_modules.split(","),
    ) if args.use_lora else None

    # Bits-and-bytes
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if args.use_4bit else None

    torch.cuda.empty_cache()

    # Load model
    model, tok, vllm = load_model_and_tokenizer(
        model_name=args.model_name,
        lora_cfg=lora_cfg,
        bnb_cfg=bnb_cfg, # No quantization for eval
        is_eval=True,
        max_model_len=args.max_model_len
    )

    ensembled_kwargs = {
        "local_model": model,
        "local_tokenizer": tok,
        "vllm_base_model": vllm,
        "assistant_generation_kwargs": args.assistant_generation_kwargs,
        "user_generation_kwargs": args.user_generation_kwargs,
    }
    # Run local inference
    local_results = []
    for ex in tqdm(testset, desc=f"Rank {rank} processing examples"):
        chat_history = ChatSessionSimulator().run_chat_simulation(
            num_samples=1, # sample one conversation
            chat_history=None,
            max_new_turns=args.max_turns,
            task_desc=task_desc,
            single_turn_prompt=ex["single_turn_prompt"],
            add_system_prompt_ratio=1.0 if args.add_system_prompt else 0.0,
            verbose=False,
            **ensembled_kwargs
        )
        chat_history = chat_history[0]

        metrics = multiturn_aware_reward(
            num_samples=1, 
            max_new_turns=0, # evaluate the current chat
            task_desc=task_desc,
            single_turn_prompt=ex["single_turn_prompt"],
            single_turn_completion=ex["single_turn_completion"],
            metadata=ex["single_turn_metadata"],
            reward_generation_kwargs=args.eval_generation_kwargs,
            chat_history=chat_history,
            metric_names=args.eval_metric_names,
            metric_weights=[0] * len(args.eval_metric_names),
            max_metric_workers=args.max_metric_workers,
            **ensembled_kwargs
        )

        metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
        local_results.append((metrics, chat_history))

    # Gather full results across all ranks
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, local_results)

    if rank == 0:
        merged = [entry[0] for shard in gathered_results for entry in shard]
        all_pairs = [entry for shard in gathered_results for entry in shard]

        # Also log global average
        metric_names = args.eval_metric_names
        metrics = {k: [] for k in metric_names}
        for item in merged:
            for k in metric_names:
                metrics[k].append(item[k])
        avg_metrics = {k: float(np.mean(metrics[k])) for k in metric_names}
        avg_metrics.update(**vars(args))

        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump(avg_metrics, f, indent=2)
        
        with open(os.path.join(args.output_dir, "eval_details.json"), "w") as f:
            json.dump(all_pairs, f, indent=2)

    dist.destroy_process_group()

if __name__ == "__main__":
    load_dotenv(".env")
    main()
