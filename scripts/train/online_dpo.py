#!/usr/bin/env python3
"""
Preparation: To run the following, you need to generate multiturn data from `scripts.engine.build_dataset`

DPO train a causal-LM + LoRA adapter on a multi-turn dataset.
Example (on 8 NVIDIA A100-SXM4-80GB GPUs):
-------
ENABLE_COLLABLLM_LOGGING=0 LLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    torchrun --master_port=56500 --nnodes=1 --nproc_per_node=8 -m scripts.train.online_dpo \
    --dataset_name math-hard \
    --metric_names "accuracy" "interactivity" "token_amount" \
    --metric_weights 1 1 -0.5 \
    --user_generation_kwargs '{"model": "gpt-4o-mini"}' \
    --assistant_generation_kwargs '{"model": "sft-math-hard-Llama-3.1-8B-Instruct", "temperature": 0.6}' \
    --reward_generation_kwargs '{"model": "claude-3-5-sonnet-latest"}' \
    --dataset_repo collabllm/collabllm-multiturn-math-hard \
    --model_name outputs/sft/collabllm-multiturn-math-hard \
    --output_dir outputs/online_dpo/collabllm-multiturn-math-hard \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_total_limit 10 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --gpu_memory_utilization 0.6 \
    --eval_steps 1 \
    --logging_steps 1 \
    --wandb_entity dsp-team \
    --wandb_project collabllm \
    --num_samples 3 \
    --max_new_turns 4 \
    --max_metric_workers 2 \
    --use_lora
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

from trl import OnlineDPOConfig, OnlineDPOTrainer
from trl.trainer.judges import BasePairwiseJudge

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
    p = argparse.ArgumentParser("Parameter-free multiturn DPO trainer")

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
    p.add_argument("--peft_r",     type=int,   default=32)
    p.add_argument("--peft_alpha", type=int,   default=16)
    p.add_argument("--peft_dropout", type=float, default=0.1)
    p.add_argument("--target_modules",
                   type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Optim & schedule
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0)        
    p.add_argument("--logging_steps", type=int, default=1)         
    p.add_argument("--max_model_len", type=int, default=8196)
    p.add_argument("--max_new_tokens", type=int, default=2048) 
    p.add_argument("--minimum_gap", type=float, default=0.1) 
    p.add_argument("--max_metric_workers", type=int, default=4)
    
    # Precision / hardware
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_4bit", action="store_true", default=False)
    p.add_argument("--use_lora", action="store_true", default=False)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.6)

    # Tracking
    p.add_argument("--wandb_project", type=str)
    p.add_argument("--wandb_entity",  type=str)
    p.add_argument("--push_to_hub",   action="store_true")
    p.add_argument("--hf_org",        type=str)

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
            max_model_len=max_model_len
        )
    except ImportError:
        vllm_base_model = None
    return model, tok, vllm_base_model

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Imporant for initializing vllm base model per GPU
    local_rank = int(os.environ['LOCAL_RANK'])
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
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights="gaussian",
        target_modules=args.target_modules.split(","),
    ) if args.use_lora else None

    # Load model
    model, tok, vllm = load_model_and_tokenizer(
        args.model_name,
        bnb_cfg=bnb_cfg,
        lora_cfg=lora_cfg,
        device=args.device,
        is_eval=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )

    # DeepSpeed zero
    ds_cfg = {
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

    # Trainer config
    train_args = OnlineDPOConfig(
        beta=0.1,
        loss_type="sigmoid",
        max_grad_norm=1.0,
        optim="adamw_torch",
        report_to="wandb",
        do_eval=False,
        eval_steps=args.eval_steps, 
        save_strategy='steps',
        save_steps=1,
        eval_strategy="no",
        gradient_checkpointing=True,  
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        max_new_tokens=args.max_new_tokens, 
        max_length=args.max_model_len,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        run_name=args.output_dir,
        output_dir=args.output_dir,
        deepspeed=ds_cfg, 
        use_vllm=False,
        fp16=not torch.cuda.is_bf16_supported(), 
        bf16=torch.cuda.is_bf16_supported(),
        save_total_limit=args.save_total_limit,
    )

    # W&B
    if args.wandb_project and os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.output_dir.replace("/", "_"),
            config=train_args.to_dict(),
            save_code=True,
            job_type="train",
        )

    ######################## JUDGE ########################
    def compute_hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()
    
    str_prompt_to_multiturn_data_map = {}
    def process(row):
        str_prompt = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        str_prompt_to_multiturn_data_map.setdefault(
            compute_hash(str_prompt), 
            {k: row[k] for k in ["single_turn_prompt", "single_turn_completion", "single_turn_metadata", "prompt"]}
        )
        row["prompt"] = str_prompt
        return row

    ds["train"] = ds["train"].map(process, load_from_cache_file=False)
    collabllm_model_kwargs = {
        "local_model": model,
        "local_tokenizer": tok,
        "vllm_base_model": vllm,
    }
    class MultiturnRewardJudge(BasePairwiseJudge):
        def judge(self, prompts, completion_pairs, shuffle_order=False):
            rank_of_the_first_completion = []
            for prompt, completion_pair in zip(prompts, completion_pairs):
                multiturn_data = str_prompt_to_multiturn_data_map.get(compute_hash(prompt))
                chat_histories = [
                    multiturn_data["prompt"] + 
                    [{"role": "assistant", "content": completion}] for completion in completion_pair
                ]
                pair_rewards = []
                for chat_history in chat_histories:
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
                    pair_rewards.append(np.mean(reward_info["MR"]))
                rank_of_the_first_completion.append(np.argmax(pair_rewards).item())
                logger.info(
                    f"\n({local_rank=}) [Response 1] {completion_pair[0]}\n\n[Response 2] {completion_pair[1]}\nRewards: {pair_rewards}\n"
                )
            return torch.tensor(rank_of_the_first_completion)
            
                
    judge = MultiturnRewardJudge()  

    ######################## [Hack] Override vLLM for OnlineDPOTrainer ########################
    def _generate_vllm(self, model, prompts):
        eos_token_id = tok.eos_token_id
        pad_token_id = tok.pad_token_id
        
        generation_kwargs = copy.deepcopy(args.assistant_generation_kwargs)
        generation_kwargs.update({"n": 2, "top_k": 50, "top_p": 1.0, "detokenize": False})

        outputs = ChatSessionSimulator()._batch_generate_with_vllm(
            batch_messages=prompts,
            vllm_base_model=vllm,
            local_model=model,
            model_name=generation_kwargs.get("model"),
            generation_kwargs=generation_kwargs,
            return_outputs=True
        )

        completion_ids = [list(output.outputs[i].token_ids) for i in range(2) for output in outputs]
        prompt_ids = [list(output.prompt_token_ids) for _ in range(2) for output in outputs]

        # Create mask and pad the prompt and completion
        max_prompt_length = max(len(ids) for ids in prompt_ids)
        prompt_mask = [[0] * (max_prompt_length - len(ids)) + [1] * len(ids) for ids in prompt_ids]
        prompt_ids = [[pad_token_id] * (max_prompt_length - len(ids)) + ids for ids in prompt_ids]
        max_tokens = args.max_new_tokens
        completion_mask = [[1] * len(ids) + [0] * (max_tokens - len(ids)) for ids in completion_ids]
        completion_ids = [
            ids + [eos_token_id] if ids[-1] != eos_token_id and len(ids) < max_tokens else ids
            for ids in completion_ids
        ]
        completion_ids = [ids + [pad_token_id] * (max_tokens - len(ids)) for ids in completion_ids]

        # Convert to tensors
        prompt_ids = torch.tensor(prompt_ids, device=self.accelerator.device)
        prompt_mask = torch.tensor(prompt_mask, device=self.accelerator.device)
        completion_ids = torch.tensor(completion_ids, device=self.accelerator.device)
        completion_mask = torch.tensor(completion_mask, device=self.accelerator.device)

        return prompt_ids, prompt_mask, completion_ids, completion_mask

    OnlineDPOTrainer._generate_vllm = _generate_vllm

    ######################## Trainer ########################
    trainer = OnlineDPOTrainer(
        model=model,
        judge=judge,
        train_dataset=ds["train"],
        processing_class=tok,
        args=train_args,
    )
    trainer.args.use_vllm = True if vllm else False

    trainer.train(resume_from_checkpoint=args.resume_ckpt_dir)

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hf_org:
        repo = f"offline_dpo-{args.dataset_repo.replace('/', '_')}"
        trainer.model.push_to_hub(f"{args.hf_org}/{repo}", private=True)
        tok.push_to_hub(f"{args.hf_org}/{repo}", private=True)

    if args.wandb_project:
        wandb.finish()

if __name__ == "__main__":
    load_dotenv(".env")

    main()
